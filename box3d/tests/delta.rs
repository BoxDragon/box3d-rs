//! World delta rollback: restore + re-step reproduces the original run bit
//! for bit, ids follow the restored state, and mesh geometry a restored
//! shape points at stays alive.

#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;

const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: i32 = 4;

/// A ground slab and a small pile of boxes, some still falling.
fn pile() -> (World, Vec<BodyId>) {
    let mut w = World::new(&WorldDef::default()).unwrap();
    let g = w.create_body(&BodyDef::fixed(Pos::new(0.0, -0.5, 0.0)));
    w.create_box_shape(g, &ShapeDef::default(), Vec3::new(20.0, 0.5, 20.0));
    let mut bodies = Vec::new();
    for i in 0..24 {
        let (x, y) = ((i % 4) as f32 * 0.9 - 1.4, 0.5 + (i / 4) as f32 * 1.05);
        let b = w.create_body(&BodyDef::dynamic(Pos::new(x, y, 0.1 * (i % 3) as f32)));
        w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.4));
        bodies.push(b);
    }
    (w, bodies)
}

/// Pose and velocity bits of every body.
fn snapshot(w: &World, bodies: &[BodyId]) -> Vec<[u32; 10]> {
    bodies
        .iter()
        .map(|&b| {
            let (p, q) = w.body_transform(b);
            let v = w.body_linear_velocity(b);
            [
                (p.x as f32).to_bits(),
                (p.y as f32).to_bits(),
                (p.z as f32).to_bits(),
                q.x.to_bits(),
                q.y.to_bits(),
                q.z.to_bits(),
                q.w.to_bits(),
                v.x.to_bits(),
                v.y.to_bits(),
                v.z.to_bits(),
            ]
        })
        .collect()
}

#[test]
fn restore_and_restep_is_bit_identical() {
    let (mut w, bodies) = pile();
    w.enable_delta(16).unwrap();
    assert_eq!(w.delta().unwrap().newest_tick(), 0);

    let mut hashes = vec![w.state_hash()];
    let mut poses = vec![snapshot(&w, &bodies)];
    for t in 1..=30 {
        if t == 22 {
            // A mid-run API write belongs to the next capture.
            w.apply_linear_impulse_to_center(bodies[3], Vec3::new(400.0, 0.0, 0.0), true);
        }
        w.step(DT, SUBSTEPS);
        assert_eq!(w.capture_delta(), Ok(t));
        hashes.push(w.state_hash());
        poses.push(snapshot(&w, &bodies));
    }
    let d = w.delta().unwrap();
    assert_eq!((d.oldest_tick(), d.newest_tick()), (14, 30));
    assert!(d.stats().log_bytes > 0);
    assert_ne!(hashes[30], hashes[20], "the pile is moving");

    for &(t, again) in &[(20, 10), (25, 5), (30, 0), (14, 16)] {
        w.restore_delta(t).unwrap();
        assert_eq!(w.state_hash(), hashes[t as usize], "hash after restore to {t}");
        assert_eq!(snapshot(&w, &bodies), poses[t as usize], "poses after restore to {t}");
        for k in 1..=again {
            let tick = t + k;
            if tick == 22 {
                w.apply_linear_impulse_to_center(bodies[3], Vec3::new(400.0, 0.0, 0.0), true);
            }
            w.step(DT, SUBSTEPS);
            assert_eq!(w.capture_delta(), Ok(tick));
            assert_eq!(w.state_hash(), hashes[tick as usize], "hash at re-stepped {tick}");
            assert_eq!(snapshot(&w, &bodies), poses[tick as usize], "poses at re-stepped {tick}");
        }
    }
}

#[test]
fn restore_window_and_refusals() {
    let (mut w, _) = pile();
    assert_eq!(w.restore_delta(0), Err(Error::InvalidArgument("world has no delta tracker")));
    assert!(w.capture_delta().is_err());
    assert!(w.enable_delta(0).is_err());
    w.enable_delta(4).unwrap();
    assert!(w.enable_delta(4).is_err(), "one tracker per world");
    for _ in 0..10 {
        w.step(DT, SUBSTEPS);
        w.capture_delta().unwrap();
    }
    assert_eq!(w.restore_delta(5), Err(Error::RestoreRefused), "older than the window");
    assert_eq!(w.restore_delta(11), Err(Error::RestoreRefused), "not captured yet");
    w.restore_delta(6).unwrap();
    assert_eq!(w.delta().unwrap().newest_tick(), 6, "later ticks are discarded");
    assert_eq!(w.restore_delta(7), Err(Error::RestoreRefused));

    // Disable and re-enable: tracking restarts at tick 0.
    w.disable_delta();
    assert!(w.delta().is_none());
    w.enable_delta(4).unwrap();
    w.step(DT, SUBSTEPS);
    assert_eq!(w.capture_delta(), Ok(1));
    // Dropping a tracked world is fine.
    drop(w);
}

#[test]
fn ids_follow_the_restored_state() {
    let (mut w, bodies) = pile();
    w.enable_delta(8).unwrap();
    w.step(DT, SUBSTEPS);
    w.capture_delta().unwrap();

    let doomed = bodies[5];
    w.destroy_body(doomed);
    let born = w.create_body(&BodyDef::dynamic(Pos::new(5.0, 3.0, 0.0)));
    w.create_box_shape(born, &ShapeDef::default(), Vec3::splat(0.25));
    // Queued requests are undone by the restore too.
    w.queue_destroy_body(bodies[6]);
    w.step(DT, SUBSTEPS);
    w.capture_delta().unwrap();
    assert!(!w.is_body_valid(doomed) && w.is_body_valid(born) && !w.is_body_valid(bodies[6]));

    w.restore_delta(1).unwrap();
    assert!(w.is_body_valid(doomed), "destroyed after the tick: back, same id");
    assert!(!w.is_body_valid(born), "created after the tick: gone");
    assert!(w.is_body_valid(bodies[6]));
    w.queue_destroy_body(bodies[7]);
    w.restore_delta(1).unwrap();
    w.step(DT, SUBSTEPS);
    assert!(w.is_body_valid(bodies[7]), "restore cleared the destroy queue");
}

#[test]
fn restored_mesh_shapes_keep_their_geometry() {
    let mut w = World::new(&WorldDef::default()).unwrap();
    let ball = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 2.0, 0.0)));
    w.create_sphere_shape(ball, &ShapeDef::default(), Vec3::ZERO, 0.3);
    w.enable_delta(8).unwrap();

    // Tick 1: a mesh floor exists.
    let floor_mesh = Mesh::grid(8, 8, 1.0, true).unwrap();
    let floor = w.create_body(&BodyDef::fixed(Pos::new(-4.0, 0.0, -4.0)));
    let floor_shape = w.create_mesh_shape(floor, &ShapeDef::default(), &floor_mesh, Vec3::ONE);
    w.step(DT, SUBSTEPS);
    w.capture_delta().unwrap();
    assert_eq!(floor_mesh.strong_count(), 2, "handle + world");

    // Tick 2: the floor is destroyed, another mesh shape is created.
    w.destroy_body(floor);
    assert_eq!(floor_mesh.strong_count(), 2, "kept while tick 1 is restorable");
    let late_mesh = Mesh::grid(2, 2, 1.0, true).unwrap();
    let late = w.create_body(&BodyDef::fixed(Pos::new(20.0, 0.0, 0.0)));
    w.create_mesh_shape(late, &ShapeDef::default(), &late_mesh, Vec3::ONE);
    w.step(DT, SUBSTEPS);
    w.capture_delta().unwrap();

    // Back to tick 1: the floor shape (and its mesh) is live again; the late
    // shape is gone and its mesh released.
    w.restore_delta(1).unwrap();
    assert!(w.is_shape_valid(floor_shape));
    assert_eq!(late_mesh.strong_count(), 1);
    drop(floor_mesh);
    assert_eq!(w.shape_mesh(floor_shape).unwrap().strong_count(), 2, "world + the clone");
    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
        w.capture_delta().unwrap();
    }
    assert!((w.body_position(ball).y as f32 - 0.3).abs() < 0.03, "the ball rests on the restored mesh");

    // Destroyed again and outside the window: released.
    let probe = w.shape_mesh(floor_shape).unwrap();
    w.destroy_body(floor);
    for _ in 0..10 {
        w.step(DT, SUBSTEPS);
        w.capture_delta().unwrap();
    }
    assert_eq!(probe.strong_count(), 1, "released once no restorable tick has the shape");
}

#[test]
fn verify_finds_no_misses_and_hash_compares_worlds() {
    let (mut a, _) = pile();
    let (mut b, bodies) = pile();
    assert_eq!(a.state_hash(), b.state_hash());
    a.enable_delta(8).unwrap();
    a.set_delta_verify(true);
    for t in 0..40 {
        a.step(DT, SUBSTEPS);
        a.capture_delta().unwrap();
        assert_eq!(a.delta().unwrap().stats().misses, 0, "candidate set missed a change at {t}");
        b.step(DT, SUBSTEPS);
        assert_eq!(a.state_hash(), b.state_hash(), "tracking does not change the simulation");
    }
    b.apply_linear_impulse_to_center(bodies[0], Vec3::new(0.0, 1.0, 0.0), true);
    b.step(DT, SUBSTEPS);
    a.step(DT, SUBSTEPS);
    assert_ne!(a.state_hash(), b.state_hash());
}

#[test]
fn state_hash_covers_more_than_poses_and_velocities() {
    let (mut a, bodies) = pile();
    let (b, _) = pile();
    // Same pose and velocity, different gravity scale.
    a.set_body_gravity_scale(bodies[3], 0.5);
    assert_ne!(a.state_hash(), b.state_hash(), "gravity scale");
    a.set_body_gravity_scale(bodies[3], 1.0);
    assert_eq!(a.state_hash(), b.state_hash());
    // Same pose, zero velocity, asleep instead of awake.
    let (mut a, bodies) = pile();
    let (mut b, b_bodies) = pile();
    a.set_body_linear_velocity(bodies[0], Vec3::ZERO);
    b.set_body_linear_velocity(b_bodies[0], Vec3::ZERO);
    assert_eq!(a.state_hash(), b.state_hash());
    a.set_body_awake(bodies[0], false);
    assert_ne!(a.state_hash(), b.state_hash(), "sleep state");
}
