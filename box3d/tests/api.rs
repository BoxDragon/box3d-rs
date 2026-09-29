//! Events, closure queries, destroy queue and callbacks.
// `.into()` on f32 coordinates is identity in single precision but converts to
// f64 with the `double-precision` feature.
#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]
use box3d::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: i32 = 4;

fn world() -> World {
    World::new(&WorldDef::default()).expect("world")
}

fn ground(w: &mut World) -> (BodyId, ShapeId) {
    let g = w.create_body(&BodyDef::fixed(Pos::new(0.0, -0.5, 0.0)).user_data(7));
    let s = w.create_box_shape(g, &ShapeDef::default().user_data(70), Vec3::new(50.0, 0.5, 50.0));
    (g, s)
}

#[test]
fn body_move_events_carry_user_data_and_sleep() {
    let mut w = world();
    ground(&mut w);
    let entity_bits = 0xDEAD_BEEF_0000_0042u64; // e.g. hecs Entity::to_bits()
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 2.0, 0.0)).user_data(entity_bits));
    w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5));
    assert_eq!(w.body_user_data(b), entity_bits);

    w.step(DT, SUBSTEPS);
    let ev = w.body_events();
    assert_eq!(ev.moves().len(), 1);
    let m = &ev.moves()[0];
    assert_eq!(m.body(), b);
    assert_eq!(m.user_data(), entity_bits);
    assert!(m.position().y < 2.0);
    assert_eq!(m.transform(), w.body_transform(b));

    let mut asleep = false;
    for _ in 0..600 {
        w.step(DT, SUBSTEPS);
        asleep |= w.body_events().moves().iter().any(|m| m.fell_asleep() && m.body() == b);
    }
    assert!(asleep, "expected a fell_asleep move event");
    w.step(DT, SUBSTEPS);
    assert!(w.body_events().moves().is_empty(), "sleeping bodies report no moves");
}

#[test]
fn hit_events_report_impacts() {
    let mut w = world();
    ground(&mut w);
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 3.0, 0.0)).linear_velocity(Vec3::new(0.0, -10.0, 0.0)));
    let s = w.create_sphere_shape(b, &ShapeDef::default().enable_hit_events(true), Vec3::ZERO, 0.5);
    let mut best = 0.0f32;
    for _ in 0..60 {
        w.step(DT, SUBSTEPS);
        for h in w.contact_events().hit() {
            assert!(h.shape_a() == s || h.shape_b() == s);
            best = best.max(h.approach_speed());
        }
    }
    assert!(best > 5.0, "expected an impact hit event, best approach speed {best}");
}

#[test]
fn sensor_events_begin_and_end() {
    let mut w = world();
    let zone = w.create_body(&BodyDef::fixed(Pos::new(0.0, 5.0, 0.0)));
    let sensor =
        w.create_box_shape(zone, &ShapeDef::default().sensor(true).enable_sensor_events(true), Vec3::splat(1.0));
    let ball = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 8.0, 0.0)));
    let visitor = w.create_sphere_shape(ball, &ShapeDef::default().enable_sensor_events(true), Vec3::ZERO, 0.25);
    let (mut began, mut ended) = (false, false);
    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
        let e = w.sensor_events();
        began |= e.begin().iter().any(|e| e.sensor_shape() == sensor && e.visitor_shape() == visitor);
        ended |= e.end().iter().any(|e| e.sensor_shape() == sensor && e.visitor_shape() == visitor);
    }
    assert!(began && ended, "ball falls through the sensor: began={began} ended={ended}");
}

#[test]
fn joint_events_fire_past_force_threshold() {
    let mut w = world();
    let anchor = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let load = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 8.0, 0.0)));
    w.create_box_shape(load, &ShapeDef::default(), Vec3::splat(0.5)); // 1000 kg
    let rope = w.create_distance_joint(&DistanceJointDef::rod(anchor, load, 2.0).force_threshold(5000.0).user_data(99));
    let mut fired = false;
    for _ in 0..30 {
        w.step(DT, SUBSTEPS);
        for e in w.joint_events().events() {
            assert_eq!(e.joint(), rope.joint());
            assert_eq!(e.user_data(), 99);
            fired = true;
        }
    }
    assert!(fired, "~9800 N load should exceed the 5000 N threshold");
    assert_eq!(w.joint_user_data(rope), 99);
}

#[test]
fn cast_ray_closure_sees_all_hits_and_can_clip() {
    let mut w = world();
    let (_, gs) = ground(&mut w);
    let mut shapes = Vec::new();
    for i in 0..3 {
        let b = w.create_body(&BodyDef::fixed(Pos::new(0.0, (2.0 + 2.0 * i as f32).into(), 0.0)));
        shapes.push(w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5)));
    }
    let origin = Pos::new(0.0, 20.0, 0.0);
    let down = Vec3::new(0.0, -40.0, 0.0);

    // Return 1: keep going, every shape on the ray reports.
    let mut all = Vec::new();
    w.cast_ray(origin, down, QueryFilter::default(), |h| {
        all.push(h.shape);
        1.0
    });
    assert_eq!(all.len(), 4);
    assert!(all.contains(&gs) && shapes.iter().all(|s| all.contains(s)));

    // Return the fraction: clip, so the final best is the closest.
    let mut best: Option<RayHit> = None;
    w.cast_ray(origin, down, QueryFilter::default(), |h| {
        best = Some(*h);
        h.fraction
    });
    let closest = w.cast_ray_closest(origin, down, QueryFilter::default()).unwrap();
    assert_eq!(best.unwrap().shape, closest.shape);
    assert_eq!(closest.shape, shapes[2]);
    assert!((closest.point.y - 6.5).abs() < 1e-3);

    // Return 0: stop at the first reported hit.
    let mut n = 0;
    w.cast_ray(origin, down, QueryFilter::default(), |_| {
        n += 1;
        0.0
    });
    assert_eq!(n, 1);

    // Filters: ground only. (Default category bits are all ones, so the crates
    // need an explicit category to be excluded.)
    for &s in &shapes {
        w.set_shape_filter(s, Filter { category_bits: 1, ..Filter::default() }, false);
    }
    w.set_shape_filter(gs, Filter { category_bits: 2, ..Filter::default() }, false);
    let mut only_ground = 0;
    w.cast_ray(origin, down, QueryFilter { category_bits: u64::MAX, mask_bits: 2 }, |h| {
        assert_eq!(h.shape, gs);
        only_ground += 1;
        1.0
    });
    assert_eq!(only_ground, 1);
}

#[test]
fn overlap_and_shape_cast_queries() {
    let mut w = world();
    let (_, gs) = ground(&mut w);
    let b = w.create_body(&BodyDef::fixed(Pos::new(5.0, 1.0, 0.0)));
    let crate_shape = w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5));

    let mut found = Vec::new();
    w.overlap_aabb(Vec3::new(4.0, 0.5, -1.0), Vec3::new(6.0, 1.5, 1.0), QueryFilter::default(), |s| {
        found.push(s);
        true
    });
    assert_eq!(found, vec![crate_shape]);

    let mut count = 0;
    w.overlap_aabb(Vec3::splat(-100.0), Vec3::splat(100.0), QueryFilter::default(), |_| {
        count += 1;
        false // stop after the first
    });
    assert_eq!(count, 1);

    // A 0.3 m sphere proxy touching the crate but floating above the ground.
    let point = [Vec3::ZERO];
    let sphere = ShapeProxy::new(&point, 0.3).unwrap();
    let mut over = Vec::new();
    w.overlap_shape(Pos::new(5.0, 1.7, 0.0), &sphere, QueryFilter::default(), |s| {
        over.push(s);
        true
    });
    assert_eq!(over, vec![crate_shape]);

    // Sweep a capsule proxy from the left into the crate.
    let seg = [Vec3::new(0.0, 0.0, -0.3), Vec3::new(0.0, 0.0, 0.3)];
    let capsule = ShapeProxy::new(&seg, 0.2).unwrap();
    let mut first: Option<RayHit> = None;
    w.cast_shape(Pos::new(0.0, 1.0, 0.0), &capsule, Vec3::new(10.0, 0.0, 0.0), QueryFilter::default(), |h| {
        if h.shape != gs {
            first = Some(*h);
            return h.fraction;
        }
        -1.0 // ignore the ground
    });
    let h = first.expect("capsule sweep hits the crate");
    assert_eq!(h.shape, crate_shape);
    assert!((h.fraction * 10.0 - 4.3).abs() < 0.02, "contact at x≈4.3, fraction {}", h.fraction);

    assert!(ShapeProxy::new(&[], 0.1).is_err());
    assert!(ShapeProxy::new(&point, -1.0).is_err());
}

#[test]
fn query_closure_panic_propagates_after_the_call() {
    let mut w = world();
    ground(&mut w);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.cast_ray(Pos::new(0.0, 5.0, 0.0), Vec3::new(0.0, -10.0, 0.0), QueryFilter::default(), |_| {
            panic!("boom in ray callback")
        });
    }));
    let msg = r.unwrap_err().downcast::<&str>().unwrap();
    assert_eq!(*msg, "boom in ray callback");
    // World still usable.
    w.step(DT, SUBSTEPS);
    assert!(w.cast_ray_closest(Pos::new(0.0, 5.0, 0.0), Vec3::new(0.0, -10.0, 0.0), QueryFilter::default()).is_some());
}

#[test]
fn destroy_queue_defers_until_step() {
    let mut w = world();
    ground(&mut w);
    let a = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 2.0, 0.0)));
    w.create_sphere_shape(a, &ShapeDef::default(), Vec3::ZERO, 0.5);
    let b = w.create_body(&BodyDef::dynamic(Pos::new(3.0, 2.0, 0.0)));
    let bs = w.create_sphere_shape(b, &ShapeDef::default(), Vec3::ZERO, 0.5);
    let bs2 = w.create_sphere_shape(b, &ShapeDef::default(), Vec3::new(0.0, 1.0, 0.0), 0.5);
    let j = w.create_distance_joint(&DistanceJointDef::rod(a, b, 3.0));

    w.queue_destroy_joint(j, true);
    w.queue_destroy_shape(bs2, true);
    w.queue_destroy_body(a);
    w.queue_destroy_body(a); // double-queued: second is skipped as stale
    assert!(w.is_body_valid(a) && w.is_joint_valid(j.joint()), "nothing happens until flush");

    w.step(DT, SUBSTEPS);
    assert!(!w.is_body_valid(a));
    assert!(!w.is_joint_valid(j.joint()));
    assert!(!w.is_shape_valid(bs2));
    assert!(w.is_shape_valid(bs));
    assert_eq!(w.body_shape_count(b), 1);
    let mut shapes = Vec::new();
    w.body_shapes(b, &mut shapes);
    assert_eq!(shapes, vec![bs]);
}

#[test]
fn custom_filter_and_pre_solve_callbacks() {
    let mut w = world();
    let (_, gs) = ground(&mut w);
    let calls = Arc::new(AtomicUsize::new(0));

    // Ghost falls through the ground via the custom filter.
    let ghost = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 1.0, 0.0)));
    let ghost_s = w.create_sphere_shape(ghost, &ShapeDef::default().enable_custom_filtering(true), Vec3::ZERO, 0.25);
    let c = calls.clone();
    w.set_custom_filter(move |a, b| {
        c.fetch_add(1, Ordering::Relaxed);
        !(a == ghost_s || b == ghost_s)
    });

    // One-way platform style: pre-solve disables contact while moving up.
    let rocket = w.create_body(
        &BodyDef::dynamic(Pos::new(5.0, 0.3, 0.0)).linear_velocity(Vec3::new(0.0, 0.0, 0.0)).gravity_scale(0.0),
    );
    let rocket_s = w.create_sphere_shape(rocket, &ShapeDef::default().enable_pre_solve_events(true), Vec3::ZERO, 0.25);
    w.set_pre_solve(move |a, b, _point, _normal| {
        assert!(a == rocket_s || b == rocket_s || a == gs || b == gs);
        false // never solve rocket contacts
    });
    w.set_body_linear_velocity(rocket, Vec3::new(0.0, -2.0, 0.0));

    for _ in 0..60 {
        w.step(DT, SUBSTEPS);
    }
    assert!(calls.load(Ordering::Relaxed) > 0);
    assert!(w.body_position(ghost).y < -1.0, "ghost should pass through the ground");
    assert!(w.body_position(rocket).y < -1.0, "pre-solve false disables the contact");

    w.clear_custom_filter();
    w.clear_pre_solve();
}

#[test]
fn callback_panic_is_resumed_after_step() {
    let mut w = world();
    ground(&mut w);
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 0.4, 0.0)));
    w.create_sphere_shape(b, &ShapeDef::default().enable_custom_filtering(true), Vec3::ZERO, 0.5);
    w.set_custom_filter(|_, _| panic!("filter exploded"));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.step(DT, SUBSTEPS)));
    assert_eq!(*r.unwrap_err().downcast::<&str>().unwrap(), "filter exploded");
    w.clear_custom_filter();
    w.step(DT, SUBSTEPS); // no stale panic left behind
}

#[test]
fn defs_are_values_and_worlds_configurable() {
    let def = WorldDef::default()
        .gravity(Vec3::new(0.0, -1.0, 0.0))
        .worker_count(1)
        .capacity(Capacity {
            static_shapes: 8,
            dynamic_shapes: 512,
            static_bodies: 8,
            dynamic_bodies: 512,
            contacts: 4096,
        })
        .destroy_queue_capacity(1024);
    let mut w = World::new(&def).unwrap();
    assert_eq!(w.gravity(), Vec3::new(0.0, -1.0, 0.0));
    assert_eq!(w.worker_count(), 1);
    assert!(bytes_allocated() > 0);
    assert_ne!(w.id().to_bits(), 0);

    let mat = SurfaceMaterial::default().friction(0.3).restitution(0.5).rolling_resistance(0.1).user_material_id(4);
    let b = w.create_body(&BodyDef::dynamic(Pos::ZERO));
    let s = w.create_sphere_shape(b, &ShapeDef::default().material(mat), Vec3::ZERO, 0.5);
    assert_eq!(w.shape_surface_material(s), mat);
    assert_eq!(w.shape_friction(s), 0.3);
    assert_eq!(std::mem::size_of::<BodyId>(), 8);
    assert_eq!(std::mem::size_of::<ShapeId>(), 8);
    assert_eq!(std::mem::size_of::<JointId>(), 8);
}
