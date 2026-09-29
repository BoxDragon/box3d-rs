// `.into()` on f32 coordinates is identity in single precision but converts to
// f64 with the `double-precision` feature.
#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;

const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: i32 = 4;

fn world() -> World {
    World::new(&WorldDef::default()).expect("world")
}

/// Static ground slab whose top surface is at y = 0.
fn ground(world: &mut World) -> BodyId {
    let g = world.create_body(&BodyDef::fixed(Pos::new(0.0, -0.5, 0.0)));
    world.create_box_shape(g, &ShapeDef::default(), Vec3::new(50.0, 0.5, 50.0));
    g
}

fn dist(a: Pos, b: Pos) -> f32 {
    (a - b).length() as f32
}

#[test]
fn falling_box_lands_on_ground() {
    let mut w = world();
    let g = ground(&mut w);
    let crate_body = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 5.0, 0.0)));
    let crate_shape =
        w.create_box_shape(crate_body, &ShapeDef::default().enable_contact_events(true), Vec3::splat(0.5));

    let mut began = false;
    for _ in 0..240 {
        w.step(DT, SUBSTEPS);
        for e in w.contact_events().begin() {
            if e.shape_a() == crate_shape || e.shape_b() == crate_shape {
                let other = if e.shape_a() == crate_shape { e.shape_b() } else { e.shape_a() };
                assert_eq!(w.shape_body(other), g);
                began = true;
            }
        }
    }
    let p = w.body_position(crate_body);
    println!("box rest height = {}", p.y);
    assert!(began, "expected a contact begin event");
    assert!((p.y - 0.5).abs() < 0.02, "box should rest on ground at y=0.5, got {}", p.y);
    assert!(w.body_linear_velocity(crate_body).length() < 0.01);
    assert!((w.body_mass(crate_body) - 1000.0).abs() < 1.0, "1 m^3 at 1000 kg/m^3");
    // Settled boxes fall asleep.
    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
    }
    assert!(!w.is_body_awake(crate_body));

    // Lifting it off the ground produces an end event.
    w.set_body_transform(crate_body, Pos::new(0.0, 10.0, 0.0), Quat::IDENTITY);
    w.set_body_awake(crate_body, true);
    w.step(DT, SUBSTEPS);
    let ended = w.contact_events().end().iter().any(|e| e.shape_a() == crate_shape || e.shape_b() == crate_shape);
    assert!(ended, "expected a contact end event");
}

#[test]
fn distance_joint_pendulum_keeps_length() {
    let mut w = world();
    let anchor = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let bob = w.create_body(&BodyDef::dynamic(Pos::new(3.0, 10.0, 0.0)));
    w.create_sphere_shape(bob, &ShapeDef::default(), Vec3::ZERO, 0.25);

    let rod = w.create_distance_joint(&DistanceJointDef::rod(anchor, bob, 3.0));
    assert_eq!(w.joint_type(rod), JointType::Distance);
    assert_eq!(DistanceJointId::from_joint(&w, rod.joint()), Some(rod));

    let mut max_err = 0.0f32;
    let mut min_y = f32::MAX;
    for _ in 0..300 {
        w.step(DT, SUBSTEPS);
        let p = w.body_position(bob);
        let anchor_p = w.body_position(anchor);
        max_err = max_err.max((dist(p, anchor_p) - 3.0).abs());
        min_y = min_y.min(p.y as f32);
    }
    println!("pendulum max length error = {max_err}, lowest y = {min_y}");
    assert!(max_err < 0.03, "rod length drifted by {max_err}");
    assert!((min_y - 7.0).abs() < 0.05, "bob should swing down to y≈7, got {min_y}");
    assert!((rod.current_length(&w) - 3.0).abs() < 0.03);
    assert!((rod.length(&w) - 3.0).abs() < 1e-6);
}

#[test]
fn distance_joint_rope_is_slack_and_winchable() {
    let mut w = world();
    let anchor = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    // Start 1 m below the anchor: well inside the 3 m rope.
    let load = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 9.0, 0.0)));
    w.create_box_shape(load, &ShapeDef::default().density(200.0), Vec3::splat(0.25));
    let rope = w.create_distance_joint(&DistanceJointDef::rope(anchor, load, 3.0));

    // Slack: the load free-falls at first (no spring force).
    for _ in 0..10 {
        w.step(DT, SUBSTEPS);
    }
    let v = w.body_linear_velocity(load);
    assert!(v.y < -1.5, "slack rope should let the load fall freely, vy={}", v.y);

    for _ in 0..240 {
        w.step(DT, SUBSTEPS);
    }
    let hang = rope.current_length(&w);
    println!("rope hang length = {hang}");
    assert!((hang - 3.0).abs() < 0.05, "load should hang at max length, got {hang}");

    // Winch in to 1.5 m.
    rope.set_length_range(&mut w, 0.0, 1.5);
    w.set_body_awake(load, true);
    for _ in 0..240 {
        w.step(DT, SUBSTEPS);
    }
    let reeled = rope.current_length(&w);
    println!("rope reeled length = {reeled} (max {})", rope.max_length(&w));
    assert!((reeled - 1.5).abs() < 0.05, "winch should pull to 1.5, got {reeled}");
}

#[test]
fn rolling_resistance_capsule_comes_to_rest() {
    let mut w = world();
    ground(&mut w);
    let r: f32 = 0.25;
    let v0: f32 = 3.0;

    let spawn = |w: &mut World, z: f32, rr: f32| {
        // Rolling without slipping around -z for +x travel.
        let def = BodyDef::dynamic(Pos::new(0.0, r.into(), z.into()))
            .linear_velocity(Vec3::new(v0, 0.0, 0.0))
            .angular_velocity(Vec3::new(0.0, 0.0, -v0 / r));
        let b = w.create_body(&def);
        let sd = ShapeDef::default().friction(0.8).rolling_resistance(rr);
        w.create_capsule_shape(b, &sd, Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, 0.5), r);
        b
    };
    let damped = spawn(&mut w, 0.0, 0.5);
    let free = spawn(&mut w, 5.0, 0.0);

    for _ in 0..300 {
        w.step(DT, SUBSTEPS);
    }
    let vd = w.body_linear_velocity(damped).length();
    let vf = w.body_linear_velocity(free).length();
    let xd = w.body_position(damped).x;
    println!("after 5 s: rolling-resistance speed = {vd}, free speed = {vf}, stop x = {xd}");
    assert!(vd < 0.05, "capsule with rolling resistance should stop, speed={vd}");
    assert!(vf > 2.0, "capsule without rolling resistance keeps rolling, speed={vf}");
}

#[test]
fn barrel_cylinder_hull_rolls_on_its_side() {
    let mut w = world();
    ground(&mut w);
    let hull = Hull::cylinder(1.0, 0.3, -0.5, 16).expect("cylinder");
    assert_eq!(hull.vertex_count(), 32);
    let barrel = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 1.0, 0.0)));
    // Lay the Y-axis cylinder along Z.
    let side = Transform::new(Vec3::ZERO, Quat::from_rotation_x(core::f32::consts::FRAC_PI_2));
    w.create_transformed_hull_shape(barrel, &ShapeDef::default(), &hull, side, Vec3::ONE).expect("barrel hull");
    drop(hull); // the world holds its own copy

    w.apply_linear_impulse_to_center(barrel, Vec3::new(w.body_mass(barrel) * 2.0, 0.0, 0.0), true);
    for _ in 0..60 {
        w.step(DT, SUBSTEPS);
    }
    let p = w.body_position(barrel);
    let wz = w.body_angular_velocity(barrel).z;
    println!("barrel at x={} y={} wz={wz}", p.x, p.y);
    assert!((p.y - 0.3).abs() < 0.05, "barrel should lie on its side, y={}", p.y);
    assert!(p.x > 0.5 && wz < -1.0, "barrel should roll in +x");
}

#[test]
fn ray_cast_closest_hits_crate() {
    let mut w = world();
    let g = ground(&mut w);
    let b = w.create_body(&BodyDef::fixed(Pos::new(0.0, 2.0, 0.0)));
    let s = w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5));
    let hit = w.cast_ray_closest(Pos::new(0.0, 10.0, 0.0), Vec3::new(0.0, -20.0, 0.0), QueryFilter::default()).unwrap();
    assert_eq!(hit.shape, s);
    assert_eq!(w.shape_body(hit.shape), b);
    assert!((hit.point.y - 2.5).abs() < 1e-3);
    assert!((hit.normal - Vec3::Y).length() < 1e-3);
    let miss = w.cast_ray_closest(Pos::new(3.0, 10.0, 0.0), Vec3::new(0.0, -5.0, 0.0), QueryFilter::default());
    assert!(miss.is_none());
    let ground_hit = w.cast_ray_closest(Pos::new(3.0, 10.0, 0.0), Vec3::new(0.0, -20.0, 0.0), QueryFilter::default());
    assert_eq!(w.shape_body(ground_hit.unwrap().shape), g);
}

#[test]
fn ids_are_generational() {
    let mut w = world();
    let b = w.create_body(&BodyDef::dynamic(Pos::ZERO));
    let s = w.create_sphere_shape(b, &ShapeDef::default(), Vec3::ZERO, 0.5);
    assert!(w.is_body_valid(b) && w.is_shape_valid(s));
    assert_eq!(BodyId::from_bits(b.to_bits()), b);
    w.destroy_body(b);
    assert!(!w.is_body_valid(b));
    assert!(!w.is_shape_valid(s), "shapes die with their body");
    let b2 = w.create_body(&BodyDef::dynamic(Pos::ZERO));
    assert_ne!(b, b2, "reused slot gets a new generation");
    assert!(!w.is_body_valid(b));

    // Ids from another world are rejected.
    let other = world();
    assert!(!other.is_body_valid(b2));
}

#[test]
#[should_panic(expected = "invalid or foreign")]
fn stale_id_panics_instead_of_ub() {
    let mut w = world();
    let b = w.create_body(&BodyDef::dynamic(Pos::ZERO));
    w.destroy_body(b);
    let _ = w.body_position(b);
}

#[test]
fn joints_of_every_type_simulate() {
    let mut w = world();
    ground(&mut w);
    let mk = |w: &mut World, x: f32| {
        let b = w.create_body(&BodyDef::dynamic(Pos::new(x.into(), 3.0, 0.0)));
        w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.2));
        b
    };
    let post = w.create_body(&BodyDef::fixed(Pos::new(0.0, 3.0, 0.0)));

    let a = mk(&mut w, 1.0);
    let sph = w.create_spherical_joint(
        &SphericalJointDef::new(post, a)
            .anchors(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO)
            .enable_cone_limit(true)
            .cone_angle(0.3)
            .enable_twist_limit(true)
            .twist_range(-0.2, 0.2),
    );

    let b = mk(&mut w, 2.0);
    let rev = w.create_revolute_joint(
        &RevoluteJointDef::new(post, b)
            .anchors(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO)
            .enable_motor(true)
            .max_motor_torque(1000.0)
            .motor_speed(1.0)
            .enable_limit(true)
            .angle_range(-1.0, 1.0),
    );

    let c = mk(&mut w, 3.0);
    let pri = w.create_prismatic_joint(
        &PrismaticJointDef::new(post, c)
            .anchors(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO)
            .enable_limit(true)
            .translation_range(-0.5, 0.5),
    );

    let d = mk(&mut w, 4.0);
    let weld = w.create_weld_joint(&WeldJointDef::new(post, d).anchors(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO));

    let e = mk(&mut w, 5.0);
    let wheel = w.create_wheel_joint(&WheelJointDef::new(post, e).anchors(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO));

    let f = mk(&mut w, 6.0);
    let mot = w.create_motor_joint(
        &MotorJointDef::new(post, f)
            .anchors(Vec3::new(6.0, 0.0, 0.0), Vec3::ZERO)
            .linear_hertz(5.0)
            .linear_damping_ratio(1.0)
            .max_spring_force(1.0e5)
            .angular_hertz(5.0)
            .angular_damping_ratio(1.0)
            .max_spring_torque(1.0e5),
    );

    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
    }
    assert!(sph.cone_angle(&w) <= 0.3 + 0.02);
    assert!(rev.angle(&w) > 0.9 && rev.angle(&w) <= 1.02, "motor drives to upper limit, angle={}", rev.angle(&w));
    assert!(pri.translation(&w).abs() <= 0.52);
    assert!(dist(w.body_position(d), Pos::new(4.0, 3.0, 0.0)) < 0.05, "weld holds");
    assert!(dist(w.body_position(f), Pos::new(6.0, 3.0, 0.0)) < 0.05, "motor spring holds");
    let _ = wheel.spin_speed(&w);
    assert_eq!(w.joint_bodies(mot), (post, f));
    w.destroy_joint(weld, true);
    assert!(!w.is_joint_valid(weld.joint()));
}

#[test]
fn many_worlds_on_many_threads() {
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..20 {
                    let mut w = world();
                    let b = w.create_body(&BodyDef::dynamic(Pos::ZERO));
                    w.create_sphere_shape(b, &ShapeDef::default(), Vec3::ZERO, 0.5);
                    w.step(DT, SUBSTEPS);
                    assert!(w.body_position(b).y < 0.0);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}
