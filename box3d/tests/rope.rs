// `.into()` on f32 coordinates is identity in single precision but converts to
// f64 with the `double-precision` feature.
#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;

const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: i32 = 4;

fn world() -> World {
    World::new(&WorldDef::default()).expect("world")
}

/// A 0.5 m box of 25 kg at `p`.
fn load(w: &mut World, p: Pos) -> BodyId {
    let b = w.create_body(&BodyDef::dynamic(p));
    w.create_box_shape(b, &ShapeDef::default().density(200.0), Vec3::splat(0.25));
    b
}

fn settle(w: &mut World, steps: usize) {
    for _ in 0..steps {
        w.step(DT, SUBSTEPS);
    }
}

fn gravity(w: &World) -> f32 {
    -w.gravity().y
}

#[test]
fn rope_hangs_at_its_length_and_carries_the_weight() {
    let mut w = world();
    let ceiling = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let b = load(&mut w, Pos::new(0.0, 9.0, 0.0));
    let rope = w.create_rope_joint(&RopeJointDef::new(ceiling, b).path(&[(Vec3::ZERO, false), (Vec3::ZERO, true)]).length(3.0));
    assert_eq!(w.joint_type(rope), JointType::Rope);
    assert_eq!(RopeJointId::from_joint(&w, rope.joint()), Some(rope));

    // Slack: free fall, no tension.
    settle(&mut w, 10);
    assert!(w.body_linear_velocity(b).y < -1.5, "slack rope lets the load fall");

    settle(&mut w, 240);
    let y = w.body_position(b).y as f32;
    let weight = w.body_mass(b) * gravity(&w);
    println!("hang y = {y}, tension = {}, weight = {weight}", rope.tension(&w));
    assert!((y - 7.0).abs() < 0.02, "hangs 3 m below the ceiling, got {y}");
    assert!((rope.tension(&w) - weight).abs() < 0.02 * weight, "tension carries the weight");
    assert!((rope.current_length(&w) - 3.0).abs() < 0.02);

    // Winch in.
    rope.set_length(&mut w, 1.5);
    w.set_body_awake(b, true);
    settle(&mut w, 240);
    let y = w.body_position(b).y as f32;
    assert!((y - 8.5).abs() < 0.02, "winched to 1.5 m, got {y}");
}

#[test]
fn rope_over_a_sheave_shares_one_tension() {
    // End fixed on the ceiling at x = -2, over a sheave at the origin of the
    // ceiling's top, down to the load: 2 m across, the rest hangs.
    let mut w = world();
    let ceiling = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let b = load(&mut w, Pos::new(0.0, 8.0, 0.0));
    let path = [(Vec3::new(-2.0, 0.0, 0.0), false), (Vec3::ZERO, false), (Vec3::ZERO, true)];
    let rope = w.create_rope_joint(&RopeJointDef::new(ceiling, b).path(&path).length(5.0));
    settle(&mut w, 300);
    let p = w.body_position(b);
    println!("sheave hang = {p:?}, tension = {}", rope.tension(&w));
    assert!((p.y as f32 - 7.0).abs() < 0.02, "5 m less the 2 m across hangs 3 m, got {}", p.y);
    assert!((p.x as f32).abs() < 0.02, "hangs straight below the sheave");
    let weight = w.body_mass(b) * gravity(&w);
    assert!((rope.tension(&w) - weight).abs() < 0.02 * weight);

    let mut points = Vec::new();
    rope.world_points(&w, &mut points);
    assert_eq!(points.len(), 3);
    assert!((points[0] - Pos::new(-2.0, 10.0, 0.0)).length() < 1e-4);
}

#[test]
fn block_and_tackle_halves_the_pull() {
    // Lifting block: the rope runs from the ceiling down under a sheave on the
    // load and back up; both legs carry the load, each with half its weight.
    let mut w = world();
    let ceiling = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let b = load(&mut w, Pos::new(0.0, 8.0, 0.0));
    let path = [
        (Vec3::new(-0.25, 0.0, 0.0), false),
        (Vec3::new(-0.25, 0.25, 0.0), true),
        (Vec3::new(0.25, 0.25, 0.0), true),
        (Vec3::new(0.25, 0.0, 0.0), false),
    ];
    let rope = w.create_rope_joint(&RopeJointDef::new(ceiling, b).path(&path).length(4.5));
    settle(&mut w, 300);
    let y = w.body_position(b).y as f32;
    let weight = w.body_mass(b) * gravity(&w);
    println!("tackle y = {y}, tension = {} of weight {weight}", rope.tension(&w));
    // Legs of (4.5 - 0.5) / 2 = 2 m: the sheave's top at 8 m, the centre 0.25 m below.
    assert!((y - 7.75).abs() < 0.02, "got {y}");
    assert!((rope.tension(&w) - 0.5 * weight).abs() < 0.02 * weight, "each leg carries half");
}

#[test]
fn stiff_rope_stretches_by_its_load() {
    // EA = 3000 N; 3 m of it under 25 kg (250 N at g = 10) stretches 0.25 m.
    let mut w = world();
    let ceiling = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let b = load(&mut w, Pos::new(0.0, 7.0, 0.0));
    let ea = 3000.0;
    let rope = w.create_rope_joint(
        &RopeJointDef::new(ceiling, b).path(&[(Vec3::ZERO, false), (Vec3::ZERO, true)]).length(3.0).stiffness(ea).damping_ratio(1.0),
    );
    settle(&mut w, 600);
    let weight = w.body_mass(b) * gravity(&w);
    let stretch = 3.0 * weight / ea;
    let y = w.body_position(b).y as f32;
    println!("stretch = {}, expected {stretch}", 7.0 - y);
    assert!((7.0 - y - stretch).abs() < 0.01, "stretched {} for {stretch}", 7.0 - y);
    assert!((rope.tension(&w) - weight).abs() < 0.02 * weight);
}

#[test]
fn rope_between_two_dynamic_bodies_pulls_both() {
    // No gravity: two bodies drifting apart stop at the rope's length.
    let mut w = World::new(&WorldDef::default().gravity(Vec3::ZERO)).expect("world");
    let a = load(&mut w, Pos::new(0.0, 0.0, 0.0));
    let b = load(&mut w, Pos::new(1.0, 0.0, 0.0));
    w.set_body_linear_velocity(a, Vec3::new(-1.0, 0.0, 0.0));
    w.set_body_linear_velocity(b, Vec3::new(1.0, 0.0, 0.0));
    let rope = w.create_rope_joint(&RopeJointDef::new(a, b).path(&[(Vec3::ZERO, false), (Vec3::ZERO, true)]).length(2.0));
    settle(&mut w, 120);
    let d = (w.body_position(b) - w.body_position(a)).length() as f32;
    assert!(d <= 2.02, "held within the length, got {d}");
    let p = w.body_linear_velocity(a) + w.body_linear_velocity(b);
    assert!(p.length() < 1e-3, "momentum stays zero");
    assert!(rope.tension(&w) >= 0.0);
}
