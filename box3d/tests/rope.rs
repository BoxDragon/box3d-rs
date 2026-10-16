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

/// A crane hook (10 kg) hung on a hoist rope, a 25 kg crate slung under it by
/// one sling from eye to eye through the hook's seat, `friction` at the seat.
/// The hook is kicked sideways; returns how much rope still runs through the
/// seat (the legs' length difference, max - min over the last second) and the
/// slip energy the seat took over the run.
fn kicked_sling(friction: f32) -> (f32, f32) {
    let mut w = world();
    let ceiling = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let hook = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 7.9, 0.0)));
    w.create_box_shape(hook, &ShapeDef::default().density(1250.0), Vec3::splat(0.1));
    let crate_ = load(&mut w, Pos::new(0.0, 6.582, 0.0));
    w.create_rope_joint(&RopeJointDef::new(ceiling, hook).path(&[(Vec3::ZERO, false), (Vec3::new(0.0, 0.1, 0.0), true)]).length(2.0));
    let path = [(Vec3::new(-0.25, 0.25, 0.0), false), (Vec3::new(0.0, -0.1, 0.0), true), (Vec3::new(0.25, 0.25, 0.0), false)];
    let sling = w.create_rope_joint(&RopeJointDef::new(crate_, hook).path(&path).length(2.0).friction(1, friction));
    w.set_body_linear_velocity(hook, Vec3::new(1.0, 0.0, 0.5));

    let mut slipped = 0.0;
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    let mut points = Vec::new();
    for step in 0..600 {
        w.step(DT, SUBSTEPS);
        slipped += sling.slip_energies(&w)[1];
        if step >= 540 {
            sling.world_points(&w, &mut points);
            let legs = ((points[1] - points[0]).length() - (points[2] - points[1]).length()) as f32;
            lo = lo.min(legs);
            hi = hi.max(legs);
        }
        // Capstan: one leg holds at most e^(friction * wrap) times the other, the
        // wrap under pi here.
        let tensions = sling.segment_tensions(&w);
        let bound = (friction * std::f32::consts::PI).exp() * 1.01;
        assert!(tensions[0] <= bound * tensions[1] + 1.0 && tensions[1] <= bound * tensions[0] + 1.0, "{tensions:?}");
    }
    (hi - lo, slipped)
}

#[test]
fn sling_slides_freely_without_friction() {
    let (sliding, slipped) = kicked_sling(0.0);
    println!("frictionless sling: legs vary {sliding} m");
    assert!(sliding > 0.02, "the hook keeps riding the sling, legs vary {sliding} m");
    assert_eq!(slipped, 0.0);
}

#[test]
fn friction_at_the_hook_stops_the_sling_sliding() {
    let (sliding, slipped) = kicked_sling(0.3);
    println!("sling at 0.3: legs vary {sliding} m, slip energy {slipped} J");
    assert!(sliding < 0.002, "the hook settles on the sling, legs vary {sliding} m");
    // Out of the kick's 6.25 J; the rest swings the whole rig.
    assert!(slipped > 0.1 && slipped < 6.25, "friction took {slipped} J");
}

#[test]
fn capstan_holds_unequal_legs() {
    // A crate slung from a fixed hook with its centre off to one side of the
    // eyes: without friction it slides until the legs pull alike; with enough
    // friction the legs hold different tensions instead.
    let tensions = |friction: f32| {
        let mut w = world();
        let ceiling = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
        let b = load(&mut w, Pos::new(0.0, 8.5, 0.0));
        let path = [(Vec3::new(-0.25, 0.25, 0.0), true), (Vec3::ZERO, false), (Vec3::new(0.25, 0.25, 0.0), true)];
        // Taut as placed: two 1.27 m legs.
        let length = 2.0 * (0.25f32 * 0.25 + 1.25 * 1.25).sqrt();
        let rope = w.create_rope_joint(&RopeJointDef::new(ceiling, b).path(&path).length(length).friction(1, friction));
        // Push the crate's end down on one side: a 5 kg weight hung off the right eye.
        let weight = w.create_body(&BodyDef::dynamic(Pos::new(0.25, 8.0, 0.0)));
        w.create_box_shape(weight, &ShapeDef::default().density(200.0 * 0.2), Vec3::splat(0.25));
        w.create_rope_joint(
            &RopeJointDef::new(b, weight).path(&[(Vec3::new(0.25, 0.0, 0.0), false), (Vec3::new(0.0, 0.25, 0.0), true)]).length(0.25),
        );
        settle(&mut w, 600);
        let t = rope.segment_tensions(&w);
        (t, w.body_rotation(b))
    };
    let (free, _) = tensions(0.0);
    println!("frictionless legs {free:?}");
    assert!((free[0] - free[1]).abs() < 0.02 * free[0], "one tension without friction");
    let (held, q) = tensions(1.5);
    println!("legs at 1.5 {held:?}, crate {q:?}");
    assert!(held[1] > 1.1 * held[0], "the loaded leg pulls harder: {held:?}");
}

/// A 25 kg load hung 3 m under a sheave (the rope's end 2 m along the
/// ceiling), kicked sideways at 2 m/s, out of the sheave's plane. Returns the
/// most it swings out (m) over the first second and over the 10th.
fn kicked_hook(bend_damping: f32) -> (f32, f32) {
    let mut w = world();
    let ceiling = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let b = load(&mut w, Pos::new(0.0, 7.0, 0.0));
    let path = [(Vec3::new(-2.0, 0.0, 0.0), false), (Vec3::ZERO, false), (Vec3::ZERO, true)];
    w.create_rope_joint(&RopeJointDef::new(ceiling, b).path(&path).length(5.0).bend_damping(bend_damping));
    w.set_body_linear_velocity(b, Vec3::new(0.0, 0.0, 2.0));
    let mut first = 0.0f32;
    let mut tenth = 0.0f32;
    for step in 0..600 {
        w.step(DT, SUBSTEPS);
        let out = (w.body_position(b).z as f32).abs();
        match step {
            0..60 => first = first.max(out),
            540.. => tenth = tenth.max(out),
            _ => {}
        }
    }
    (first, tenth)
}

#[test]
fn bending_damping_stills_a_swinging_hook() {
    let (first, tenth) = kicked_hook(0.0);
    println!("free: {first} m, then {tenth} m");
    assert!(first > 0.5 && tenth > 0.9 * first, "a frictionless rope swings on: {first} then {tenth}");

    // 400 N m² s on 3 m: 133 N m s against the swing's 225 kg m², about 0.3/s.
    let (first, tenth) = kicked_hook(400.0);
    println!("damped: {first} m, then {tenth} m");
    assert!(first > 0.4, "the kick still swings it: {first}");
    assert!(tenth < 0.15 * first, "the swing dies down: {first} then {tenth}");
}

/// Swing angle a steady side force holds a 25 kg load at, 3 m under a fixed
/// point, with the rope's bending stiffness `ei`.
fn side_pushed(ei: f32) -> f32 {
    let mut w = world();
    let ceiling = w.create_body(&BodyDef::fixed(Pos::new(0.0, 10.0, 0.0)));
    let b = load(&mut w, Pos::new(0.0, 7.0, 0.0));
    w.create_rope_joint(
        &RopeJointDef::new(ceiling, b)
            .path(&[(Vec3::ZERO, false), (Vec3::ZERO, true)])
            .length(3.0)
            .bend_stiffness(ei)
            .bend_damping(600.0),
    );
    for _ in 0..900 {
        w.apply_force_to_center(b, Vec3::new(0.4 * w.body_mass(b) * gravity(&w), 0.0, 0.0), true);
        w.step(DT, SUBSTEPS);
    }
    let p = w.body_position(b);
    (p.x as f32).atan2(10.0 - p.y as f32)
}

#[test]
fn bending_stiffness_straightens_the_rope() {
    // Free: tan(angle) = side / weight = 0.4.
    let free = side_pushed(0.0);
    println!("free angle {free}");
    assert!((free - 0.4f32.atan()).abs() < 0.01, "got {free}");

    // EI 1125 on 3 m: 375 N m per radian at the ceiling. About the ceiling,
    // 0.4 W L cos(a) = W L sin(a) + 375 a, with W L = 750 N m: a = 0.26.
    // (The load turns freely to the rope: its end takes no moment once still.)
    let stiff = side_pushed(1125.0);
    println!("stiff angle {stiff}");
    assert!((stiff - 0.2597).abs() < 0.01, "got {stiff}");
}
