//! Soft contacts: SurfaceMaterial::contact_stiffness turns a contact into a
//! physical spring-damper. A resting load overlaps by load / stiffness, no
//! matter how light the soft body is, and stiffness 0 is the old rigid contact.

// `.into()` on f32 coordinates is identity in single precision but converts to
// f64 with the `double-precision` feature.
#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;

const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: i32 = 4;
const HALF: f32 = 0.25;

fn world() -> World {
    World::new(&WorldDef::default()).expect("world")
}

fn gravity(w: &World) -> f32 {
    -w.gravity().y
}

/// Static ground whose top face is y = 0.
fn ground(w: &mut World, material: SurfaceMaterial) -> BodyId {
    let g = w.create_body(&BodyDef::fixed(Pos::new(0.0, -1.0, 0.0)));
    w.create_box_shape(g, &ShapeDef::default().material(material), Vec3::new(10.0, 1.0, 10.0));
    g
}

/// A 0.5 m box of `mass` kg resting just above y = 0.
fn crate_box(w: &mut World, mass: f32, material: SurfaceMaterial, y: f32) -> BodyId {
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.0, y.into(), 0.0)).enable_sleep(false));
    let density = mass / (8.0 * HALF * HALF * HALF);
    w.create_box_shape(b, &ShapeDef::default().density(density).material(material), Vec3::splat(HALF));
    b
}

fn steps(w: &mut World, n: usize) {
    for _ in 0..n {
        w.step(DT, SUBSTEPS);
    }
}

/// Penetration of a box resting on y = 0, from its height.
fn depth(w: &World, b: BodyId) -> f32 {
    HALF - w.body_position(b).y as f32
}

/// Deepest overlap among `body`'s touching manifold points, from the contact readback.
fn contact_depth(w: &World, body: BodyId, scratch: &mut ContactScratch) -> f32 {
    let mut deepest = 0.0f32;
    w.visit_body_contacts(body, scratch, |c| {
        for m in c.manifolds() {
            for p in m.points() {
                deepest = deepest.max(-p.separation());
            }
        }
    });
    deepest
}

fn soft(k: f32) -> SurfaceMaterial {
    SurfaceMaterial::default().contact_stiffness(k)
}

#[test]
fn resting_box_overlaps_by_weight_over_stiffness_and_holds() {
    let k = 1.0e5;
    let mut w = world();
    ground(&mut w, soft(k));
    let b = crate_box(&mut w, 100.0, SurfaceMaterial::default(), HALF + 0.01);
    let expected = w.body_mass(b) * gravity(&w) / k;

    steps(&mut w, 120);
    let d2 = depth(&w, b);
    let mut scratch = ContactScratch::new();
    let cd = contact_depth(&w, b, &mut scratch);
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for _ in 0..360 {
        w.step(DT, SUBSTEPS);
        let d = depth(&w, b);
        lo = lo.min(d);
        hi = hi.max(d);
    }
    let d8 = depth(&w, b);
    println!(
        "expected {expected:.6}, at 2 s {d2:.6} (contacts {cd:.6}), at 8 s {d8:.6}, range over 6 s [{lo:.6}, {hi:.6}]"
    );
    assert!((d2 - expected).abs() < 0.05 * expected, "depth {d2} vs m g / k = {expected}");
    assert!((cd - expected).abs() < 0.05 * expected, "contact readback {cd} vs {expected}");
    assert!((d8 - d2).abs() < 1.0e-4, "no drift: {d2} -> {d8}");
    assert!(hi - lo < 1.0e-4, "steady: range {lo}..{hi}");
}

#[test]
fn resting_on_a_soft_mesh_overlaps_by_weight_over_stiffness() {
    // Mesh contacts take the scalar (non-SIMD) solver path, the same code as overflow contacts.
    let k = 1.0e5;
    let mut w = world();
    let v = [
        Vec3::new(-10.0, 0.0, -10.0),
        Vec3::new(10.0, 0.0, -10.0),
        Vec3::new(-10.0, 0.0, 10.0),
        Vec3::new(10.0, 0.0, 10.0),
    ];
    let t = [[0u32, 2, 1], [1, 2, 3]];
    let mesh = Mesh::new(&MeshDef::new(&v, &t).identify_edges(true)).expect("mesh");
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    w.create_mesh_shape(g, &ShapeDef::default().material(soft(k)), &mesh, Vec3::ONE);
    let b = crate_box(&mut w, 100.0, SurfaceMaterial::default(), HALF + 0.01);
    let expected = w.body_mass(b) * gravity(&w) / k;

    // Measured from the contact readback: box-vs-mesh collision keeps a small skin, so the
    // body height is not the overlap here.
    let mut scratch = ContactScratch::new();
    steps(&mut w, 120);
    let d2 = contact_depth(&w, b, &mut scratch);
    steps(&mut w, 360);
    let d8 = contact_depth(&w, b, &mut scratch);
    println!("mesh: expected {expected:.6}, at 2 s {d2:.6}, at 8 s {d8:.6}");
    assert!((d2 - expected).abs() < 0.05 * expected, "depth {d2} vs m g / k = {expected}");
    assert!((d8 - d2).abs() < 1.0e-4, "no drift: {d2} -> {d8}");
}

/// A small pile on rigid materials, stepped and hashed.
fn rigid_pile(material: SurfaceMaterial) -> (u64, Vec<[u32; 3]>) {
    let mut w = world();
    ground(&mut w, material);
    let mut bodies = Vec::new();
    for i in 0..12 {
        let b = w.create_body(&BodyDef::dynamic(Pos::new(
            (0.1 * (i % 3) as f32).into(),
            (0.3 + 0.55 * i as f32).into(),
            (0.07 * (i % 2) as f32).into(),
        )));
        w.create_box_shape(b, &ShapeDef::default().material(material), Vec3::splat(HALF));
        bodies.push(b);
    }
    steps(&mut w, 300);
    let positions = bodies
        .iter()
        .map(|&b| {
            let p = w.body_position(b);
            [(p.x as f32).to_bits(), (p.y as f32).to_bits(), (p.z as f32).to_bits()]
        })
        .collect();
    (w.state_hash(), positions)
}

#[test]
fn zero_stiffness_is_bit_identical_to_rigid() {
    let default = rigid_pile(SurfaceMaterial::default());
    // The damping ratio is ignored without a stiffness.
    let explicit = rigid_pile(SurfaceMaterial::default().contact_stiffness(0.0).contact_damping_ratio(0.3));
    assert_eq!(default, explicit);
    assert_eq!(SurfaceMaterial::default().stiffness(), 0.0);
    assert_eq!(SurfaceMaterial::default().damping_ratio(), 1.0);
}

#[test]
fn two_soft_shapes_act_as_springs_in_series() {
    let (ka, kb) = (2.0e5, 1.0e5);
    let mut w = world();
    ground(&mut w, soft(kb));
    let b = crate_box(&mut w, 100.0, soft(ka), HALF + 0.01);
    let k = ka * kb / (ka + kb);
    let expected = w.body_mass(b) * gravity(&w) / k;
    steps(&mut w, 240);
    let d = depth(&w, b);
    println!("series: expected {expected:.6} (k = {k}), got {d:.6}");
    assert!((d - expected).abs() < 0.05 * expected, "depth {d} vs m g / k_series = {expected}");
}

#[test]
fn light_fender_between_wall_and_heavy_hull_compresses_by_force_over_stiffness() {
    // No gravity: a 50 t hull pushed by a force squeezes a 20 kg soft fender
    // against a static wall. Both contacts carry F, so both overlap F / k even
    // though the fender is 2500 times lighter than the hull.
    //
    // The damping ratio is relative to each contact point's effective mass, which
    // the light fender sets, so the hull bouncing on the fender (k / 2 in series
    // with 50 t: a 2 s period) is only lightly damped, as physics says. A step load
    // rings for half a minute; ramp it in like a hull being warped in.
    let k = 1.0e6;
    let force = 5.0e4;
    let mut w = World::new(&WorldDef::default().gravity(Vec3::ZERO)).expect("world");

    let wall = w.create_body(&BodyDef::fixed(Pos::new(-1.0, 0.0, 0.0)));
    w.create_box_shape(wall, &ShapeDef::default(), Vec3::new(1.0, 2.0, 2.0)); // face at x = 0

    let fender = w.create_body(&BodyDef::dynamic(Pos::new(HALF.into(), 0.0, 0.0)).enable_sleep(false));
    let fender_density = 20.0 / (8.0 * HALF * HALF * HALF);
    w.create_box_shape(fender, &ShapeDef::default().density(fender_density).material(soft(k)), Vec3::splat(HALF));

    let hull_half = 1.0;
    let hull =
        w.create_body(&BodyDef::dynamic(Pos::new((2.0 * HALF + hull_half).into(), 0.0, 0.0)).enable_sleep(false));
    let hull_density = 50_000.0 / (8.0 * hull_half * hull_half * hull_half);
    w.create_box_shape(hull, &ShapeDef::default().density(hull_density), Vec3::splat(hull_half));
    assert!((w.body_mass(fender) - 20.0).abs() < 0.01 && (w.body_mass(hull) - 50_000.0).abs() < 1.0);

    let expected = force / k;
    let ramp_steps = 600;
    let push = |w: &mut World, i: usize| {
        let t = (i as f32 / ramp_steps as f32).min(1.0);
        let f = force * t * t * (3.0 - 2.0 * t);
        w.apply_force_to_center(hull, Vec3::new(-f, 0.0, 0.0), true)
    };
    for i in 0..900 {
        push(&mut w, i);
        w.step(DT, SUBSTEPS);
    }

    // Speed: the slow hull-on-fender mode still ringing from the ramp. Jitter: the
    // largest step-to-step velocity change, which that slow mode keeps tiny.
    let mut speed = [0.0f32; 2];
    let mut jitter = [0.0f32; 2];
    let mut last = [w.body_linear_velocity(fender), w.body_linear_velocity(hull)];
    let mut lo = [f32::MAX; 2];
    let mut hi = [f32::MIN; 2];
    for _ in 0..120 {
        push(&mut w, ramp_steps);
        w.step(DT, SUBSTEPS);
        for (i, b) in [fender, hull].into_iter().enumerate() {
            let v = w.body_linear_velocity(b);
            speed[i] = speed[i].max(v.length());
            jitter[i] = jitter[i].max((v - last[i]).length());
            last[i] = v;
        }
        let fx = w.body_position(fender).x as f32;
        let hx = w.body_position(hull).x as f32;
        let wall_depth = HALF - fx;
        let hull_depth = (fx + HALF) - (hx - hull_half);
        for (i, d) in [wall_depth, hull_depth].into_iter().enumerate() {
            lo[i] = lo[i].min(d);
            hi[i] = hi[i].max(d);
        }
    }
    let fx = w.body_position(fender).x as f32;
    let hx = w.body_position(hull).x as f32;
    let wall_depth = HALF - fx;
    let hull_depth = (fx + HALF) - (hx - hull_half);
    println!(
        "fender: expected {expected:.6}, wall contact {wall_depth:.6} [{:.6}, {:.6}], hull contact {hull_depth:.6} [{:.6}, {:.6}], max |v| fender {:.2e} hull {:.2e}, max |dv| per step fender {:.2e} hull {:.2e}",
        lo[0], hi[0], lo[1], hi[1], speed[0], speed[1], jitter[0], jitter[1]
    );
    assert!((wall_depth - expected).abs() < 0.05 * expected, "wall contact {wall_depth} vs F / k = {expected}");
    assert!((hull_depth - expected).abs() < 0.05 * expected, "hull contact {hull_depth} vs F / k = {expected}");
    assert!(hi[0] - lo[0] < 0.02 * expected && hi[1] - lo[1] < 0.02 * expected, "steady");
    assert!(jitter[0] < 1.0e-3 && jitter[1] < 1.0e-3, "no jitter");
}

/// Drops a 100 kg box 0.3 m onto a soft ground; returns (times the box left the
/// ground after first touching it, final depth, expected depth, deepest overlap).
fn drop_test(zeta: f32) -> (usize, f32, f32, f32) {
    let k = 1.0e5;
    let mut w = world();
    ground(&mut w, soft(k).contact_damping_ratio(zeta));
    let b = crate_box(&mut w, 100.0, SurfaceMaterial::default(), HALF + 0.3);
    let expected = w.body_mass(b) * gravity(&w) / k;
    let mut touched = false;
    let mut in_contact = false;
    let mut separations = 0;
    let mut deepest = 0.0f32;
    for _ in 0..240 {
        w.step(DT, SUBSTEPS);
        let d = depth(&w, b);
        deepest = deepest.max(d);
        if d > 0.0 {
            touched = true;
            in_contact = true;
        } else if touched && in_contact {
            in_contact = false;
            separations += 1;
        }
    }
    (separations, depth(&w, b), expected, deepest)
}

#[test]
fn critically_damped_impact_settles_without_bouncing() {
    let (bounces, d, expected, deepest) = drop_test(1.0);
    println!("zeta 1: bounces {bounces}, deepest {deepest:.4}, settled {d:.6} vs {expected:.6}");
    assert_eq!(bounces, 0, "critically damped: no bounce");
    assert!((d - expected).abs() < 0.05 * expected, "settles at m g / k");
    // Impact compresses deeper than the static load.
    assert!(deepest > 2.0 * expected, "dynamic compression {deepest} vs static {expected}");

    let (bounces, _, _, deepest) = drop_test(0.05);
    println!("zeta 0.05: bounces {bounces}, deepest {deepest:.4}");
    assert!(bounces > 0, "a lightly damped spring bounces");
}
