//! Kinematic character mover: collide + solve planes + cast, on box, mesh and
//! height-field ground, sliding along walls.

#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;

const DT: f32 = 1.0 / 60.0;

fn world() -> World {
    World::new(&WorldDef::default()).expect("world")
}

/// A pure query character (no body): feet at `pos`.
struct Character {
    pos: Pos,
    vel: Vec3,
    capsule: Capsule,
    planes: [CollisionPlane; 16],
    plane_count: usize,
    on_ground: bool,
}

impl Character {
    fn new(pos: Pos) -> Self {
        Self {
            pos,
            vel: Vec3::ZERO,
            capsule: Capsule::standing(1.8, 0.3),
            planes: [CollisionPlane::default(); 16],
            plane_count: 0,
            on_ground: false,
        }
    }

    /// One update: gravity, walk at `walk` (horizontal velocity), up to 5
    /// collide/solve/cast iterations, then clip the velocity. Allocation free.
    fn update(&mut self, w: &World, walk: Vec3) {
        self.vel.x = walk.x;
        self.vel.z = walk.z;
        if self.on_ground {
            self.vel.y = 0.0;
        }
        self.vel.y -= 10.0 * DT;
        let target = self.pos + Pos::from(self.vel * DT);
        for _ in 0..5 {
            self.plane_count = 0;
            let (planes, count) = (&mut self.planes, &mut self.plane_count);
            w.collide_mover(self.pos, &self.capsule, QueryFilter::default(), |_shape, results| {
                for r in results {
                    if *count < planes.len() {
                        planes[*count] = r.to_collision_plane();
                        *count += 1;
                    }
                }
                true
            });
            let delta = to_vec3(target - self.pos);
            let solved = solve_planes(delta, &mut self.planes[..self.plane_count]);
            let fraction = w.cast_mover(self.pos, &self.capsule, solved.delta, QueryFilter::default());
            let step = solved.delta * fraction;
            self.pos += Pos::from(step);
            if step.length_squared() < 1e-4 {
                break;
            }
        }
        let planes = &self.planes[..self.plane_count];
        self.on_ground = planes.iter().any(|p| p.plane.normal.y > 0.7);
        self.vel = clip_vector(self.vel, planes);
    }
}

/// `Pos` is `Vec3` or `DVec3` depending on the feature.
fn to_vec3(p: Pos) -> Vec3 {
    Vec3::new(p.x as f32, p.y as f32, p.z as f32)
}

fn box_ground(w: &mut World) {
    let g = w.create_body(&BodyDef::fixed(Pos::new(0.0, -0.5, 0.0)));
    w.create_box_shape(g, &ShapeDef::default(), Vec3::new(50.0, 0.5, 50.0));
}

#[test]
fn character_stands_on_box_ground() {
    let mut w = world();
    box_ground(&mut w);
    let mut c = Character::new(Pos::new(0.0, 1.0, 0.0));
    for _ in 0..120 {
        c.update(&w, Vec3::ZERO);
    }
    println!("character feet at {}", c.pos);
    assert!(c.on_ground);
    assert!((c.pos.y as f32).abs() < 0.05, "feet should rest on y=0, got {}", c.pos.y);
    assert!(c.vel.y.abs() < 0.5);
}

#[test]
fn character_walks_into_wall_and_slides_along_it() {
    let mut w = world();
    box_ground(&mut w);
    // Wall: the face at x = 5, spanning z in [-20, 20].
    let wall = w.create_body(&BodyDef::fixed(Pos::new(5.5, 2.0, 0.0)));
    w.create_box_shape(wall, &ShapeDef::default(), Vec3::new(0.5, 2.0, 20.0));

    let mut c = Character::new(Pos::new(0.0, 0.0, 0.0));
    // Walk straight into the wall.
    for _ in 0..180 {
        c.update(&w, Vec3::new(4.0, 0.0, 0.0));
    }
    let x = c.pos.x as f32;
    println!("blocked at x = {x}, z = {}", c.pos.z);
    assert!(x < 5.0 - 0.3 + 0.02 && x > 5.0 - 0.3 - 0.1, "should stop against the wall at x ~ 4.7, got {x}");
    assert!(c.vel.x.abs() < 1e-3, "clip_vector should remove velocity into the wall, got {}", c.vel.x);
    assert!((c.pos.y as f32).abs() < 0.05, "still standing on the ground, y = {}", c.pos.y);

    // Walk diagonally: blocked in x, slides in +z.
    let z0 = c.pos.z as f32;
    for _ in 0..60 {
        c.update(&w, Vec3::new(3.0, 0.0, 3.0));
    }
    let (x, z) = (c.pos.x as f32, c.pos.z as f32);
    println!("slid to x = {x}, z = {z}");
    assert!(x < 4.72, "must not penetrate the wall, x = {x}");
    assert!(z - z0 > 2.5, "should slide ~3 m along the wall, moved {}", z - z0);
}

#[test]
fn character_on_mesh_and_height_field_ground() {
    // Left half: mesh ground at y = 0. Right half: height field at y = 0.5.
    let mut w = world();
    let mesh = Mesh::grid(20, 20, 1.0, true).unwrap(); // [-10, 10]^2
    let g = w.create_body(&BodyDef::fixed(Pos::new(-10.0, 0.0, 0.0)));
    w.create_mesh_shape(g, &ShapeDef::default(), &mesh, Vec3::ONE);
    drop(mesh);
    let n = 21u32;
    let heights = vec![0.5f32; (n * n) as usize];
    let hf = HeightField::new(&HeightFieldDef::new(&heights, n, n, Vec3::ONE)).unwrap();
    let g = w.create_body(&BodyDef::fixed(Pos::new(0.0, 0.0, -10.0)));
    w.create_height_field_shape(g, &ShapeDef::default(), &hf);
    drop(hf);

    let mut on_mesh = Character::new(Pos::new(-10.0, 1.0, 0.0));
    let mut on_field = Character::new(Pos::new(10.0, 1.5, 0.0));
    for _ in 0..120 {
        on_mesh.update(&w, Vec3::new(0.0, 0.0, 1.0));
        on_field.update(&w, Vec3::new(0.0, 0.0, -1.0));
    }
    println!("mesh feet {}, field feet {}", on_mesh.pos, on_field.pos);
    assert!(on_mesh.on_ground && on_field.on_ground);
    assert!((on_mesh.pos.y as f32).abs() < 0.05);
    assert!((on_field.pos.y as f32 - 0.5).abs() < 0.05);
    assert!((on_mesh.pos.z as f32 - 2.0).abs() < 0.1, "walked 2 m, z = {}", on_mesh.pos.z);
}

#[test]
fn plane_solver_and_clip() {
    let ground = Plane::new(Vec3::Y, 0.0);
    let wall = Plane::new(-Vec3::X, -1.0); // x <= 1 is free
    assert!((wall.separation(Vec3::new(0.5, 0.0, 0.0)) - 0.5).abs() < 1e-6);
    let mut planes = [CollisionPlane::rigid(ground), CollisionPlane::rigid(wall)];
    let r = solve_planes(Vec3::new(2.0, -1.0, 0.5), &mut planes);
    println!("{r:?} {planes:?}");
    // The solver allows B3_LINEAR_SLOP (5 mm) of penetration.
    assert!(r.delta.y.abs() < 0.01, "ground push, delta {:?}", r.delta);
    assert!((r.delta.x - 1.0).abs() < 0.01, "wall push, delta {:?}", r.delta);
    assert!((r.delta.z - 0.5).abs() < 1e-5, "free axis untouched");
    assert!(planes[0].push > 0.0 && planes[1].push > 0.0);
    assert!(r.iteration_count >= 1);

    let v = clip_vector(Vec3::new(3.0, -2.0, 1.0), &planes);
    assert!(v.x.abs() < 1e-5 && v.y.abs() < 1e-5 && (v.z - 1.0).abs() < 1e-5, "{v}");

    // Soft planes push at most their limit and never clip.
    let mut soft = [CollisionPlane::soft(ground, 0.1)];
    let r = solve_planes(Vec3::new(0.0, -1.0, 0.0), &mut soft);
    assert!((soft[0].push - 0.1).abs() < 1e-6);
    assert!((r.delta.y + 0.9).abs() < 1e-5);
    assert_eq!(clip_vector(Vec3::new(0.0, -1.0, 0.0), &soft), Vec3::new(0.0, -1.0, 0.0));

    // Empty slices are fine.
    assert_eq!(solve_planes(Vec3::X, &mut []).delta, Vec3::X);
    assert_eq!(clip_vector(Vec3::X, &[]), Vec3::X);
}

#[test]
fn collide_mover_reports_shapes_and_planes() {
    let mut w = world();
    box_ground(&mut w);
    let crate_body = w.create_body(&BodyDef::fixed(Pos::new(0.55, 0.5, 0.0)));
    let crate_shape = w.create_box_shape(crate_body, &ShapeDef::default(), Vec3::splat(0.5));
    // Mover overlapping the crate's -x face by 0.05 and resting on the ground.
    let cap = Capsule::standing(1.8, 0.3);
    let origin = Pos::new(-0.2, 0.0, 0.0);
    let mut seen = Vec::new();
    w.collide_mover(origin, &cap, QueryFilter::default(), |shape, planes| {
        seen.push((shape, planes.to_vec()));
        true
    });
    let crate_planes = seen.iter().find(|(s, _)| *s == crate_shape).expect("crate plane");
    let p = crate_planes.1[0];
    assert!(p.plane.normal.dot(-Vec3::X) > 0.99, "crate plane points at the mover: {p:?}");
    assert_eq!(p.triangle_index, 0);
    let s = p.plane.separation(Vec3::ZERO);
    assert!(s < 0.0 && s > -0.1, "mover overlaps the crate by ~0.05: {s}");

    // Stop after the first shape.
    let mut calls = 0;
    w.collide_mover(origin, &cap, QueryFilter::default(), |_, _| {
        calls += 1;
        false
    });
    assert_eq!(calls, 1);

    // cast_mover: the crate blocks a sweep in +x, unless filtered out.
    let free = Pos::new(-3.0, 0.1, 0.0);
    let f = w.cast_mover(free, &cap, Vec3::new(4.0, 0.0, 0.0), QueryFilter::default());
    assert!(f < 1.0 && (free.x as f32 + 4.0 * f + 0.3 - 0.05).abs() < 0.05, "fraction {f}");
    let f = w.cast_mover_filtered(free, &cap, Vec3::new(4.0, 0.0, 0.0), QueryFilter::default(), |s| s != crate_shape);
    assert_eq!(f, 1.0);

    // Body-level versions.
    let mut out = [BodyPlaneResult::default(); 4];
    let n = w.body_collide_mover(
        crate_body,
        origin,
        &cap,
        QueryFilter::default(),
        Pos::new(0.55, 0.5, 0.0),
        Quat::IDENTITY,
        &mut out,
    );
    assert_eq!(n, 1);
    assert_eq!(out[0].shape, crate_shape);
    assert!(out[0].result.plane.normal.x < -0.99);
    // Zero capacity writes nothing.
    assert_eq!(
        w.body_collide_mover(crate_body, origin, &cap, QueryFilter::default(), Pos::ZERO, Quat::IDENTITY, &mut []),
        0
    );

    let crate_xf = w.body_transform(crate_body);
    let hit = w
        .body_time_of_impact_mover(
            crate_body,
            free,
            &cap,
            Vec3::new(4.0, 0.0, 0.0),
            QueryFilter::default(),
            crate_xf,
            crate_xf,
        )
        .expect("toi hit");
    assert_eq!(hit.shape, crate_shape);
    assert!(hit.fraction > 0.0 && hit.fraction < 1.0);
    assert!(hit.normal.x > 0.9 || hit.normal.x < -0.9, "{hit:?}");
    assert!((hit.point.x as f32 - 0.05).abs() < 0.05, "{hit:?}");
    // Sweeping away misses.
    assert!(
        w.body_time_of_impact_mover(
            crate_body,
            free,
            &cap,
            Vec3::new(-4.0, 0.0, 0.0),
            QueryFilter::default(),
            crate_xf,
            crate_xf
        )
        .is_none()
    );
}

#[test]
#[should_panic(expected = "boom")]
fn collide_mover_resumes_panics() {
    let mut w = world();
    box_ground(&mut w);
    w.collide_mover(Pos::ZERO, &Capsule::standing(1.8, 0.3), QueryFilter::default(), |_, _| panic!("boom"));
}

#[test]
#[should_panic(expected = "finite")]
fn non_finite_mover_panics() {
    let w = world();
    let cap = Capsule::new(Vec3::ZERO, Vec3::Y, f32::NAN);
    w.cast_mover(Pos::ZERO, &cap, Vec3::X, QueryFilter::default());
}
