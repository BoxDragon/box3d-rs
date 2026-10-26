//! Triangle meshes and height fields: resting contact, ray casts, holes,
//! materials, keepalive and argument validation.

// `.into()` on f32 coordinates is identity in single precision but converts to
// f64 with the `double-precision` feature.
#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;

const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: i32 = 4;

fn world() -> World {
    World::new(&WorldDef::default()).expect("world")
}

/// `n` x `n` quads of `cell` metres on y = 0, centered on the origin, CCW
/// seen from above (+Y normals). Two triangles per quad.
fn grid(n: u32, cell: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let half = 0.5 * n as f32 * cell;
    let mut v = Vec::new();
    for z in 0..=n {
        for x in 0..=n {
            v.push(Vec3::new(x as f32 * cell - half, 0.0, z as f32 * cell - half));
        }
    }
    let row = n + 1;
    let mut t = Vec::new();
    for z in 0..n {
        for x in 0..n {
            let a = z * row + x;
            let (b, c, d) = (a + 1, a + row, a + row + 1);
            // (x,z), (x,z+1), (x+1,z) is CCW from +Y.
            t.push([a, c, b]);
            t.push([b, c, d]);
        }
    }
    (v, t)
}

fn mesh_ground(w: &mut World, def: &ShapeDef<'_>) -> (BodyId, ShapeId, Mesh) {
    let (v, t) = grid(20, 1.0);
    let mesh = Mesh::new(&MeshDef::new(&v, &t).identify_edges(true)).expect("mesh");
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    let s = w.create_mesh_shape(g, def, &mesh, Vec3::ONE);
    (g, s, mesh)
}

/// 21 x 21 height field with 1 m cells covering [-10, 10]^2, flat at
/// `height`, with an optional hole in the cell whose min corner is (0, 0).
fn height_field(height: f32, hole: bool) -> HeightField {
    let n = 21u32;
    let heights = vec![height; (n * n) as usize];
    let mut mats = vec![0u8; ((n - 1) * (n - 1)) as usize];
    if hole {
        // Cell (x=10, z=10) spans world [0, 1] x [0, 1] once offset by -10.
        mats[(10 * (n - 1) + 10) as usize] = HEIGHT_FIELD_HOLE;
    }
    HeightField::new(&HeightFieldDef::new(&heights, n, n, Vec3::ONE).material_indices(&mats).height_range(-5.0, 5.0))
        .expect("height field")
}

fn settle(w: &mut World, body: BodyId, steps: usize) {
    for _ in 0..steps {
        w.step(DT, SUBSTEPS);
    }
    let _ = w.body_position(body);
}

#[test]
fn box_rests_on_mesh_ground() {
    let mut w = world();
    let (_g, s, _mesh) = mesh_ground(&mut w, &ShapeDef::default());
    assert_eq!(w.shape_type(s), ShapeType::Mesh);
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.3, 3.0, -0.2)));
    w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5));
    settle(&mut w, b, 240);
    let p = w.body_position(b);
    println!("box on mesh rests at y = {}", p.y);
    assert!((p.y as f32 - 0.5).abs() < 0.03, "box should rest on the mesh at y=0.5, got {}", p.y);
    assert!(w.body_linear_velocity(b).length() < 0.05);
}

#[test]
fn sphere_rolls_across_mesh_without_falling_through() {
    // Fast rolling across many triangle edges; identify_edges avoids bumps.
    let mut w = world();
    mesh_ground(&mut w, &ShapeDef::default());
    let b = w.create_body(&BodyDef::dynamic(Pos::new(-8.0, 0.25, 0.3)).linear_velocity(Vec3::new(6.0, 0.0, 0.0)));
    w.create_sphere_shape(b, &ShapeDef::default(), Vec3::ZERO, 0.25);
    let mut min_y = f32::MAX;
    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
        min_y = min_y.min(w.body_position(b).y as f32);
    }
    println!("rolling sphere min y = {min_y}");
    assert!(min_y > 0.2, "sphere sank into the mesh: {min_y}");
}

#[test]
fn box_rests_on_height_field_ground() {
    let mut w = world();
    let hf = height_field(1.0, false);
    let g = w.create_body(&BodyDef::fixed(Pos::new(-10.0, 0.0, -10.0)));
    let s = w.create_height_field_shape(g, &ShapeDef::default(), &hf);
    assert_eq!(w.shape_type(s), ShapeType::HeightField);
    assert!(HeightField::ptr_eq(&w.shape_height_field(s).unwrap(), &hf));
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.4, 4.0, -3.3)));
    w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5));
    settle(&mut w, b, 240);
    let p = w.body_position(b);
    println!("box on height field rests at y = {}", p.y);
    assert!((p.y as f32 - 1.5).abs() < 0.03, "box should rest on the field (h=1) at y=1.5, got {}", p.y);
}

#[test]
fn box_rests_on_sloped_height_field() {
    // Heights rise along x: h = 0.25 * x. The box must stay above the slope.
    let n = 11u32;
    let heights: Vec<f32> = (0..n * n).map(|i| 0.25 * (i % n) as f32).collect();
    let hf = HeightField::new(&HeightFieldDef::new(&heights, n, n, Vec3::new(2.0, 1.0, 2.0))).unwrap();
    let (lo, hi) = hf.bounds();
    assert!((lo.y - 0.0).abs() < 1e-3 && (hi.y - 2.5).abs() < 1e-3, "bounds {lo} {hi}");
    assert_eq!(hf.counts(), (11, 11));
    let mut w = world();
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    w.create_height_field_shape(g, &ShapeDef::default().friction(1.0), &hf);
    let b = w.create_body(&BodyDef::dynamic(Pos::new(10.0, 5.0, 10.0)));
    w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5));
    settle(&mut w, b, 300);
    let p = w.body_position(b);
    let ground = 0.125 * p.x as f32; // slope 0.25 per 2 m cell
    println!("box on slope at {p}, ground {ground}");
    assert!(p.y as f32 > ground, "box fell through the sloped field");
    assert!((p.y as f32 - ground) < 1.0);
}

#[test]
fn ray_hits_mesh_with_per_triangle_material() {
    let mut w = world();
    let (v, t) = grid(4, 1.0);
    // Left half (x < 0) material 0, right half material 1.
    let mats: Vec<u8> = t.iter().map(|tri| u8::from(tri.iter().map(|&i| v[i as usize].x).sum::<f32>() > 0.0)).collect();
    let mesh = Mesh::new(&MeshDef::new(&v, &t).material_indices(&mats)).unwrap();
    assert_eq!(mesh.material_count(), 2);
    assert_eq!(mesh.triangle_count(), 32);
    let materials = [SurfaceMaterial::default().user_material_id(7), SurfaceMaterial::default().user_material_id(9)];
    let g = w.create_body(&BodyDef::fixed(Pos::new(0.0, 1.0, 0.0)));
    let s = w.create_mesh_shape(g, &ShapeDef::default().mesh_materials(&materials), &mesh, Vec3::new(2.0, 1.0, 2.0));
    assert_eq!(w.shape_mesh_material_count(s), 2);
    assert_eq!(w.shape_mesh_material(s, 1).as_raw().userMaterialId, 9);
    assert_eq!(w.shape_mesh_scale(s), Vec3::new(2.0, 1.0, 2.0));
    w.step(DT, SUBSTEPS);

    let down = Vec3::new(0.0, -10.0, 0.0);
    let hit = w.cast_ray_closest(Pos::new(-2.5, 5.0, 0.7), down, QueryFilter::default()).expect("hit left");
    assert_eq!(hit.shape, s);
    assert!((hit.point.y as f32 - 1.0).abs() < 1e-4);
    assert!(hit.normal.dot(Vec3::Y) > 0.999);
    assert_eq!(hit.user_material_id, 7);
    let tri = hit.triangle_index;
    assert!(tri >= 0 && (tri as usize) < mesh.triangle_count());
    assert_eq!(mesh.material_indices()[tri as usize], 0);

    let hit = w.cast_ray_closest(Pos::new(2.5, 5.0, 0.7), down, QueryFilter::default()).expect("hit right");
    assert_eq!(hit.user_material_id, 9);
    assert_eq!(mesh.material_indices()[hit.triangle_index as usize], 1);

    // Outside the scaled mesh (half extent 4) -> miss.
    assert!(w.cast_ray_closest(Pos::new(4.5, 5.0, 0.0), down, QueryFilter::default()).is_none());
    // Meshes are one-sided: a ray from below passes through.
    assert!(w.cast_ray_closest(Pos::new(0.3, -5.0, 0.3), -down, QueryFilter::default()).is_none());

    // Change a mesh material at run time.
    w.set_shape_mesh_material(s, 1, SurfaceMaterial::default().user_material_id(11));
    let hit = w.cast_ray_closest(Pos::new(2.5, 5.0, 0.7), down, QueryFilter::default()).unwrap();
    assert_eq!(hit.user_material_id, 11);
}

#[test]
fn ray_hits_height_field_and_passes_through_hole() {
    let mut w = world();
    let hf = height_field(0.5, true);
    let g = w.create_body(&BodyDef::fixed(Pos::new(-10.0, 0.0, -10.0)));
    let s = w.create_height_field_shape(g, &ShapeDef::default(), &hf);
    // A box floor below the field catches rays through the hole.
    let floor = w.create_body(&BodyDef::fixed(Pos::new(0.0, -3.0, 0.0)));
    let floor_shape = w.create_box_shape(floor, &ShapeDef::default(), Vec3::new(20.0, 0.5, 20.0));
    w.step(DT, SUBSTEPS);

    let down = Vec3::new(0.0, -20.0, 0.0);
    let hit = w.cast_ray_closest(Pos::new(3.3, 5.0, -2.6), down, QueryFilter::default()).expect("field hit");
    assert_eq!(hit.shape, s);
    assert!((hit.point.y as f32 - 0.5).abs() < 1e-3, "hit y {}", hit.point.y);
    assert!(hit.normal.dot(Vec3::Y) > 0.999);
    // Cell (13, 7): triangle 2 * (7 * 20 + 13) or + 1.
    let cell = 7 * 20 + 13;
    assert!(hit.triangle_index == 2 * cell || hit.triangle_index == 2 * cell + 1, "tri {}", hit.triangle_index);

    // Straight through the hole cell [0,1]^2 onto the floor.
    let hit = w.cast_ray_closest(Pos::new(0.5, 5.0, 0.5), down, QueryFilter::default()).expect("floor hit");
    assert_eq!(hit.shape, floor_shape, "ray should pass through the hole");
    assert!((hit.point.y as f32 + 2.5).abs() < 1e-3);

    // Next to the hole is solid.
    let hit = w.cast_ray_closest(Pos::new(1.5, 5.0, 0.5), down, QueryFilter::default()).unwrap();
    assert_eq!(hit.shape, s);

    // A ball dropped into the hole falls through onto the floor.
    let ball = w.create_body(&BodyDef::dynamic(Pos::new(0.5, 2.0, 0.5)));
    w.create_sphere_shape(ball, &ShapeDef::default(), Vec3::ZERO, 0.2);
    for _ in 0..180 {
        w.step(DT, SUBSTEPS);
    }
    let y = w.body_position(ball).y as f32;
    assert!((y + 2.3).abs() < 0.05, "ball should drop through the hole to the floor, y = {y}");
}

#[test]
fn mesh_keepalive_and_release() {
    let mut w = world();
    let mesh = Mesh::grid(8, 8, 1.0, true).unwrap();
    let probe = mesh.clone();
    assert_eq!(probe.strong_count(), 2);

    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    let s1 = w.create_mesh_shape(g, &ShapeDef::default(), &mesh, Vec3::ONE);
    let s2 = w.create_mesh_shape(g, &ShapeDef::default(), &mesh, Vec3::new(1.0, 1.0, -1.0));
    assert_eq!(probe.strong_count(), 4, "user + probe + two shapes");
    drop(mesh); // the user's handle goes away right after shape creation
    assert_eq!(probe.strong_count(), 3);
    assert!(Mesh::ptr_eq(&w.shape_mesh(s1).unwrap(), &probe));

    // Still collides and is hit by rays.
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.2, 2.0, 0.1)));
    w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5));
    settle(&mut w, b, 180);
    assert!((w.body_position(b).y as f32 - 0.5).abs() < 0.03);
    assert!(w.cast_ray_closest(Pos::new(1.1, 3.0, 1.2), Vec3::new(0.0, -5.0, 0.0), QueryFilter::default()).is_some());

    // Destroying one shape releases its reference immediately.
    w.destroy_shape(s2, true);
    assert_eq!(probe.strong_count(), 2);
    assert!(w.is_shape_valid(s1));

    // Destroying the body releases the rest.
    w.destroy_body(g);
    assert!(!w.is_shape_valid(s1));
    assert_eq!(probe.strong_count(), 1);

    // Queued destruction releases at the next step.
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    w.create_mesh_shape(g, &ShapeDef::default(), &probe, Vec3::ONE);
    assert_eq!(probe.strong_count(), 2);
    w.queue_destroy_body(g);
    assert_eq!(probe.strong_count(), 2);
    w.step(DT, SUBSTEPS);
    assert_eq!(probe.strong_count(), 1);

    // Dropping the world releases too; one mesh shared by two worlds.
    let mut w2 = world();
    let g1 = w.create_body(&BodyDef::fixed(Pos::ZERO));
    let g2 = w2.create_body(&BodyDef::fixed(Pos::ZERO));
    w.create_mesh_shape(g1, &ShapeDef::default(), &probe, Vec3::ONE);
    w2.create_mesh_shape(g2, &ShapeDef::default(), &probe, Vec3::ONE);
    assert_eq!(probe.strong_count(), 3);
    drop(w);
    assert_eq!(probe.strong_count(), 2);
    drop(w2);
    assert_eq!(probe.strong_count(), 1);
}

#[test]
fn height_field_keepalive_and_release() {
    let mut w = world();
    let hf = height_field(0.0, false);
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    let s = w.create_height_field_shape(g, &ShapeDef::default(), &hf);
    assert_eq!(hf.strong_count(), 2);
    w.queue_destroy_shape(s, false);
    w.flush_destroys();
    assert_eq!(hf.strong_count(), 1);
    w.create_height_field_shape(g, &ShapeDef::default(), &hf);
    assert_eq!(hf.strong_count(), 2);
    drop(w);
    assert_eq!(hf.strong_count(), 1);
}

#[test]
fn mesh_on_kinematic_body_is_queryable() {
    // Box3D allows meshes on non-static bodies (queries, sensors, movers);
    // only static meshes are meant for contacts.
    let mut w = world();
    let mesh = Mesh::cuboid(Vec3::ZERO, Vec3::splat(1.0), true).unwrap();
    assert_eq!(mesh.triangle_count(), 12);
    let k = w.create_body(&BodyDef::kinematic(Pos::new(0.0, 2.0, 0.0)).linear_velocity(Vec3::new(1.0, 0.0, 0.0)));
    let s = w.create_mesh_shape(k, &ShapeDef::default(), &mesh, Vec3::ONE);
    for _ in 0..60 {
        w.step(DT, SUBSTEPS);
    }
    let hit = w.cast_ray_closest(Pos::new(1.0, 10.0, 0.0), Vec3::new(0.0, -20.0, 0.0), QueryFilter::default());
    let hit = hit.expect("moving mesh box should be hit");
    assert_eq!(hit.shape, s);
    assert!((hit.point.y as f32 - 3.0).abs() < 1e-3);
    assert!(w.body_mass(k) == 0.0);
}

#[test]
fn degenerate_triangles_are_reported() {
    let v = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    // Triangle 1 is collinear (zero area, distinct indices) -> reported.
    // Triangle 2 repeats an index -> dropped silently.
    let t = [[0, 1, 2], [2, 3, 4], [0, 0, 1]];
    let mut degenerate = Vec::new();
    let mesh = Mesh::new_with_degenerates(&MeshDef::new(&v, &t), &mut degenerate).unwrap();
    assert_eq!(degenerate, vec![1]);
    assert_eq!(mesh.degenerate_count(), 1);
    assert_eq!(mesh.triangle_count(), 1);

    // Flat indices, clockwise input: the stored triangle is CCW.
    let flat = [0, 2, 1];
    let cw = Mesh::new(&MeshDef::from_flat_indices(&v, &flat).clockwise(true)).unwrap();
    let tri = cw.triangles()[0];
    let [a, b, c] = tri.map(|i| cw.vertices()[i as usize]);
    assert!((b - a).cross(c - a).y > 0.0, "stored winding should face +Y");

    // All degenerate -> Box3D refuses.
    assert_eq!(Mesh::new(&MeshDef::new(&v, &[[2, 3, 4]])).unwrap_err(), Error::MeshCreationFailed);
}

#[test]
fn welding_merges_duplicate_vertices() {
    // Two triangles with their own copies of the shared edge.
    let v = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
    ];
    let t = [[0, 1, 2], [3, 4, 5]];
    let welded = Mesh::new(&MeshDef::new(&v, &t).weld_vertices(0.001).median_split(true)).unwrap();
    assert_eq!(welded.vertex_count(), 4);
    let raw = Mesh::new(&MeshDef::new(&v, &t)).unwrap();
    assert_eq!(raw.vertex_count(), 6);
    assert!((welded.surface_area() - 1.0).abs() < 1e-5);
    let (lo, hi) = welded.bounds();
    assert_eq!((lo, hi), (Vec3::ZERO, Vec3::new(1.0, 0.0, 1.0)));
}

#[test]
fn invalid_mesh_arguments() {
    let v = [Vec3::ZERO, Vec3::Z, Vec3::X];
    let ok = [[0u32, 1, 2]];
    let inv = |r: Result<Mesh, Error>| matches!(r, Err(Error::InvalidArgument(_)));
    assert!(inv(Mesh::new(&MeshDef::new(&v[..2], &[[0, 1, 1]]))), "too few vertices");
    assert!(inv(Mesh::new(&MeshDef::new(&v, &[]))), "no triangles");
    assert!(inv(Mesh::new(&MeshDef::new(&v, &[[0, 1, 3]]))), "index out of range");
    assert!(inv(Mesh::new(&MeshDef::from_flat_indices(&v, &[0, 1]))), "not a multiple of 3");
    assert!(inv(Mesh::new(&MeshDef::from_flat_indices(&v, &[0, -1, 2]))), "negative index");
    assert!(inv(Mesh::new(&MeshDef::new(&v, &ok).material_indices(&[0, 1]))), "material count");
    assert!(inv(Mesh::new(&MeshDef::new(&v, &ok).weld_vertices(f32::NAN))), "weld tolerance");
    let nan = [Vec3::ZERO, Vec3::Z, Vec3::new(f32::NAN, 0.0, 0.0)];
    assert!(inv(Mesh::new(&MeshDef::new(&nan, &ok))), "non-finite vertex");
    assert!(inv(Mesh::grid(0, 4, 1.0, false)));
    assert!(inv(Mesh::grid(4, 4, -1.0, false)));
    assert!(inv(Mesh::cuboid(Vec3::ZERO, Vec3::new(1.0, 0.0, 1.0), false)));
    assert!(Mesh::new(&MeshDef::new(&v, &ok)).is_ok());
}

#[test]
fn invalid_height_field_arguments() {
    let h = [0.0f32; 12]; // 4 x 3
    let inv = |r: Result<HeightField, Error>| matches!(r, Err(Error::InvalidArgument(_)));
    assert!(HeightField::new(&HeightFieldDef::new(&h, 4, 3, Vec3::ONE)).is_ok());
    assert!(inv(HeightField::new(&HeightFieldDef::new(&h, 3, 3, Vec3::ONE))), "len != count_x * count_z");
    assert!(inv(HeightField::new(&HeightFieldDef::new(&h[..6], 6, 1, Vec3::ONE))), "count < 2");
    assert!(inv(HeightField::new(&HeightFieldDef::new(&h, 4, 3, Vec3::new(1.0, 0.0, 1.0)))), "zero scale");
    assert!(inv(HeightField::new(&HeightFieldDef::new(&h, 4, 3, Vec3::new(1.0, -1.0, 1.0)))), "negative scale");
    let ok_mats = [0u8; 6]; // 3 x 2 cells
    assert!(HeightField::new(&HeightFieldDef::new(&h, 4, 3, Vec3::ONE).material_indices(&ok_mats)).is_ok());
    assert!(
        inv(HeightField::new(&HeightFieldDef::new(&h, 4, 3, Vec3::ONE).material_indices(&ok_mats[..5]))),
        "cell material count"
    );
    assert!(inv(HeightField::new(&HeightFieldDef::new(&h, 4, 3, Vec3::ONE).height_range(1.0, 0.0))), "min > max");
    let mut nan = h;
    nan[3] = f32::INFINITY;
    assert!(inv(HeightField::new(&HeightFieldDef::new(&nan, 4, 3, Vec3::ONE))), "non-finite height");
}

#[test]
#[should_panic(expected = "static body")]
fn height_field_on_dynamic_body_panics() {
    let mut w = world();
    let hf = height_field(0.0, false);
    let b = w.create_body(&BodyDef::dynamic(Pos::ZERO));
    w.create_height_field_shape(b, &ShapeDef::default(), &hf);
}

#[test]
#[should_panic(expected = "must stay static")]
fn height_field_body_cannot_become_dynamic() {
    let mut w = world();
    let hf = height_field(0.0, false);
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    w.create_height_field_shape(g, &ShapeDef::default(), &hf);
    w.set_body_type(g, BodyType::Static); // no-op is fine
    w.set_body_type(g, BodyType::Dynamic);
}

#[test]
fn mesh_body_type_can_change() {
    let mut w = world();
    let (g, s, _mesh) = mesh_ground(&mut w, &ShapeDef::default());
    w.set_body_type(g, BodyType::Kinematic);
    assert_eq!(w.body_type(g), BodyType::Kinematic);
    // The world is still usable (not left locked).
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 5.0, 0.0)));
    assert!(w.is_body_valid(b));
    assert!(w.is_shape_valid(s));
}

#[test]
#[should_panic(expected = "out of range")]
fn mesh_material_index_out_of_range_panics() {
    let mut w = world();
    let (_g, s, _mesh) = mesh_ground(&mut w, &ShapeDef::default());
    assert_eq!(w.shape_mesh_material_count(s), 1);
    w.shape_mesh_material(s, 1);
}

/// 21 x 21 flat field at y = 0 over [-10, 10]^2 once its body sits at
/// (-10, 0, -10): cells with world z < 0 take material 0, the rest material 1.
fn two_material_field() -> HeightField {
    let n = 21u32;
    let heights = vec![0.0; (n * n) as usize];
    let mats: Vec<u8> = (0..(n - 1) * (n - 1)).map(|c| u8::from(c / (n - 1) >= 10)).collect();
    HeightField::new(&HeightFieldDef::new(&heights, n, n, Vec3::ONE).material_indices(&mats)).expect("height field")
}

/// A ball of radius 0.5 started at 3 m/s along +x from x = -8 at world `z`.
fn rolling_ball(w: &mut World, z: f32) -> BodyId {
    let b = w.create_body(&BodyDef::dynamic(Pos::new(-8.0, 0.5, z.into())).linear_velocity(Vec3::new(3.0, 0.0, 0.0)));
    w.create_sphere_shape(b, &ShapeDef::default(), Vec3::ZERO, 0.5);
    b
}

#[test]
fn rolling_resistance_follows_height_field_cells() {
    let mut w = world();
    let hf = two_material_field();
    let g = w.create_body(&BodyDef::fixed(Pos::new(-10.0, 0.0, -10.0)));
    let materials = [SurfaceMaterial::default(), SurfaceMaterial::default().rolling_resistance(0.3)];
    w.create_height_field_shape(g, &ShapeDef::default().mesh_materials(&materials), &hf);
    let free = rolling_ball(&mut w, -5.0);
    let held = rolling_ball(&mut w, 5.0);
    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
    }
    let (xf, xh) = (w.body_position(free).x as f32, w.body_position(held).x as f32);
    let (vf, vh) = (w.body_linear_velocity(free).length(), w.body_linear_velocity(held).length());
    println!("free ball x {xf} v {vf}, held ball x {xh} v {vh}");
    assert!(xf > -4.5 && vf > 1.0, "a ball on material 0 rolls on: x {xf} v {vf}");
    assert!(vh < 0.05, "a ball on material 1 (rolling resistance 0.3) stops: v {vh}");
    assert!(xh < xf - 2.0, "the held ball stops short: {xh} vs {xf}");
}

#[test]
fn rolling_resistance_follows_mesh_triangles() {
    let mut w = world();
    let (v, t) = grid(20, 1.0);
    // Triangles with z > 0 take material 1.
    let mats: Vec<u8> = t.iter().map(|tri| u8::from(tri.iter().map(|&i| v[i as usize].z).sum::<f32>() > 0.0)).collect();
    let mesh = Mesh::new(&MeshDef::new(&v, &t).identify_edges(true).material_indices(&mats)).unwrap();
    let materials = [SurfaceMaterial::default(), SurfaceMaterial::default().rolling_resistance(0.3)];
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    w.create_mesh_shape(g, &ShapeDef::default().mesh_materials(&materials), &mesh, Vec3::ONE);
    let free = rolling_ball(&mut w, -5.0);
    let held = rolling_ball(&mut w, 5.0);
    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
    }
    let (vf, vh) = (w.body_linear_velocity(free).length(), w.body_linear_velocity(held).length());
    println!("mesh: free ball v {vf}, held ball v {vh}");
    assert!(vf > 1.0, "a ball on material 0 rolls on: v {vf}");
    assert!(vh < 0.05, "a ball on material 1 stops: v {vh}");
}

#[test]
fn rolling_resistance_of_a_single_material_mesh_mixes_as_the_max() {
    // As in convex contacts, the larger of the two materials' resistances acts:
    // the ground's stops a ball that has none of its own.
    let mut w = world();
    mesh_ground(&mut w, &ShapeDef::default().rolling_resistance(0.3));
    let b = rolling_ball(&mut w, 0.3);
    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
    }
    let v = w.body_linear_velocity(b).length();
    assert!(v < 0.05, "the mesh's rolling resistance stops the ball: v {v}");
}

#[test]
fn contacts_report_the_height_field_cell_material() {
    let mut w = world();
    let hf = two_material_field();
    let g = w.create_body(&BodyDef::fixed(Pos::new(-10.0, 0.0, -10.0)));
    let materials = [SurfaceMaterial::default().user_material_id(7), SurfaceMaterial::default().user_material_id(9)];
    let field = w.create_height_field_shape(g, &ShapeDef::default().mesh_materials(&materials), &hf);
    let crate_def = ShapeDef::default().material(SurfaceMaterial::default().user_material_id(3)).enable_contact_events(true);
    // One box on material 1, one straddling the seam at z = 0.
    let on_one = w.create_body(&BodyDef::dynamic(Pos::new(2.5, 0.6, 4.5)));
    let one_shape = w.create_box_shape(on_one, &crate_def, Vec3::splat(0.5));
    let straddling = w.create_body(&BodyDef::dynamic(Pos::new(-4.5, 0.6, 0.0)));
    w.create_box_shape(straddling, &crate_def, Vec3::new(0.5, 0.5, 0.9));

    // The begin-touch event's contact names the cell's material.
    let mut begun = false;
    for _ in 0..30 {
        w.step(DT, SUBSTEPS);
        for e in w.contact_events().begin() {
            if e.shape_a() != one_shape && e.shape_b() != one_shape {
                continue;
            }
            let c = w.begin_contact(e).expect("begin contact is live");
            let field_is_a = c.shape_a() == field;
            let point = c.manifolds().iter().flat_map(|m| m.points()).next().expect("a point");
            let (a, b) = c.user_materials(point);
            assert_eq!(if field_is_a { (a, b) } else { (b, a) }, (9, 3));
            begun = true;
        }
    }
    assert!(begun, "the box began touching the field");

    // Resting contacts name each point's cell.
    let mut scratch = ContactScratch::new();
    let mut seen = Vec::new();
    for body in [on_one, straddling] {
        let mut ids = Vec::new();
        w.visit_body_contacts(body, &mut scratch, |c| {
            for p in c.manifolds().iter().flat_map(|m| m.points()) {
                let (a, b) = c.user_materials(p);
                let (ground, other) = if c.shape_a() == field { (a, b) } else { (b, a) };
                assert_eq!(other, 3);
                let cell = p.triangle_index() >> 1;
                assert_eq!(ground, if cell / 20 >= 10 { 9 } else { 7 }, "triangle {}", p.triangle_index());
                ids.push(ground);
            }
        });
        ids.sort();
        ids.dedup();
        seen.push(ids);
    }
    assert_eq!(seen, [vec![9], vec![7, 9]], "the straddling box touches both cells' materials");
}
