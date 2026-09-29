//! Mesh / height-field memory is released exactly when the last shape (and
//! handle) is gone, measured with Box3D's global byte counter. A single test
//! in its own binary so no other test allocates through Box3D concurrently.

#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;

#[test]
fn mesh_and_height_field_bytes_are_released_with_their_shapes() {
    assert_eq!(bytes_allocated(), 0);
    let mut w = World::new(&WorldDef::default()).unwrap();
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    let base = bytes_allocated();

    let mesh = Mesh::grid(32, 32, 0.5, true).unwrap();
    let mesh_bytes = mesh.byte_count() as i64;
    assert_eq!(bytes_allocated() - base, mesh_bytes, "the mesh blob is Box3D-allocated");
    let s = w.create_mesh_shape(g, &ShapeDef::default(), &mesh, Vec3::ONE);
    drop(mesh); // world keeps it alive
    let with_shape = bytes_allocated();
    assert!(with_shape - base >= mesh_bytes, "mesh must still be allocated");

    // Still usable after the handle is gone.
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.0, 2.0, 0.0)));
    w.create_sphere_shape(b, &ShapeDef::default(), Vec3::ZERO, 0.3);
    for _ in 0..120 {
        w.step(1.0 / 60.0, 4);
    }
    assert!((w.body_position(b).y as f32 - 0.3).abs() < 0.03);
    assert!(w.is_shape_valid(s));

    let heights = vec![0.0f32; 64 * 64];
    let hf = HeightField::new(&HeightFieldDef::new(&heights, 64, 64, Vec3::ONE)).unwrap();
    let hf_bytes = hf.byte_count() as i64;
    let g2 = w.create_body(&BodyDef::fixed(Pos::new(100.0, 0.0, 0.0)));
    w.create_height_field_shape(g2, &ShapeDef::default(), &hf);
    drop(hf);

    let before = bytes_allocated();
    w.destroy_body(g);
    let freed = before - bytes_allocated();
    println!("mesh {mesh_bytes} bytes, destroy_body freed {freed}");
    assert!(freed >= mesh_bytes, "destroying the body frees the mesh ({freed} < {mesh_bytes})");

    let before = bytes_allocated();
    w.destroy_body(g2);
    let freed = before - bytes_allocated();
    assert!(freed >= hf_bytes, "destroying the body frees the height field ({freed} < {hf_bytes})");

    drop(w);
    assert_eq!(bytes_allocated(), 0, "everything released with the world");
}
