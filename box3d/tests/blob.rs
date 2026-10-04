//! Hull and mesh blobs: `as_bytes` / `from_bytes` round trips are byte-exact, stale
//! versions and truncated or corrupt bytes are rejected, and loaded geometry simulates.

// `.into()` on f32 coordinates is identity in single precision but converts to
// f64 with the `double-precision` feature.
#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;
use core::mem::offset_of;

/// Deterministic pseudo-random sequence (xorshift64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn rock_points() -> Vec<Vec3> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    (0..64).map(|_| Vec3::new(rng.unit() - 0.5, 0.6 * (rng.unit() - 0.5), 0.8 * (rng.unit() - 0.5))).collect()
}

fn rock() -> Hull {
    Hull::from_points(&rock_points(), 32).unwrap()
}

fn hulls() -> Vec<(&'static str, fn() -> Hull)> {
    vec![
        ("rock", rock),
        ("cylinder", || Hull::cylinder(1.0, 0.4, -0.5, 12).unwrap()),
        ("cuboid", || Hull::cuboid(Vec3::new(0.5, 0.25, 1.0))),
        ("transformed cuboid", || {
            Hull::transformed_cuboid(
                Vec3::new(0.3, 0.2, 0.1),
                Transform::new(Vec3::new(0.1, 0.2, 0.3), Quat::from_rotation_y(0.7)),
            )
        }),
    ]
}

/// A bumpy 16 x 16 grid with two materials and identified edges.
fn terrain() -> Mesh {
    let n = 16u32;
    let mut v = Vec::new();
    for z in 0..=n {
        for x in 0..=n {
            let (fx, fz) = (x as f32 - 8.0, z as f32 - 8.0);
            v.push(Vec3::new(fx, 0.1 * (fx * 0.7).sin() * (fz * 0.5).cos(), fz));
        }
    }
    let mut t = Vec::new();
    for z in 0..n {
        for x in 0..n {
            let i = z * (n + 1) + x;
            t.push([i, i + n + 1, i + 1]);
            t.push([i + 1, i + n + 1, i + n + 2]);
        }
    }
    let m: Vec<u8> = (0..t.len()).map(|i| (i % 2) as u8).collect();
    Mesh::new(&MeshDef::new(&v, &t).material_indices(&m).identify_edges(true)).unwrap()
}

fn meshes() -> Vec<(&'static str, fn() -> Mesh)> {
    vec![
        ("terrain", terrain),
        ("grid", || Mesh::grid(40, 30, 0.5, true).unwrap()),
        ("box", || Mesh::cuboid(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 1.0, 2.0), false).unwrap()),
    ]
}

/// `bytes` at an odd address, so loading must not assume alignment.
fn unaligned(bytes: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(bytes.len() + 1);
    v.push(0);
    v.extend_from_slice(bytes);
    v
}

#[test]
fn round_trip_equals_a_fresh_build() {
    for (name, make) in hulls() {
        let built = make();
        let shifted = unaligned(built.as_bytes());
        let loaded = Hull::from_bytes(&shifted[1..]).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(loaded.as_bytes(), make().as_bytes(), "{name}");
        assert_eq!(loaded.vertex_count(), built.vertex_count(), "{name}");
        assert_eq!(loaded.as_raw() as usize % 16, 0, "{name}: Box3D allocation is 16-byte aligned");
    }
    for (name, make) in meshes() {
        let built = make();
        let shifted = unaligned(built.as_bytes());
        let loaded = Mesh::from_bytes(&shifted[1..]).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(loaded.as_bytes(), make().as_bytes(), "{name}");
        assert_eq!(loaded.triangles(), built.triangles(), "{name}");
        assert_eq!(loaded.as_raw() as usize % 16, 0, "{name}: Box3D allocation is 16-byte aligned");
    }
}

#[test]
fn version_mismatch_is_rejected() {
    let mut hull = rock().as_bytes().to_vec();
    hull[0] ^= 1;
    assert_eq!(Hull::from_bytes(&hull).unwrap_err(), Error::BlobVersionMismatch);
    let mut mesh = terrain().as_bytes().to_vec();
    mesh[7] ^= 0x80;
    assert_eq!(Mesh::from_bytes(&mesh).unwrap_err(), Error::BlobVersionMismatch);
    assert_eq!(Hull::VERSION, sys::B3_HULL_VERSION);
    assert_eq!(Mesh::VERSION, 0xAAAB_9A00_F1A8_AAF7);
}

/// Set the little-endian i32 at `offset`.
fn put_i32(bytes: &mut [u8], offset: usize, v: i32) {
    bytes[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
}

fn get_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

// Header field offsets.
const HULL_EDGE_COUNT: usize = offset_of!(sys::b3HullData, edgeCount);
const HULL_EDGE_OFFSET: usize = offset_of!(sys::b3HullData, edgeOffset);
const HULL_BYTE_COUNT: usize = offset_of!(sys::b3HullData, byteCount);
const MESH_BYTE_COUNT: usize = offset_of!(sys::b3MeshData, byteCount);
const MESH_NODE_OFFSET: usize = offset_of!(sys::b3MeshData, nodeOffset);
const MESH_TRIANGLE_OFFSET: usize = offset_of!(sys::b3MeshData, triangleOffset);

#[test]
fn truncated_bytes_are_rejected() {
    let hull = rock();
    let bytes = hull.as_bytes();
    for len in 0..bytes.len() {
        assert!(matches!(Hull::from_bytes(&bytes[..len]), Err(Error::InvalidBlob(_))), "hull prefix {len}");
        // Also with byteCount patched to claim the shorter length: the arrays no longer fit.
        if len >= HULL_BYTE_COUNT + 4 {
            let mut cut = bytes[..len].to_vec();
            put_i32(&mut cut, HULL_BYTE_COUNT, len as i32);
            assert!(matches!(Hull::from_bytes(&cut), Err(Error::InvalidBlob(_))), "patched hull prefix {len}");
        }
    }
    let mesh = terrain();
    let bytes = mesh.as_bytes();
    for len in (0..bytes.len()).step_by(7) {
        assert!(matches!(Mesh::from_bytes(&bytes[..len]), Err(Error::InvalidBlob(_))), "mesh prefix {len}");
        if len >= MESH_BYTE_COUNT + 4 {
            let mut cut = bytes[..len].to_vec();
            put_i32(&mut cut, MESH_BYTE_COUNT, len as i32);
            assert!(matches!(Mesh::from_bytes(&cut), Err(Error::InvalidBlob(_))), "patched mesh prefix {len}");
        }
    }
}

/// Store a correct content hash in edited blob bytes (as a buggy cook would), so the
/// structural checks are what rejects them.
fn rehash(bytes: &mut [u8], update: unsafe extern "C" fn(*mut u8) -> u64) {
    let mut aligned = vec![0u64; bytes.len().div_ceil(8)];
    // SAFETY: aligned holds at least bytes.len() bytes.
    let buf = unsafe { core::slice::from_raw_parts_mut(aligned.as_mut_ptr() as *mut u8, bytes.len()) };
    buf.copy_from_slice(bytes);
    // SAFETY: an 8-byte aligned blob whose byteCount equals its length (callers keep it).
    unsafe { update(buf.as_mut_ptr()) };
    bytes.copy_from_slice(buf);
}

unsafe extern "C" fn update_hull(p: *mut u8) -> u64 {
    // SAFETY: forwarded from rehash.
    unsafe { sys::b3UpdateHullHash(p.cast()) }
}

unsafe extern "C" fn update_mesh(p: *mut u8) -> u64 {
    // SAFETY: forwarded from rehash.
    unsafe { sys::b3UpdateMeshHash(p.cast()) }
}

#[test]
fn corrupt_bytes_are_rejected() {
    let reject_hull = |mut bytes: Vec<u8>, what: &str| {
        assert_eq!(Hull::from_bytes(&bytes).unwrap_err(), Error::InvalidBlob("hash mismatch"), "{what}");
        rehash(&mut bytes, update_hull);
        let err = Hull::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidBlob(m) if m != "hash mismatch"), "{what}: {err}");
    };
    let good = rock().as_bytes().to_vec();
    let edges = get_i32(&good, HULL_EDGE_OFFSET) as usize;

    let mut b = good.clone();
    let past = b.len() as i32 - 4;
    put_i32(&mut b, HULL_EDGE_OFFSET, past);
    reject_hull(b, "edge array past the end");
    let mut b = good.clone();
    put_i32(&mut b, HULL_EDGE_OFFSET, -8);
    reject_hull(b, "negative offset");
    let mut b = good.clone();
    put_i32(&mut b, HULL_EDGE_COUNT, 1000);
    reject_hull(b, "edge count over the limit");
    let mut b = good.clone();
    b[edges] = 255; // first half-edge's `next`
    reject_hull(b, "edge next out of range");
    let mut b = good.clone();
    b[edges + 1] = 7; // first half-edge's `twin`
    reject_hull(b, "twin mismatch");

    let reject_mesh = |mut bytes: Vec<u8>, what: &str| {
        assert_eq!(Mesh::from_bytes(&bytes).unwrap_err(), Error::InvalidBlob("hash mismatch"), "{what}");
        rehash(&mut bytes, update_mesh);
        let err = Mesh::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidBlob(m) if m != "hash mismatch"), "{what}: {err}");
    };
    let good = terrain().as_bytes().to_vec();
    let nodes = get_i32(&good, MESH_NODE_OFFSET) as usize;
    let triangles = get_i32(&good, MESH_TRIANGLE_OFFSET) as usize;

    let mut b = good.clone();
    put_i32(&mut b, triangles, 1 << 20);
    reject_mesh(b, "triangle vertex index out of range");
    let mut b = good.clone();
    let first = get_i32(&b, triangles);
    put_i32(&mut b, triangles + 4, first);
    reject_mesh(b, "repeated triangle vertex");
    // The root is internal: its tag/childOffset word follows lowerBound.
    let mut b = good.clone();
    put_i32(&mut b, nodes + 12, 0x7fff_fff0);
    reject_mesh(b, "right child past the node array");
    let mut b = good.clone();
    put_i32(&mut b, nodes + 12, 1 << 2);
    reject_mesh(b, "right child equal to the left child");
    let mut b = good.clone();
    b[nodes..nodes + 4].copy_from_slice(&1.0e9f32.to_le_bytes());
    reject_mesh(b, "root does not bound its children");
}

/// Drop `hull` on a box and `mesh` under a sphere, and cast rays at both.
fn simulate(hull: Option<&Hull>, mesh: Option<&Mesh>) {
    let mut w = World::new(&WorldDef::default()).unwrap();
    let g = w.create_body(&BodyDef::fixed(Pos::ZERO));
    if let Some(m) = mesh {
        w.create_mesh_shape(g, &ShapeDef::default(), m, Vec3::ONE);
    } else {
        w.create_box_shape(g, &ShapeDef::default(), Vec3::new(5.0, 0.1, 5.0));
    }
    let b = w.create_body(&BodyDef::dynamic(Pos::new(0.3, 1.0, 0.2)));
    match hull {
        Some(h) => w.create_hull_shape(b, &ShapeDef::default(), h),
        None => w.create_sphere_shape(b, &ShapeDef::default(), Vec3::ZERO, 0.3),
    };
    for _ in 0..10 {
        w.step(1.0 / 60.0, 4);
        w.cast_ray_closest(Pos::new(0.0, 5.0, 0.0), Vec3::new(0.1, -10.0, 0.0), QueryFilter::default());
    }
}

/// Bit flips: every flip of a stored blob fails its hash. With the hash forged to match,
/// loading must still never read out of bounds or panic, and what it accepts must simulate.
#[test]
fn bit_flips_never_crash() {
    let good = rock().as_bytes().to_vec();
    let mut accepted = 0;
    for i in 0..good.len() {
        for bit in 0..8 {
            let mut b = good.clone();
            b[i] ^= 1 << bit;
            assert!(Hull::from_bytes(&b).is_err(), "hull flip at {i}:{bit} accepted");
            if (HULL_BYTE_COUNT..HULL_BYTE_COUNT + 4).contains(&i) {
                continue; // rehash trusts byteCount
            }
            rehash(&mut b, update_hull);
            if let Ok(h) = Hull::from_bytes(&b) {
                accepted += 1;
                simulate(Some(&h), None);
            }
        }
    }
    let good = terrain().as_bytes().to_vec();
    let mut rng = Rng(42);
    for _ in 0..4000 {
        let mut b = good.clone();
        let i = (rng.next() % b.len() as u64) as usize;
        b[i] ^= 1 << (rng.next() % 8);
        assert!(Mesh::from_bytes(&b).is_err(), "mesh flip at {i} accepted");
        if (MESH_BYTE_COUNT..MESH_BYTE_COUNT + 4).contains(&i) {
            continue;
        }
        rehash(&mut b, update_mesh);
        if let Ok(m) = Mesh::from_bytes(&b) {
            accepted += 1;
            simulate(None, Some(&m));
        }
    }
    // Forged flips in floats, flags or padding can pass the structural checks.
    println!("accepted {accepted} forged blobs");
}

#[test]
fn loaded_geometry_simulates() {
    let mesh = Mesh::from_bytes(Mesh::grid(20, 20, 1.0, true).unwrap().as_bytes()).unwrap();
    let hull = Hull::from_bytes(Hull::cuboid(Vec3::splat(0.25)).as_bytes()).unwrap();
    let rock = Hull::from_bytes(rock().as_bytes()).unwrap();

    let mut w = World::new(&WorldDef::default()).unwrap();
    let ground = w.create_body(&BodyDef::fixed(Pos::ZERO));
    w.create_mesh_shape(ground, &ShapeDef::default(), &mesh, Vec3::ONE);
    drop(mesh); // the world keeps the loaded mesh alive
    let cube = w.create_body(&BodyDef::dynamic(Pos::new(-2.0, 1.0, 0.0)));
    w.create_hull_shape(cube, &ShapeDef::default(), &hull);
    let stone = w.create_body(&BodyDef::dynamic(Pos::new(2.0, 1.0, 0.0)));
    w.create_hull_shape(stone, &ShapeDef::default(), &rock);
    for _ in 0..180 {
        w.step(1.0 / 60.0, 4);
    }
    let y = w.body_position(cube).y as f32;
    assert!((y - 0.25).abs() < 0.02, "cube rests on the loaded mesh, y = {y}");
    let y = w.body_position(stone).y as f32;
    assert!(y > 0.0 && y < 0.4, "rock rests on the loaded mesh, y = {y}");
    assert!(w.body_linear_velocity(stone).length() < 0.05);
}
