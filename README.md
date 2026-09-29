# box3d-rs

Rust bindings for [Box3D](https://github.com/erincatto/box3d), Erin Catto's
3D rigid-body physics engine (C17, MIT).

- **`box3d-sys`**: raw FFI. Box3D is vendored as a git submodule and compiled with `cc`. Pregenerated bindings are the default, so no libclang is needed.
- **`box3d`**: a thin safe wrapper using glam 0.30. It is designed so a game loop allocates nothing after warm-up.

See [PRIOR_WORK.md](PRIOR_WORK.md) for the other Rust bindings and how this one compares.

## Pinned upstream

`box3d-sys/box3d` is a submodule of `https://github.com/erincatto/box3d`, pinned to
**`9f998c862d54c03a633ecea3831937385c78b532`** (2026-09-27). At that commit `b3GetVersion()` reports 0.2.0.

## Build

```sh
git clone --recursive <this repo>        # or: git submodule update --init
cargo build
cargo test                               # physics, API, terrain, mover and no-alloc tests
cargo test --features box3d/double-precision
cargo run -p box3d --example hello       # drops a crate, prints its height
```

A C17 compiler (gcc, clang or MSVC) is the only native requirement.

Box3D is slow when built unoptimized. This workspace therefore sets `[profile.dev.package.box3d-sys] opt-level = 3`. Add the same override to the workspace that consumes these crates.

### C build settings

These mirror Box3D's CMake setup:

- `-std=gnu17 -ffp-contract=off` on gcc/clang, and `/std:c17` on MSVC. `-ffp-contract=off` keeps results cross-platform deterministic, as CMake does.
- SIMD comes from the compiler's predefines: SSE2 on x86-64 and NEON on aarch64. There is no AVX2 path at this commit.
- `NDEBUG` is defined whenever Rust debug assertions are off.
- `BOX3D_VALIDATE` is only defined with the `validate` feature, so it is always off in release.
- Links `m` and `pthread` on Unix.

### Features

| crate | feature | effect |
|---|---|---|
| both | `glam` (default) | `From`/`Into` between `b3Vec3`/`b3Quat`/`b3Pos` and glam. The safe crate always uses glam; the feature only adds the conversions on the sys types. |
| both | `double-precision` | Defines `BOX3D_DOUBLE_PRECISION`. `b3Pos` becomes f64, and `box3d::Pos` becomes `DVec3` instead of `Vec3`. This changes the ABI; matching bindings are committed. |
| both | `bindgen` | Generate bindings at build time instead of using the committed ones. |
| both | `validate` | `BOX3D_VALIDATE`: heavy internal checks, debug only. |
| both | `disable-simd` | Scalar fallback. |
| sys | `enable-assert` | Keep Box3D asserts on in release. |

### Regenerating bindings

Do this after moving the submodule to a new commit. It needs libclang, plus rustfmt for readable output.

```sh
git -C box3d-sys/box3d checkout <new-commit>
BOX3D_SYS_UPDATE_BINDINGS=1 cargo build -p box3d-sys --features bindgen
BOX3D_SYS_UPDATE_BINDINGS=1 cargo build -p box3d-sys --features bindgen,double-precision
```

These write `box3d-sys/src/bindings.rs` and `bindings_double.rs`.

The binding generation works as follows:

- The allowlist is `b3.*` and `B3_.*`.
- Enums are emitted as constants, for example `sys::b3BodyType_b3_dynamicBody`.
- Static-const globals such as `b3Vec3_zero` and `b3_nullBodyId` are blocklisted and re-provided in `box3d-sys/src/consts.rs`.

## The safe API in one screen

```rust
use box3d::*;

let mut world = World::new(&WorldDef::default().gravity(Vec3::new(0.0, -10.0, 0.0)))?;
let ground = world.create_body(&BodyDef::fixed(Pos::new(0.0, -0.5, 0.0)));
world.create_box_shape(ground, &ShapeDef::default(), Vec3::new(50.0, 0.5, 50.0));

let body = world.create_body(&BodyDef::dynamic(Pos::new(0.0, 4.0, 0.0)).user_data(entity.to_bits()));
world.create_box_shape(body, &ShapeDef::default().density(500.0).enable_contact_events(true), Vec3::splat(0.5));

// Static terrain: the world keeps the mesh alive for as long as the shape exists.
let mesh = Mesh::new(&MeshDef::new(&vertices, &triangles).identify_edges(true))?;
world.create_mesh_shape(ground, &ShapeDef::default(), &mesh, Vec3::ONE);
drop(mesh);

loop {
    world.step(1.0 / 60.0, 4);                       // flushes the destroy queue first
    for m in world.body_events().moves() {           // &[BodyMoveEvent], borrowed, no copy
        sync(m.user_data(), m.position(), m.rotation());
    }
    for c in world.contact_events().begin() { /* c.shape_a(), c.shape_b() */ }
    let hit = world.cast_ray_closest(origin, dir * 100.0, QueryFilter::default());
    world.cast_ray(origin, dir * 100.0, QueryFilter::default(), |h| { /* ... */ h.fraction });
}
```

### Design

- **Handles.**
  - `BodyId`, `ShapeId` and `JointId` are 8-byte `repr(transparent)` `Copy` ids, and `WorldId` is 4 bytes.
  - Typed joint ids (`DistanceJointId`, ...) are `Into<JointId>`.
  - There is no registry: ids are Box3D's own generational ids.
  - Every call validates the id (`b3*_IsValid` plus a same-world check). A stale or foreign id panics in Rust instead of hitting UB in C.
- **User data.** A `u64` on body, shape and joint defs, meant for something like hecs `Entity::to_bits()`. Pointers are never stored there.
- **Defs.** `BodyDef`, `ShapeDef`, `SurfaceMaterial` and the joint defs are `repr(transparent)` value wrappers. Their `Default` comes from `b3Default*Def()`, and they take by-value builder setters. `ShapeDef<'m>` borrows mesh materials as a slice.
- **Events.** `body_events()`, `contact_events()` (begin, end, hit), `sensor_events()` and `joint_events()` return slices that point straight into Box3D's buffers.
  - `step(&mut self)` makes the borrow checker reject holding them across a step. A `compile_fail` doctest checks this.
- **Queries.** `cast_ray`, `cast_shape`, `overlap_aabb` and `overlap_shape` take a generic `FnMut`.
  - The closure goes to C as `&mut F` through a monomorphized `extern "C"` trampoline, with no boxing.
  - `cast_ray_closest` returns by value.
  - Collection helpers such as `body_shapes` fill any `&mut impl Extend`.
- **Meshes and height fields.** Box3D copies hulls into the world, but a mesh or height-field shape keeps a *raw pointer* to its `b3MeshData` / `b3HeightFieldData`.
  - `Mesh` and `HeightField` are immutable `Arc` handles, so `Clone` is a refcount bump. One mesh can back any number of shapes, each with its own scale, in any number of worlds.
  - `create_mesh_shape(body, def, &mesh, scale)` and `create_height_field_shape(body, def, &hf)` store a clone in the `World`. The clone is released when the shape goes away: `destroy_shape`, `destroy_body`, the destroy queue (at `flush_destroys`/`step`), or dropping the world. So the user's handle can be dropped right after creating shapes. Pruning is a non-allocating `retain` in those calls; `step` never touches it unless the destroy queue had entries.
  - Build meshes with `Mesh::new(&MeshDef::new(&vertices, &[[u32; 3]]))` or `MeshDef::from_flat_indices(&vertices, &[i32])`, plus `material_indices`, `weld_vertices(tol)`, `median_split`, `identify_edges` and `clockwise`. `Mesh::new_with_degenerates` also reports the dropped degenerate triangles. `Mesh::grid` and `Mesh::cuboid` wrap Box3D's helpers.
  - Height fields: `HeightField::new(&HeightFieldDef::new(&heights, count_x, count_z, scale))`, with `heights[z * count_x + x]`, optional per-cell `material_indices` (`HEIGHT_FIELD_HOLE` = 255 for holes) and `height_range(min, max)` for the 16-bit quantization.
  - Counts, index ranges, lengths and finiteness are validated up front and return `Error::InvalidArgument`, because Box3D only asserts on them.
  - Per-triangle materials come from `ShapeDef::mesh_materials`. `shape_mesh_material(_count)` and `set_shape_mesh_material` access them at run time.
- **Character mover.** `collide_mover(origin, &capsule, filter, |shape, &[PlaneResult]| ..)`, `cast_mover` / `cast_mover_filtered(.., |shape| keep)`, `solve_planes(delta, &mut [CollisionPlane])` and `clip_vector(v, &[CollisionPlane])`, plus `body_collide_mover` and `body_time_of_impact_mover`.
  - `Capsule`, `Plane`, `PlaneResult`, `CollisionPlane` and `BodyPlaneResult` are `repr(C)` and layout-identical to the C types (checked at compile time). Plane slices are passed to and from C without copying, and a mover update into a fixed-size `[CollisionPlane; N]` allocates nothing.
  - Everything is relative to the query `origin`, as in Box3D, so it stays precise in large worlds.
- **Callbacks.** `set_custom_filter` and `set_pre_solve` take `Box<dyn Fn .. + Send + Sync>`, stored in the `World`. They may run on worker threads when `worker_count > 1`.
- **Panics.** Panics in any Rust callback are caught at the C boundary. For queries they are resumed after the call returns, and for callbacks after `step`.
- **Destruction.** No id has a destroying `Drop`; only `World`'s `Drop` destroys simulation objects. `Hull`, `Mesh` and `HeightField` free their own geometry (meshes and height fields once the last shape using them is gone).
  - Destroy right away with `destroy_body`, `destroy_shape` or `destroy_joint`.
  - Or defer with `queue_destroy_*`. The queue is pre-sized (`WorldDef::destroy_queue_capacity`, default 256), reused, and drained at the start of `step`. Stale entries are skipped.
- **Allocator.** `unsafe fn set_allocator(alloc, free)` wraps `b3SetAllocator`, and `bytes_allocated()` wraps `b3GetByteCount`.
  - It is `unsafe` because the compiler cannot check an allocator's contract.
  - It returns `Error::AllocatorBusy` while Box3D holds memory.
- **World setup.** `WorldDef` exposes `capacity(Capacity { .. })` (`b3Capacity`) and `worker_count` (default 1, so no internal threads).
- **Threading.** `World` is `Send` and `!Sync`. World create and destroy go through a global lock, because Box3D's world table is a global array.
- **glam conversions.** These happen by value; glam's `Quat` is never pointer-cast.

### Box3D quirks worth knowing

- **Distance joints.**
  - A rod is `enable_spring = false`, the default: rigid at `length`.
  - A rope is `DistanceJointDef::rope(a, b, max)`: spring on with `hertz = 0` (zero stiffness, so slack) and the limit on, giving a length range of `[0, max]`.
  - Winch a rope with `set_length_range`. `length` alone does nothing useful for a rope.
- **Rolling resistance.**
  - Set it on the shape's `SurfaceMaterial`. Pairs mix as `max(a, b)`, scaled by the larger "rolling radius" of the pair: the sphere or capsule radius, or `0.25 * inner_radius` for a hull. The C header says it applies to spheres and capsules only, but hulls get it with that reduced radius. Mesh contacts use only the non-mesh shape's value.
  - It is a cap on an angular impulse that opposes *all* relative angular velocity at the contact, including spin about the normal. The cap is `rr * radius * normal_impulse`. So it only acts while touching and under load; there is no damping in free flight.
  - `0.0` disables it. With `0.5` and friction 0.8, a 0.25 m capsule rolling at 3 m/s stops in under 5 s; without it, it keeps rolling (see the test).
- **Hulls.**
  - `Hull::cuboid`, `Hull::cylinder` and `Hull::from_points` build a hull that Rust owns: `b3DestroyHull` for computed hulls, and a boxed `b3BoxHull` for boxes.
  - Creating a shape clones the hull into the world's hull database. The database is deduplicated by content and refcounted, so 300 boxes of the same size cost one copy. A `Hull` can be dropped or reused right after `create_hull_shape`.
  - `create_box_shape` builds its `b3BoxHull` on the stack, so no Rust allocation is needed.
  - `from_points` can fail on degenerate input, and so can `create_transformed_hull_shape`, which re-hulls the baked geometry. Both return `Err(HullCreationFailed)` rather than a null id.
  - Cylinders are hulls with 3 to 32 sides, aligned to the Y axis.
- **Shape filters.** Default category and mask bits are all ones. A query mask only excludes shapes that were given a narrower category.
- **Sensors.** `enable_sensor_events` defaults to false on *both* the sensor and the visitor, so set it on both.
- **Meshes.**
  - Box3D says "mesh collision only creates contacts on static bodies". It does not enforce that, and neither does the wrapper: a mesh on a kinematic body works for rays, sensors and movers. A mesh has no mass, and mesh-vs-mesh/height-field pairs never collide. Keep collision meshes static.
  - Meshes are one-sided: rays, shape casts and movers ignore back faces.
  - `b3CreateMesh` reorders triangles into BVH order, and it drops degenerate triangles. The `triangle_index` in `RayHit`/`PlaneResult` indexes `Mesh::triangles()`, **not** your input order. Use `Mesh::triangles()`/`material_indices()` to map a hit back.
  - Triangles that repeat a vertex index are dropped silently and are not counted as degenerate.
  - `identify_edges(true)` is worth it for ground meshes: it avoids internal-edge bumps.
- **Height fields.**
  - Static-only. `b3CreateHeightFieldShape` returns a null id on a non-static body; the wrapper panics instead.
  - **Box3D bug:** `b3Body_SetType` to a non-static type on a body with a height-field (or compound) shape returns early *with the world still locked*. After that every create/destroy silently fails (null ids in release; an assert in debug). `set_body_type` therefore panics for bodies carrying a height field created through this API.
  - Cell `(x, z)` has triangles `2 * (z * (count_x - 1) + x)` and `+ 1`. The field spans `[0, scale.x * (count_x - 1)] x [0, scale.z * (count_z - 1)]` in local space, and heights are scaled by `scale.y`.
- **Character mover.**
  - `b3World_CollideMover` reports the mover's own shapes too, so filter them in the closure. `cast_mover` ignores shapes the mover already overlaps.
  - `solve_planes` leaves up to `B3_LINEAR_SLOP` (5 mm) of penetration by design.
  - `PlaneResult::triangle_index` is 0 (not -1) for convex shapes.
  - The body-level `body_collide_mover` / `body_time_of_impact_mover` only consider convex shapes (spheres, capsules, hulls) on the body.

### Allocation behaviour

`tests/no_alloc.rs` installs a counting `b3SetAllocator` and a counting `#[global_allocator]`. It builds a 10x10x3 wall of 1 m boxes on the ground plus a distance-joint pendulum, and warms up for 120 steps. It then measures 600 steps; each step does `step`, iterates `body_events().moves()` and the contact events, and makes 10 `cast_ray` closure calls.

- **Steady state** (300 awake resting boxes, a swinging pendulum): **0 Rust allocations and 0 Box3D allocations.**
- **Impact** (a heavy bob keeps smashing the wall, so the contact set keeps changing): 0 Rust allocations.
  - Box3D made 1 allocation of 256 bytes when a constraint-graph color array reached a new high-water mark. The path is `b3World_Step`, then `b3Collide`, then `b3AddContactToGraph`, then `b3Array_Push(color->convexContacts)`, then `b3GrowAlloc`.
  - `b3Capacity` does not pre-size these per-color arrays; `b3CreateGraph` reserves only the body bitsets. It is amortized doubling, so rare and bounded.
  - The test prints a backtrace for any such allocation.
- **Terrain** (a mesh ground and a height-field ground with 50 awake boxes resting on them, rays hitting both, and a character-mover update every step: `collide_mover` into a fixed `[CollisionPlane; 32]`, `solve_planes`, `cast_mover_filtered` and `clip_vector`): **0 Rust allocations and 0 Box3D allocations.** The `Mesh`/`HeightField` handles are dropped right after shape creation, so this also exercises the world keeping them alive.

### Not wrapped (use `box3d::sys`)

- Compound shapes, and `b3Shape_*` accessors beyond the common ones.
- Debug draw.
- Recording and replay.
- Explosions.
- Parallel and filter joints.
- Task-system callbacks (`enqueueTask`/`finishTask`). With `worker_count > 1`, Box3D's built-in scheduler is used.
- Friction and restitution mixing callbacks.
- Contact-data queries (`b3Body_GetContactData`, `b3Shape_GetContactData`) and sensor-overlap queries (`b3Shape_GetSensorOverlaps`).
- Body-level and standalone geometry casts (`b3ShapeCast`, `b3ShapeDistance`, ...) and dynamic-tree access.
- World explode, joint wake helpers beyond `wake_joint_bodies`, and world snapshots.
