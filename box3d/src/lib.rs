//! Thin safe wrapper over [Box3D](https://github.com/erincatto/box3d), Erin
//! Catto's 3D rigid body physics engine.
//!
//! The API mirrors the C API closely, uses [glam] 0.30 types, and is designed
//! so that a game loop allocates nothing after warm-up (see
//! `tests/no_alloc.rs`).
//!
//! # Ownership model
//! - [`World`] owns the native world. Dropping it destroys every body, shape
//!   and joint in it. Nothing else destroys simulation objects on `Drop`
//!   ([`Hull`], [`Mesh`] and [`HeightField`] only free their own geometry).
//! - [`BodyId`], [`ShapeId`], [`JointId`] (8 bytes, `repr(transparent)` over
//!   the C ids) and the typed joint ids such as [`DistanceJointId`] are `Copy`
//!   handles with no registry behind them. They are **generational**: once
//!   the object is destroyed the id is permanently invalid, even if the slot
//!   is reused.
//! - Every method that takes an id validates it first (`b3*_IsValid` plus a
//!   same-world check). A stale or foreign id **panics on the Rust side**
//!   (like an out-of-bounds index) instead of reaching Box3D, where it would
//!   be undefined behaviour in release builds. Use [`World::is_body_valid`]
//!   and friends to check first.
//! - Destroy immediately with [`World::destroy_body`] etc., or defer with
//!   [`World::queue_destroy_body`] etc.: the queue is pre-sized
//!   ([`WorldDef::destroy_queue_capacity`]), reused, and flushed at the start
//!   of [`World::step`] (or by [`World::flush_destroys`]). Stale entries are
//!   skipped, so queueing twice is harmless.
//! - User data is a plain `u64` on bodies, shapes and joints (for example a
//!   hecs `Entity::to_bits()`). The wrapper never stores pointers there.
//!
//! # Meshes and height fields
//! [`Hull`]s are copied into the world, but Box3D keeps a *raw pointer* to
//! the `b3MeshData` / `b3HeightFieldData` of every mesh or height-field
//! shape. [`Mesh`] and [`HeightField`] are therefore immutable,
//! reference-counted (`Arc`) handles: `Clone` is cheap, one mesh can back any
//! number of shapes (each with its own scale) in any number of worlds, and
//! [`World::create_mesh_shape`] / [`World::create_height_field_shape`] store
//! a clone in the world. That clone is released when the shape is destroyed
//! directly, with its body, through the destroy queue, or with the world, so
//! dropping your handle right after creating shapes is fine. Releasing
//! happens in `destroy_*` / [`World::flush_destroys`], never inside the
//! solver step, and never allocates. Height fields are static-only (creating
//! one on a non-static body panics, as does making its body non-static);
//! meshes are allowed on any body type, as in Box3D, but only collide
//! properly when static. Shapes created through [`sys`] directly are not
//! tracked: keeping their geometry alive is up to you.
//!
//! # Character mover
//! [`World::collide_mover`], [`World::cast_mover`], [`solve_planes`] and
//! [`clip_vector`] wrap Box3D's kinematic capsule mover. The plane types are
//! `repr(C)` and layout-identical to the C ones, so the solver works in place
//! on a caller-owned `&mut [CollisionPlane]` and a mover update allocates
//! nothing.
//!
//! # Events and queries
//! [`World::body_events`], [`World::contact_events`],
//! [`World::sensor_events`] and [`World::joint_events`] return slices that
//! borrow Box3D's own buffers: no copy, no allocation. Because
//! [`World::step`] takes `&mut self`, the borrow checker stops them from
//! outliving the step. Callback queries ([`World::cast_ray`],
//! [`World::cast_shape`], [`World::overlap_aabb`], [`World::overlap_shape`],
//! [`World::collide_mover`], [`World::cast_mover_filtered`])
//! take a generic closure passed to C through a monomorphized trampoline;
//! nothing is boxed.
//!
//! # Threading
//! [`World`] is `Send` but not `Sync`. World creation and destruction are
//! serialized through a global lock because Box3D keeps its world table in a
//! global array. The default worker count is 1 (no internal threads). The
//! custom-filter and pre-solve callbacks may run on Box3D worker threads when
//! `worker_count > 1`, hence their `Send + Sync` bound.
//!
//! # FFI safety
//! All `unsafe` is confined to calls into [`sys`]. Rust callbacks called by C
//! (queries, mover callbacks, custom filter, pre-solve) catch panics at the boundary; the
//! panic is resumed on the Rust side once the C call returns. The one
//! `unsafe` public function is [`set_allocator`], whose contract the
//! compiler cannot check.

pub use box3d_sys as sys;
pub use glam;
pub use glam::{Mat3, Quat, Vec3};

mod blob;
mod body;
mod delta;
mod events;
mod height_field;
mod hull;
mod id;
mod joint;
mod math;
mod mesh;
mod mover;
mod query;
mod shape;
mod world;

pub use body::{BodyDef, BodyType, MassData, TIME_TO_SLEEP};
pub use delta::{DeltaStats, WorldDelta};
pub use events::{
    BodyEvents, BodyMoveEvent, ContactBeginEvent, ContactEndEvent, ContactEvents, ContactHitEvent, JointEvent,
    JointEvents, SensorBeginEvent, SensorEndEvent, SensorEvents,
};
pub use height_field::{HEIGHT_FIELD_HOLE, HeightField, HeightFieldDef};
pub use hull::Hull;
pub use id::{BodyId, JointId, ShapeId, WorldId};
pub use joint::{
    DistanceJointDef, DistanceJointId, FilterJointDef, JointType, MotorJointDef, MotorJointId, PrismaticJointDef, PrismaticJointId,
    RevoluteJointDef, RevoluteJointId, SphericalJointDef, SphericalJointId, WeldJointDef, WeldJointId, WheelJointDef,
    WheelJointId,
};
pub use math::{Pos, Transform};
pub use mesh::{Mesh, MeshDef};
pub use mover::{
    BodyPlaneResult, Capsule, CollisionPlane, MoverHit, Plane, PlaneResult, PlaneSolverResult, clip_vector,
    solve_planes,
};
pub use query::{QueryFilter, RayHit, ShapeProxy};
pub use shape::{Filter, ShapeDef, ShapeType, SurfaceMaterial};
pub use world::{AllocFn, Capacity, FreeFn, World, WorldDef, bytes_allocated, set_allocator};

/// Errors from fallible constructors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Box3D supports at most `B3_MAX_WORLDS` (128) simultaneous worlds.
    TooManyWorlds,
    /// Hull construction failed (degenerate input or limits exceeded).
    HullCreationFailed,
    /// Mesh construction failed (e.g. every triangle is degenerate, or the
    /// BVH got too deep).
    MeshCreationFailed,
    /// An argument was out of the range Box3D accepts.
    InvalidArgument(&'static str),
    /// [`set_allocator`] was called while Box3D still held memory.
    AllocatorBusy,
    /// [`World::restore_delta`]: the tick is outside the tracker's window, or
    /// the world is recording.
    RestoreRefused,
    /// [`World::capture_delta`]: the tracker refused (world locked or detached).
    CaptureRefused,
    /// [`Hull::from_bytes`] / [`Mesh::from_bytes`]: the blob was written by another Box3D
    /// hull or mesh version (`Hull::VERSION`, `Mesh::VERSION`). Re-cook it.
    BlobVersionMismatch,
    /// [`Hull::from_bytes`] / [`Mesh::from_bytes`]: truncated or inconsistent bytes.
    InvalidBlob(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::TooManyWorlds => write!(f, "too many Box3D worlds (max {})", sys::B3_MAX_WORLDS),
            Error::HullCreationFailed => write!(f, "Box3D hull creation failed"),
            Error::MeshCreationFailed => write!(f, "Box3D mesh creation failed"),
            Error::InvalidArgument(what) => write!(f, "invalid argument: {what}"),
            Error::AllocatorBusy => write!(f, "Box3D allocator can only be replaced while nothing is allocated"),
            Error::RestoreRefused => write!(f, "world delta restore refused (tick outside the window, or recording)"),
            Error::CaptureRefused => write!(f, "world delta capture refused (world locked or detached)"),
            Error::BlobVersionMismatch => write!(f, "hull or mesh blob from another Box3D version"),
            Error::InvalidBlob(what) => write!(f, "invalid hull or mesh blob: {what}"),
        }
    }
}

impl std::error::Error for Error {}

/// Box3D library version.
pub fn version() -> (i32, i32, i32) {
    // SAFETY: pure getter, no preconditions.
    let v = unsafe { sys::b3GetVersion() };
    (v.major, v.minor, v.revision)
}
