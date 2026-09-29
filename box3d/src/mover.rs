//! Kinematic character mover: capsule casts and collision planes
//! (`b3World_CastMover`, `b3World_CollideMover`, `b3SolvePlanes`,
//! `b3ClipVector`).
//!
//! Typical update (see `tests/mover.rs`), all origin-relative so it stays
//! precise far from the world origin:
//!
//! 1. [`World::collide_mover`] at the current position, copying each
//!    [`PlaneResult`] into a caller-owned fixed-capacity
//!    `[CollisionPlane; N]` (skip the character's own shapes).
//! 2. [`solve_planes`] with the desired displacement to get a delta that
//!    resolves penetration and slides along the planes.
//! 3. [`World::cast_mover`] along that delta to avoid tunnelling, then move
//!    by `fraction * delta`.
//! 4. [`clip_vector`] the velocity against the planes that pushed.
//!
//! [`Capsule`], [`Plane`], [`PlaneResult`], [`CollisionPlane`] and
//! [`BodyPlaneResult`] are `repr(C)` and layout-identical to their C
//! counterparts, so slices go to and from Box3D without copying.

use crate::id::{BodyId, ShapeId};
use crate::math::{from_pos, from_v3, pos, quat, v3};
use crate::query::{Ctx, overlap_tramp};
use crate::{Pos, Quat, QueryFilter, Vec3, World, sys};
use core::ffi::c_void;
use core::mem::{align_of, offset_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// A plane: `separation(p) = dot(normal, p) - offset` (`b3Plane`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Plane {
    pub normal: Vec3,
    pub offset: f32,
}

impl Plane {
    pub fn new(normal: Vec3, offset: f32) -> Self {
        Self { normal, offset }
    }

    /// Signed distance of `point` above the plane.
    #[inline]
    pub fn separation(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.offset
    }
}

/// A capsule: the segment `center1..center2` inflated by `radius`
/// (`b3Capsule`). For movers, the points are relative to the query origin.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Capsule {
    pub center1: Vec3,
    pub center2: Vec3,
    pub radius: f32,
}

impl Capsule {
    pub fn new(center1: Vec3, center2: Vec3, radius: f32) -> Self {
        Self { center1, center2, radius }
    }

    /// Upright (Y axis) capsule standing on the origin: total height
    /// `height` (including both caps), so its lowest point is at y = 0 and its
    /// highest at `height`. Requires `height >= 2 * radius`.
    pub fn standing(height: f32, radius: f32) -> Self {
        let half = (0.5 * height - radius).max(0.0);
        let mid = 0.5 * height;
        Self { center1: Vec3::new(0.0, mid - half, 0.0), center2: Vec3::new(0.0, mid + half, 0.0), radius }
    }

    #[track_caller]
    fn check(&self) {
        assert!(
            self.center1.is_finite() && self.center2.is_finite() && self.radius.is_finite() && self.radius >= 0.0,
            "mover capsule must be finite with radius >= 0, got {self:?}"
        );
    }

    fn as_raw(&self) -> *const sys::b3Capsule {
        self as *const Capsule as *const sys::b3Capsule
    }
}

/// A plane between the mover and one shape, from [`World::collide_mover`]
/// (`b3PlaneResult`). Origin-relative.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PlaneResult {
    /// Outward-pointing plane (away from the shape, towards the mover).
    pub plane: Plane,
    /// Closest point on the shape (may not be unique).
    pub point: Vec3,
    /// Mesh/height-field triangle index (0 for convex shapes).
    pub triangle_index: i32,
    /// Compound child index.
    pub child_index: i32,
    /// Material index into the shape's materials (clamped to the valid range).
    pub material_index: i32,
}

impl PlaneResult {
    /// A rigid collision plane that also clips velocity.
    #[inline]
    pub fn to_collision_plane(&self) -> CollisionPlane {
        CollisionPlane::rigid(self.plane)
    }
}

/// Input/output of [`solve_planes`] and [`clip_vector`] (`b3CollisionPlane`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct CollisionPlane {
    pub plane: Plane,
    /// Maximum push (length units). `f32::MAX` is rigid; smaller values make
    /// the plane soft (e.g. to push through small dynamic objects).
    pub push_limit: f32,
    /// Output: the push applied by [`solve_planes`].
    pub push: f32,
    /// Whether [`clip_vector`] clips against this plane. Use `false` for
    /// soft planes.
    pub clip_velocity: bool,
}

impl CollisionPlane {
    /// Rigid plane (`push_limit = f32::MAX`) that clips velocity.
    #[inline]
    pub fn rigid(plane: Plane) -> Self {
        Self { plane, push_limit: f32::MAX, push: 0.0, clip_velocity: true }
    }

    /// Soft plane that pushes at most `push_limit` and does not clip velocity.
    #[inline]
    pub fn soft(plane: Plane, push_limit: f32) -> Self {
        Self { plane, push_limit, push: 0.0, clip_velocity: false }
    }
}

impl Default for CollisionPlane {
    fn default() -> Self {
        Self::rigid(Plane::default())
    }
}

/// Result of [`solve_planes`] (`b3PlaneSolverResult`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaneSolverResult {
    /// The final translation, relative to the position used to collect the planes.
    pub delta: Vec3,
    /// Solver iterations used (diagnostic, at most 20).
    pub iteration_count: i32,
}

/// A plane against one shape of a specific body, from
/// [`World::body_collide_mover`] (`b3BodyPlaneResult`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct BodyPlaneResult {
    pub shape: ShapeId,
    pub result: PlaneResult,
}

/// Hit from [`World::body_time_of_impact_mover`] (`b3BodyTOIResult`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoverHit {
    /// World-space hit point.
    pub point: Pos,
    /// Points from the body towards the mover.
    pub normal: Vec3,
    /// Fraction of the mover translation, `(0, 1)`.
    pub fraction: f32,
    pub shape: ShapeId,
}

// Layout checks: these types are passed to C by pointer.
const _: () = {
    assert!(size_of::<Plane>() == size_of::<sys::b3Plane>());
    assert!(align_of::<Plane>() == align_of::<sys::b3Plane>());
    assert!(offset_of!(Plane, offset) == offset_of!(sys::b3Plane, offset));

    assert!(size_of::<Capsule>() == size_of::<sys::b3Capsule>());
    assert!(align_of::<Capsule>() == align_of::<sys::b3Capsule>());
    assert!(offset_of!(Capsule, center2) == offset_of!(sys::b3Capsule, center2));
    assert!(offset_of!(Capsule, radius) == offset_of!(sys::b3Capsule, radius));

    assert!(size_of::<PlaneResult>() == size_of::<sys::b3PlaneResult>());
    assert!(align_of::<PlaneResult>() == align_of::<sys::b3PlaneResult>());
    assert!(offset_of!(PlaneResult, point) == offset_of!(sys::b3PlaneResult, point));
    assert!(offset_of!(PlaneResult, triangle_index) == offset_of!(sys::b3PlaneResult, triangleIndex));
    assert!(offset_of!(PlaneResult, child_index) == offset_of!(sys::b3PlaneResult, childIndex));
    assert!(offset_of!(PlaneResult, material_index) == offset_of!(sys::b3PlaneResult, materialIndex));

    assert!(size_of::<CollisionPlane>() == size_of::<sys::b3CollisionPlane>());
    assert!(align_of::<CollisionPlane>() == align_of::<sys::b3CollisionPlane>());
    assert!(offset_of!(CollisionPlane, push_limit) == offset_of!(sys::b3CollisionPlane, pushLimit));
    assert!(offset_of!(CollisionPlane, push) == offset_of!(sys::b3CollisionPlane, push));
    assert!(offset_of!(CollisionPlane, clip_velocity) == offset_of!(sys::b3CollisionPlane, clipVelocity));

    assert!(size_of::<BodyPlaneResult>() == size_of::<sys::b3BodyPlaneResult>());
    assert!(align_of::<BodyPlaneResult>() == align_of::<sys::b3BodyPlaneResult>());
    assert!(offset_of!(BodyPlaneResult, result) == offset_of!(sys::b3BodyPlaneResult, result));
};

/// Find the displacement closest to `target_delta` that satisfies every
/// plane (`b3SolvePlanes`). Writes each plane's `push`. Operates in place on
/// the caller's slice; no allocation.
pub fn solve_planes(target_delta: Vec3, planes: &mut [CollisionPlane]) -> PlaneSolverResult {
    let count = planes.len().min(i32::MAX as usize) as i32;
    // SAFETY: CollisionPlane is layout-identical to b3CollisionPlane (checked
    // above); Box3D reads/writes exactly `count` entries and only writes `push`.
    let r = unsafe { sys::b3SolvePlanes(v3(target_delta), planes.as_mut_ptr() as *mut sys::b3CollisionPlane, count) };
    PlaneSolverResult { delta: from_v3(r.delta), iteration_count: r.iterationCount }
}

/// Remove the components of `vector` that point into the planes that pushed
/// in the last [`solve_planes`] (`push != 0` and `clip_velocity`)
/// (`b3ClipVector`).
pub fn clip_vector(vector: Vec3, planes: &[CollisionPlane]) -> Vec3 {
    let count = planes.len().min(i32::MAX as usize) as i32;
    // SAFETY: layout-identical, read-only, `count` entries.
    from_v3(unsafe { sys::b3ClipVector(v3(vector), planes.as_ptr() as *const sys::b3CollisionPlane, count) })
}

unsafe extern "C" fn plane_tramp<F: FnMut(ShapeId, &[PlaneResult]) -> bool>(
    shape: sys::b3ShapeId,
    planes: *const sys::b3PlaneResult,
    count: i32,
    ctx: *mut c_void,
) -> bool {
    // SAFETY: ctx is the &mut Ctx<F> passed by collide_mover.
    let ctx = unsafe { &mut *(ctx as *mut Ctx<'_, F>) };
    if ctx.panic.is_some() {
        return false;
    }
    let planes: &[PlaneResult] = if planes.is_null() || count <= 0 {
        &[]
    } else {
        // SAFETY: Box3D passes `count` plane results on its stack, valid for
        // the call; PlaneResult is layout-identical to b3PlaneResult.
        unsafe { core::slice::from_raw_parts(planes as *const PlaneResult, count as usize) }
    };
    match catch_unwind(AssertUnwindSafe(|| (ctx.f)(ShapeId(shape), planes))) {
        Ok(v) => v,
        Err(p) => {
            ctx.panic = Some(p);
            false
        }
    }
}

/// Character mover API.
impl World {
    /// Sweep the capsule `mover` (relative to `origin`) along `translation`
    /// and return the fraction of `translation` that is free
    /// (`b3World_CastMover`). Handles sliding contact without clipping;
    /// shapes the mover already overlaps are ignored, as are back faces of
    /// meshes and height fields. Use [`Self::collide_mover`] to learn what
    /// the mover touches.
    ///
    /// # Panics
    /// If the capsule or translation is not finite.
    #[track_caller]
    pub fn cast_mover(&self, origin: Pos, mover: &Capsule, translation: Vec3, filter: QueryFilter) -> f32 {
        mover.check();
        assert!(translation.is_finite(), "mover translation must be finite");
        // SAFETY: valid world, layout-identical capsule, no callback.
        unsafe {
            sys::b3World_CastMover(
                self.id,
                pos(origin),
                mover.as_raw(),
                v3(translation),
                filter.to_raw(),
                None,
                core::ptr::null_mut(),
            )
        }
    }

    /// [`Self::cast_mover`] with a per-shape filter closure: return `false`
    /// to ignore a shape (e.g. the character's own body).
    #[track_caller]
    pub fn cast_mover_filtered<F: FnMut(ShapeId) -> bool>(
        &self,
        origin: Pos,
        mover: &Capsule,
        translation: Vec3,
        filter: QueryFilter,
        mut f: F,
    ) -> f32 {
        mover.check();
        assert!(translation.is_finite(), "mover translation must be finite");
        let mut ctx = Ctx { f: &mut f, panic: None };
        // SAFETY: valid world; ctx outlives the synchronous query and matches
        // the trampoline's type parameter (b3MoverFilterFcn has the same
        // signature as an overlap callback).
        let fraction = unsafe {
            sys::b3World_CastMover(
                self.id,
                pos(origin),
                mover.as_raw(),
                v3(translation),
                filter.to_raw(),
                Some(overlap_tramp::<F>),
                &mut ctx as *mut Ctx<'_, F> as *mut c_void,
            )
        };
        ctx.finish();
        fraction
    }

    /// Gather collision planes between the capsule `mover` (relative to
    /// `origin`) and every overlapping shape (`b3World_CollideMover`). The
    /// closure receives each shape with its planes (origin-relative, a slice
    /// borrowed from Box3D's stack); return `false` to stop. Includes the
    /// character's own shapes if it has a body, so filter those out.
    ///
    /// # Panics
    /// If the capsule is not finite; re-raises a panic from `f`.
    #[track_caller]
    pub fn collide_mover<F: FnMut(ShapeId, &[PlaneResult]) -> bool>(
        &self,
        origin: Pos,
        mover: &Capsule,
        filter: QueryFilter,
        mut f: F,
    ) {
        mover.check();
        let mut ctx = Ctx { f: &mut f, panic: None };
        // SAFETY: as in cast_mover_filtered.
        unsafe {
            sys::b3World_CollideMover(
                self.id,
                pos(origin),
                mover.as_raw(),
                filter.to_raw(),
                Some(plane_tramp::<F>),
                &mut ctx as *mut Ctx<'_, F> as *mut c_void,
            );
        }
        ctx.finish();
    }

    /// Collide the mover with the convex shapes of one body placed at
    /// `body_position`/`body_rotation` (`b3Body_CollideMover`), writing at
    /// most `out.len()` results (one per shape). Returns the count written.
    ///
    /// # Panics
    /// If `body` is stale or foreign, or the capsule is not finite.
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn body_collide_mover(
        &self,
        body: BodyId,
        origin: Pos,
        mover: &Capsule,
        filter: QueryFilter,
        body_position: Pos,
        body_rotation: Quat,
        out: &mut [BodyPlaneResult],
    ) -> usize {
        let b = self.body(body);
        mover.check();
        let xf = sys::b3WorldTransform { p: pos(body_position), q: quat(body_rotation) };
        let cap = out.len().min(i32::MAX as usize) as i32;
        // SAFETY: validated body; BodyPlaneResult is layout-identical to
        // b3BodyPlaneResult and Box3D writes at most `cap` entries.
        let n = unsafe {
            sys::b3Body_CollideMover(
                b,
                out.as_mut_ptr() as *mut sys::b3BodyPlaneResult,
                cap,
                pos(origin),
                mover.as_raw(),
                filter.to_raw(),
                xf,
            )
        };
        n.clamp(0, cap) as usize
    }

    /// Time of impact of the mover sweeping by `translation` against the
    /// convex shapes of one body moving from `body_from` to `body_to`
    /// (`b3Body_TimeOfImpactMover`). Shapes that initially overlap are
    /// ignored. `None` if nothing is hit.
    ///
    /// # Panics
    /// If `body` is stale or foreign, or the capsule/translation is not finite.
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn body_time_of_impact_mover(
        &self,
        body: BodyId,
        origin: Pos,
        mover: &Capsule,
        translation: Vec3,
        filter: QueryFilter,
        body_from: (Pos, Quat),
        body_to: (Pos, Quat),
    ) -> Option<MoverHit> {
        let b = self.body(body);
        mover.check();
        assert!(translation.is_finite(), "mover translation must be finite");
        let xf1 = sys::b3WorldTransform { p: pos(body_from.0), q: quat(body_from.1) };
        let xf2 = sys::b3WorldTransform { p: pos(body_to.0), q: quat(body_to.1) };
        // SAFETY: validated body; plain values.
        let r = unsafe {
            sys::b3Body_TimeOfImpactMover(b, pos(origin), mover.as_raw(), v3(translation), filter.to_raw(), xf1, xf2)
        };
        (r.fraction < 1.0 && r.shapeId.index1 != 0).then(|| MoverHit {
            point: from_pos(r.point),
            normal: from_v3(r.normal),
            fraction: r.fraction,
            shape: ShapeId(r.shapeId),
        })
    }
}
