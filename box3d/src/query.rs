//! Spatial queries. Callback queries use a monomorphized `extern "C"`
//! trampoline per closure type with `&mut F` as the C context: no boxing, no
//! allocation. A panic in the closure is caught before it reaches C, the
//! query is terminated, and the panic is resumed after Box3D returns.

use crate::id::ShapeId;
use crate::math::{from_pos, from_v3, pos, v3};
use crate::{Error, Pos, Vec3, World, sys};
use core::ffi::c_void;
use core::marker::PhantomData;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

/// Category/mask filter for queries. Default hits everything.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryFilter {
    pub category_bits: u64,
    pub mask_bits: u64,
}

impl Default for QueryFilter {
    fn default() -> Self {
        // SAFETY: returns a value.
        let f = unsafe { sys::b3DefaultQueryFilter() };
        Self { category_bits: f.categoryBits, mask_bits: f.maskBits }
    }
}

impl QueryFilter {
    fn to_raw(self) -> sys::b3QueryFilter {
        // SAFETY: returns a value; the recording name stays null.
        let mut f = unsafe { sys::b3DefaultQueryFilter() };
        f.categoryBits = self.category_bits;
        f.maskBits = self.mask_bits;
        f
    }
}

/// A ray or shape-cast hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub shape: ShapeId,
    pub point: Pos,
    pub normal: Vec3,
    /// Fraction of `translation` at the hit, `[0, 1]`.
    pub fraction: f32,
    pub user_material_id: u64,
    /// Triangle index for mesh/height-field shapes, otherwise -1.
    pub triangle_index: i32,
    /// Child index for compound shapes.
    pub child_index: i32,
}

/// A convex point cloud plus radius for shape overlap/cast queries
/// (`b3ShapeProxy`). Borrows the points; at most 128 of them. Points are
/// relative to the query `origin`.
#[derive(Clone, Copy, Debug)]
pub struct ShapeProxy<'a>(sys::b3ShapeProxy, PhantomData<&'a [Vec3]>);

// glam::Vec3 is #[repr(C)] { x, y, z: f32 }, identical to b3Vec3.
const _: () = {
    assert!(core::mem::size_of::<Vec3>() == core::mem::size_of::<sys::b3Vec3>());
    assert!(core::mem::align_of::<Vec3>() == core::mem::align_of::<sys::b3Vec3>());
};

impl<'a> ShapeProxy<'a> {
    /// `points.len()` must be in `1..=128`; `radius >= 0`. A single point with
    /// a radius is a sphere, two points a capsule.
    pub fn new(points: &'a [Vec3], radius: f32) -> Result<Self, Error> {
        if points.is_empty() || points.len() > sys::B3_MAX_SHAPE_CAST_POINTS as usize {
            return Err(Error::InvalidArgument("shape proxy needs 1..=128 points"));
        }
        if !(radius.is_finite() && radius >= 0.0) {
            return Err(Error::InvalidArgument("shape proxy radius must be >= 0"));
        }
        Ok(Self(
            sys::b3ShapeProxy { points: points.as_ptr() as *const sys::b3Vec3, count: points.len() as i32, radius },
            PhantomData,
        ))
    }
}

struct Ctx<'f, F> {
    f: &'f mut F,
    panic: Option<Box<dyn Any + Send>>,
}

impl<F> Ctx<'_, F> {
    fn finish(self) {
        if let Some(p) = self.panic {
            resume_unwind(p);
        }
    }
}

unsafe extern "C" fn cast_tramp<F: FnMut(&RayHit) -> f32>(
    shape: sys::b3ShapeId,
    point: sys::b3Pos,
    normal: sys::b3Vec3,
    fraction: f32,
    user_material_id: u64,
    triangle_index: i32,
    child_index: i32,
    ctx: *mut c_void,
) -> f32 {
    // SAFETY: ctx is the &mut Ctx<F> passed by the enclosing query call.
    let ctx = unsafe { &mut *(ctx as *mut Ctx<'_, F>) };
    if ctx.panic.is_some() {
        return 0.0;
    }
    let hit = RayHit {
        shape: ShapeId(shape),
        point: from_pos(point),
        normal: from_v3(normal),
        fraction,
        user_material_id,
        triangle_index,
        child_index,
    };
    match catch_unwind(AssertUnwindSafe(|| (ctx.f)(&hit))) {
        Ok(v) => v,
        Err(p) => {
            ctx.panic = Some(p);
            0.0
        }
    }
}

unsafe extern "C" fn overlap_tramp<F: FnMut(ShapeId) -> bool>(shape: sys::b3ShapeId, ctx: *mut c_void) -> bool {
    // SAFETY: ctx is the &mut Ctx<F> passed by the enclosing query call.
    let ctx = unsafe { &mut *(ctx as *mut Ctx<'_, F>) };
    if ctx.panic.is_some() {
        return false;
    }
    match catch_unwind(AssertUnwindSafe(|| (ctx.f)(ShapeId(shape)))) {
        Ok(v) => v,
        Err(p) => {
            ctx.panic = Some(p);
            false
        }
    }
}

impl World {
    /// Closest hit along `origin .. origin + translation`, returned by value.
    /// Ignores shapes the ray starts inside and back faces of meshes.
    pub fn cast_ray_closest(&self, origin: Pos, translation: Vec3, filter: QueryFilter) -> Option<RayHit> {
        // SAFETY: valid world, no callbacks involved.
        let r = unsafe { sys::b3World_CastRayClosest(self.id, pos(origin), v3(translation), filter.to_raw()) };
        r.hit.then(|| RayHit {
            shape: ShapeId(r.shapeId),
            point: from_pos(r.point),
            normal: from_v3(r.normal),
            fraction: r.fraction,
            user_material_id: r.userMaterialId,
            triangle_index: r.triangleIndex,
            child_index: r.childIndex,
        })
    }

    /// Ray cast with a callback per hit shape (in no particular order). The
    /// closure's return value steers the cast: `-1` ignore this shape, `0`
    /// stop, `hit.fraction` clip the ray (closest hit), `1` continue (all hits).
    pub fn cast_ray<F: FnMut(&RayHit) -> f32>(&self, origin: Pos, translation: Vec3, filter: QueryFilter, mut f: F) {
        let mut ctx = Ctx { f: &mut f, panic: None };
        // SAFETY: valid world; ctx outlives the synchronous query and matches
        // the trampoline's type parameter.
        unsafe {
            sys::b3World_CastRay(
                self.id,
                pos(origin),
                v3(translation),
                filter.to_raw(),
                Some(cast_tramp::<F>),
                &mut ctx as *mut Ctx<'_, F> as *mut c_void,
            );
        }
        ctx.finish();
    }

    /// Sweep `proxy` (relative to `origin`) along `translation`. Same callback
    /// protocol as [`Self::cast_ray`].
    pub fn cast_shape<F: FnMut(&RayHit) -> f32>(
        &self,
        origin: Pos,
        proxy: &ShapeProxy<'_>,
        translation: Vec3,
        filter: QueryFilter,
        mut f: F,
    ) {
        let mut ctx = Ctx { f: &mut f, panic: None };
        // SAFETY: as in cast_ray; proxy points are borrowed for the call.
        unsafe {
            sys::b3World_CastShape(
                self.id,
                pos(origin),
                &proxy.0,
                v3(translation),
                filter.to_raw(),
                Some(cast_tramp::<F>),
                &mut ctx as *mut Ctx<'_, F> as *mut c_void,
            );
        }
        ctx.finish();
    }

    /// Every shape whose bounding box *may* overlap `[lower, upper]`. Return
    /// `false` from the closure to stop early.
    pub fn overlap_aabb<F: FnMut(ShapeId) -> bool>(&self, lower: Vec3, upper: Vec3, filter: QueryFilter, mut f: F) {
        let aabb = sys::b3AABB { lowerBound: v3(lower.min(upper)), upperBound: v3(upper.max(lower)) };
        let mut ctx = Ctx { f: &mut f, panic: None };
        // SAFETY: as in cast_ray.
        unsafe {
            sys::b3World_OverlapAABB(
                self.id,
                aabb,
                filter.to_raw(),
                Some(overlap_tramp::<F>),
                &mut ctx as *mut Ctx<'_, F> as *mut c_void,
            );
        }
        ctx.finish();
    }

    /// Every shape that overlaps `proxy` (points relative to `origin`).
    /// Return `false` from the closure to stop early.
    pub fn overlap_shape<F: FnMut(ShapeId) -> bool>(
        &self,
        origin: Pos,
        proxy: &ShapeProxy<'_>,
        filter: QueryFilter,
        mut f: F,
    ) {
        let mut ctx = Ctx { f: &mut f, panic: None };
        // SAFETY: as in cast_shape.
        unsafe {
            sys::b3World_OverlapShape(
                self.id,
                pos(origin),
                &proxy.0,
                filter.to_raw(),
                Some(overlap_tramp::<F>),
                &mut ctx as *mut Ctx<'_, F> as *mut c_void,
            );
        }
        ctx.finish();
    }
}
