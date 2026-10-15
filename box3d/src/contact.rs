//! Touching contacts of a body, read back after a step.
//!
//! [`World::visit_body_contacts`] copies Box3D's contact headers into a
//! reusable [`ContactScratch`] (no allocation once it has grown) and hands
//! each one to a callback. The manifolds are borrowed straight from Box3D's
//! internal storage; the higher-ranked callback keeps them from escaping, so
//! nothing outlives the next step.

use crate::id::{BodyId, ShapeId};
use crate::math::from_v3;
use crate::{Vec3, World, sys};

/// Reusable buffer for [`World::visit_body_contacts`].
#[derive(Default)]
pub struct ContactScratch(Vec<sys::b3ContactData>);

impl ContactScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// One touching contact between two shapes. By convention the manifold
/// normals point from shape A to shape B; the body being visited may be
/// either side (compare [`World::shape_body`] of [`Self::shape_a`]).
#[derive(Clone, Copy)]
pub struct ContactData<'a>(&'a sys::b3ContactData);

impl<'a> ContactData<'a> {
    pub fn shape_a(&self) -> ShapeId {
        ShapeId(self.0.shapeIdA)
    }
    pub fn shape_b(&self) -> ShapeId {
        ShapeId(self.0.shapeIdB)
    }
    /// One manifold for convex pairs; mesh and height-field contacts can have several.
    pub fn manifolds(&self) -> &'a [Manifold] {
        let (ptr, count) = (self.0.manifolds, self.0.manifoldCount);
        if ptr.is_null() || count <= 0 {
            &[]
        } else {
            // SAFETY: Box3D's manifold array for this contact, valid until the next
            // step; `Manifold` is repr(transparent) over `b3Manifold`, and the
            // borrow cannot leave the visitor callback.
            unsafe { core::slice::from_raw_parts(ptr as *const Manifold, count as usize) }
        }
    }
}

/// A contact manifold: a normal and up to four points.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Manifold(sys::b3Manifold);

impl Manifold {
    /// World unit normal, from shape A to shape B.
    pub fn normal(&self) -> Vec3 {
        from_v3(self.0.normal)
    }
    pub fn points(&self) -> &[ManifoldPoint] {
        let n = (self.0.pointCount.max(0) as usize).min(self.0.points.len());
        // SAFETY: ManifoldPoint is repr(transparent) over b3ManifoldPoint.
        unsafe { &*(&self.0.points[..n] as *const [sys::b3ManifoldPoint] as *const [ManifoldPoint]) }
    }
    pub fn as_raw(&self) -> &sys::b3Manifold {
        &self.0
    }
}

/// A manifold point, as computed by the narrow phase at the start of the last
/// step, with the impulses of that step's solve.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ManifoldPoint(sys::b3ManifoldPoint);

impl ManifoldPoint {
    /// Contact point relative to body A's center of mass, world axes.
    pub fn anchor_a(&self) -> Vec3 {
        from_v3(self.0.anchorA)
    }
    /// Contact point relative to body B's center of mass, world axes.
    pub fn anchor_b(&self) -> Vec3 {
        from_v3(self.0.anchorB)
    }
    /// Negative when overlapping.
    pub fn separation(&self) -> f32 {
        self.0.separation
    }
    /// Normal impulse of the last sub-step (N s). Divide by the sub-step to get a force.
    pub fn normal_impulse(&self) -> f32 {
        self.0.normalImpulse
    }
    /// Normal impulse summed over the whole step, including restitution.
    pub fn total_normal_impulse(&self) -> f32 {
        self.0.totalNormalImpulse
    }
    pub fn as_raw(&self) -> &sys::b3ManifoldPoint {
        &self.0
    }
}

impl World {
    /// Calls `f` for every touching contact of `body` (manifold point count may
    /// still be 0 for some mesh manifolds; check [`Manifold::points`]).
    /// `scratch` is reused: after it has grown, this does not allocate.
    #[track_caller]
    pub fn visit_body_contacts(&self, body: BodyId, scratch: &mut ContactScratch, mut f: impl FnMut(ContactData<'_>)) {
        let raw = self.body(body);
        // SAFETY: valid body id (checked above).
        let capacity = unsafe { sys::b3Body_GetContactCapacity(raw) }.max(0) as usize;
        let buf = &mut scratch.0;
        buf.clear();
        if capacity == 0 {
            return;
        }
        buf.reserve(capacity);
        // SAFETY: `buf` has room for `capacity` entries; Box3D writes at most that
        // many and returns the count it wrote.
        let n = unsafe { sys::b3Body_GetContactData(raw, buf.as_mut_ptr(), capacity as i32) }.max(0) as usize;
        // SAFETY: the first `n` entries were initialized by Box3D.
        unsafe { buf.set_len(n.min(capacity)) };
        for c in buf.iter() {
            f(ContactData(c));
        }
    }
}
