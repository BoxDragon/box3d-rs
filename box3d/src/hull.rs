use crate::math::{from_v3, v3, xf};
use crate::{Error, Transform, Vec3, sys};
use core::ptr::NonNull;

enum Storage {
    /// Heap hull from `b3CreateHull`/`b3CreateCylinder`; freed with `b3DestroyHull`.
    Heap(NonNull<sys::b3HullData>),
    /// Box hull returned by value (`b3MakeBoxHull`); must NOT be passed to
    /// `b3DestroyHull`. Its internal offsets are relative to the struct, so
    /// it is fine to move the box around.
    Boxed(Box<sys::b3BoxHull>),
}

/// An owned convex hull, usable for any number of shapes.
///
/// Shape creation copies the hull into the world's (reference counted,
/// content-deduplicated) hull database, so a `Hull` may be dropped right
/// after creating shapes from it.
pub struct Hull(Storage);

// SAFETY: the hull data is immutable after construction and owned uniquely.
unsafe impl Send for Hull {}
// SAFETY: only read through shared references.
unsafe impl Sync for Hull {}

impl Hull {
    /// Axis-aligned box centered on the body origin (`b3MakeBoxHull`).
    /// Half extents are clamped to a small minimum.
    pub fn cuboid(half_extents: Vec3) -> Hull {
        // SAFETY: returns a value.
        let h = unsafe { sys::b3MakeBoxHull(half_extents.x, half_extents.y, half_extents.z) };
        Hull(Storage::Boxed(Box::new(h)))
    }

    /// Box with a local transform (`b3MakeTransformedBoxHull`).
    pub fn transformed_cuboid(half_extents: Vec3, transform: Transform) -> Hull {
        // SAFETY: returns a value; `xf` normalizes the rotation.
        let h = unsafe { sys::b3MakeTransformedBoxHull(half_extents.x, half_extents.y, half_extents.z, xf(transform)) };
        Hull(Storage::Boxed(Box::new(h)))
    }

    /// Tessellated cylinder along +Y from `y_offset` to `y_offset + height`
    /// (`b3CreateCylinder`). `sides` must be in `3..=32`.
    ///
    /// For a barrel centered on the body, use `y_offset = -height / 2`. To lay
    /// it on its side use [`crate::World::create_transformed_hull_shape`].
    pub fn cylinder(height: f32, radius: f32, y_offset: f32, sides: u32) -> Result<Hull, Error> {
        if !(height.is_finite() && height > 0.0) {
            return Err(Error::InvalidArgument("cylinder height must be > 0"));
        }
        if !(radius.is_finite() && radius > 0.0) {
            return Err(Error::InvalidArgument("cylinder radius must be > 0"));
        }
        if !(3..=32).contains(&sides) {
            return Err(Error::InvalidArgument("cylinder sides must be in 3..=32"));
        }
        // SAFETY: arguments satisfy b3CreateCylinder's asserts.
        let p = unsafe { sys::b3CreateCylinder(height, radius, y_offset, sides as i32) };
        NonNull::new(p).map(|p| Hull(Storage::Heap(p))).ok_or(Error::HullCreationFailed)
    }

    /// Quickhull over a point cloud (`b3CreateHull`). Needs at least 4
    /// non-coplanar points; `max_vertices` is clamped to `[4, 128]`.
    pub fn from_points(points: &[Vec3], max_vertices: usize) -> Result<Hull, Error> {
        if points.len() < 4 {
            return Err(Error::InvalidArgument("hull needs at least 4 points"));
        }
        if points.len() > i32::MAX as usize || points.iter().any(|p| !p.is_finite()) {
            return Err(Error::InvalidArgument("hull points must be finite"));
        }
        let pts: Vec<sys::b3Vec3> = points.iter().copied().map(v3).collect();
        let max = max_vertices.min(sys::B3_MAX_HULL_VERTICES as usize) as i32;
        // SAFETY: pts is a valid array of pts.len() points.
        let p = unsafe { sys::b3CreateHull(pts.as_ptr(), pts.len() as i32, max) };
        NonNull::new(p).map(|p| Hull(Storage::Heap(p))).ok_or(Error::HullCreationFailed)
    }

    /// Pointer to the C hull. Valid while `self` lives.
    pub fn as_raw(&self) -> *const sys::b3HullData {
        match &self.0 {
            Storage::Heap(p) => p.as_ptr(),
            Storage::Boxed(b) => &b.base,
        }
    }

    fn data(&self) -> &sys::b3HullData {
        // SAFETY: as_raw is valid and non-null while self lives.
        unsafe { &*self.as_raw() }
    }

    pub fn vertex_count(&self) -> usize {
        self.data().vertexCount.max(0) as usize
    }

    pub fn face_count(&self) -> usize {
        self.data().faceCount.max(0) as usize
    }

    pub fn volume(&self) -> f32 {
        self.data().volume
    }

    /// Local centroid.
    pub fn center(&self) -> Vec3 {
        from_v3(self.data().center)
    }

    /// Radius of the largest sphere at the center (rolling radius uses 1/4 of this).
    pub fn inner_radius(&self) -> f32 {
        self.data().innerRadius
    }
}

impl Clone for Hull {
    fn clone(&self) -> Self {
        match &self.0 {
            Storage::Boxed(b) => Hull(Storage::Boxed(b.clone())),
            Storage::Heap(p) => {
                // SAFETY: valid hull; returns a fresh heap copy.
                let c = unsafe { sys::b3CloneHull(p.as_ptr()) };
                Hull(Storage::Heap(NonNull::new(c).expect("b3CloneHull failed (out of memory?)")))
            }
        }
    }
}

impl Drop for Hull {
    fn drop(&mut self) {
        if let Storage::Heap(p) = self.0 {
            // SAFETY: allocated by Box3D's hull creator, freed exactly once.
            unsafe { sys::b3DestroyHull(p.as_ptr()) }
        }
    }
}

impl core::fmt::Debug for Hull {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Hull")
            .field("vertices", &self.vertex_count())
            .field("faces", &self.face_count())
            .field("volume", &self.volume())
            .finish()
    }
}
