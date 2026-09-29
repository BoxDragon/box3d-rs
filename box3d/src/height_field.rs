//! Height fields (`b3HeightFieldData`) and height-field shapes.
//!
//! Same ownership scheme as [`crate::Mesh`]: Box3D keeps a raw pointer in the
//! shape, so [`HeightField`] is a reference-counted handle and the world
//! retains a clone per height-field shape.

use crate::id::{BodyId, ShapeId};
use crate::math::{from_v3, v3};
use crate::world::Resource;
use crate::{BodyType, Error, ShapeDef, Vec3, World, sys};
use core::ptr::NonNull;
use std::sync::Arc;

/// Material index that marks a height-field cell as a hole
/// (`B3_HEIGHT_FIELD_HOLE`): nothing collides with it or is hit by queries.
pub const HEIGHT_FIELD_HOLE: u8 = sys::B3_HEIGHT_FIELD_HOLE as u8;

/// Height-field creation parameters (`b3HeightFieldDef`). Borrows the input
/// arrays; nothing is retained after [`HeightField::new`] returns.
///
/// Grid layout, in shape-local space: grid point `(x, z)` (with
/// `x < count_x`, `z < count_z`) is `heights[z * count_x + x]` and sits at
/// `scale * (x, height, z)`. The field spans `[0, scale.x * (count_x - 1)]` by
/// `[0, scale.z * (count_z - 1)]`; position it with the body transform.
/// Cell `(x, z)` (`x < count_x - 1`, `z < count_z - 1`) is split into two
/// triangles, `2 * (z * (count_x - 1) + x)` and that `+ 1`, which is the
/// `triangle_index` reported by queries. Faces point +Y unless
/// [`Self::clockwise`] is set.
///
/// Heights are quantized to 16 bits over `[min_height, max_height]`
/// (by default the input's own range) and clamped to it.
#[derive(Clone, Copy, Debug)]
pub struct HeightFieldDef<'a> {
    heights: &'a [f32],
    count_x: u32,
    count_z: u32,
    scale: Vec3,
    material_indices: Option<&'a [u8]>,
    height_range: Option<(f32, f32)>,
    clockwise: bool,
}

impl<'a> HeightFieldDef<'a> {
    /// `heights.len()` must equal `count_x * count_z`, both counts at least 2;
    /// `scale` components must be positive. Checked by [`HeightField::new`].
    pub fn new(heights: &'a [f32], count_x: u32, count_z: u32, scale: Vec3) -> Self {
        Self { heights, count_x, count_z, scale, material_indices: None, height_range: None, clockwise: false }
    }

    /// One material index per *cell* (`(count_x - 1) * (count_z - 1)`, row
    /// major like the heights), indexing into [`ShapeDef::mesh_materials`].
    /// Use [`HEIGHT_FIELD_HOLE`] for holes.
    pub fn material_indices(mut self, indices: &'a [u8]) -> Self {
        self.material_indices = Some(indices);
        self
    }

    /// Global quantization range (unscaled heights). Give adjacent tiles the
    /// same range so their edges line up exactly. Heights are clamped to it.
    pub fn height_range(mut self, min_height: f32, max_height: f32) -> Self {
        self.height_range = Some((min_height, max_height));
        self
    }

    /// Clockwise winding: effectively flips the field upside down (faces -Y).
    pub fn clockwise(mut self, enabled: bool) -> Self {
        self.clockwise = enabled;
        self
    }

    fn to_raw(self) -> Result<sys::b3HeightFieldDef, Error> {
        const MAX_POINTS: u64 = 1 << 26;
        if self.count_x < 2 || self.count_z < 2 {
            return Err(Error::InvalidArgument("height field counts must be at least 2"));
        }
        let points = self.count_x as u64 * self.count_z as u64;
        if points > MAX_POINTS {
            return Err(Error::InvalidArgument("height field has too many grid points"));
        }
        if self.heights.len() as u64 != points {
            return Err(Error::InvalidArgument("height field heights.len() must equal count_x * count_z"));
        }
        let cells = (self.count_x as u64 - 1) * (self.count_z as u64 - 1);
        if let Some(m) = self.material_indices
            && m.len() as u64 != cells
        {
            return Err(Error::InvalidArgument(
                "height field material_indices.len() must equal (count_x - 1) * (count_z - 1)",
            ));
        }
        if !(self.scale.is_finite() && self.scale.cmpgt(Vec3::ZERO).all()) {
            return Err(Error::InvalidArgument("height field scale must be positive and finite"));
        }
        if self.heights.iter().any(|h| !h.is_finite()) {
            return Err(Error::InvalidArgument("height field heights must be finite"));
        }
        let (min, max) = match self.height_range {
            Some((min, max)) => {
                if !(min.is_finite() && max.is_finite() && min <= max) {
                    return Err(Error::InvalidArgument("height field range must be finite with min <= max"));
                }
                (min, max)
            }
            None => self.heights.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h))),
        };
        Ok(sys::b3HeightFieldDef {
            // Box3D only reads through these pointers.
            heights: self.heights.as_ptr() as *mut f32,
            materialIndices: self.material_indices.map_or(core::ptr::null_mut(), |m| m.as_ptr() as *mut u8),
            scale: v3(self.scale),
            countX: self.count_x as i32,
            countZ: self.count_z as i32,
            globalMinimumHeight: min,
            globalMaximumHeight: max,
            clockwiseWinding: self.clockwise,
        })
    }
}

struct HeightFieldData(NonNull<sys::b3HeightFieldData>);

// SAFETY: immutable after construction; Box3D only reads it.
unsafe impl Send for HeightFieldData {}
// SAFETY: as above.
unsafe impl Sync for HeightFieldData {}

impl Drop for HeightFieldData {
    fn drop(&mut self) {
        // SAFETY: created by b3CreateHeightField, freed exactly once after
        // every shape using it is gone (the worlds hold clones of the Arc).
        unsafe { sys::b3DestroyHeightField(self.0.as_ptr()) }
    }
}

/// An immutable, compressed height field, shareable across shapes and
/// worlds. `Clone` is a reference-count bump; the world retains a clone per
/// height-field shape (see [`crate::Mesh`] for the rationale).
#[derive(Clone)]
pub struct HeightField(Arc<HeightFieldData>);

impl HeightField {
    /// Build a height field (`b3CreateHeightField`). Fails with
    /// [`Error::InvalidArgument`] on inconsistent counts or non-finite input.
    pub fn new(def: &HeightFieldDef<'_>) -> Result<HeightField, Error> {
        let raw = def.to_raw()?;
        // SAFETY: counts and array lengths validated; slices outlive the call.
        let p = unsafe { sys::b3CreateHeightField(&raw) };
        let p = NonNull::new(p).expect("b3CreateHeightField returned null");
        Ok(HeightField(Arc::new(HeightFieldData(p))))
    }

    /// Pointer to the C height field. Valid while any clone of `self` lives.
    pub fn as_raw(&self) -> *const sys::b3HeightFieldData {
        self.0.0.as_ptr()
    }

    fn data(&self) -> &sys::b3HeightFieldData {
        // SAFETY: non-null, immutable, alive while self lives.
        unsafe { &*self.as_raw() }
    }

    /// Number of live handles: user clones plus one per height-field shape
    /// the worlds are keeping alive.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// `(count_x, count_z)` grid points.
    pub fn counts(&self) -> (u32, u32) {
        let d = self.data();
        (d.columnCount.max(0) as u32, d.rowCount.max(0) as u32)
    }

    pub fn scale(&self) -> Vec3 {
        from_v3(self.data().scale)
    }

    /// Local bounds `(lower, upper)`, scaled.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let b = self.data().aabb;
        (from_v3(b.lowerBound), from_v3(b.upperBound))
    }

    /// Size of the blob (counted by [`crate::bytes_allocated`]).
    pub fn byte_count(&self) -> usize {
        self.data().byteCount.max(0) as usize
    }

    pub fn ptr_eq(a: &HeightField, b: &HeightField) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }
}

impl core::fmt::Debug for HeightField {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HeightField")
            .field("counts", &self.counts())
            .field("scale", &self.scale())
            .field("bytes", &self.byte_count())
            .finish()
    }
}

/// Height-field shape API.
impl World {
    /// Attach a height field to a **static** body. The world keeps a clone of
    /// `height_field` until the shape is destroyed (directly, with its body,
    /// or with the world). Per-cell materials come from
    /// [`ShapeDef::mesh_materials`].
    ///
    /// # Panics
    /// If `body` is stale, foreign, or not static (Box3D only supports height
    /// fields on static bodies). While it carries a height field the body
    /// cannot be made non-static either: see [`World::set_body_type`].
    #[track_caller]
    pub fn create_height_field_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef<'_>,
        height_field: &HeightField,
    ) -> ShapeId {
        let b = self.body(body);
        assert_eq!(self.body_type(body), BodyType::Static, "height fields require a static body");
        self.prune_resources();
        // SAFETY: validated static body; the pointer stays valid for the life
        // of the shape because the world retains a clone below.
        let id = ShapeId(unsafe { sys::b3CreateHeightFieldShape(b, &def.0, height_field.as_raw()) });
        if !id.is_null() {
            self.retain_resource(id, Resource::HeightField(height_field.clone()));
        }
        id
    }
}
