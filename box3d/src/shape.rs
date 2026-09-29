use crate::hull::Hull;
use crate::id::{BodyId, ShapeId};
use crate::math::{v3, xf};
use crate::{Error, Transform, Vec3, World, sys};
use core::marker::PhantomData;

/// Surface material: a value wrapper over `b3SurfaceMaterial`.
/// `Default` is `b3DefaultSurfaceMaterial` (friction 0.6, everything else 0).
/// Read fields through [`Self::as_raw`].
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct SurfaceMaterial(pub(crate) sys::b3SurfaceMaterial);

impl Default for SurfaceMaterial {
    fn default() -> Self {
        // SAFETY: returns a value.
        Self(unsafe { sys::b3DefaultSurfaceMaterial() })
    }
}

impl PartialEq for SurfaceMaterial {
    fn eq(&self, o: &Self) -> bool {
        let (a, b) = (&self.0, &o.0);
        a.friction == b.friction
            && a.restitution == b.restitution
            && a.rollingResistance == b.rollingResistance
            && a.tangentVelocity.x == b.tangentVelocity.x
            && a.tangentVelocity.y == b.tangentVelocity.y
            && a.tangentVelocity.z == b.tangentVelocity.z
            && a.userMaterialId == b.userMaterialId
            && a.customColor == b.customColor
    }
}

impl SurfaceMaterial {
    /// Coulomb friction, usually `[0, 1]`. Pairs mix as `sqrt(a * b)`.
    pub fn friction(mut self, friction: f32) -> Self {
        self.0.friction = friction;
        self
    }
    /// Restitution (bounce), usually `[0, 1]`. Pairs mix as `max(a, b)`.
    pub fn restitution(mut self, restitution: f32) -> Self {
        self.0.restitution = restitution;
        self
    }
    /// Rolling resistance, usually `[0, 1]`. Pairs mix as `max(a, b)`, then
    /// scale by the larger rolling radius of the pair: sphere/capsule radius,
    /// or `0.25 * inner_radius` for hulls (the C header says "spheres and
    /// capsules only", but hulls get the reduced radius too). The resulting
    /// resisting torque is capped at `rr * radius * normal_impulse`.
    pub fn rolling_resistance(mut self, rolling_resistance: f32) -> Self {
        self.0.rollingResistance = rolling_resistance;
        self
    }
    /// Conveyor-belt velocity, shape-local, projected onto the contact plane.
    pub fn tangent_velocity(mut self, v: Vec3) -> Self {
        self.0.tangentVelocity = v3(v);
        self
    }
    /// Opaque id reported by ray/shape casts and hit events.
    pub fn user_material_id(mut self, id: u64) -> Self {
        self.0.userMaterialId = id;
        self
    }
    pub fn as_raw(&self) -> &sys::b3SurfaceMaterial {
        &self.0
    }
}

/// Collision filter (category/mask/group). Default: category 1, mask all, group 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Filter {
    pub category_bits: u64,
    pub mask_bits: u64,
    /// Same non-zero group: negative never collide, positive always collide.
    pub group_index: i32,
}

impl Default for Filter {
    fn default() -> Self {
        // SAFETY: returns a value.
        Self::from_raw(unsafe { sys::b3DefaultFilter() })
    }
}

impl Filter {
    pub(crate) fn from_raw(f: sys::b3Filter) -> Self {
        Self { category_bits: f.categoryBits, mask_bits: f.maskBits, group_index: f.groupIndex }
    }

    pub(crate) fn to_raw(self) -> sys::b3Filter {
        sys::b3Filter { categoryBits: self.category_bits, maskBits: self.mask_bits, groupIndex: self.group_index }
    }
}

/// Shape creation parameters: a value wrapper over `b3ShapeDef`.
/// `Default` is `b3DefaultShapeDef`: density 1000 kg/m^3, all events **off**.
///
/// `'m` borrows the optional per-triangle mesh material slice.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ShapeDef<'m>(pub(crate) sys::b3ShapeDef, PhantomData<&'m [SurfaceMaterial]>);

// SAFETY: `name` is never set, `userData` holds an integer, and `materials`
// borrows an immutable `&'m [SurfaceMaterial]` (plain data).
unsafe impl Send for ShapeDef<'_> {}
// SAFETY: as above.
unsafe impl Sync for ShapeDef<'_> {}

impl Default for ShapeDef<'_> {
    fn default() -> Self {
        // SAFETY: returns a value with the validity cookie set.
        Self(unsafe { sys::b3DefaultShapeDef() }, PhantomData)
    }
}

impl<'m> ShapeDef<'m> {
    /// Base surface material (friction, restitution, rolling resistance, ...).
    pub fn material(mut self, m: SurfaceMaterial) -> Self {
        self.0.baseMaterial = m.0;
        self
    }
    pub fn friction(mut self, friction: f32) -> Self {
        self.0.baseMaterial.friction = friction;
        self
    }
    pub fn restitution(mut self, restitution: f32) -> Self {
        self.0.baseMaterial.restitution = restitution;
        self
    }
    /// See [`SurfaceMaterial::rolling_resistance`].
    pub fn rolling_resistance(mut self, rr: f32) -> Self {
        self.0.baseMaterial.rollingResistance = rr;
        self
    }
    /// Per-triangle materials for mesh/height-field shapes (ignored for convex
    /// shapes). Box3D copies them at shape creation.
    pub fn mesh_materials(mut self, materials: &'m [SurfaceMaterial]) -> Self {
        // SurfaceMaterial is repr(transparent) over b3SurfaceMaterial.
        self.0.materials = materials.as_ptr() as *mut sys::b3SurfaceMaterial;
        self.0.materialCount = materials.len().min(i32::MAX as usize) as i32;
        self
    }
    /// kg/m^3.
    pub fn density(mut self, density: f32) -> Self {
        self.0.density = density;
        self
    }
    pub fn filter(mut self, filter: Filter) -> Self {
        self.0.filter = filter.to_raw();
        self
    }
    /// Sensor: reports overlaps, never collides. Needs `enable_sensor_events`
    /// on the visitor shapes too.
    pub fn sensor(mut self, is_sensor: bool) -> Self {
        self.0.isSensor = is_sensor;
        self
    }
    pub fn enable_sensor_events(mut self, enabled: bool) -> Self {
        self.0.enableSensorEvents = enabled;
        self
    }
    /// Begin/end touch events (needed on at least one shape of a pair).
    pub fn enable_contact_events(mut self, enabled: bool) -> Self {
        self.0.enableContactEvents = enabled;
        self
    }
    /// Hit events above [`crate::WorldDef::hit_event_threshold`].
    pub fn enable_hit_events(mut self, enabled: bool) -> Self {
        self.0.enableHitEvents = enabled;
        self
    }
    /// Call the world's pre-solve callback for this shape's contacts.
    pub fn enable_pre_solve_events(mut self, enabled: bool) -> Self {
        self.0.enablePreSolveEvents = enabled;
        self
    }
    /// Call the world's custom filter callback for pairs involving this shape.
    pub fn enable_custom_filtering(mut self, enabled: bool) -> Self {
        self.0.enableCustomFiltering = enabled;
        self
    }
    /// Recompute body mass when the shape is added (default true).
    pub fn update_body_mass(mut self, update: bool) -> Self {
        self.0.updateBodyMass = update;
        self
    }
    /// Opaque 64-bit user value (e.g. hecs `Entity::to_bits().get()`). Never a pointer.
    pub fn user_data(mut self, value: u64) -> Self {
        self.0.userData = value as usize as *mut core::ffi::c_void;
        self
    }
    pub fn as_raw(&self) -> &sys::b3ShapeDef {
        &self.0
    }
}

/// Shape geometry type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapeType {
    Sphere,
    Capsule,
    Hull,
    Mesh,
    HeightField,
    Compound,
}

impl ShapeType {
    fn from_raw(t: sys::b3ShapeType) -> Self {
        match t {
            sys::b3ShapeType_b3_sphereShape => ShapeType::Sphere,
            sys::b3ShapeType_b3_capsuleShape => ShapeType::Capsule,
            sys::b3ShapeType_b3_hullShape => ShapeType::Hull,
            sys::b3ShapeType_b3_meshShape => ShapeType::Mesh,
            sys::b3ShapeType_b3_heightShape => ShapeType::HeightField,
            _ => ShapeType::Compound,
        }
    }
}

#[track_caller]
fn check_positive(v: f32, what: &'static str) {
    assert!(v.is_finite() && v > 0.0, "{what} must be positive and finite, got {v}");
}

/// Shape API. Every method panics if an id is stale or belongs to another world.
impl World {
    /// Attach a sphere of `radius` centered at body-local `center`.
    #[track_caller]
    pub fn create_sphere_shape(&mut self, body: BodyId, def: &ShapeDef<'_>, center: Vec3, radius: f32) -> ShapeId {
        let b = self.body(body);
        check_positive(radius, "sphere radius");
        let geom = sys::b3Sphere { center: v3(center), radius };
        // SAFETY: validated body; def/geometry are copied by Box3D.
        ShapeId(unsafe { sys::b3CreateSphereShape(b, &def.0, &geom) })
    }

    /// Attach a capsule between body-local `center1` and `center2`.
    #[track_caller]
    pub fn create_capsule_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef<'_>,
        center1: Vec3,
        center2: Vec3,
        radius: f32,
    ) -> ShapeId {
        let b = self.body(body);
        check_positive(radius, "capsule radius");
        let geom = sys::b3Capsule { center1: v3(center1), center2: v3(center2), radius };
        // SAFETY: validated body; def/geometry are copied by Box3D.
        ShapeId(unsafe { sys::b3CreateCapsuleShape(b, &def.0, &geom) })
    }

    /// Attach a convex hull. Box3D copies the hull into the world's
    /// content-deduplicated hull database, so `hull` may be dropped (or reused
    /// for many shapes) afterwards.
    #[track_caller]
    pub fn create_hull_shape(&mut self, body: BodyId, def: &ShapeDef<'_>, hull: &Hull) -> ShapeId {
        let b = self.body(body);
        // SAFETY: validated body; hull points to a valid b3HullData that Box3D clones.
        ShapeId(unsafe { sys::b3CreateHullShape(b, &def.0, hull.as_raw()) })
    }

    /// Attach a hull with a baked local `transform` and (possibly non-uniform
    /// or mirrored) `scale`, scale applied first. E.g. lay a Y-axis cylinder on
    /// its side for a rolling barrel.
    ///
    /// Box3D re-computes the hull for the baked geometry; in extreme cases
    /// (e.g. a near-zero scale axis) that fails and
    /// [`Error::HullCreationFailed`] is returned.
    #[track_caller]
    pub fn create_transformed_hull_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef<'_>,
        hull: &Hull,
        transform: Transform,
        scale: Vec3,
    ) -> Result<ShapeId, Error> {
        let b = self.body(body);
        // SAFETY: validated body; hull is cloned+baked by Box3D.
        let id =
            ShapeId(unsafe { sys::b3CreateTransformedHullShape(b, &def.0, hull.as_raw(), xf(transform), v3(scale)) });
        if id.is_null() { Err(Error::HullCreationFailed) } else { Ok(id) }
    }

    /// Axis-aligned box centered on the body origin. Same as
    /// `create_hull_shape(body, def, &Hull::cuboid(half_extents))` but builds
    /// the box hull on the stack (no Rust allocation).
    #[track_caller]
    pub fn create_box_shape(&mut self, body: BodyId, def: &ShapeDef<'_>, half_extents: Vec3) -> ShapeId {
        check_positive(half_extents.x, "half_extents.x");
        check_positive(half_extents.y, "half_extents.y");
        check_positive(half_extents.z, "half_extents.z");
        let b = self.body(body);
        // SAFETY: returns a value. b3BoxHull addresses its arrays by offsets
        // from `base`, so it is position independent; Box3D clones it.
        let hull = unsafe { sys::b3MakeBoxHull(half_extents.x, half_extents.y, half_extents.z) };
        // SAFETY: validated body; `hull.base` heads a valid box hull.
        ShapeId(unsafe { sys::b3CreateHullShape(b, &def.0, &hull.base) })
    }

    /// Destroy a shape. With `update_body_mass = false`, call
    /// [`World::apply_mass_from_shapes`] yourself.
    #[track_caller]
    pub fn destroy_shape(&mut self, id: ShapeId, update_body_mass: bool) {
        let s = self.shape(id);
        // SAFETY: validated id.
        unsafe { sys::b3DestroyShape(s, update_body_mass) }
    }

    #[track_caller]
    pub fn shape_type(&self, id: ShapeId) -> ShapeType {
        // SAFETY: validated id.
        ShapeType::from_raw(unsafe { sys::b3Shape_GetType(self.shape(id)) })
    }

    /// The body a shape is attached to.
    #[track_caller]
    pub fn shape_body(&self, id: ShapeId) -> BodyId {
        // SAFETY: validated id.
        BodyId(unsafe { sys::b3Shape_GetBody(self.shape(id)) })
    }

    #[track_caller]
    pub fn shape_friction(&self, id: ShapeId) -> f32 {
        // SAFETY: validated id.
        unsafe { sys::b3Shape_GetFriction(self.shape(id)) }
    }

    #[track_caller]
    pub fn set_shape_friction(&mut self, id: ShapeId, friction: f32) {
        let s = self.shape(id);
        // SAFETY: validated id.
        unsafe { sys::b3Shape_SetFriction(s, friction) }
    }

    #[track_caller]
    pub fn shape_restitution(&self, id: ShapeId) -> f32 {
        // SAFETY: validated id.
        unsafe { sys::b3Shape_GetRestitution(self.shape(id)) }
    }

    #[track_caller]
    pub fn set_shape_restitution(&mut self, id: ShapeId, restitution: f32) {
        let s = self.shape(id);
        // SAFETY: validated id.
        unsafe { sys::b3Shape_SetRestitution(s, restitution) }
    }

    #[track_caller]
    pub fn shape_surface_material(&self, id: ShapeId) -> SurfaceMaterial {
        // SAFETY: validated id.
        SurfaceMaterial(unsafe { sys::b3Shape_GetSurfaceMaterial(self.shape(id)) })
    }

    /// Replace friction, restitution, rolling resistance and tangent velocity at once.
    #[track_caller]
    pub fn set_shape_surface_material(&mut self, id: ShapeId, material: SurfaceMaterial) {
        let s = self.shape(id);
        // SAFETY: validated id.
        unsafe { sys::b3Shape_SetSurfaceMaterial(s, material.0) }
    }

    #[track_caller]
    pub fn shape_density(&self, id: ShapeId) -> f32 {
        // SAFETY: validated id.
        unsafe { sys::b3Shape_GetDensity(self.shape(id)) }
    }

    #[track_caller]
    pub fn set_shape_density(&mut self, id: ShapeId, density: f32, update_body_mass: bool) {
        let s = self.shape(id);
        // SAFETY: validated id.
        unsafe { sys::b3Shape_SetDensity(s, density, update_body_mass) }
    }

    #[track_caller]
    pub fn shape_filter(&self, id: ShapeId) -> Filter {
        // SAFETY: validated id.
        Filter::from_raw(unsafe { sys::b3Shape_GetFilter(self.shape(id)) })
    }

    /// Change the collision filter (almost as expensive as recreating the shape).
    #[track_caller]
    pub fn set_shape_filter(&mut self, id: ShapeId, filter: Filter, invoke_contacts: bool) {
        let s = self.shape(id);
        // SAFETY: validated id.
        unsafe { sys::b3Shape_SetFilter(s, filter.to_raw(), invoke_contacts) }
    }

    #[track_caller]
    pub fn enable_shape_contact_events(&mut self, id: ShapeId, enabled: bool) {
        let s = self.shape(id);
        // SAFETY: validated id.
        unsafe { sys::b3Shape_EnableContactEvents(s, enabled) }
    }

    #[track_caller]
    pub fn shape_user_data(&self, id: ShapeId) -> u64 {
        // SAFETY: validated id; pointer used only as an integer.
        unsafe { sys::b3Shape_GetUserData(self.shape(id)) as usize as u64 }
    }

    #[track_caller]
    pub fn set_shape_user_data(&mut self, id: ShapeId, value: u64) {
        let s = self.shape(id);
        // SAFETY: validated id; Box3D never dereferences user data.
        unsafe { sys::b3Shape_SetUserData(s, value as usize as *mut core::ffi::c_void) }
    }
}
