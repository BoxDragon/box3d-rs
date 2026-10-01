use crate::id::{BodyId, JointId, ShapeId};
use crate::math::{from_mat3, from_pos, from_quat, from_v3, mat3, pos, quat, v3};
use crate::{Mat3, Pos, Quat, Vec3, World, sys};

/// Seconds every body of an island must rest below its sleep threshold
/// before the island falls asleep (`B3_TIME_TO_SLEEP`).
pub const TIME_TO_SLEEP: f32 = sys::B3_TIME_TO_SLEEP as f32;

/// Body simulation type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BodyType {
    /// Zero mass, zero velocity, may be moved manually.
    #[default]
    Static,
    /// Zero mass, velocity set by the user, moved by the solver.
    Kinematic,
    /// Positive mass, moved by forces and contacts.
    Dynamic,
}

impl BodyType {
    pub(crate) fn to_raw(self) -> sys::b3BodyType {
        match self {
            BodyType::Static => sys::b3BodyType_b3_staticBody,
            BodyType::Kinematic => sys::b3BodyType_b3_kinematicBody,
            BodyType::Dynamic => sys::b3BodyType_b3_dynamicBody,
        }
    }

    pub(crate) fn from_raw(t: sys::b3BodyType) -> Self {
        match t {
            sys::b3BodyType_b3_kinematicBody => BodyType::Kinematic,
            sys::b3BodyType_b3_dynamicBody => BodyType::Dynamic,
            _ => BodyType::Static,
        }
    }
}

/// Body creation parameters: a value wrapper over `b3BodyDef` with by-value
/// builder setters. `Default` is `b3DefaultBodyDef` (static body at origin).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct BodyDef(pub(crate) sys::b3BodyDef);

// SAFETY: `name` is never set (null) and `userData` only ever holds an integer.
unsafe impl Send for BodyDef {}
// SAFETY: as above.
unsafe impl Sync for BodyDef {}

impl Default for BodyDef {
    fn default() -> Self {
        // SAFETY: returns a value with the validity cookie set.
        Self(unsafe { sys::b3DefaultBodyDef() })
    }
}

impl BodyDef {
    pub fn new(body_type: BodyType, position: Pos) -> Self {
        Self::default().body_type(body_type).position(position)
    }
    pub fn dynamic(position: Pos) -> Self {
        Self::new(BodyType::Dynamic, position)
    }
    pub fn kinematic(position: Pos) -> Self {
        Self::new(BodyType::Kinematic, position)
    }
    pub fn fixed(position: Pos) -> Self {
        Self::new(BodyType::Static, position)
    }

    pub fn body_type(mut self, t: BodyType) -> Self {
        self.0.type_ = t.to_raw();
        self
    }
    pub fn position(mut self, p: Pos) -> Self {
        self.0.position = pos(p);
        self
    }
    pub fn rotation(mut self, q: Quat) -> Self {
        self.0.rotation = quat(q);
        self
    }
    pub fn linear_velocity(mut self, v: Vec3) -> Self {
        self.0.linearVelocity = v3(v);
        self
    }
    pub fn angular_velocity(mut self, w: Vec3) -> Self {
        self.0.angularVelocity = v3(w);
        self
    }
    pub fn linear_damping(mut self, d: f32) -> Self {
        self.0.linearDamping = d;
        self
    }
    pub fn angular_damping(mut self, d: f32) -> Self {
        self.0.angularDamping = d;
        self
    }
    pub fn gravity_scale(mut self, s: f32) -> Self {
        self.0.gravityScale = s;
        self
    }
    /// Sleep speed threshold, m/s (default 0.05).
    pub fn sleep_threshold(mut self, speed: f32) -> Self {
        self.0.sleepThreshold = speed;
        self
    }
    pub fn enable_sleep(mut self, enabled: bool) -> Self {
        self.0.enableSleep = enabled;
        self
    }
    pub fn awake(mut self, awake: bool) -> Self {
        self.0.isAwake = awake;
        self
    }
    /// Continuous collision against other dynamic bodies. Use sparingly.
    pub fn bullet(mut self, bullet: bool) -> Self {
        self.0.isBullet = bullet;
        self
    }
    /// Skip rotation speed clamping (wheels, rolling barrels).
    pub fn allow_fast_rotation(mut self, allow: bool) -> Self {
        self.0.allowFastRotation = allow;
        self
    }
    /// Opaque 64-bit user value, e.g. a hecs `Entity::to_bits().get()`.
    /// Returned by [`World::body_user_data`] and [`crate::BodyMoveEvent::user_data`].
    /// Never a pointer.
    pub fn user_data(mut self, value: u64) -> Self {
        self.0.userData = value as usize as *mut core::ffi::c_void;
        self
    }
    pub fn as_raw(&self) -> &sys::b3BodyDef {
        &self.0
    }
}

/// Mass properties of a body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassData {
    /// Mass, kg.
    pub mass: f32,
    /// Center of mass in body-local space.
    pub center: Vec3,
    /// Rotational inertia about the center of mass, kg*m^2.
    pub rotational_inertia: Mat3,
}

impl MassData {
    pub(crate) fn from_raw(m: sys::b3MassData) -> Self {
        Self { mass: m.mass, center: from_v3(m.center), rotational_inertia: from_mat3(m.inertia) }
    }

    pub(crate) fn to_raw(self) -> sys::b3MassData {
        sys::b3MassData { mass: self.mass, center: v3(self.center), inertia: mat3(self.rotational_inertia) }
    }
}

/// Body API. Every method panics if the id is stale or belongs to another world.
impl World {
    /// Create a rigid body. Add shapes to give it collision and mass.
    pub fn create_body(&mut self, def: &BodyDef) -> BodyId {
        // SAFETY: valid world, def from b3DefaultBodyDef, not inside a callback.
        BodyId(unsafe { sys::b3CreateBody(self.id, &def.0) })
    }

    /// Destroy a body and all shapes and joints attached to it.
    #[track_caller]
    pub fn destroy_body(&mut self, id: BodyId) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3DestroyBody(b) }
        // Release meshes / height fields only used by this body's shapes.
        self.prune_resources();
    }

    #[track_caller]
    pub fn body_type(&self, id: BodyId) -> BodyType {
        // SAFETY: validated id.
        BodyType::from_raw(unsafe { sys::b3Body_GetType(self.body(id)) })
    }

    /// Change the body type (expensive). Updates mass properties.
    ///
    /// # Panics
    /// If the id is stale/foreign, or if the body carries a height-field shape
    /// and `body_type` is not [`BodyType::Static`]. (Box3D does not support
    /// that change, and its early return in `b3Body_SetType` would leave the
    /// world permanently locked.)
    #[track_caller]
    pub fn set_body_type(&mut self, id: BodyId, body_type: BodyType) {
        let b = self.body(id);
        assert!(
            body_type == BodyType::Static || !self.body_has_height_field(id),
            "a body with a height-field shape must stay static"
        );
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetType(b, body_type.to_raw()) }
    }

    /// Body origin in world space.
    #[track_caller]
    pub fn body_position(&self, id: BodyId) -> Pos {
        // SAFETY: validated id.
        from_pos(unsafe { sys::b3Body_GetPosition(self.body(id)) })
    }

    #[track_caller]
    pub fn body_rotation(&self, id: BodyId) -> Quat {
        // SAFETY: validated id.
        from_quat(unsafe { sys::b3Body_GetRotation(self.body(id)) })
    }

    /// `(position, rotation)` of the body origin.
    #[track_caller]
    pub fn body_transform(&self, id: BodyId) -> (Pos, Quat) {
        // SAFETY: validated id.
        let t = unsafe { sys::b3Body_GetTransform(self.body(id)) };
        (from_pos(t.p), from_quat(t.q))
    }

    /// Teleport a body. Fairly expensive; prefer creating bodies in place and,
    /// for kinematic bodies, [`Self::set_body_target_transform`].
    #[track_caller]
    pub fn set_body_transform(&mut self, id: BodyId, position: Pos, rotation: Quat) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetTransform(b, pos(position), quat(rotation)) }
    }

    /// Set velocities so a (kinematic) body reaches `position`/`rotation`
    /// after `time_step`. Good for driving crane parts.
    #[track_caller]
    pub fn set_body_target_transform(&mut self, id: BodyId, position: Pos, rotation: Quat, time_step: f32, wake: bool) {
        let b = self.body(id);
        let target = sys::b3WorldTransform { p: pos(position), q: quat(rotation) };
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetTargetTransform(b, target, time_step, wake) }
    }

    /// World point from a body-local point.
    #[track_caller]
    pub fn body_world_point(&self, id: BodyId, local_point: Vec3) -> Pos {
        // SAFETY: validated id.
        from_pos(unsafe { sys::b3Body_GetWorldPoint(self.body(id), v3(local_point)) })
    }

    /// Body-local point from a world point.
    #[track_caller]
    pub fn body_local_point(&self, id: BodyId, world_point: Pos) -> Vec3 {
        // SAFETY: validated id.
        from_v3(unsafe { sys::b3Body_GetLocalPoint(self.body(id), pos(world_point)) })
    }

    /// Linear velocity of the center of mass, m/s.
    #[track_caller]
    pub fn body_linear_velocity(&self, id: BodyId) -> Vec3 {
        // SAFETY: validated id.
        from_v3(unsafe { sys::b3Body_GetLinearVelocity(self.body(id)) })
    }

    #[track_caller]
    pub fn set_body_linear_velocity(&mut self, id: BodyId, velocity: Vec3) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetLinearVelocity(b, v3(velocity)) }
    }

    /// Angular velocity, rad/s (world space).
    #[track_caller]
    pub fn body_angular_velocity(&self, id: BodyId) -> Vec3 {
        // SAFETY: validated id.
        from_v3(unsafe { sys::b3Body_GetAngularVelocity(self.body(id)) })
    }

    #[track_caller]
    pub fn set_body_angular_velocity(&mut self, id: BodyId, velocity: Vec3) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetAngularVelocity(b, v3(velocity)) }
    }

    /// Apply a force (N) at a world point. Ignored if the body is asleep and `wake` is false.
    #[track_caller]
    pub fn apply_force(&mut self, id: BodyId, force: Vec3, point: Pos, wake: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_ApplyForce(b, v3(force), pos(point), wake) }
    }

    #[track_caller]
    pub fn apply_force_to_center(&mut self, id: BodyId, force: Vec3, wake: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_ApplyForceToCenter(b, v3(force), wake) }
    }

    /// Apply a world-space torque (N*m).
    #[track_caller]
    pub fn apply_torque(&mut self, id: BodyId, torque: Vec3, wake: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_ApplyTorque(b, v3(torque), wake) }
    }

    /// Apply a linear impulse (N*s) at a world point. One-shot; use forces for steady pushes.
    #[track_caller]
    pub fn apply_linear_impulse(&mut self, id: BodyId, impulse: Vec3, point: Pos, wake: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_ApplyLinearImpulse(b, v3(impulse), pos(point), wake) }
    }

    #[track_caller]
    pub fn apply_linear_impulse_to_center(&mut self, id: BodyId, impulse: Vec3, wake: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_ApplyLinearImpulseToCenter(b, v3(impulse), wake) }
    }

    /// Apply a world-space angular impulse (kg*m^2/s).
    #[track_caller]
    pub fn apply_angular_impulse(&mut self, id: BodyId, impulse: Vec3, wake: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_ApplyAngularImpulse(b, v3(impulse), wake) }
    }

    /// Mass in kg.
    #[track_caller]
    pub fn body_mass(&self, id: BodyId) -> f32 {
        // SAFETY: validated id.
        unsafe { sys::b3Body_GetMass(self.body(id)) }
    }

    #[track_caller]
    pub fn body_mass_data(&self, id: BodyId) -> MassData {
        // SAFETY: validated id.
        MassData::from_raw(unsafe { sys::b3Body_GetMassData(self.body(id)) })
    }

    /// Override mass properties. Lost when shapes are added/removed or the type changes.
    #[track_caller]
    pub fn set_body_mass_data(&mut self, id: BodyId, mass_data: MassData) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetMassData(b, mass_data.to_raw()) }
    }

    /// Recompute mass from the attached shapes' densities.
    #[track_caller]
    pub fn apply_mass_from_shapes(&mut self, id: BodyId) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_ApplyMassFromShapes(b) }
    }

    /// Center of mass in world space.
    #[track_caller]
    pub fn body_world_center(&self, id: BodyId) -> Pos {
        // SAFETY: validated id.
        from_pos(unsafe { sys::b3Body_GetWorldCenter(self.body(id)) })
    }

    #[track_caller]
    pub fn set_body_linear_damping(&mut self, id: BodyId, damping: f32) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetLinearDamping(b, damping) }
    }

    #[track_caller]
    pub fn set_body_angular_damping(&mut self, id: BodyId, damping: f32) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetAngularDamping(b, damping) }
    }

    #[track_caller]
    pub fn set_body_gravity_scale(&mut self, id: BodyId, scale: f32) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetGravityScale(b, scale) }
    }

    #[track_caller]
    pub fn is_body_awake(&self, id: BodyId) -> bool {
        // SAFETY: validated id.
        unsafe { sys::b3Body_IsAwake(self.body(id)) }
    }

    /// Wake (or put to sleep) the body's whole island.
    #[track_caller]
    pub fn set_body_awake(&mut self, id: BodyId, awake: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetAwake(b, awake) }
    }

    /// Allow or forbid this body from sleeping.
    #[track_caller]
    pub fn enable_body_sleep(&mut self, id: BodyId, enabled: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_EnableSleep(b, enabled) }
    }

    /// Seconds the body has rested below its sleep threshold. Its island
    /// falls asleep at the end of a step in which every body in it has
    /// rested [`TIME_TO_SLEEP`].
    #[track_caller]
    pub fn body_sleep_time(&self, id: BodyId) -> f32 {
        // SAFETY: validated id.
        unsafe { sys::b3Body_GetSleepTime(self.body(id)) }
    }

    /// Set the body's sleep time, e.g. copied from another world so its
    /// island falls asleep on the same step there. A sleeping body ignores a
    /// time of at least [`TIME_TO_SLEEP`]; a smaller one wakes its island
    /// first (waking resets the island's sleep times to zero). Static and
    /// disabled bodies ignore it.
    #[track_caller]
    pub fn set_body_sleep_time(&mut self, id: BodyId, seconds: f32) {
        assert!(seconds.is_finite() && seconds >= 0.0, "sleep time {seconds} must be finite and non-negative");
        let b = self.body(id);
        // SAFETY: validated id and time.
        unsafe { sys::b3Body_SetSleepTime(b, seconds) }
    }

    #[track_caller]
    pub fn set_body_bullet(&mut self, id: BodyId, bullet: bool) {
        let b = self.body(id);
        // SAFETY: validated id.
        unsafe { sys::b3Body_SetBullet(b, bullet) }
    }

    #[track_caller]
    pub fn body_user_data(&self, id: BodyId) -> u64 {
        // SAFETY: validated id. The pointer is only used as an integer.
        unsafe { sys::b3Body_GetUserData(self.body(id)) as usize as u64 }
    }

    #[track_caller]
    pub fn set_body_user_data(&mut self, id: BodyId, value: u64) {
        let b = self.body(id);
        // SAFETY: validated id. Box3D never dereferences user data.
        unsafe { sys::b3Body_SetUserData(b, value as usize as *mut core::ffi::c_void) }
    }

    /// Number of shapes attached to a body.
    #[track_caller]
    pub fn body_shape_count(&self, id: BodyId) -> usize {
        // SAFETY: validated id.
        unsafe { sys::b3Body_GetShapeCount(self.body(id)) }.max(0) as usize
    }

    /// Append the body's shapes to `out`. Uses a stack buffer; only bodies
    /// with more than 64 shapes need a temporary heap buffer.
    #[track_caller]
    pub fn body_shapes(&self, id: BodyId, out: &mut impl Extend<ShapeId>) {
        let b = self.body(id);
        // SAFETY: validated id; `buf` holds `cap` elements.
        collect_ids(
            unsafe { sys::b3Body_GetShapeCount(b) },
            sys::b3ShapeId { index1: 0, world0: 0, generation: 0 },
            |buf, cap| unsafe { sys::b3Body_GetShapes(b, buf, cap) },
            |raw| out.extend(raw.iter().copied().map(ShapeId)),
        );
    }

    /// Number of joints attached to a body.
    #[track_caller]
    pub fn body_joint_count(&self, id: BodyId) -> usize {
        // SAFETY: validated id.
        unsafe { sys::b3Body_GetJointCount(self.body(id)) }.max(0) as usize
    }

    /// Append the body's joints to `out` (stack buffer up to 64 joints).
    #[track_caller]
    pub fn body_joints(&self, id: BodyId, out: &mut impl Extend<JointId>) {
        let b = self.body(id);
        // SAFETY: validated id; `buf` holds `cap` elements.
        collect_ids(
            unsafe { sys::b3Body_GetJointCount(b) },
            sys::b3JointId { index1: 0, world0: 0, generation: 0 },
            |buf, cap| unsafe { sys::b3Body_GetJoints(b, buf, cap) },
            |raw| out.extend(raw.iter().copied().map(JointId)),
        );
    }
}

const STACK_IDS: usize = 64;

fn collect_ids<T: Copy>(count: i32, zero: T, fill: impl FnOnce(*mut T, i32) -> i32, sink: impl FnOnce(&[T])) {
    let count = count.max(0) as usize;
    let mut stack = [zero; STACK_IDS];
    let mut heap;
    let buf: &mut [T] = if count <= STACK_IDS {
        &mut stack[..count]
    } else {
        heap = vec![zero; count];
        &mut heap[..]
    };
    let n = fill(buf.as_mut_ptr(), buf.len() as i32).clamp(0, buf.len() as i32) as usize;
    sink(&buf[..n]);
}
