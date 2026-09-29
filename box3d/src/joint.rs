//! Joints. Creation is on [`World`]; per-type operations are methods on the
//! typed ids, taking `&World` for getters and `&mut World` for setters.
//! Every method panics if the joint is stale or belongs to another world.
//!
//! Joint definitions are value wrappers over the C `b3*JointDef` structs with
//! by-value builder setters; `Default` is the matching `b3Default*JointDef`.
//! Local frames are relative to each body's **origin** (not center of mass).

use crate::id::{BodyId, JointId};
use crate::math::{from_v3, from_xf, quat, v3, xf};
use crate::{Quat, Transform, Vec3, World, sys};

/// Joint type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JointType {
    Parallel,
    Distance,
    Filter,
    Motor,
    Prismatic,
    Revolute,
    Spherical,
    Weld,
    Wheel,
}

impl JointType {
    fn from_raw(t: sys::b3JointType) -> Self {
        match t {
            sys::b3JointType_b3_parallelJoint => JointType::Parallel,
            sys::b3JointType_b3_distanceJoint => JointType::Distance,
            sys::b3JointType_b3_filterJoint => JointType::Filter,
            sys::b3JointType_b3_motorJoint => JointType::Motor,
            sys::b3JointType_b3_prismaticJoint => JointType::Prismatic,
            sys::b3JointType_b3_revoluteJoint => JointType::Revolute,
            sys::b3JointType_b3_sphericalJoint => JointType::Spherical,
            sys::b3JointType_b3_weldJoint => JointType::Weld,
            _ => JointType::Wheel,
        }
    }
}

/// Declares a joint def wrapper with the shared `b3JointDef` builders.
macro_rules! joint_def {
    ($(#[$m:meta])* $name:ident, $raw:ty, $default:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $raw);

        // SAFETY: the only pointer (`base.userData`) holds an integer.
        unsafe impl Send for $name {}
        // SAFETY: as above.
        unsafe impl Sync for $name {}

        impl Default for $name {
            fn default() -> Self {
                // SAFETY: returns a value with the validity cookie set.
                Self(unsafe { sys::$default() })
            }
        }

        impl $name {
            /// Default def connecting `body_a` and `body_b` at their origins.
            pub fn new(body_a: BodyId, body_b: BodyId) -> Self {
                Self::default().bodies(body_a, body_b)
            }
            pub fn bodies(mut self, body_a: BodyId, body_b: BodyId) -> Self {
                self.0.base.bodyIdA = body_a.0;
                self.0.base.bodyIdB = body_b.0;
                self
            }
            /// Joint frame on body A, relative to A's origin.
            pub fn local_frame_a(mut self, frame: Transform) -> Self {
                self.0.base.localFrameA = xf(frame);
                self
            }
            /// Joint frame on body B, relative to B's origin.
            pub fn local_frame_b(mut self, frame: Transform) -> Self {
                self.0.base.localFrameB = xf(frame);
                self
            }
            /// Set only the frame origins (body-local anchor points).
            pub fn anchors(mut self, anchor_a: Vec3, anchor_b: Vec3) -> Self {
                self.0.base.localFrameA.p = v3(anchor_a);
                self.0.base.localFrameB.p = v3(anchor_b);
                self
            }
            /// Let the connected bodies collide with each other.
            pub fn collide_connected(mut self, collide: bool) -> Self {
                self.0.base.collideConnected = collide;
                self
            }
            /// Force (N) above which a joint event is reported (default `f32::MAX`).
            pub fn force_threshold(mut self, force: f32) -> Self {
                self.0.base.forceThreshold = force;
                self
            }
            /// Torque (N*m) above which a joint event is reported (default `f32::MAX`).
            pub fn torque_threshold(mut self, torque: f32) -> Self {
                self.0.base.torqueThreshold = torque;
                self
            }
            /// Opaque 64-bit user value (e.g. hecs `Entity::to_bits().get()`). Never a pointer.
            pub fn user_data(mut self, value: u64) -> Self {
                self.0.base.userData = value as usize as *mut core::ffi::c_void;
                self
            }
            pub fn as_raw(&self) -> &$raw {
                &self.0
            }
            #[track_caller]
            fn validated(&self, world: &World) -> $raw {
                let a = BodyId(self.0.base.bodyIdA);
                let b = BodyId(self.0.base.bodyIdB);
                world.body(a);
                world.body(b);
                assert!(a != b, "joint bodies must differ");
                self.0
            }
        }
    };
}

/// By-value builder for one C field.
macro_rules! field {
    ($(#[$m:meta])* $fname:ident, $cfield:ident, $ty:ty) => {
        $(#[$m])*
        pub fn $fname(mut self, value: $ty) -> Self {
            self.0.$cfield = value;
            self
        }
    };
    ($(#[$m:meta])* $fname:ident, $cfield:ident, $ty:ty, $conv:path) => {
        $(#[$m])*
        pub fn $fname(mut self, value: $ty) -> Self {
            self.0.$cfield = $conv(value);
            self
        }
    };
}

macro_rules! typed_joint {
    ($(#[$m:meta])* $name:ident, $ty:expr) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name(JointId);

        impl $name {
            /// The untyped id, for the generic joint API.
            #[inline]
            pub fn joint(self) -> JointId {
                self.0
            }

            /// Downcast an untyped joint id; `None` if stale, foreign, or of another type.
            pub fn from_joint(world: &World, id: JointId) -> Option<Self> {
                // SAFETY: validity checked first.
                (world.is_joint_valid(id) && unsafe { sys::b3Joint_GetType(id.0) } == $ty).then_some(Self(id))
            }

            #[inline]
            #[track_caller]
            fn raw(self, world: &World) -> sys::b3JointId {
                // Type is guaranteed by construction; generation check guarantees
                // the slot has not been reused.
                world.joint(self.0)
            }
        }

        impl From<$name> for JointId {
            fn from(j: $name) -> JointId {
                j.0
            }
        }
    };
}

/// Generates `pub fn name(self, world: &World) -> T { f(sys_fn(raw)) }`
macro_rules! getter {
    ($(#[$m:meta])* $fname:ident, $sys:ident, $ret:ty) => {
        $(#[$m])*
        #[track_caller]
        pub fn $fname(self, world: &World) -> $ret {
            // SAFETY: validated id of the right type.
            unsafe { sys::$sys(self.raw(world)) }
        }
    };
    ($(#[$m:meta])* $fname:ident, $sys:ident, $ret:ty, $conv:path) => {
        $(#[$m])*
        #[track_caller]
        pub fn $fname(self, world: &World) -> $ret {
            // SAFETY: validated id of the right type.
            $conv(unsafe { sys::$sys(self.raw(world)) })
        }
    };
}

/// Generates `pub fn name(self, world: &mut World, a: A, ...) { sys_fn(raw, a, ...) }`
macro_rules! setter {
    ($(#[$m:meta])* $fname:ident, $sys:ident $(, $arg:ident : $aty:ty => $conv:expr)*) => {
        $(#[$m])*
        #[track_caller]
        pub fn $fname(self, world: &mut World $(, $arg: $aty)*) {
            let j = self.raw(world);
            // SAFETY: validated id of the right type.
            unsafe { sys::$sys(j $(, $conv)*) }
        }
    };
}

// ---------------------------------------------------------------------------
// Generic joint API

impl World {
    /// Destroy a joint. `wake_attached` wakes both bodies.
    #[track_caller]
    pub fn destroy_joint(&mut self, id: impl Into<JointId>, wake_attached: bool) {
        let j = self.joint(id.into());
        // SAFETY: validated id.
        unsafe { sys::b3DestroyJoint(j, wake_attached) }
    }

    #[track_caller]
    pub fn joint_type(&self, id: impl Into<JointId>) -> JointType {
        // SAFETY: validated id.
        JointType::from_raw(unsafe { sys::b3Joint_GetType(self.joint(id.into())) })
    }

    /// `(body_a, body_b)`.
    #[track_caller]
    pub fn joint_bodies(&self, id: impl Into<JointId>) -> (BodyId, BodyId) {
        let j = self.joint(id.into());
        // SAFETY: validated id.
        unsafe { (BodyId(sys::b3Joint_GetBodyA(j)), BodyId(sys::b3Joint_GetBodyB(j))) }
    }

    #[track_caller]
    pub fn joint_local_frame_a(&self, id: impl Into<JointId>) -> Transform {
        // SAFETY: validated id.
        from_xf(unsafe { sys::b3Joint_GetLocalFrameA(self.joint(id.into())) })
    }

    #[track_caller]
    pub fn joint_local_frame_b(&self, id: impl Into<JointId>) -> Transform {
        // SAFETY: validated id.
        from_xf(unsafe { sys::b3Joint_GetLocalFrameB(self.joint(id.into())) })
    }

    /// Move the attachment frame on body A (e.g. slide a hook along a beam).
    #[track_caller]
    pub fn set_joint_local_frame_a(&mut self, id: impl Into<JointId>, frame: Transform) {
        let j = self.joint(id.into());
        // SAFETY: validated id.
        unsafe { sys::b3Joint_SetLocalFrameA(j, xf(frame)) }
    }

    #[track_caller]
    pub fn set_joint_local_frame_b(&mut self, id: impl Into<JointId>, frame: Transform) {
        let j = self.joint(id.into());
        // SAFETY: validated id.
        unsafe { sys::b3Joint_SetLocalFrameB(j, xf(frame)) }
    }

    #[track_caller]
    pub fn set_joint_collide_connected(&mut self, id: impl Into<JointId>, collide: bool) {
        let j = self.joint(id.into());
        // SAFETY: validated id.
        unsafe { sys::b3Joint_SetCollideConnected(j, collide) }
    }

    #[track_caller]
    pub fn wake_joint_bodies(&mut self, id: impl Into<JointId>) {
        let j = self.joint(id.into());
        // SAFETY: validated id.
        unsafe { sys::b3Joint_WakeBodies(j) }
    }

    /// Constraint force from the last step (N), world space.
    #[track_caller]
    pub fn joint_constraint_force(&self, id: impl Into<JointId>) -> Vec3 {
        // SAFETY: validated id.
        from_v3(unsafe { sys::b3Joint_GetConstraintForce(self.joint(id.into())) })
    }

    /// Constraint torque from the last step (N*m), world space.
    #[track_caller]
    pub fn joint_constraint_torque(&self, id: impl Into<JointId>) -> Vec3 {
        // SAFETY: validated id.
        from_v3(unsafe { sys::b3Joint_GetConstraintTorque(self.joint(id.into())) })
    }

    /// The joint's user value (see the defs' `user_data`).
    #[track_caller]
    pub fn joint_user_data(&self, id: impl Into<JointId>) -> u64 {
        // SAFETY: validated id; pointer used only as an integer.
        unsafe { sys::b3Joint_GetUserData(self.joint(id.into())) as usize as u64 }
    }
}

// ---------------------------------------------------------------------------
// Distance joint

typed_joint!(
    /// Distance joint: keeps two anchor points at a distance. Used for ropes,
    /// chains, winch cables and springs.
    DistanceJointId,
    sys::b3JointType_b3_distanceJoint
);

joint_def!(
    /// Distance joint definition. The C default is a **rigid rod** of length 1
    /// (spring disabled, which also disables limit and motor).
    ///
    /// For a rope, use [`DistanceJointDef::rope`]: spring enabled with zero
    /// stiffness (no force while slack) plus a `[min, max]` length limit.
    DistanceJointDef,
    sys::b3DistanceJointDef,
    b3DefaultDistanceJointDef
);

impl DistanceJointDef {
    /// Rigid rod of `length` between the anchors.
    pub fn rod(body_a: BodyId, body_b: BodyId, length: f32) -> Self {
        Self::new(body_a, body_b).length(length)
    }

    /// Slack rope: no force below `max_length`, hard stop at `max_length`.
    /// Winch it at runtime with [`DistanceJointId::set_length_range`].
    pub fn rope(body_a: BodyId, body_b: BodyId, max_length: f32) -> Self {
        Self::new(body_a, body_b)
            .length(max_length)
            .enable_spring(true)
            .hertz(0.0)
            .damping_ratio(0.0)
            .enable_limit(true)
            .length_range(0.0, max_length)
    }

    field!(/// Rest length, clamped to `[0.005 m (B3_LINEAR_SLOP), B3_HUGE]`.
        length, length, f32);
    field!(/// If false the joint is rigid and ignores limit and motor.
        enable_spring, enableSpring, bool);
    field!(/// Spring stiffness (Hz). 0 = no spring force.
        hertz, hertz, f32);
    field!(damping_ratio, dampingRatio, f32);
    field!(/// Max tension the spring applies (negative N). Default `-f32::MAX`.
        lower_spring_force, lowerSpringForce, f32);
    field!(/// Max compression the spring applies (N). Default `f32::MAX`.
        upper_spring_force, upperSpringForce, f32);
    field!(/// Only effective with the spring enabled.
        enable_limit, enableLimit, bool);
    field!(enable_motor, enableMotor, bool);
    field!(max_motor_force, maxMotorForce, f32);
    field!(/// m/s, positive lengthens.
        motor_speed, motorSpeed, f32);

    /// `[min, max]` length limit.
    pub fn length_range(mut self, min: f32, max: f32) -> Self {
        self.0.minLength = min.min(max);
        self.0.maxLength = max.max(min);
        self
    }
}

impl World {
    #[track_caller]
    pub fn create_distance_joint(&mut self, def: &DistanceJointDef) -> DistanceJointId {
        let d = def.validated(self);
        // SAFETY: bodies validated and owned by this world.
        DistanceJointId(JointId(unsafe { sys::b3CreateDistanceJoint(self.id, &d) }))
    }
}

impl DistanceJointId {
    getter!(/// Rest length.
        length, b3DistanceJoint_GetLength, f32);
    setter!(/// Set rest length (clamped to >= 0.005 m).
        set_length, b3DistanceJoint_SetLength, length: f32 => length);
    getter!(/// Current anchor distance.
        current_length, b3DistanceJoint_GetCurrentLength, f32);
    getter!(min_length, b3DistanceJoint_GetMinLength, f32);
    getter!(max_length, b3DistanceJoint_GetMaxLength, f32);
    setter!(/// Set `[min, max]` (each clamped to >= 0.005 m, swapped if reversed).
        set_length_range, b3DistanceJoint_SetLengthRange, min: f32 => min, max: f32 => max);
    getter!(is_spring_enabled, b3DistanceJoint_IsSpringEnabled, bool);
    setter!(/// When disabled the joint is rigid (limit/motor ignored).
        enable_spring, b3DistanceJoint_EnableSpring, enabled: bool => enabled);
    setter!(set_spring_hertz, b3DistanceJoint_SetSpringHertz, hertz: f32 => hertz);
    setter!(set_spring_damping_ratio, b3DistanceJoint_SetSpringDampingRatio, ratio: f32 => ratio);
    setter!(/// Tension (lower, negative) and compression (upper) force limits.
        set_spring_force_range, b3DistanceJoint_SetSpringForceRange, lower: f32 => lower, upper: f32 => upper);
    getter!(is_limit_enabled, b3DistanceJoint_IsLimitEnabled, bool);
    setter!(/// Only has an effect when the spring is enabled.
        enable_limit, b3DistanceJoint_EnableLimit, enabled: bool => enabled);
    setter!(enable_motor, b3DistanceJoint_EnableMotor, enabled: bool => enabled);
    setter!(set_motor_speed, b3DistanceJoint_SetMotorSpeed, speed: f32 => speed);
    setter!(set_max_motor_force, b3DistanceJoint_SetMaxMotorForce, force: f32 => force);
    getter!(motor_force, b3DistanceJoint_GetMotorForce, f32);
}

// ---------------------------------------------------------------------------
// Spherical joint

typed_joint!(
    /// Ball-and-socket joint with optional cone and twist limits.
    SphericalJointId,
    sys::b3JointType_b3_sphericalJoint
);

joint_def!(
    /// Spherical joint definition. The cone is centered on frame A's z-axis,
    /// the twist on frame B's z-axis.
    SphericalJointDef,
    sys::b3SphericalJointDef,
    b3DefaultSphericalJointDef
);

impl SphericalJointDef {
    field!(enable_cone_limit, enableConeLimit, bool);
    field!(/// Radians, `[0, pi]`.
        cone_angle, coneAngle, f32);
    field!(enable_twist_limit, enableTwistLimit, bool);
    field!(enable_spring, enableSpring, bool);
    field!(hertz, hertz, f32);
    field!(damping_ratio, dampingRatio, f32);
    field!(/// Frame B relative to frame A.
        target_rotation, targetRotation, Quat, quat);
    field!(enable_motor, enableMotor, bool);
    field!(max_motor_torque, maxMotorTorque, f32);
    field!(/// Relative angular velocity `w_b - w_a`, world space.
        motor_velocity, motorVelocity, Vec3, v3);

    /// Twist limits in radians, within `[-0.99 pi, 0.99 pi]`.
    pub fn twist_range(mut self, lower: f32, upper: f32) -> Self {
        self.0.lowerTwistAngle = lower;
        self.0.upperTwistAngle = upper;
        self
    }
}

impl World {
    #[track_caller]
    pub fn create_spherical_joint(&mut self, def: &SphericalJointDef) -> SphericalJointId {
        let d = def.validated(self);
        // SAFETY: bodies validated and owned by this world.
        SphericalJointId(JointId(unsafe { sys::b3CreateSphericalJoint(self.id, &d) }))
    }
}

impl SphericalJointId {
    setter!(enable_cone_limit, b3SphericalJoint_EnableConeLimit, enabled: bool => enabled);
    getter!(is_cone_limit_enabled, b3SphericalJoint_IsConeLimitEnabled, bool);
    setter!(/// Radians.
        set_cone_limit, b3SphericalJoint_SetConeLimit, angle: f32 => angle);
    getter!(cone_limit, b3SphericalJoint_GetConeLimit, f32);
    getter!(/// Current cone angle, radians.
        cone_angle, b3SphericalJoint_GetConeAngle, f32);
    setter!(enable_twist_limit, b3SphericalJoint_EnableTwistLimit, enabled: bool => enabled);
    getter!(is_twist_limit_enabled, b3SphericalJoint_IsTwistLimitEnabled, bool);
    setter!(/// Radians.
        set_twist_limits, b3SphericalJoint_SetTwistLimits, lower: f32 => lower, upper: f32 => upper);
    getter!(/// Current twist angle, radians.
        twist_angle, b3SphericalJoint_GetTwistAngle, f32);
    setter!(enable_spring, b3SphericalJoint_EnableSpring, enabled: bool => enabled);
    setter!(set_spring_hertz, b3SphericalJoint_SetSpringHertz, hertz: f32 => hertz);
    setter!(set_spring_damping_ratio, b3SphericalJoint_SetSpringDampingRatio, ratio: f32 => ratio);
    setter!(set_target_rotation, b3SphericalJoint_SetTargetRotation, rotation: Quat => quat(rotation));
    setter!(enable_motor, b3SphericalJoint_EnableMotor, enabled: bool => enabled);
    setter!(set_motor_velocity, b3SphericalJoint_SetMotorVelocity, velocity: Vec3 => v3(velocity));
    setter!(set_max_motor_torque, b3SphericalJoint_SetMaxMotorTorque, torque: f32 => torque);
}

// ---------------------------------------------------------------------------
// Revolute joint

typed_joint!(
    /// Hinge about the frames' z-axis, with optional spring, limits and motor.
    RevoluteJointId,
    sys::b3JointType_b3_revoluteJoint
);

joint_def!(
    /// Revolute (hinge) joint definition. Rotation is about the local frames' z-axis.
    RevoluteJointDef,
    sys::b3RevoluteJointDef,
    b3DefaultRevoluteJointDef
);

impl RevoluteJointDef {
    field!(/// Spring target angle, radians.
        target_angle, targetAngle, f32);
    field!(enable_spring, enableSpring, bool);
    field!(hertz, hertz, f32);
    field!(damping_ratio, dampingRatio, f32);
    field!(enable_limit, enableLimit, bool);
    field!(enable_motor, enableMotor, bool);
    field!(max_motor_torque, maxMotorTorque, f32);
    field!(/// rad/s.
        motor_speed, motorSpeed, f32);

    /// Angle limits in radians, within `[-0.99 pi, 0.99 pi]`.
    pub fn angle_range(mut self, lower: f32, upper: f32) -> Self {
        self.0.lowerAngle = lower;
        self.0.upperAngle = upper;
        self
    }
}

impl World {
    #[track_caller]
    pub fn create_revolute_joint(&mut self, def: &RevoluteJointDef) -> RevoluteJointId {
        let d = def.validated(self);
        // SAFETY: bodies validated and owned by this world.
        RevoluteJointId(JointId(unsafe { sys::b3CreateRevoluteJoint(self.id, &d) }))
    }
}

impl RevoluteJointId {
    getter!(/// Current angle, radians.
        angle, b3RevoluteJoint_GetAngle, f32);
    setter!(enable_limit, b3RevoluteJoint_EnableLimit, enabled: bool => enabled);
    setter!(/// Radians.
        set_limits, b3RevoluteJoint_SetLimits, lower: f32 => lower, upper: f32 => upper);
    getter!(lower_limit, b3RevoluteJoint_GetLowerLimit, f32);
    getter!(upper_limit, b3RevoluteJoint_GetUpperLimit, f32);
    setter!(enable_motor, b3RevoluteJoint_EnableMotor, enabled: bool => enabled);
    setter!(set_motor_speed, b3RevoluteJoint_SetMotorSpeed, speed: f32 => speed);
    getter!(motor_speed, b3RevoluteJoint_GetMotorSpeed, f32);
    setter!(set_max_motor_torque, b3RevoluteJoint_SetMaxMotorTorque, torque: f32 => torque);
    getter!(motor_torque, b3RevoluteJoint_GetMotorTorque, f32);
    setter!(enable_spring, b3RevoluteJoint_EnableSpring, enabled: bool => enabled);
    setter!(set_spring_hertz, b3RevoluteJoint_SetSpringHertz, hertz: f32 => hertz);
    setter!(set_spring_damping_ratio, b3RevoluteJoint_SetSpringDampingRatio, ratio: f32 => ratio);
    setter!(set_target_angle, b3RevoluteJoint_SetTargetAngle, radians: f32 => radians);
}

// ---------------------------------------------------------------------------
// Prismatic joint

typed_joint!(
    /// Slider along frame A's x-axis, no relative rotation.
    PrismaticJointId,
    sys::b3JointType_b3_prismaticJoint
);

joint_def!(
    /// Prismatic (slider) joint definition. Body B slides along frame A's x-axis.
    PrismaticJointDef,
    sys::b3PrismaticJointDef,
    b3DefaultPrismaticJointDef
);

impl PrismaticJointDef {
    field!(enable_spring, enableSpring, bool);
    field!(hertz, hertz, f32);
    field!(damping_ratio, dampingRatio, f32);
    field!(target_translation, targetTranslation, f32);
    field!(enable_limit, enableLimit, bool);
    field!(enable_motor, enableMotor, bool);
    field!(max_motor_force, maxMotorForce, f32);
    field!(motor_speed, motorSpeed, f32);

    pub fn translation_range(mut self, lower: f32, upper: f32) -> Self {
        self.0.lowerTranslation = lower;
        self.0.upperTranslation = upper;
        self
    }
}

impl World {
    #[track_caller]
    pub fn create_prismatic_joint(&mut self, def: &PrismaticJointDef) -> PrismaticJointId {
        let d = def.validated(self);
        // SAFETY: bodies validated and owned by this world.
        PrismaticJointId(JointId(unsafe { sys::b3CreatePrismaticJoint(self.id, &d) }))
    }
}

impl PrismaticJointId {
    getter!(translation, b3PrismaticJoint_GetTranslation, f32);
    getter!(speed, b3PrismaticJoint_GetSpeed, f32);
    setter!(enable_limit, b3PrismaticJoint_EnableLimit, enabled: bool => enabled);
    setter!(set_limits, b3PrismaticJoint_SetLimits, lower: f32 => lower, upper: f32 => upper);
    setter!(enable_motor, b3PrismaticJoint_EnableMotor, enabled: bool => enabled);
    setter!(set_motor_speed, b3PrismaticJoint_SetMotorSpeed, speed: f32 => speed);
    setter!(set_max_motor_force, b3PrismaticJoint_SetMaxMotorForce, force: f32 => force);
    getter!(motor_force, b3PrismaticJoint_GetMotorForce, f32);
    setter!(enable_spring, b3PrismaticJoint_EnableSpring, enabled: bool => enabled);
    setter!(set_spring_hertz, b3PrismaticJoint_SetSpringHertz, hertz: f32 => hertz);
    setter!(set_spring_damping_ratio, b3PrismaticJoint_SetSpringDampingRatio, ratio: f32 => ratio);
    setter!(set_target_translation, b3PrismaticJoint_SetTargetTranslation, translation: f32 => translation);
}

// ---------------------------------------------------------------------------
// Weld joint

typed_joint!(
    /// Weld: locks relative translation and rotation, optionally springy.
    WeldJointId,
    sys::b3JointType_b3_weldJoint
);

joint_def!(
    /// Weld joint definition. Zero hertz = as rigid as the solver allows.
    WeldJointDef,
    sys::b3WeldJointDef,
    b3DefaultWeldJointDef
);

impl WeldJointDef {
    field!(linear_hertz, linearHertz, f32);
    field!(angular_hertz, angularHertz, f32);
    field!(linear_damping_ratio, linearDampingRatio, f32);
    field!(angular_damping_ratio, angularDampingRatio, f32);
}

impl World {
    #[track_caller]
    pub fn create_weld_joint(&mut self, def: &WeldJointDef) -> WeldJointId {
        let d = def.validated(self);
        // SAFETY: bodies validated and owned by this world.
        WeldJointId(JointId(unsafe { sys::b3CreateWeldJoint(self.id, &d) }))
    }
}

impl WeldJointId {
    setter!(set_linear_hertz, b3WeldJoint_SetLinearHertz, hertz: f32 => hertz);
    setter!(set_angular_hertz, b3WeldJoint_SetAngularHertz, hertz: f32 => hertz);
    setter!(set_linear_damping_ratio, b3WeldJoint_SetLinearDampingRatio, ratio: f32 => ratio);
    setter!(set_angular_damping_ratio, b3WeldJoint_SetAngularDampingRatio, ratio: f32 => ratio);
}

// ---------------------------------------------------------------------------
// Filter joint

joint_def!(
    /// Filter joint definition: no constraint, only disables collision between
    /// the two bodies (e.g. a carried crate and its carrier). Leave
    /// `collide_connected` off.
    FilterJointDef,
    sys::b3FilterJointDef,
    b3DefaultFilterJointDef
);

impl World {
    #[track_caller]
    pub fn create_filter_joint(&mut self, def: &FilterJointDef) -> JointId {
        let d = def.validated(self);
        // SAFETY: bodies validated and owned by this world.
        JointId(unsafe { sys::b3CreateFilterJoint(self.id, &d) })
    }
}

// ---------------------------------------------------------------------------
// Wheel joint

typed_joint!(
    /// Wheel: body B spins about frame B's z-axis and suspends along frame A's x-axis.
    WheelJointId,
    sys::b3JointType_b3_wheelJoint
);

joint_def!(
    /// Wheel joint definition (body A = chassis, body B = wheel). C defaults:
    /// suspension spring on at 1 Hz / 0.7 damping.
    WheelJointDef,
    sys::b3WheelJointDef,
    b3DefaultWheelJointDef
);

impl WheelJointDef {
    field!(enable_suspension_spring, enableSuspensionSpring, bool);
    field!(suspension_hertz, suspensionHertz, f32);
    field!(suspension_damping_ratio, suspensionDampingRatio, f32);
    field!(enable_suspension_limit, enableSuspensionLimit, bool);
    field!(enable_spin_motor, enableSpinMotor, bool);
    field!(max_spin_torque, maxSpinTorque, f32);
    field!(/// rad/s.
        spin_speed, spinSpeed, f32);
    field!(enable_steering, enableSteering, bool);
    field!(steering_hertz, steeringHertz, f32);
    field!(steering_damping_ratio, steeringDampingRatio, f32);
    field!(target_steering_angle, targetSteeringAngle, f32);
    field!(max_steering_torque, maxSteeringTorque, f32);
    field!(enable_steering_limit, enableSteeringLimit, bool);

    pub fn suspension_range(mut self, lower: f32, upper: f32) -> Self {
        self.0.lowerSuspensionLimit = lower;
        self.0.upperSuspensionLimit = upper;
        self
    }

    pub fn steering_range(mut self, lower: f32, upper: f32) -> Self {
        self.0.lowerSteeringLimit = lower;
        self.0.upperSteeringLimit = upper;
        self
    }
}

impl World {
    #[track_caller]
    pub fn create_wheel_joint(&mut self, def: &WheelJointDef) -> WheelJointId {
        let d = def.validated(self);
        // SAFETY: bodies validated and owned by this world.
        WheelJointId(JointId(unsafe { sys::b3CreateWheelJoint(self.id, &d) }))
    }
}

impl WheelJointId {
    setter!(enable_suspension, b3WheelJoint_EnableSuspension, enabled: bool => enabled);
    setter!(set_suspension_hertz, b3WheelJoint_SetSuspensionHertz, hertz: f32 => hertz);
    setter!(set_suspension_damping_ratio, b3WheelJoint_SetSuspensionDampingRatio, ratio: f32 => ratio);
    setter!(enable_suspension_limit, b3WheelJoint_EnableSuspensionLimit, enabled: bool => enabled);
    setter!(set_suspension_limits, b3WheelJoint_SetSuspensionLimits, lower: f32 => lower, upper: f32 => upper);
    setter!(enable_spin_motor, b3WheelJoint_EnableSpinMotor, enabled: bool => enabled);
    setter!(set_spin_motor_speed, b3WheelJoint_SetSpinMotorSpeed, speed: f32 => speed);
    setter!(set_max_spin_torque, b3WheelJoint_SetMaxSpinTorque, torque: f32 => torque);
    getter!(spin_speed, b3WheelJoint_GetSpinSpeed, f32);
    getter!(spin_torque, b3WheelJoint_GetSpinTorque, f32);
    setter!(enable_steering, b3WheelJoint_EnableSteering, enabled: bool => enabled);
    setter!(set_target_steering_angle, b3WheelJoint_SetTargetSteeringAngle, radians: f32 => radians);
    setter!(set_steering_limits, b3WheelJoint_SetSteeringLimits, lower: f32 => lower, upper: f32 => upper);
    setter!(enable_steering_limit, b3WheelJoint_EnableSteeringLimit, enabled: bool => enabled);
    setter!(set_max_steering_torque, b3WheelJoint_SetMaxSteeringTorque, torque: f32 => torque);
    getter!(steering_angle, b3WheelJoint_GetSteeringAngle, f32);
}

// ---------------------------------------------------------------------------
// Motor joint

typed_joint!(
    /// Motor joint: drives the pose/velocity of B relative to A with
    /// force/torque limits while staying responsive to collisions (carrying).
    MotorJointId,
    sys::b3JointType_b3_motorJoint
);

joint_def!(
    /// Motor joint definition. The spring pulls frame B toward frame A; the
    /// velocity motor drives the relative velocity. C defaults are all zero.
    MotorJointDef,
    sys::b3MotorJointDef,
    b3DefaultMotorJointDef
);

impl MotorJointDef {
    field!(linear_velocity, linearVelocity, Vec3, v3);
    field!(max_velocity_force, maxVelocityForce, f32);
    field!(angular_velocity, angularVelocity, Vec3, v3);
    field!(max_velocity_torque, maxVelocityTorque, f32);
    field!(linear_hertz, linearHertz, f32);
    field!(linear_damping_ratio, linearDampingRatio, f32);
    field!(max_spring_force, maxSpringForce, f32);
    field!(angular_hertz, angularHertz, f32);
    field!(angular_damping_ratio, angularDampingRatio, f32);
    field!(max_spring_torque, maxSpringTorque, f32);
}

impl World {
    #[track_caller]
    pub fn create_motor_joint(&mut self, def: &MotorJointDef) -> MotorJointId {
        let d = def.validated(self);
        // SAFETY: bodies validated and owned by this world.
        MotorJointId(JointId(unsafe { sys::b3CreateMotorJoint(self.id, &d) }))
    }
}

impl MotorJointId {
    setter!(set_linear_velocity, b3MotorJoint_SetLinearVelocity, velocity: Vec3 => v3(velocity));
    setter!(set_angular_velocity, b3MotorJoint_SetAngularVelocity, velocity: Vec3 => v3(velocity));
    setter!(set_max_velocity_force, b3MotorJoint_SetMaxVelocityForce, force: f32 => force);
    setter!(set_max_velocity_torque, b3MotorJoint_SetMaxVelocityTorque, torque: f32 => torque);
    setter!(set_linear_hertz, b3MotorJoint_SetLinearHertz, hertz: f32 => hertz);
    setter!(set_linear_damping_ratio, b3MotorJoint_SetLinearDampingRatio, ratio: f32 => ratio);
    setter!(set_angular_hertz, b3MotorJoint_SetAngularHertz, hertz: f32 => hertz);
    setter!(set_angular_damping_ratio, b3MotorJoint_SetAngularDampingRatio, ratio: f32 => ratio);
    setter!(set_max_spring_force, b3MotorJoint_SetMaxSpringForce, force: f32 => force);
    setter!(set_max_spring_torque, b3MotorJoint_SetMaxSpringTorque, torque: f32 => torque);
}
