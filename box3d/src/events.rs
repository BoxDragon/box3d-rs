//! Zero-copy post-step events.
//!
//! Every accessor returns a slice borrowed straight from Box3D's internal
//! event buffers (`slice::from_raw_parts`, no copying, no allocation). The
//! borrow is tied to `&World`, and [`World::step`] takes `&mut World`, so the
//! compiler guarantees no slice outlives the step that invalidates it.
//! Collect what you need before mutating the world.

use crate::id::{BodyId, JointId, ShapeId};
use crate::math::{from_pos, from_quat, from_v3};
use crate::{Pos, Quat, Vec3, World, sys};

/// # Safety
/// `ptr` must point to `count` initialized `C` values that stay valid for
/// `'a`, and `W` must be `repr(transparent)` over `C`.
unsafe fn slice<'a, C, W>(ptr: *const C, count: i32) -> &'a [W] {
    debug_assert_eq!(core::mem::size_of::<C>(), core::mem::size_of::<W>());
    if ptr.is_null() || count <= 0 {
        &[]
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { core::slice::from_raw_parts(ptr as *const W, count as usize) }
    }
}

macro_rules! event {
    ($(#[$m:meta])* $name:ident, $raw:ty) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $raw);

        impl $name {
            pub fn as_raw(&self) -> &$raw {
                &self.0
            }
        }
    };
}

event!(
    /// A body moved during the step (only moved bodies are reported; the
    /// cheapest way to sync render transforms).
    BodyMoveEvent,
    sys::b3BodyMoveEvent
);

impl BodyMoveEvent {
    pub fn body(&self) -> BodyId {
        BodyId(self.0.bodyId)
    }
    /// The body's user value (see [`crate::BodyDef::user_data`]).
    pub fn user_data(&self) -> u64 {
        self.0.userData as usize as u64
    }
    pub fn position(&self) -> Pos {
        from_pos(self.0.transform.p)
    }
    pub fn rotation(&self) -> Quat {
        from_quat(self.0.transform.q)
    }
    /// `(position, rotation)` of the body origin.
    pub fn transform(&self) -> (Pos, Quat) {
        (self.position(), self.rotation())
    }
    /// The body went to sleep this step.
    pub fn fell_asleep(&self) -> bool {
        self.0.fellAsleep
    }
}

event!(
    /// Two shapes started touching. Needs `enable_contact_events` on at least
    /// one of them (off by default).
    ContactBeginEvent,
    sys::b3ContactBeginTouchEvent
);

impl ContactBeginEvent {
    pub fn shape_a(&self) -> ShapeId {
        ShapeId(self.0.shapeIdA)
    }
    pub fn shape_b(&self) -> ShapeId {
        ShapeId(self.0.shapeIdB)
    }
}

event!(
    /// Two shapes stopped touching. Either shape may already be destroyed:
    /// check with [`World::is_shape_valid`].
    ContactEndEvent,
    sys::b3ContactEndTouchEvent
);

impl ContactEndEvent {
    pub fn shape_a(&self) -> ShapeId {
        ShapeId(self.0.shapeIdA)
    }
    pub fn shape_b(&self) -> ShapeId {
        ShapeId(self.0.shapeIdB)
    }
}

event!(
    /// Impact faster than the world's hit threshold on shapes with
    /// `enable_hit_events` (e.g. crate-drop sounds).
    ContactHitEvent,
    sys::b3ContactHitEvent
);

impl ContactHitEvent {
    pub fn shape_a(&self) -> ShapeId {
        ShapeId(self.0.shapeIdA)
    }
    pub fn shape_b(&self) -> ShapeId {
        ShapeId(self.0.shapeIdB)
    }
    pub fn point(&self) -> Pos {
        from_pos(self.0.point)
    }
    /// From shape A to shape B.
    pub fn normal(&self) -> Vec3 {
        from_v3(self.0.normal)
    }
    /// Always positive, m/s.
    pub fn approach_speed(&self) -> f32 {
        self.0.approachSpeed
    }
    pub fn user_material_a(&self) -> u64 {
        self.0.userMaterialIdA
    }
    pub fn user_material_b(&self) -> u64 {
        self.0.userMaterialIdB
    }
}

event!(
    /// A visitor shape started overlapping a sensor shape.
    SensorBeginEvent,
    sys::b3SensorBeginTouchEvent
);

impl SensorBeginEvent {
    pub fn sensor_shape(&self) -> ShapeId {
        ShapeId(self.0.sensorShapeId)
    }
    pub fn visitor_shape(&self) -> ShapeId {
        ShapeId(self.0.visitorShapeId)
    }
}

event!(
    /// A visitor stopped overlapping a sensor. Either may be destroyed.
    SensorEndEvent,
    sys::b3SensorEndTouchEvent
);

impl SensorEndEvent {
    pub fn sensor_shape(&self) -> ShapeId {
        ShapeId(self.0.sensorShapeId)
    }
    pub fn visitor_shape(&self) -> ShapeId {
        ShapeId(self.0.visitorShapeId)
    }
}

event!(
    /// An awake joint exceeded its force or torque threshold (e.g. a rope
    /// that should snap).
    JointEvent,
    sys::b3JointEvent
);

impl JointEvent {
    pub fn joint(&self) -> JointId {
        JointId(self.0.jointId)
    }
    /// The joint's user value (see the joint defs' `user_data`).
    pub fn user_data(&self) -> u64 {
        self.0.userData as usize as u64
    }
}

/// Body events of the last step.
#[derive(Clone, Copy, Debug)]
pub struct BodyEvents<'w> {
    moves: &'w [BodyMoveEvent],
}

impl<'w> BodyEvents<'w> {
    pub fn moves(&self) -> &'w [BodyMoveEvent] {
        self.moves
    }
}

/// Contact events of the last step.
#[derive(Clone, Copy, Debug)]
pub struct ContactEvents<'w> {
    begin: &'w [ContactBeginEvent],
    end: &'w [ContactEndEvent],
    hit: &'w [ContactHitEvent],
}

impl<'w> ContactEvents<'w> {
    pub fn begin(&self) -> &'w [ContactBeginEvent] {
        self.begin
    }
    pub fn end(&self) -> &'w [ContactEndEvent] {
        self.end
    }
    pub fn hit(&self) -> &'w [ContactHitEvent] {
        self.hit
    }
}

/// Sensor events of the last step.
#[derive(Clone, Copy, Debug)]
pub struct SensorEvents<'w> {
    begin: &'w [SensorBeginEvent],
    end: &'w [SensorEndEvent],
}

impl<'w> SensorEvents<'w> {
    pub fn begin(&self) -> &'w [SensorBeginEvent] {
        self.begin
    }
    pub fn end(&self) -> &'w [SensorEndEvent] {
        self.end
    }
}

/// Joint events of the last step.
#[derive(Clone, Copy, Debug)]
pub struct JointEvents<'w> {
    events: &'w [JointEvent],
}

impl<'w> JointEvents<'w> {
    pub fn events(&self) -> &'w [JointEvent] {
        self.events
    }
}

// SAFETY (all four getters below): the arrays live inside the world and are
// only cleared or reallocated by b3World_Step or other world mutation, all of
// which require `&mut World`; the returned borrow of `&self` therefore keeps
// them valid. Event wrappers are repr(transparent) over the C structs.
impl World {
    /// Bodies that moved during the last step.
    pub fn body_events(&self) -> BodyEvents<'_> {
        // SAFETY: see above.
        unsafe {
            let e = sys::b3World_GetBodyEvents(self.id);
            BodyEvents { moves: slice(e.moveEvents, e.moveCount) }
        }
    }

    /// Contact begin/end/hit events of the last step.
    pub fn contact_events(&self) -> ContactEvents<'_> {
        // SAFETY: see above.
        unsafe {
            let e = sys::b3World_GetContactEvents(self.id);
            ContactEvents {
                begin: slice(e.beginEvents, e.beginCount),
                end: slice(e.endEvents, e.endCount),
                hit: slice(e.hitEvents, e.hitCount),
            }
        }
    }

    /// Sensor overlap begin/end events of the last step.
    pub fn sensor_events(&self) -> SensorEvents<'_> {
        // SAFETY: see above.
        unsafe {
            let e = sys::b3World_GetSensorEvents(self.id);
            SensorEvents { begin: slice(e.beginEvents, e.beginCount), end: slice(e.endEvents, e.endCount) }
        }
    }

    /// Joint threshold events of the last step.
    pub fn joint_events(&self) -> JointEvents<'_> {
        // SAFETY: see above.
        unsafe {
            let e = sys::b3World_GetJointEvents(self.id);
            JointEvents { events: slice(e.jointEvents, e.count) }
        }
    }
}
