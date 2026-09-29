//! Re-implementations of header-only `static const` values, which have no
//! linkable symbol.

use crate::*;

pub const b3Vec3_zero: b3Vec3 = b3Vec3 { x: 0.0, y: 0.0, z: 0.0 };
pub const b3Vec3_one: b3Vec3 = b3Vec3 { x: 1.0, y: 1.0, z: 1.0 };
pub const b3Quat_identity: b3Quat = b3Quat { v: b3Vec3_zero, s: 1.0 };
pub const b3Transform_identity: b3Transform = b3Transform { p: b3Vec3_zero, q: b3Quat_identity };

pub const b3_nullWorldId: b3WorldId = b3WorldId { index1: 0, generation: 0 };
pub const b3_nullBodyId: b3BodyId = b3BodyId { index1: 0, world0: 0, generation: 0 };
pub const b3_nullShapeId: b3ShapeId = b3ShapeId { index1: 0, world0: 0, generation: 0 };
pub const b3_nullJointId: b3JointId = b3JointId { index1: 0, world0: 0, generation: 0 };
pub const b3_nullContactId: b3ContactId = b3ContactId { index1: 0, world0: 0, padding: 0, generation: 0 };

#[cfg(not(feature = "double-precision"))]
pub const b3Pos_zero: b3Pos = b3Vec3_zero;
#[cfg(feature = "double-precision")]
pub const b3Pos_zero: b3Pos = b3Pos { x: 0.0, y: 0.0, z: 0.0 };

#[cfg(not(feature = "double-precision"))]
pub const b3WorldTransform_identity: b3WorldTransform = b3Transform_identity;
#[cfg(feature = "double-precision")]
pub const b3WorldTransform_identity: b3WorldTransform = b3WorldTransform { p: b3Pos_zero, q: b3Quat_identity };
