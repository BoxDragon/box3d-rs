use crate::sys;
use glam::{Mat3, Quat, Vec3};

/// World-space position. `Vec3` normally, `DVec3` with `double-precision`.
#[cfg(not(feature = "double-precision"))]
pub type Pos = glam::Vec3;
/// World-space position. `Vec3` normally, `DVec3` with `double-precision`.
#[cfg(feature = "double-precision")]
pub type Pos = glam::DVec3;

/// A rigid transform in a body's local space (used for joint frames).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self { position: Vec3::ZERO, rotation: Quat::IDENTITY };

    pub fn from_translation(position: Vec3) -> Self {
        Self { position, rotation: Quat::IDENTITY }
    }

    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }
}

#[inline]
pub(crate) fn v3(v: Vec3) -> sys::b3Vec3 {
    sys::b3Vec3 { x: v.x, y: v.y, z: v.z }
}

#[inline]
pub(crate) fn from_v3(v: sys::b3Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

#[inline]
pub(crate) fn quat(q: Quat) -> sys::b3Quat {
    // Box3D asserts on non-normalized rotations; normalize defensively.
    let q = if q.is_normalized() { q } else { q.normalize() };
    sys::b3Quat { v: sys::b3Vec3 { x: q.x, y: q.y, z: q.z }, s: q.w }
}

#[inline]
pub(crate) fn from_quat(q: sys::b3Quat) -> Quat {
    Quat::from_xyzw(q.v.x, q.v.y, q.v.z, q.s)
}

#[inline]
pub(crate) fn xf(t: Transform) -> sys::b3Transform {
    sys::b3Transform { p: v3(t.position), q: quat(t.rotation) }
}

#[inline]
pub(crate) fn from_xf(t: sys::b3Transform) -> Transform {
    Transform { position: from_v3(t.p), rotation: from_quat(t.q) }
}

#[inline]
pub(crate) fn mat3(m: Mat3) -> sys::b3Matrix3 {
    sys::b3Matrix3 { cx: v3(m.x_axis), cy: v3(m.y_axis), cz: v3(m.z_axis) }
}

#[inline]
pub(crate) fn from_mat3(m: sys::b3Matrix3) -> Mat3 {
    Mat3::from_cols(from_v3(m.cx), from_v3(m.cy), from_v3(m.cz))
}

#[cfg(not(feature = "double-precision"))]
#[inline]
pub(crate) fn pos(p: Pos) -> sys::b3Pos {
    v3(p)
}

#[cfg(not(feature = "double-precision"))]
#[inline]
pub(crate) fn from_pos(p: sys::b3Pos) -> Pos {
    from_v3(p)
}

#[cfg(feature = "double-precision")]
#[inline]
pub(crate) fn pos(p: Pos) -> sys::b3Pos {
    sys::b3Pos { x: p.x, y: p.y, z: p.z }
}

#[cfg(feature = "double-precision")]
#[inline]
pub(crate) fn from_pos(p: sys::b3Pos) -> Pos {
    glam::DVec3::new(p.x, p.y, p.z)
}
