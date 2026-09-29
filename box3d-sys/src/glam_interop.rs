//! `From`/`Into` between Box3D math structs and glam 0.30.

use crate::{b3Quat, b3Vec3};

impl From<glam::Vec3> for b3Vec3 {
    #[inline]
    fn from(v: glam::Vec3) -> Self {
        b3Vec3 { x: v.x, y: v.y, z: v.z }
    }
}

impl From<b3Vec3> for glam::Vec3 {
    #[inline]
    fn from(v: b3Vec3) -> Self {
        glam::Vec3::new(v.x, v.y, v.z)
    }
}

impl From<glam::Vec3A> for b3Vec3 {
    #[inline]
    fn from(v: glam::Vec3A) -> Self {
        b3Vec3 { x: v.x, y: v.y, z: v.z }
    }
}

impl From<b3Vec3> for glam::Vec3A {
    #[inline]
    fn from(v: b3Vec3) -> Self {
        glam::Vec3A::new(v.x, v.y, v.z)
    }
}

impl From<glam::Quat> for b3Quat {
    #[inline]
    fn from(q: glam::Quat) -> Self {
        b3Quat { v: b3Vec3 { x: q.x, y: q.y, z: q.z }, s: q.w }
    }
}

impl From<b3Quat> for glam::Quat {
    #[inline]
    fn from(q: b3Quat) -> Self {
        glam::Quat::from_xyzw(q.v.x, q.v.y, q.v.z, q.s)
    }
}

#[cfg(feature = "double-precision")]
mod double {
    use crate::b3Pos;

    impl From<glam::DVec3> for b3Pos {
        #[inline]
        fn from(v: glam::DVec3) -> Self {
            b3Pos { x: v.x, y: v.y, z: v.z }
        }
    }

    impl From<b3Pos> for glam::DVec3 {
        #[inline]
        fn from(v: b3Pos) -> Self {
            glam::DVec3::new(v.x, v.y, v.z)
        }
    }
}
