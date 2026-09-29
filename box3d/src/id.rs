use crate::sys;
use core::fmt;
use core::hash::{Hash, Hasher};

macro_rules! id_type {
    ($(#[$m:meta])* $name:ident, $raw:ty) => {
        $(#[$m])*
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $raw);

        impl $name {
            /// The null id. Never valid.
            pub const NULL: Self = Self(<$raw>::ZERO);

            /// The underlying C id.
            #[inline]
            pub fn raw(self) -> $raw {
                self.0
            }

            /// Wrap a C id. Safe: every use is validated by the [`World`](crate::World).
            #[inline]
            pub fn from_raw(raw: $raw) -> Self {
                Self(raw)
            }

            #[inline]
            pub fn is_null(self) -> bool {
                self.0.index1 == 0
            }

            /// Pack into a `u64` (same layout as Box3D's `b3Store*Id`), e.g. to
            /// store in an ECS component or engine handle.
            #[inline]
            pub fn to_bits(self) -> u64 {
                ((self.0.index1 as u32 as u64) << 32)
                    | ((self.0.world0 as u64) << 16)
                    | self.0.generation as u64
            }

            /// Inverse of [`Self::to_bits`].
            #[inline]
            pub fn from_bits(bits: u64) -> Self {
                Self(<$raw>::from_parts((bits >> 32) as u32 as i32, (bits >> 16) as u16, bits as u16))
            }

            #[inline]
            pub(crate) fn world0(self) -> u16 {
                self.0.world0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::NULL
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.to_bits() == other.to_bits()
            }
        }

        impl Eq for $name {}

        impl Hash for $name {
            #[inline]
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.to_bits().hash(state)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    concat!(stringify!($name), "({}:{}v{})"),
                    self.0.world0, self.0.index1, self.0.generation
                )
            }
        }
    };
}

trait RawId {
    const ZERO: Self;
    fn from_parts(index1: i32, world0: u16, generation: u16) -> Self;
}

macro_rules! raw_id {
    ($raw:ty) => {
        impl RawId for $raw {
            const ZERO: Self = Self { index1: 0, world0: 0, generation: 0 };
            fn from_parts(index1: i32, world0: u16, generation: u16) -> Self {
                Self { index1, world0, generation }
            }
        }
    };
}

raw_id!(sys::b3BodyId);
raw_id!(sys::b3ShapeId);
raw_id!(sys::b3JointId);

id_type!(
    /// Handle to a rigid body. Generational; invalid once the body or its world is destroyed.
    BodyId,
    sys::b3BodyId
);
id_type!(
    /// Handle to a shape. Generational; invalid once the shape, its body, or its world is destroyed.
    ShapeId,
    sys::b3ShapeId
);
id_type!(
    /// Handle to a joint of any type. Generational; invalid once the joint, either body, or the
    /// world is destroyed.
    JointId,
    sys::b3JointId
);

/// Handle to a world (4 bytes). Only useful for identification; the
/// [`World`](crate::World) value owns the simulation.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct WorldId(pub(crate) sys::b3WorldId);

impl WorldId {
    #[inline]
    pub fn raw(self) -> sys::b3WorldId {
        self.0
    }

    #[inline]
    pub fn to_bits(self) -> u32 {
        ((self.0.index1 as u32) << 16) | self.0.generation as u32
    }
}

impl PartialEq for WorldId {
    fn eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl Eq for WorldId {}

impl Hash for WorldId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_bits().hash(state)
    }
}

impl fmt::Debug for WorldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WorldId({}v{})", self.0.index1, self.0.generation)
    }
}

const _: () = {
    assert!(core::mem::size_of::<BodyId>() == 8);
    assert!(core::mem::size_of::<ShapeId>() == 8);
    assert!(core::mem::size_of::<JointId>() == 8);
    assert!(core::mem::size_of::<WorldId>() == 4);
};
