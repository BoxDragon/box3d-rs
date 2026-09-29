//! Raw FFI bindings to [Box3D](https://github.com/erincatto/box3d).
//!
//! Everything here is `unsafe` and mirrors the C API one-to-one (`b3*`
//! functions and types, `B3_*` constants). See the `box3d` crate for a safe
//! wrapper.
//!
//! By default the committed, pregenerated bindings in `src/bindings.rs`
//! (or `src/bindings_double.rs` with `double-precision`) are used so no
//! libclang is required. Enable the `bindgen` feature to regenerate them at
//! build time.
//!
//! # Safety notes
//! - Box3D validates ids only through `B3_ASSERT`, which is compiled out in
//!   release builds. Passing a stale or foreign id to any function other than
//!   the `b3*_IsValid` family is undefined behaviour in release.
//! - World creation/destruction touches a global world table without locking.
//! - Header `static const` values (`b3Vec3_zero`, `b3_nullBodyId`, ...) and
//!   `static inline` helpers (`b3Add`, `b3MakeRot`, ...) have no linkable
//!   symbol and are therefore not bound; a few are re-implemented in [`consts`].

#![no_std]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]
#![allow(rustdoc::all)]

#[cfg(all(feature = "bindgen", not(docsrs)))]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(all(not(all(feature = "bindgen", not(docsrs))), not(feature = "double-precision")))]
mod ffi {
    include!("bindings.rs");
}

#[cfg(all(not(all(feature = "bindgen", not(docsrs))), feature = "double-precision"))]
mod ffi {
    include!("bindings_double.rs");
}

pub use ffi::*;

pub mod consts;

#[cfg(feature = "glam")]
mod glam_interop;

/// In double-precision mode the C header renames `b3CreateWorld` to
/// `b3CreateWorldDoublePrecision` (to force a link error on a precision
/// mismatch). This alias keeps call sites identical.
#[cfg(feature = "double-precision")]
pub use ffi::b3CreateWorldDoublePrecision as b3CreateWorld;
