//! Shared checks for loading serialized hull and mesh blobs ([`crate::Hull::from_bytes`],
//! [`crate::Mesh::from_bytes`]).
//!
//! A blob is a header struct followed by arrays addressed by byte offsets from the header.
//! Loading copies the bytes into Box3D memory (16-byte aligned) and then checks every offset,
//! count and index the C code follows, so a corrupt blob is rejected instead of read out of
//! bounds.

use crate::Error;
use core::mem::{align_of, size_of};

/// Header checks on the caller's (possibly unaligned) bytes, before copying: length, version
/// (first field of both blob headers), and the header's own byte count.
pub(crate) fn check_header<H>(bytes: &[u8], version: u64, fields: impl Fn(&H) -> (u64, i32)) -> Result<(), Error> {
    if bytes.len() < size_of::<H>() {
        return Err(Error::InvalidBlob("shorter than its header"));
    }
    if bytes.len() > i32::MAX as usize {
        return Err(Error::InvalidBlob("larger than i32::MAX bytes"));
    }
    // SAFETY: at least size_of::<H>() readable bytes; H is plain C data (any bit pattern is
    // valid), read unaligned.
    let header = unsafe { (bytes.as_ptr() as *const H).read_unaligned() };
    let (found, byte_count) = fields(&header);
    if found != version {
        return Err(Error::BlobVersionMismatch);
    }
    if byte_count as usize != bytes.len() {
        return Err(Error::InvalidBlob("byteCount does not match the length"));
    }
    Ok(())
}

/// `count` values of `T` at byte `offset` of the blob `header`, which is `len` bytes long.
/// `None` unless the array lies after the header, inside the blob, and aligned for `T`.
///
/// `T` must be plain C data (any bit pattern valid).
pub(crate) fn array<H, T>(header: &H, len: usize, offset: i32, count: usize) -> Option<&[T]> {
    let offset = usize::try_from(offset).ok()?;
    let end = count.checked_mul(size_of::<T>())?.checked_add(offset)?;
    if offset < size_of::<H>() || end > len || offset % align_of::<T>() != 0 {
        return None;
    }
    // SAFETY: the range lies inside the `len`-byte allocation that starts at `header` (the
    // caller's contract), and the allocation is 16-byte aligned, so `offset` aligned for T
    // gives an aligned pointer.
    Some(unsafe { core::slice::from_raw_parts((header as *const H as *const u8).add(offset) as *const T, count) })
}

/// All components finite.
pub(crate) fn finite(v: &crate::sys::b3Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// Finite and ordered.
pub(crate) fn sane_aabb(b: &crate::sys::b3AABB) -> bool {
    let (l, u) = (&b.lowerBound, &b.upperBound);
    finite(l) && finite(u) && l.x <= u.x && l.y <= u.y && l.z <= u.z
}
