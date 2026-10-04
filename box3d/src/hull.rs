use crate::math::{from_v3, v3, xf};
use crate::{Error, Transform, Vec3, blob, sys};
use core::ptr::NonNull;

enum Storage {
    /// Heap hull from `b3CreateHull`/`b3CreateCylinder`/`b3CopyHullBytes`; freed with
    /// `b3DestroyHull`.
    Heap(NonNull<sys::b3HullData>),
    /// Box hull returned by value (`b3MakeBoxHull`); must NOT be passed to
    /// `b3DestroyHull`. Its internal offsets are relative to the struct, so
    /// it is fine to move the box around.
    Boxed(Box<sys::b3BoxHull>),
}

/// An owned convex hull, usable for any number of shapes.
///
/// Shape creation copies the hull into the world's (reference counted,
/// content-deduplicated) hull database, so a `Hull` may be dropped right
/// after creating shapes from it.
pub struct Hull(Storage);

// SAFETY: the hull data is immutable after construction and owned uniquely.
unsafe impl Send for Hull {}
// SAFETY: only read through shared references.
unsafe impl Sync for Hull {}

impl Hull {
    /// Axis-aligned box centered on the body origin (`b3MakeBoxHull`).
    /// Half extents are clamped to a small minimum.
    pub fn cuboid(half_extents: Vec3) -> Hull {
        // SAFETY: returns a value.
        let h = unsafe { sys::b3MakeBoxHull(half_extents.x, half_extents.y, half_extents.z) };
        Hull(Storage::Boxed(Box::new(h)))
    }

    /// Box with a local transform (`b3MakeTransformedBoxHull`).
    pub fn transformed_cuboid(half_extents: Vec3, transform: Transform) -> Hull {
        // SAFETY: returns a value; `xf` normalizes the rotation.
        let h = unsafe { sys::b3MakeTransformedBoxHull(half_extents.x, half_extents.y, half_extents.z, xf(transform)) };
        Hull(Storage::Boxed(Box::new(h)))
    }

    /// Tessellated cylinder along +Y from `y_offset` to `y_offset + height`
    /// (`b3CreateCylinder`). `sides` must be in `3..=32`.
    ///
    /// For a barrel centered on the body, use `y_offset = -height / 2`. To lay
    /// it on its side use [`crate::World::create_transformed_hull_shape`].
    pub fn cylinder(height: f32, radius: f32, y_offset: f32, sides: u32) -> Result<Hull, Error> {
        if !(height.is_finite() && height > 0.0) {
            return Err(Error::InvalidArgument("cylinder height must be > 0"));
        }
        if !(radius.is_finite() && radius > 0.0) {
            return Err(Error::InvalidArgument("cylinder radius must be > 0"));
        }
        if !(3..=32).contains(&sides) {
            return Err(Error::InvalidArgument("cylinder sides must be in 3..=32"));
        }
        // SAFETY: arguments satisfy b3CreateCylinder's asserts.
        let p = unsafe { sys::b3CreateCylinder(height, radius, y_offset, sides as i32) };
        NonNull::new(p).map(|p| Hull(Storage::Heap(p))).ok_or(Error::HullCreationFailed)
    }

    /// Quickhull over a point cloud (`b3CreateHull`). Needs at least 4
    /// non-coplanar points; `max_vertices` is clamped to `[4, 128]`.
    pub fn from_points(points: &[Vec3], max_vertices: usize) -> Result<Hull, Error> {
        if points.len() < 4 {
            return Err(Error::InvalidArgument("hull needs at least 4 points"));
        }
        if points.len() > i32::MAX as usize || points.iter().any(|p| !p.is_finite()) {
            return Err(Error::InvalidArgument("hull points must be finite"));
        }
        let pts: Vec<sys::b3Vec3> = points.iter().copied().map(v3).collect();
        let max = max_vertices.min(sys::B3_MAX_HULL_VERTICES as usize) as i32;
        // SAFETY: pts is a valid array of pts.len() points.
        let p = unsafe { sys::b3CreateHull(pts.as_ptr(), pts.len() as i32, max) };
        NonNull::new(p).map(|p| Hull(Storage::Heap(p))).ok_or(Error::HullCreationFailed)
    }

    /// `b3HullData::version` of hulls this build reads and writes (`B3_HULL_VERSION`).
    pub const VERSION: u64 = sys::B3_HULL_VERSION;

    /// Load a hull from bytes written by [`Self::as_bytes`] (possibly unaligned), copied into
    /// Box3D memory. The result is byte-identical to the hull that wrote them.
    ///
    /// Fails with [`Error::BlobVersionMismatch`] if the blob is from another Box3D hull
    /// version, and [`Error::InvalidBlob`] if it is truncated, does not match its content hash
    /// (corruption), or is inconsistent: every offset, count and topology index is checked (the
    /// checks of Box3D's `b3IsValidHull`, bounded), every float must be finite, and the mass
    /// properties must be ones the points allow.
    pub fn from_bytes(bytes: &[u8]) -> Result<Hull, Error> {
        blob::check_header::<sys::b3HullData>(bytes, Self::VERSION, |h| (h.version, h.byteCount))?;
        // SAFETY: bytes is valid for bytes.len() reads; the size was checked above.
        let p = unsafe { sys::b3CopyHullBytes(bytes.as_ptr().cast(), bytes.len() as i32) };
        // Owned from here, so a rejected copy is freed (with its checked byteCount).
        let hull = Hull(Storage::Heap(NonNull::new(p).ok_or(Error::InvalidBlob("copy failed"))?));
        let saved = hull.data().hash;
        // SAFETY: our own copy, byteCount checked; rewrites the hash, equal when intact.
        if unsafe { sys::b3UpdateHullHash(p) } != saved {
            return Err(Error::InvalidBlob("hash mismatch"));
        }
        validate_hull(hull.data(), bytes.len()).map_err(Error::InvalidBlob)?;
        Ok(hull)
    }

    /// The hull's bytes (`byteCount` of them), as Box3D lays them out: self-relative and
    /// pointer-free, ready for [`Self::from_bytes`].
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: a hull owns byteCount contiguous bytes from its header (b3BoxHull's byteCount
        // is its sizeof); immutable while self lives.
        unsafe { core::slice::from_raw_parts(self.as_raw() as *const u8, self.data().byteCount as usize) }
    }

    /// Pointer to the C hull. Valid while `self` lives.
    pub fn as_raw(&self) -> *const sys::b3HullData {
        match &self.0 {
            Storage::Heap(p) => p.as_ptr(),
            Storage::Boxed(b) => &b.base,
        }
    }

    fn data(&self) -> &sys::b3HullData {
        // SAFETY: as_raw is valid and non-null while self lives.
        unsafe { &*self.as_raw() }
    }

    pub fn vertex_count(&self) -> usize {
        self.data().vertexCount.max(0) as usize
    }

    pub fn face_count(&self) -> usize {
        self.data().faceCount.max(0) as usize
    }

    pub fn volume(&self) -> f32 {
        self.data().volume
    }

    /// Local centroid.
    pub fn center(&self) -> Vec3 {
        from_v3(self.data().center)
    }

    /// Radius of the largest sphere at the center (rolling radius uses 1/4 of this).
    pub fn inner_radius(&self) -> f32 {
        self.data().innerRadius
    }

    /// Call `f(a, b)` once per edge with hull-local endpoints (wireframes).
    pub fn for_each_edge(&self, f: impl FnMut(Vec3, Vec3)) {
        for_each_hull_edge(self.data(), f);
    }
}

/// Each edge of `data` once: a half-edge pair is reported from the lower index.
pub(crate) fn for_each_hull_edge(data: &sys::b3HullData, mut f: impl FnMut(Vec3, Vec3)) {
    if data.pointOffset == 0 || data.edgeOffset == 0 || data.vertexCount <= 0 || data.edgeCount <= 0 {
        return;
    }
    let base = data as *const sys::b3HullData as *const u8;
    // SAFETY: like b3GetHullPoints/b3GetHullEdges: the offsets are byte offsets
    // from the struct into its own allocation, with one point per vertex.
    let (points, edges) = unsafe {
        (
            core::slice::from_raw_parts(
                base.add(data.pointOffset as usize) as *const sys::b3Vec3,
                data.vertexCount as usize,
            ),
            core::slice::from_raw_parts(
                base.add(data.edgeOffset as usize) as *const sys::b3HullHalfEdge,
                data.edgeCount as usize,
            ),
        )
    };
    for (i, e) in edges.iter().enumerate() {
        let twin = e.twin as usize;
        if twin < i || twin >= edges.len() {
            continue;
        }
        let (a, b) = (e.origin as usize, edges[twin].origin as usize);
        if a < points.len() && b < points.len() {
            f(from_v3(points[a]), from_v3(points[b]));
        }
    }
}

/// `b3IsValidHull` with bounds: every array inside the `len`-byte blob, every index in range,
/// face loops closed within the edge count, finite floats, SOA copies equal to the points and
/// normals, mass properties the points allow. `data` heads a `len`-byte, 16-byte aligned
/// allocation.
fn validate_hull(data: &sys::b3HullData, len: usize) -> Result<(), &'static str> {
    let (v, e, f) = (data.vertexCount, data.edgeCount, data.faceCount);
    if !(4..=sys::B3_MAX_HULL_VERTICES as i32).contains(&v)
        || !(4..=sys::B3_MAX_HULL_FACES as i32).contains(&f)
        || !(6..=2 * sys::B3_MAX_HULL_EDGES as i32).contains(&e)
        || e % 2 != 0
    {
        return Err("hull counts out of range");
    }
    if v - e / 2 + f != 2 {
        return Err("hull counts break Euler's formula");
    }
    let (v, e, f) = (v as usize, e as usize, f as usize);
    let (soa_v, soa_f) = ((v + 3) & !3, (f + 3) & !3);
    let arrays = (
        blob::array::<_, sys::b3HullVertex>(data, len, data.vertexOffset, v),
        blob::array::<_, sys::b3Vec3>(data, len, data.pointOffset, v),
        blob::array::<_, sys::b3HullHalfEdge>(data, len, data.edgeOffset, e),
        blob::array::<_, sys::b3Plane>(data, len, data.planeOffset, f),
        blob::array::<_, sys::b3HullFace>(data, len, data.faceOffset, f),
        blob::array::<_, f32>(data, len, data.soaVertexOffset, 3 * soa_v),
        blob::array::<_, f32>(data, len, data.soaNormalOffset, 3 * soa_f),
    );
    let (Some(vertices), Some(points), Some(edges), Some(planes), Some(faces), Some(soa_vertices), Some(soa_normals)) =
        arrays
    else {
        return Err("hull array outside the blob");
    };

    let m = &data.centralInertia;
    if !(blob::sane_aabb(&data.aabb)
        && blob::finite(&data.center)
        && blob::finite(&m.cx)
        && blob::finite(&m.cy)
        && blob::finite(&m.cz)
        && data.volume.is_finite()
        && data.surfaceArea.is_finite()
        && data.innerRadius.is_finite())
    {
        return Err("hull property not finite");
    }
    if !(data.volume > 0.0 && data.surfaceArea > 0.0 && data.innerRadius > 0.0) {
        return Err("hull volume, area or inner radius not positive");
    }
    if !points.iter().all(blob::finite) || !soa_vertices.iter().chain(soa_normals).all(|x| x.is_finite()) {
        return Err("hull point not finite");
    }

    // Mass properties within what the points allow (bodies take them as given; Box3D asserts
    // on what follows from bad ones). Bounds in f64 with 1% slack for the builders' rounding.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in points {
        for (k, x) in [p.x, p.y, p.z].into_iter().enumerate() {
            lo[k] = lo[k].min(x as f64);
            hi[k] = hi[k].max(x as f64);
        }
    }
    let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let slack = 0.01 * ext[0].max(ext[1]).max(ext[2]);
    let center = [data.center.x as f64, data.center.y as f64, data.center.z as f64];
    if (0..3).any(|k| center[k] < lo[k] - slack || center[k] > hi[k] + slack) {
        return Err("hull center outside its points");
    }
    let volume = data.volume as f64;
    if volume > 1.01 * ext[0] * ext[1] * ext[2] || data.innerRadius as f64 > 0.505 * ext[0].min(ext[1]).min(ext[2]) {
        return Err("hull volume or inner radius exceeds its points");
    }
    // Central inertia at unit density: positive semi-definite (Box3D asserts a body's det >= 0;
    // a hull alone on a body passes its own on), and each diagonal term at most the volume
    // times the largest squared distance from the axis.
    let [xx, xy, xz] = [m.cx.x, m.cy.x, m.cz.x].map(f64::from);
    let [yx, yy, yz] = [m.cx.y, m.cy.y, m.cz.y].map(f64::from);
    let [zx, zy, zz] = [m.cx.z, m.cy.z, m.cz.z].map(f64::from);
    let det = xx * (yy * zz - yz * zy) - xy * (yx * zz - yz * zx) + xz * (yx * zy - yy * zx);
    if !(xx > 0.0 && yy > 0.0 && zz > 0.0)
        || xx * yy - xy * yx < 0.0
        || xx * zz - xz * zx < 0.0
        || yy * zz - yz * zy < 0.0
        || det < 0.0
    {
        return Err("hull inertia not positive semi-definite");
    }
    let r2 = |k: usize| (hi[k] - center[k]).max(center[k] - lo[k]).powi(2);
    let bound = |a: usize, b: usize| 1.01 * volume * (r2(a) + r2(b));
    if xx > bound(1, 2) || yy > bound(0, 2) || zz > bound(0, 1) {
        return Err("hull inertia exceeds its points");
    }

    for (i, vertex) in vertices.iter().enumerate() {
        if edges.get(vertex.edge as usize).is_none_or(|edge| edge.origin as usize != i) {
            return Err("hull vertex edge does not start at the vertex");
        }
    }
    for (i, edge) in edges.iter().enumerate() {
        if edge.next as usize >= e || edge.origin as usize >= v || edge.face as usize >= f {
            return Err("hull edge index out of range");
        }
        // Half-edges come in pairs: 2k and 2k + 1 are twins.
        if edge.twin as usize != i ^ 1 {
            return Err("hull edge twin mismatch");
        }
    }
    let c = &data.center;
    for (i, (face, plane)) in faces.iter().zip(planes).enumerate() {
        let n = &plane.normal;
        if !(blob::finite(n) && plane.offset.is_finite()) {
            return Err("hull plane not finite");
        }
        if n.x * c.x + n.y * c.y + n.z * c.z - plane.offset >= 0.0 {
            return Err("hull center not behind a face");
        }
        let base = face.edge as usize;
        if base >= e {
            return Err("hull face edge out of range");
        }
        // Every edge index was checked above, so only the loop length needs bounding.
        let mut index = base;
        let mut steps = 0;
        loop {
            let edge = &edges[index];
            let next = &edges[edge.next as usize];
            let twin = &edges[edge.twin as usize];
            if edge.face as usize != i || next.origin != twin.origin {
                return Err("hull face loop inconsistent");
            }
            index = edge.next as usize;
            steps += 1;
            if index == base {
                break;
            }
            if steps == e {
                return Err("hull face loop does not close");
            }
        }
    }

    let (vx, vy, vz) = (&soa_vertices[..soa_v], &soa_vertices[soa_v..2 * soa_v], &soa_vertices[2 * soa_v..]);
    for (i, p) in points.iter().enumerate() {
        if vx[i].to_bits() != p.x.to_bits() || vy[i].to_bits() != p.y.to_bits() || vz[i].to_bits() != p.z.to_bits() {
            return Err("hull SOA vertices differ from the points");
        }
    }
    let (nx, ny, nz) = (&soa_normals[..soa_f], &soa_normals[soa_f..2 * soa_f], &soa_normals[2 * soa_f..]);
    for (i, plane) in planes.iter().enumerate() {
        let n = &plane.normal;
        if nx[i].to_bits() != n.x.to_bits() || ny[i].to_bits() != n.y.to_bits() || nz[i].to_bits() != n.z.to_bits() {
            return Err("hull SOA normals differ from the planes");
        }
    }
    Ok(())
}

impl Clone for Hull {
    fn clone(&self) -> Self {
        match &self.0 {
            Storage::Boxed(b) => Hull(Storage::Boxed(b.clone())),
            Storage::Heap(p) => {
                // SAFETY: valid hull; returns a fresh heap copy.
                let c = unsafe { sys::b3CloneHull(p.as_ptr()) };
                Hull(Storage::Heap(NonNull::new(c).expect("b3CloneHull failed (out of memory?)")))
            }
        }
    }
}

impl Drop for Hull {
    fn drop(&mut self) {
        if let Storage::Heap(p) = self.0 {
            // SAFETY: allocated by Box3D's hull creator, freed exactly once.
            unsafe { sys::b3DestroyHull(p.as_ptr()) }
        }
    }
}

impl core::fmt::Debug for Hull {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Hull")
            .field("vertices", &self.vertex_count())
            .field("faces", &self.face_count())
            .field("volume", &self.volume())
            .finish()
    }
}
