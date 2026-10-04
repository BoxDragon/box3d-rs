//! Triangle meshes (`b3MeshData`) and mesh shapes.
//!
//! Box3D does **not** copy a mesh into the shape: `b3CreateMeshShape` keeps a
//! raw pointer for the lifetime of the shape. [`Mesh`] is therefore a cheap,
//! clonable, reference-counted handle, and the [`World`] retains one clone per
//! mesh shape until that shape, its body, or the world is destroyed.

use crate::id::{BodyId, ShapeId};
use crate::math::{from_v3, v3};
use crate::shape::SurfaceMaterial;
use crate::world::Resource;
use crate::{Error, ShapeDef, ShapeType, Vec3, World, blob, sys};
use core::ptr::NonNull;
use std::sync::Arc;

/// Mesh creation parameters (`b3MeshDef`). Borrows the input arrays; nothing
/// is retained after [`Mesh::new`] returns.
///
/// Triangles are counter-clockwise (the front face, the one that collides,
/// is the one seen with CCW vertices) unless [`Self::clockwise`] is set.
#[derive(Clone, Copy, Debug)]
pub struct MeshDef<'a> {
    vertices: &'a [Vec3],
    /// Flat vertex indices, 3 per triangle. Validated in [`Mesh::new`].
    indices: IndexSlice<'a>,
    material_indices: Option<&'a [u8]>,
    weld_tolerance: Option<f32>,
    median_split: bool,
    identify_edges: bool,
    clockwise: bool,
}

#[derive(Clone, Copy, Debug)]
enum IndexSlice<'a> {
    Triangles(&'a [[u32; 3]]),
    Flat(&'a [i32]),
}

impl<'a> MeshDef<'a> {
    /// A mesh from vertices and one `[u32; 3]` per triangle.
    pub fn new(vertices: &'a [Vec3], triangles: &'a [[u32; 3]]) -> Self {
        Self::with_indices(vertices, IndexSlice::Triangles(triangles))
    }

    /// A mesh from vertices and a flat index list (3 per triangle, as Box3D
    /// takes it). `indices.len()` must be a multiple of 3.
    pub fn from_flat_indices(vertices: &'a [Vec3], indices: &'a [i32]) -> Self {
        Self::with_indices(vertices, IndexSlice::Flat(indices))
    }

    fn with_indices(vertices: &'a [Vec3], indices: IndexSlice<'a>) -> Self {
        Self {
            vertices,
            indices,
            material_indices: None,
            weld_tolerance: None,
            median_split: false,
            identify_edges: false,
            clockwise: false,
        }
    }

    /// One material index per triangle, indexing into the shape's
    /// [`ShapeDef::mesh_materials`] (out-of-range indices are clamped by
    /// Box3D to the last material). Length must equal the triangle count.
    pub fn material_indices(mut self, indices: &'a [u8]) -> Self {
        self.material_indices = Some(indices);
        self
    }

    /// Weld vertices closer than `tolerance` (length units) before building.
    pub fn weld_vertices(mut self, tolerance: f32) -> Self {
        self.weld_tolerance = Some(tolerance);
        self
    }

    /// Median split instead of SAH: faster build, good for grid-like meshes.
    pub fn median_split(mut self, enabled: bool) -> Self {
        self.median_split = enabled;
        self
    }

    /// Compute triangle adjacency from shared edges, which removes internal
    /// edge bumps (ghost collisions) when things slide across the mesh.
    /// Recommended for ground meshes.
    pub fn identify_edges(mut self, enabled: bool) -> Self {
        self.identify_edges = enabled;
        self
    }

    /// Input triangles are clockwise.
    pub fn clockwise(mut self, enabled: bool) -> Self {
        self.clockwise = enabled;
        self
    }

    fn triangle_count(&self) -> usize {
        match self.indices {
            IndexSlice::Triangles(t) => t.len(),
            IndexSlice::Flat(f) => f.len() / 3,
        }
    }

    /// Validate everything Box3D only asserts on and build the C def.
    fn to_raw(self) -> Result<sys::b3MeshDef, Error> {
        let vcount = self.vertices.len();
        if vcount < 3 {
            return Err(Error::InvalidArgument("mesh needs at least 3 vertices"));
        }
        if vcount > i32::MAX as usize {
            return Err(Error::InvalidArgument("too many mesh vertices"));
        }
        if self.vertices.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidArgument("mesh vertices must be finite"));
        }
        let indices: &[i32] = match self.indices {
            IndexSlice::Triangles(t) => {
                if t.iter().flatten().any(|&i| i as usize >= vcount) {
                    return Err(Error::InvalidArgument("mesh vertex index out of range"));
                }
                // SAFETY: [[u32; 3]] is 3 * len contiguous u32; every value is
                // < vcount <= i32::MAX so reading it as i32 keeps the value.
                unsafe { core::slice::from_raw_parts(t.as_ptr() as *const i32, t.len() * 3) }
            }
            IndexSlice::Flat(f) => {
                if f.len() % 3 != 0 {
                    return Err(Error::InvalidArgument("mesh index count must be a multiple of 3"));
                }
                if f.iter().any(|&i| i < 0 || i as usize >= vcount) {
                    return Err(Error::InvalidArgument("mesh vertex index out of range"));
                }
                f
            }
        };
        let tcount = indices.len() / 3;
        if tcount == 0 {
            return Err(Error::InvalidArgument("mesh needs at least 1 triangle"));
        }
        if tcount > i32::MAX as usize {
            return Err(Error::InvalidArgument("too many mesh triangles"));
        }
        if let Some(m) = self.material_indices
            && m.len() != tcount
        {
            return Err(Error::InvalidArgument("mesh material_indices.len() must equal the triangle count"));
        }
        if let Some(t) = self.weld_tolerance
            && !(t.is_finite() && t >= 0.0)
        {
            return Err(Error::InvalidArgument("mesh weld tolerance must be >= 0"));
        }
        Ok(sys::b3MeshDef {
            // Box3D only reads through these pointers (the def is const).
            vertices: self.vertices.as_ptr() as *mut sys::b3Vec3,
            stride: 0,
            indices: indices.as_ptr() as *mut i32,
            materialIndices: self.material_indices.map_or(core::ptr::null_mut(), |m| m.as_ptr() as *mut u8),
            weldTolerance: self.weld_tolerance.unwrap_or(0.0),
            vertexCount: vcount as i32,
            triangleCount: tcount as i32,
            weldVertices: self.weld_tolerance.is_some_and(|t| t > 0.0),
            useMedianSplit: self.median_split,
            identifyEdges: self.identify_edges,
            clockWiseWinding: self.clockwise,
        })
    }
}

struct MeshData(NonNull<sys::b3MeshData>);

// SAFETY: the mesh is immutable after construction; Box3D only reads it
// (possibly from worker threads).
unsafe impl Send for MeshData {}
// SAFETY: as above.
unsafe impl Sync for MeshData {}

impl Drop for MeshData {
    fn drop(&mut self) {
        // SAFETY: created by b3CreateMesh (or a b3Create*Mesh helper, or
        // b3CopyMeshBytes with a checked byteCount), freed exactly once,
        // after every shape using it is gone (the worlds hold clones of the
        // Arc).
        unsafe { sys::b3DestroyMesh(self.0.as_ptr()) }
    }
}

/// An immutable triangle-mesh collision BVH (`b3MeshData`), shareable across
/// any number of shapes and worlds, with any per-shape scale.
///
/// `Clone` is a reference-count bump. Box3D keeps a raw pointer to the mesh
/// in every shape made from it, so [`World::create_mesh_shape`] stores a
/// clone in the world; the mesh is freed when the last user handle *and* the
/// last shape using it are gone. Dropping your `Mesh` right after creating
/// shapes is fine.
///
/// Box3D sorts triangles into BVH order, so the `triangle_index` reported by
/// ray casts and mover planes indexes [`Self::triangles`], **not** your input
/// order (degenerate triangles are dropped too).
#[derive(Clone)]
pub struct Mesh(Arc<MeshData>);

impl Mesh {
    /// Build a mesh (`b3CreateMesh`). Degenerate (near zero-area) triangles
    /// are skipped; see [`Self::degenerate_count`].
    ///
    /// Fails with [`Error::InvalidArgument`] on bad counts or out-of-range
    /// indices, and [`Error::MeshCreationFailed`] if Box3D rejects the input
    /// (e.g. every triangle is degenerate).
    pub fn new(def: &MeshDef<'_>) -> Result<Mesh, Error> {
        let raw = def.to_raw()?;
        // SAFETY: raw was validated (counts, index ranges, finite vertices)
        // and borrows slices that outlive the call; no degenerate buffer.
        let p = unsafe { sys::b3CreateMesh(&raw, core::ptr::null_mut(), 0) };
        Self::from_ptr(p)
    }

    /// Like [`Self::new`], and also appends the input indices of the
    /// degenerate triangles that were dropped (triangles that repeat a vertex
    /// index are dropped silently and not reported, as in Box3D).
    pub fn new_with_degenerates(def: &MeshDef<'_>, degenerates: &mut impl Extend<u32>) -> Result<Mesh, Error> {
        let raw = def.to_raw()?;
        let mut buf = vec![0i32; def.triangle_count()];
        // SAFETY: as in new; buf holds triangle_count ints, the maximum Box3D
        // can report.
        let p = unsafe { sys::b3CreateMesh(&raw, buf.as_mut_ptr(), buf.len() as i32) };
        let mesh = Self::from_ptr(p)?;
        let n = (mesh.degenerate_count()).min(buf.len());
        degenerates.extend(buf[..n].iter().map(|&i| i as u32));
        Ok(mesh)
    }

    /// Axis-aligned box surface (12 triangles, outward normals) centered at
    /// `center` (`b3CreateBoxMesh`).
    pub fn cuboid(center: Vec3, half_extents: Vec3, identify_edges: bool) -> Result<Mesh, Error> {
        if !(center.is_finite() && half_extents.is_finite() && half_extents.cmpgt(Vec3::ZERO).all()) {
            return Err(Error::InvalidArgument("box mesh half extents must be positive and finite"));
        }
        // SAFETY: plain values.
        let p = unsafe { sys::b3CreateBoxMesh(v3(center), v3(half_extents), identify_edges) };
        Self::from_ptr(p)
    }

    /// Flat grid of `x_count` x `z_count` square cells of `cell_width` on the
    /// y = 0 plane, centered on the origin, normals +Y (`b3CreateGridMesh`).
    pub fn grid(x_count: u32, z_count: u32, cell_width: f32, identify_edges: bool) -> Result<Mesh, Error> {
        if x_count == 0 || z_count == 0 || x_count > 4096 || z_count > 4096 {
            return Err(Error::InvalidArgument("grid mesh cell counts must be in 1..=4096"));
        }
        if !(cell_width.is_finite() && cell_width > 0.0) {
            return Err(Error::InvalidArgument("grid mesh cell width must be > 0"));
        }
        // SAFETY: arguments in range; one material.
        let p = unsafe { sys::b3CreateGridMesh(x_count as i32, z_count as i32, cell_width, 1, identify_edges) };
        Self::from_ptr(p)
    }

    /// `b3MeshData::version` of meshes this build reads and writes (`B3_MESH_VERSION`).
    pub const VERSION: u64 = sys::B3_MESH_VERSION as u64;

    /// Load a mesh from bytes written by [`Self::as_bytes`] (possibly unaligned), copied into
    /// Box3D memory. The result is byte-identical to the mesh that wrote them.
    ///
    /// Fails with [`Error::BlobVersionMismatch`] if the blob is from another Box3D mesh
    /// version, and [`Error::InvalidBlob`] if it is truncated, does not match its content hash
    /// (corruption), or is inconsistent: every offset, count and index is checked, the BVH must
    /// be a tree in Box3D's depth-first layout whose leaves cover the triangles in order and
    /// whose bounds nest (Box3D's `b3IsValidMesh` with validation on), within Box3D's query
    /// stack depth, and every float must be finite.
    pub fn from_bytes(bytes: &[u8]) -> Result<Mesh, Error> {
        blob::check_header::<sys::b3MeshData>(bytes, Self::VERSION, |h| (h.version, h.byteCount))?;
        // SAFETY: bytes is valid for bytes.len() reads; the size was checked above.
        let p = unsafe { sys::b3CopyMeshBytes(bytes.as_ptr().cast(), bytes.len() as i32) };
        // Owned from here, so a rejected copy is freed (with its checked byteCount).
        let data = MeshData(NonNull::new(p).ok_or(Error::InvalidBlob("copy failed"))?);
        // SAFETY: a live copy; read before the rehash below writes it.
        let saved = unsafe { (*p).hash };
        // SAFETY: our own copy, byteCount checked; rewrites the hash, equal when intact.
        if unsafe { sys::b3UpdateMeshHash(p) } != saved {
            return Err(Error::InvalidBlob("hash mismatch"));
        }
        // SAFETY: a live, 16-byte aligned copy of bytes.len() bytes.
        validate_mesh(unsafe { data.0.as_ref() }, bytes.len()).map_err(Error::InvalidBlob)?;
        Ok(Mesh(Arc::new(data)))
    }

    /// The mesh's bytes (`byteCount` of them), as Box3D lays them out: self-relative and
    /// pointer-free, ready for [`Self::from_bytes`].
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: a mesh owns byteCount contiguous bytes from its header; immutable while any
        // clone lives.
        unsafe { core::slice::from_raw_parts(self.as_raw() as *const u8, self.byte_count()) }
    }

    fn from_ptr(p: *mut sys::b3MeshData) -> Result<Mesh, Error> {
        NonNull::new(p).map(|p| Mesh(Arc::new(MeshData(p)))).ok_or(Error::MeshCreationFailed)
    }

    /// Pointer to the C mesh. Valid while any clone of `self` lives.
    pub fn as_raw(&self) -> *const sys::b3MeshData {
        self.0.0.as_ptr()
    }

    fn data(&self) -> &sys::b3MeshData {
        // SAFETY: non-null, immutable, alive while self lives.
        unsafe { &*self.as_raw() }
    }

    /// # Safety
    /// `offset` / `count` must describe an array of `T` inside the mesh blob.
    unsafe fn array<T>(&self, offset: i32, count: i32) -> &[T] {
        if offset == 0 || count <= 0 {
            return &[];
        }
        // SAFETY: caller guarantees the range lies inside the allocation;
        // Box3D aligns every array to 8 bytes.
        unsafe {
            core::slice::from_raw_parts((self.as_raw() as *const u8).add(offset as usize) as *const T, count as usize)
        }
    }

    /// Number of live handles: user clones plus one per mesh shape that the
    /// worlds are keeping alive.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// Welded, deduplicated vertices.
    pub fn vertices(&self) -> &[Vec3] {
        let d = self.data();
        // SAFETY: vertexOffset/vertexCount describe the b3Vec3 array; Vec3 is
        // layout-compatible with b3Vec3.
        unsafe { self.array::<Vec3>(d.vertexOffset, d.vertexCount) }
    }

    /// Triangles in Box3D's (BVH) order: this is what `triangle_index` in
    /// query results indexes. Winding is CCW regardless of the input winding.
    pub fn triangles(&self) -> &[[i32; 3]] {
        let d = self.data();
        // SAFETY: b3MeshTriangle is repr(C) { i32, i32, i32 } == [i32; 3].
        unsafe { self.array::<[i32; 3]>(d.triangleOffset, d.triangleCount) }
    }

    /// Per-triangle material indices, in [`Self::triangles`] order.
    pub fn material_indices(&self) -> &[u8] {
        let d = self.data();
        // SAFETY: one u8 per triangle.
        unsafe { self.array::<u8>(d.materialOffset, d.triangleCount) }
    }

    pub fn triangle_count(&self) -> usize {
        self.data().triangleCount.max(0) as usize
    }

    pub fn vertex_count(&self) -> usize {
        self.data().vertexCount.max(0) as usize
    }

    /// Number of degenerate triangles dropped during construction.
    pub fn degenerate_count(&self) -> usize {
        self.data().degenerateCount.max(0) as usize
    }

    /// `max(material_indices) + 1`.
    pub fn material_count(&self) -> usize {
        self.data().materialCount.max(0) as usize
    }

    /// Local bounds `(lower, upper)` before scaling.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let b = self.data().bounds;
        (from_v3(b.lowerBound), from_v3(b.upperBound))
    }

    pub fn surface_area(&self) -> f32 {
        self.data().surfaceArea
    }

    /// Size of the mesh blob (counted by [`crate::bytes_allocated`]).
    pub fn byte_count(&self) -> usize {
        self.data().byteCount.max(0) as usize
    }

    /// True if both handles refer to the same mesh data.
    pub fn ptr_eq(a: &Mesh, b: &Mesh) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }
}

/// `B3_MESH_STACK_SIZE` (mesh.c, internal): the traversal stack of mesh queries.
const MESH_STACK_SIZE: usize = 256;

/// `B3_LEAF_NODE` (mesh.c, internal): the 2-bit node tag of a leaf.
const LEAF_NODE: u32 = 3;

fn contains(b: &sys::b3MeshNode, p: &sys::b3Vec3) -> bool {
    let (l, u) = (&b.lowerBound, &b.upperBound);
    l.x <= p.x && l.y <= p.y && l.z <= p.z && p.x <= u.x && p.y <= u.y && p.z <= u.z
}

fn contains_node(a: &sys::b3MeshNode, b: &sys::b3MeshNode) -> bool {
    contains(a, &b.lowerBound) && contains(a, &b.upperBound)
}

/// `b3IsValidMesh` (with `b3IsConsistent`) made safe for untrusted bytes: every array inside
/// the `len`-byte blob, every index in range, and the BVH a depth-first tree (left child
/// follows its parent, the right child at `childOffset`, each subtree a contiguous node range)
/// that Box3D's query stack can traverse, with leaves covering the triangles in order. `data`
/// heads a `len`-byte, 16-byte aligned allocation.
fn validate_mesh(data: &sys::b3MeshData, len: usize) -> Result<(), &'static str> {
    let (nc, vc, tc) = (data.nodeCount, data.vertexCount, data.triangleCount);
    if vc < 3 || tc < 1 || nc < 1 || nc as i64 > 2 * tc as i64 - 1 {
        return Err("mesh counts out of range");
    }
    if !(1..=256).contains(&data.materialCount) || data.degenerateCount < 0 || data.treeHeight < 0 {
        return Err("mesh counts out of range");
    }
    let (nc, vc, tc) = (nc as usize, vc as usize, tc as usize);
    let arrays = (
        blob::array::<_, sys::b3MeshNode>(data, len, data.nodeOffset, nc),
        blob::array::<_, sys::b3Vec3>(data, len, data.vertexOffset, vc),
        blob::array::<_, sys::b3MeshTriangle>(data, len, data.triangleOffset, tc),
        blob::array::<_, u8>(data, len, data.materialOffset, tc),
        blob::array::<_, u8>(data, len, data.flagsOffset, tc),
    );
    let (Some(nodes), Some(vertices), Some(triangles), Some(materials), Some(_flags)) = arrays else {
        return Err("mesh array outside the blob");
    };

    if !blob::sane_aabb(&data.bounds) || !(data.surfaceArea.is_finite() && data.surfaceArea >= 0.0) {
        return Err("mesh bounds or area not finite");
    }
    if !vertices.iter().all(blob::finite) {
        return Err("mesh vertex not finite");
    }
    for t in triangles {
        let (a, b, c) = (t.index1, t.index2, t.index3);
        if [a, b, c].iter().any(|&i| i < 0 || i as usize >= vc) || a == b || a == c || b == c {
            return Err("mesh triangle index out of range or repeated");
        }
    }
    if materials.iter().any(|&m| m as i32 >= data.materialCount) {
        return Err("mesh material index out of range");
    }
    if !nodes.iter().all(|n| blob::finite(&n.lowerBound) && blob::finite(&n.upperBound)) {
        return Err("mesh node bounds not finite");
    }

    // Walk as b3SortMeshTriangles does (right child pushed first, same depth limit), carrying
    // each subtree's node range [index, end).
    let mut stack = [(0usize, 0usize); MESH_STACK_SIZE];
    stack[0] = (0, nc);
    let mut count = 1;
    let mut next_triangle = 0;
    while count > 0 {
        count -= 1;
        let (index, end) = stack[count];
        let node = &nodes[index];
        // SAFETY: both union views are the same 32-bit bitfield; any value is valid.
        let (tag, high) = unsafe { (node.data.asLeaf.type_(), node.data.asLeaf.triangleCount()) };
        if tag == LEAF_NODE {
            let (offset, n) = (node.triangleOffset as usize, high as usize);
            if end != index + 1 || offset != next_triangle || n == 0 || n > tc - offset {
                return Err("mesh BVH leaf out of order");
            }
            for t in &triangles[offset..offset + n] {
                // Indices were checked above.
                for i in [t.index1, t.index2, t.index3] {
                    if !contains(node, &vertices[i as usize]) {
                        return Err("mesh BVH leaf does not bound its triangles");
                    }
                }
            }
            next_triangle += n;
        } else {
            let right = index + high as usize;
            if !(index + 1 < right && right < end) {
                return Err("mesh BVH child out of its subtree");
            }
            if count >= MESH_STACK_SIZE - 2 {
                return Err("mesh BVH too deep");
            }
            if !contains_node(node, &nodes[index + 1]) || !contains_node(node, &nodes[right]) {
                return Err("mesh BVH node does not bound its children");
            }
            stack[count] = (right, end);
            stack[count + 1] = (index + 1, right);
            count += 2;
        }
    }
    if next_triangle != tc {
        return Err("mesh BVH leaves do not cover the triangles");
    }
    Ok(())
}

impl core::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mesh")
            .field("vertices", &self.vertex_count())
            .field("triangles", &self.triangle_count())
            .field("degenerate", &self.degenerate_count())
            .field("bytes", &self.byte_count())
            .finish()
    }
}

/// Mesh shape API.
impl World {
    /// Attach a triangle mesh with a (possibly non-uniform or negative) local
    /// `scale`. The world keeps a clone of `mesh` until the shape is
    /// destroyed (directly, with its body, or with the world).
    ///
    /// Allowed on any body type, as in Box3D, but **only put meshes on static
    /// bodies** for collision: a mesh has no mass, never collides with other
    /// meshes/height fields, and Box3D only really supports mesh contacts
    /// against static meshes ("mesh collision only creates contacts on static
    /// bodies"). Meshes on kinematic bodies still work for queries, sensors
    /// and character movers.
    ///
    /// Mesh surfaces are one-sided: rays and movers ignore back faces.
    ///
    /// # Panics
    /// If `body` is stale or foreign, or `scale` is not finite.
    #[track_caller]
    pub fn create_mesh_shape(&mut self, body: BodyId, def: &ShapeDef<'_>, mesh: &Mesh, scale: Vec3) -> ShapeId {
        let b = self.body(body);
        assert!(scale.is_finite(), "mesh scale must be finite, got {scale}");
        self.prune_resources();
        // SAFETY: validated body; the mesh pointer stays valid for the life of
        // the shape because the world retains a clone of the Arc below.
        let id = ShapeId(unsafe { sys::b3CreateMeshShape(b, &def.0, mesh.as_raw(), v3(scale)) });
        if !id.is_null() {
            self.retain_resource(id, Resource::Mesh(mesh.clone()));
        }
        id
    }

    /// Local scale of a mesh shape.
    ///
    /// # Panics
    /// If the id is stale/foreign or the shape is not a mesh.
    #[track_caller]
    pub fn shape_mesh_scale(&self, id: ShapeId) -> Vec3 {
        assert_eq!(self.shape_type(id), ShapeType::Mesh, "not a mesh shape");
        // SAFETY: validated id of a mesh shape.
        from_v3(unsafe { sys::b3Shape_GetMesh(id.0) }.scale)
    }

    /// Number of surface materials on a shape (`b3Shape_GetMeshMaterialCount`):
    /// the length of [`ShapeDef::mesh_materials`] for mesh/height-field
    /// shapes, 1 otherwise.
    #[track_caller]
    pub fn shape_mesh_material_count(&self, id: ShapeId) -> usize {
        // SAFETY: validated id.
        unsafe { sys::b3Shape_GetMeshMaterialCount(self.shape(id)).max(0) as usize }
    }

    /// Per-triangle material `index` of a mesh/height-field shape.
    ///
    /// # Panics
    /// If the id is stale/foreign or `index >= shape_mesh_material_count`.
    #[track_caller]
    pub fn shape_mesh_material(&self, id: ShapeId, index: usize) -> SurfaceMaterial {
        let n = self.shape_mesh_material_count(id);
        assert!(index < n, "mesh material index {index} out of range (count {n})");
        // SAFETY: validated id and index.
        SurfaceMaterial(unsafe { sys::b3Shape_GetMeshSurfaceMaterial(id.0, index as i32) })
    }

    /// Replace per-triangle material `index` of a mesh/height-field shape.
    ///
    /// # Panics
    /// As [`Self::shape_mesh_material`], or if friction, restitution or
    /// rolling resistance is negative or not finite.
    #[track_caller]
    pub fn set_shape_mesh_material(&mut self, id: ShapeId, index: usize, material: SurfaceMaterial) {
        let n = self.shape_mesh_material_count(id);
        assert!(index < n, "mesh material index {index} out of range (count {n})");
        assert_ne!(self.shape_type(id), ShapeType::Compound, "compound materials are baked");
        let m = &material.0;
        for (v, what) in
            [(m.friction, "friction"), (m.restitution, "restitution"), (m.rollingResistance, "rolling resistance")]
        {
            assert!(v.is_finite() && v >= 0.0, "{what} must be >= 0 and finite, got {v}");
        }
        // SAFETY: validated id and index.
        unsafe { sys::b3Shape_SetMeshMaterial(id.0, material.0, index as i32) }
    }
}
