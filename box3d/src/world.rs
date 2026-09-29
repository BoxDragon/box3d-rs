use crate::id::{BodyId, JointId, ShapeId, WorldId};
use crate::math::{from_pos, from_v3, v3};
use crate::{Error, Pos, Vec3, sys};
use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Box3D's world table is a plain global array; creating/destroying worlds
/// on several threads at once would race. Serialize those operations (and
/// allocator installation).
static WORLD_TABLE_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn lock_world_table() -> std::sync::MutexGuard<'static, ()> {
    WORLD_TABLE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Allocator

/// Allocation callback: `(size, alignment) -> ptr`. Box3D always passes a
/// size rounded up to a multiple of `alignment` (16).
pub type AllocFn = unsafe extern "C" fn(size: usize, alignment: i32) -> *mut c_void;
/// Free callback: `(ptr, size)` with the same rounded size given to [`AllocFn`].
pub type FreeFn = unsafe extern "C" fn(mem: *mut c_void, size: usize);

/// Route all Box3D allocations through `alloc`/`free` (`b3SetAllocator`).
///
/// Fails with [`Error::AllocatorBusy`] if Box3D currently holds any memory
/// (live worlds or hulls), since those blocks would later be released through
/// the wrong allocator.
///
/// # Safety
/// `alloc` must return a non-null pointer to at least `size` writable bytes
/// aligned to `alignment`, or abort; `free` must accept exactly the pointers
/// produced by `alloc`. Both must be thread-safe (Box3D may allocate on
/// worker threads when `worker_count > 1`) and must not unwind.
pub unsafe fn set_allocator(alloc: AllocFn, free: FreeFn) -> Result<(), Error> {
    let _guard = lock_world_table();
    if bytes_allocated() != 0 {
        return Err(Error::AllocatorBusy);
    }
    // SAFETY: nothing is allocated, so no block can cross allocators; the
    // caller guarantees the functions' contract.
    unsafe { sys::b3SetAllocator(Some(alloc), Some(free)) };
    Ok(())
}

/// Bytes currently allocated by Box3D (`b3GetByteCount`), across all worlds.
pub fn bytes_allocated() -> i64 {
    // SAFETY: atomic read.
    unsafe { sys::b3GetByteCount() }
}

// ---------------------------------------------------------------------------
// WorldDef

/// Pre-sizing hints to avoid run-time allocation (`b3Capacity`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capacity {
    pub static_shapes: i32,
    pub dynamic_shapes: i32,
    pub static_bodies: i32,
    pub dynamic_bodies: i32,
    pub contacts: i32,
}

/// World creation parameters: a value wrapper over `b3WorldDef`.
/// Defaults come from `b3DefaultWorldDef`, except `worker_count = 1`.
#[derive(Clone, Copy, Debug)]
pub struct WorldDef {
    raw: sys::b3WorldDef,
    destroy_queue_capacity: usize,
}

// SAFETY: the only pointers in b3WorldDef (task/debug callbacks, user data)
// are never set through this API and stay null.
unsafe impl Send for WorldDef {}
// SAFETY: as above; the value is plain data.
unsafe impl Sync for WorldDef {}

impl Default for WorldDef {
    fn default() -> Self {
        // SAFETY: returns a value with the validity cookie set.
        let mut raw = unsafe { sys::b3DefaultWorldDef() };
        raw.workerCount = 1;
        Self { raw, destroy_queue_capacity: 256 }
    }
}

impl WorldDef {
    /// Gravity, m/s^2 (default `(0, -10, 0)`; Box3D has no fixed up axis).
    pub fn gravity(mut self, gravity: Vec3) -> Self {
        self.raw.gravity = v3(gravity);
        self
    }
    pub fn enable_sleep(mut self, enabled: bool) -> Self {
        self.raw.enableSleep = enabled;
        self
    }
    pub fn enable_continuous(mut self, enabled: bool) -> Self {
        self.raw.enableContinuous = enabled;
        self
    }
    /// Solver workers, clamped to `[1, B3_MAX_WORKERS]`. Above 1 Box3D spawns
    /// its own internal thread pool (and pre-/filter callbacks run on it).
    pub fn worker_count(mut self, count: u32) -> Self {
        self.raw.workerCount = count.clamp(1, sys::B3_MAX_WORKERS);
        self
    }
    /// Initial container sizes, to avoid growth allocations after creation.
    pub fn capacity(mut self, c: Capacity) -> Self {
        self.raw.capacity = sys::b3Capacity {
            staticShapeCount: c.static_shapes,
            dynamicShapeCount: c.dynamic_shapes,
            staticBodyCount: c.static_bodies,
            dynamicBodyCount: c.dynamic_bodies,
            contactCount: c.contacts,
        };
        self
    }
    /// Collisions faster than this (m/s) bounce.
    pub fn restitution_threshold(mut self, speed: f32) -> Self {
        self.raw.restitutionThreshold = speed;
        self
    }
    /// Approach speed (m/s) needed to emit a hit event on hit-enabled shapes.
    pub fn hit_event_threshold(mut self, speed: f32) -> Self {
        self.raw.hitEventThreshold = speed;
        self
    }
    /// Contact stiffness (Hz), damping ratio, and max push-out speed (m/s).
    pub fn contact_tuning(mut self, hertz: f32, damping_ratio: f32, push_out_speed: f32) -> Self {
        self.raw.contactHertz = hertz;
        self.raw.contactDampingRatio = damping_ratio;
        self.raw.contactSpeed = push_out_speed;
        self
    }
    pub fn maximum_linear_speed(mut self, speed: f32) -> Self {
        self.raw.maximumLinearSpeed = speed;
        self
    }
    /// Pre-sized capacity of the deferred destroy queue (default 256).
    pub fn destroy_queue_capacity(mut self, capacity: usize) -> Self {
        self.destroy_queue_capacity = capacity;
        self
    }
    /// Read access to the C definition.
    pub fn as_raw(&self) -> &sys::b3WorldDef {
        &self.raw
    }
}

// ---------------------------------------------------------------------------
// Callbacks

type CustomFilterFn = dyn Fn(ShapeId, ShapeId) -> bool + Send + Sync;
type PreSolveFn = dyn Fn(ShapeId, ShapeId, Pos, Vec3) -> bool + Send + Sync;

#[derive(Default)]
struct Callbacks {
    custom_filter: Option<Box<CustomFilterFn>>,
    pre_solve: Option<Box<PreSolveFn>>,
    panicked: AtomicBool,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Callbacks {
    fn store_panic(&self, payload: Box<dyn Any + Send>) {
        let mut slot = self.panic.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(payload);
        }
        self.panicked.store(true, Ordering::Release);
    }
}

unsafe extern "C" fn custom_filter_tramp(a: sys::b3ShapeId, b: sys::b3ShapeId, ctx: *mut c_void) -> bool {
    // SAFETY: ctx is the World's boxed Callbacks, alive for the whole step.
    let cb = unsafe { &*(ctx as *const Callbacks) };
    let Some(f) = cb.custom_filter.as_deref() else { return true };
    match catch_unwind(AssertUnwindSafe(|| f(ShapeId(a), ShapeId(b)))) {
        Ok(r) => r,
        Err(p) => {
            cb.store_panic(p);
            true
        }
    }
}

unsafe extern "C" fn pre_solve_tramp(
    a: sys::b3ShapeId,
    b: sys::b3ShapeId,
    point: sys::b3Pos,
    normal: sys::b3Vec3,
    ctx: *mut c_void,
) -> bool {
    // SAFETY: ctx is the World's boxed Callbacks, alive for the whole step.
    let cb = unsafe { &*(ctx as *const Callbacks) };
    let Some(f) = cb.pre_solve.as_deref() else { return true };
    match catch_unwind(AssertUnwindSafe(|| f(ShapeId(a), ShapeId(b), from_pos(point), from_v3(normal)))) {
        Ok(r) => r,
        Err(p) => {
            cb.store_panic(p);
            true
        }
    }
}

/// Geometry Box3D references by raw pointer from a shape, kept alive by the
/// world until the shape is gone.
pub(crate) enum Resource {
    Mesh(crate::Mesh),
    HeightField(crate::HeightField),
}

#[derive(Clone, Copy, Debug)]
enum Destroy {
    Body(BodyId),
    Shape(ShapeId, bool),
    Joint(JointId, bool),
}

// ---------------------------------------------------------------------------
// World

/// A Box3D simulation world. Owns every body, shape and joint in it; dropping
/// the world destroys them all (ids become permanently invalid).
///
/// `Send` but not `Sync`.
pub struct World {
    pub(crate) id: sys::b3WorldId,
    callbacks: Box<Callbacks>,
    destroy_queue: Vec<Destroy>,
    /// One entry per mesh / height-field shape (see [`Resource`]). Pruned when
    /// shapes or bodies are destroyed, never during `step`.
    resources: Vec<(ShapeId, Resource)>,
    _not_sync: PhantomData<Cell<()>>,
}

impl World {
    /// Create a world. Fails if 128 worlds already exist.
    pub fn new(def: &WorldDef) -> Result<World, Error> {
        let raw = def.raw;
        let callbacks = Box::<Callbacks>::default();
        let destroy_queue = Vec::with_capacity(def.destroy_queue_capacity);
        let _guard = lock_world_table();
        // SAFETY: plain getter.
        if unsafe { sys::b3GetWorldCount() } >= sys::B3_MAX_WORLDS as i32 {
            return Err(Error::TooManyWorlds);
        }
        // SAFETY: `raw` comes from b3DefaultWorldDef (valid cookie) with no
        // callbacks set; the world table is protected by WORLD_TABLE_LOCK.
        let id = unsafe { sys::b3CreateWorld(&raw) };
        if id.index1 == 0 {
            return Err(Error::TooManyWorlds);
        }
        Ok(World { id, callbacks, destroy_queue, resources: Vec::new(), _not_sync: PhantomData })
    }

    pub fn id(&self) -> WorldId {
        WorldId(self.id)
    }

    /// The raw C world id. Valid for as long as `self` lives.
    pub fn raw(&self) -> sys::b3WorldId {
        self.id
    }

    #[inline]
    pub(crate) fn index0(&self) -> u16 {
        self.id.index1 - 1
    }

    /// Flush the destroy queue, then advance the simulation by `time_step`
    /// (keep it fixed, e.g. 1/60) with `sub_steps` (usually 4, min 1).
    ///
    /// Takes `&mut self`, so event slices borrowed from the previous step
    /// cannot outlive it:
    ///
    /// ```compile_fail,E0502
    /// use box3d::*;
    /// let mut world = World::new(&WorldDef::default()).unwrap();
    /// let events = world.body_events();
    /// world.step(1.0 / 60.0, 4); // error: `world` is borrowed by `events`
    /// println!("{}", events.moves().len());
    /// ```
    ///
    /// # Panics
    /// Re-raises a panic from a pre-solve or custom-filter callback (caught
    /// on the C side of the boundary, resumed here after the step).
    pub fn step(&mut self, time_step: f32, sub_steps: i32) {
        self.flush_destroys();
        // SAFETY: world id valid for self's lifetime; &mut self guarantees no
        // outstanding borrows of event buffers and no concurrent access.
        unsafe { sys::b3World_Step(self.id, time_step, sub_steps.max(1)) }
        if self.callbacks.panicked.load(Ordering::Acquire) {
            self.callbacks.panicked.store(false, Ordering::Relaxed);
            let payload = self.callbacks.panic.lock().unwrap_or_else(|e| e.into_inner()).take();
            if let Some(p) = payload {
                resume_unwind(p);
            }
        }
    }

    // ---- deferred destruction --------------------------------------------

    /// Queue a body for destruction at the next [`Self::flush_destroys`]
    /// (called automatically at the start of [`Self::step`]). The queue is
    /// pre-sized ([`WorldDef::destroy_queue_capacity`]) and reused; it only
    /// allocates if you exceed that capacity.
    pub fn queue_destroy_body(&mut self, id: BodyId) {
        self.destroy_queue.push(Destroy::Body(id));
    }

    pub fn queue_destroy_shape(&mut self, id: ShapeId, update_body_mass: bool) {
        self.destroy_queue.push(Destroy::Shape(id, update_body_mass));
    }

    pub fn queue_destroy_joint(&mut self, id: impl Into<JointId>, wake_attached: bool) {
        self.destroy_queue.push(Destroy::Joint(id.into(), wake_attached));
    }

    /// Destroy everything queued. Ids that are already invalid (e.g. a shape
    /// whose body was destroyed earlier in the queue) are skipped silently.
    pub fn flush_destroys(&mut self) {
        if self.destroy_queue.is_empty() {
            return;
        }
        let mut queue = core::mem::take(&mut self.destroy_queue);
        for cmd in queue.drain(..) {
            match cmd {
                Destroy::Body(b) if self.is_body_valid(b) => {
                    // SAFETY: validated id.
                    unsafe { sys::b3DestroyBody(b.0) }
                }
                Destroy::Shape(s, m) if self.is_shape_valid(s) => {
                    // SAFETY: validated id.
                    unsafe { sys::b3DestroyShape(s.0, m) }
                }
                Destroy::Joint(j, w) if self.is_joint_valid(j) => {
                    // SAFETY: validated id.
                    unsafe { sys::b3DestroyJoint(j.0, w) }
                }
                _ => {}
            }
        }
        self.destroy_queue = queue;
        self.prune_resources();
    }

    // ---- mesh / height-field keepalive -----------------------------------

    /// Keep `resource` alive for as long as `shape` exists.
    pub(crate) fn retain_resource(&mut self, shape: ShapeId, resource: Resource) {
        self.resources.push((shape, resource));
    }

    /// Release meshes / height fields whose shape no longer exists. Must run
    /// only *after* Box3D destroyed the shapes. Never allocates.
    pub(crate) fn prune_resources(&mut self) {
        if self.resources.is_empty() {
            return;
        }
        let world0 = self.index0();
        // SAFETY: b3Shape_IsValid accepts any id; see is_body_valid.
        self.resources.retain(|(s, _)| s.world0() == world0 && unsafe { sys::b3Shape_IsValid(s.0) });
    }

    /// True if `body` carries a height-field shape created through this API.
    pub(crate) fn body_has_height_field(&self, body: BodyId) -> bool {
        self.resources.iter().any(|(s, r)| {
            matches!(r, Resource::HeightField(_))
                && self.is_shape_valid(*s)
                // SAFETY: validated shape id.
                && BodyId(unsafe { sys::b3Shape_GetBody(s.0) }) == body
        })
    }

    // ---- settings --------------------------------------------------------

    pub fn gravity(&self) -> Vec3 {
        // SAFETY: valid world.
        from_v3(unsafe { sys::b3World_GetGravity(self.id) })
    }

    pub fn set_gravity(&mut self, gravity: Vec3) {
        // SAFETY: valid world.
        unsafe { sys::b3World_SetGravity(self.id, v3(gravity)) }
    }

    /// Enable/disable sleeping for the whole world.
    pub fn enable_sleeping(&mut self, enabled: bool) {
        // SAFETY: valid world.
        unsafe { sys::b3World_EnableSleeping(self.id, enabled) }
    }

    pub fn is_sleeping_enabled(&self) -> bool {
        // SAFETY: valid world.
        unsafe { sys::b3World_IsSleepingEnabled(self.id) }
    }

    pub fn enable_continuous(&mut self, enabled: bool) {
        // SAFETY: valid world.
        unsafe { sys::b3World_EnableContinuous(self.id, enabled) }
    }

    /// Set the number of solver workers, clamped to `[1, B3_MAX_WORKERS]`.
    pub fn set_worker_count(&mut self, count: u32) {
        let count = count.clamp(1, sys::B3_MAX_WORKERS) as i32;
        // SAFETY: valid world, count in range.
        unsafe { sys::b3World_SetWorkerCount(self.id, count) }
    }

    pub fn worker_count(&self) -> u32 {
        // SAFETY: valid world.
        unsafe { sys::b3World_GetWorkerCount(self.id) as u32 }
    }

    pub fn awake_body_count(&self) -> usize {
        // SAFETY: valid world.
        unsafe { sys::b3World_GetAwakeBodyCount(self.id).max(0) as usize }
    }

    pub fn set_hit_event_threshold(&mut self, speed: f32) {
        // SAFETY: valid world.
        unsafe { sys::b3World_SetHitEventThreshold(self.id, speed) }
    }

    // ---- callbacks -------------------------------------------------------

    /// Register a custom contact filter, called for pairs where at least one
    /// shape has [`ShapeDef::enable_custom_filtering`](crate::ShapeDef::enable_custom_filtering).
    /// Return `false` to disable collision.
    ///
    /// **Runs during [`Self::step`], possibly on Box3D worker threads**, hence
    /// `Send + Sync`. It cannot access the world. A panic inside is caught at
    /// the FFI boundary (the pair collides) and re-raised after the step.
    pub fn set_custom_filter(&mut self, f: impl Fn(ShapeId, ShapeId) -> bool + Send + Sync + 'static) {
        self.callbacks.custom_filter = Some(Box::new(f));
        let ctx = &*self.callbacks as *const Callbacks as *mut c_void;
        // SAFETY: ctx outlives the world registration (boxed, owned by self).
        unsafe { sys::b3World_SetCustomFilterCallback(self.id, Some(custom_filter_tramp), ctx) }
    }

    pub fn clear_custom_filter(&mut self) {
        // SAFETY: valid world; unregister before dropping the closure.
        unsafe { sys::b3World_SetCustomFilterCallback(self.id, None, core::ptr::null_mut()) }
        self.callbacks.custom_filter = None;
    }

    /// Register a pre-solve callback `(shape_a, shape_b, point, normal) -> keep`,
    /// called for contacts on shapes with
    /// [`ShapeDef::enable_pre_solve_events`](crate::ShapeDef::enable_pre_solve_events).
    /// Return `false` to disable the contact for this step (one-way platforms).
    ///
    /// **Runs during [`Self::step`], possibly on Box3D worker threads.** Same
    /// panic handling as [`Self::set_custom_filter`].
    pub fn set_pre_solve(&mut self, f: impl Fn(ShapeId, ShapeId, Pos, Vec3) -> bool + Send + Sync + 'static) {
        self.callbacks.pre_solve = Some(Box::new(f));
        let ctx = &*self.callbacks as *const Callbacks as *mut c_void;
        // SAFETY: ctx outlives the world registration (boxed, owned by self).
        unsafe { sys::b3World_SetPreSolveCallback(self.id, Some(pre_solve_tramp), ctx) }
    }

    pub fn clear_pre_solve(&mut self) {
        // SAFETY: valid world; unregister before dropping the closure.
        unsafe { sys::b3World_SetPreSolveCallback(self.id, None, core::ptr::null_mut()) }
        self.callbacks.pre_solve = None;
    }

    // ---- id validation -------------------------------------------------

    /// True if `id` refers to a live body in this world.
    pub fn is_body_valid(&self, id: BodyId) -> bool {
        // SAFETY: b3Body_IsValid accepts any id; we only query our own world
        // slot, which cannot be concurrently created/destroyed.
        id.world0() == self.index0() && unsafe { sys::b3Body_IsValid(id.0) }
    }

    /// True if `id` refers to a live shape in this world.
    pub fn is_shape_valid(&self, id: ShapeId) -> bool {
        // SAFETY: see is_body_valid.
        id.world0() == self.index0() && unsafe { sys::b3Shape_IsValid(id.0) }
    }

    /// True if `id` refers to a live joint in this world.
    pub fn is_joint_valid(&self, id: JointId) -> bool {
        // SAFETY: see is_body_valid.
        id.world0() == self.index0() && unsafe { sys::b3Joint_IsValid(id.0) }
    }

    #[inline]
    #[track_caller]
    pub(crate) fn body(&self, id: BodyId) -> sys::b3BodyId {
        assert!(self.is_body_valid(id), "invalid or foreign {id:?}");
        id.0
    }

    #[inline]
    #[track_caller]
    pub(crate) fn shape(&self, id: ShapeId) -> sys::b3ShapeId {
        assert!(self.is_shape_valid(id), "invalid or foreign {id:?}");
        id.0
    }

    #[inline]
    #[track_caller]
    pub(crate) fn joint(&self, id: JointId) -> sys::b3JointId {
        assert!(self.is_joint_valid(id), "invalid or foreign {id:?}");
        id.0
    }
}

impl Drop for World {
    fn drop(&mut self) {
        let _guard = lock_world_table();
        // SAFETY: we own this world id; destroyed exactly once. Destroying the
        // world frees all bodies/shapes/joints; callbacks are dropped after.
        unsafe { sys::b3DestroyWorld(self.id) }
    }
}

impl core::fmt::Debug for World {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("World").field("id", &self.id()).finish()
    }
}

/// Mesh / height-field lookup.
impl World {
    /// The mesh of a mesh shape created with
    /// [`World::create_mesh_shape`](crate::World::create_mesh_shape) (a new
    /// handle to the same data). `None` for other shapes.
    ///
    /// # Panics
    /// If the id is stale or foreign.
    #[track_caller]
    pub fn shape_mesh(&self, id: ShapeId) -> Option<crate::Mesh> {
        self.shape(id);
        self.resources.iter().find_map(|(s, r)| match r {
            Resource::Mesh(m) if *s == id => Some(m.clone()),
            _ => None,
        })
    }

    /// The height field of a shape created with
    /// [`World::create_height_field_shape`](crate::World::create_height_field_shape).
    /// `None` for other shapes.
    ///
    /// # Panics
    /// If the id is stale or foreign.
    #[track_caller]
    pub fn shape_height_field(&self, id: ShapeId) -> Option<crate::HeightField> {
        self.shape(id);
        self.resources.iter().find_map(|(s, r)| match r {
            Resource::HeightField(h) if *s == id => Some(h.clone()),
            _ => None,
        })
    }
}
