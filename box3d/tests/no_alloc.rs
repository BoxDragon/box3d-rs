//! Steady-state stepping must not allocate: neither inside Box3D (checked via
//! `b3SetAllocator`) nor in the Rust wrapper (checked via a counting global
//! allocator).
//!
//! This binary holds a single test so that nothing else touches Box3D's
//! global allocator hooks. The Rust counter only counts allocations made on
//! the thread that has armed it, so libtest's own bookkeeping on other
//! threads cannot produce false positives; with `worker_count = 1` all Box3D
//! work happens on that thread.
// `.into()` on f32 coordinates is identity in single precision but converts to
// f64 with the `double-precision` feature.
#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]

use box3d::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ---- Rust side -------------------------------------------------------------

struct CountingAlloc;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
}
static RUST_ALLOCS: AtomicU64 = AtomicU64::new(0);

fn note_rust_alloc() {
    // try_with: the TLS slot may be gone during thread teardown.
    if ARMED.try_with(|a| a.get()).unwrap_or(false) {
        RUST_ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: forwards to System unchanged.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_rust_alloc();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_rust_alloc();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_rust_alloc();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

// ---- Box3D side ------------------------------------------------------------

static B3_ALLOCS: AtomicU64 = AtomicU64::new(0);
static B3_FREES: AtomicU64 = AtomicU64::new(0);
static B3_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Set during the measured window: every Box3D allocation records a backtrace
/// so a failure says exactly where Box3D allocated.
static B3_TRACE: AtomicBool = AtomicBool::new(false);
static B3_TRACES: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn record_b3_alloc(size: usize) {
    // Pause the Rust counter: capturing a backtrace allocates.
    let was = ARMED.try_with(|a| a.replace(false)).unwrap_or(false);
    let bt = std::backtrace::Backtrace::force_capture();
    B3_TRACES.lock().unwrap_or_else(|e| e.into_inner()).push(format!("{size} bytes at:\n{bt}"));
    let _ = ARMED.try_with(|a| a.set(was));
}

unsafe extern "C" fn b3_alloc(size: usize, alignment: i32) -> *mut c_void {
    B3_ALLOCS.fetch_add(1, Ordering::Relaxed);
    B3_ALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    if B3_TRACE.load(Ordering::Relaxed) {
        record_b3_alloc(size);
    }
    match Layout::from_size_align(size, alignment as usize) {
        // SAFETY: size > 0 (Box3D never allocates 0 bytes).
        Ok(l) => unsafe { System.alloc(l) as *mut c_void },
        Err(_) => std::process::abort(),
    }
}

unsafe extern "C" fn b3_free(mem: *mut c_void, size: usize) {
    B3_FREES.fetch_add(1, Ordering::Relaxed);
    // SAFETY: Box3D passes back the same (rounded) size; alignment is always 16.
    unsafe { System.dealloc(mem as *mut u8, Layout::from_size_align_unchecked(size, 16)) }
}

// ---- test --------------------------------------------------------------------

const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: i32 = 4;

/// 10 wide x 10 high x 3 deep wall of 1 m boxes on a ground slab, plus a
/// distance-joint pendulum. With `impact`, the (heavy) bob sweeps through
/// the top of the wall every swing and a bullet knocks the wall during
/// warm-up; otherwise the bob swings freely beside a resting wall.
fn build(impact: bool) -> World {
    let mut w = World::new(&WorldDef::default()).expect("world");

    let ground = w.create_body(&BodyDef::fixed(Pos::new(0.0, -0.5, 0.0)));
    w.create_box_shape(ground, &ShapeDef::default(), Vec3::new(50.0, 0.5, 50.0));

    let box_def = ShapeDef::default().enable_contact_events(true);
    let mut n = 0u64;
    for z in 0..3 {
        for y in 0..10 {
            for x in 0..10 {
                let p = Pos::new((x as f32 - 4.5).into(), (y as f32 + 0.5).into(), (z as f32 - 1.0).into());
                // Sleep off: all 300 boxes and their contacts stay in the
                // solver during the measured window.
                let b = w.create_body(&BodyDef::dynamic(p).enable_sleep(false).user_data(n));
                w.create_box_shape(b, &box_def, Vec3::splat(0.5));
                n += 1;
            }
        }
    }

    let (pivot, start, radius, density) = if impact {
        (Pos::new(0.0, 14.0, 0.0), Pos::new(4.0, 14.0, 0.0), 0.75, 20000.0)
    } else {
        (Pos::new(12.0, 8.0, 0.0), Pos::new(16.0, 8.0, 0.0), 0.25, 1000.0)
    };
    let pivot = w.create_body(&BodyDef::fixed(pivot));
    let bob = w.create_body(&BodyDef::dynamic(start).enable_sleep(false).user_data(u64::MAX));
    w.create_sphere_shape(bob, &box_def.density(density), Vec3::ZERO, radius);
    w.create_distance_joint(&DistanceJointDef::rod(pivot, bob, 4.0));

    if impact {
        w.create_body(
            &BodyDef::dynamic(Pos::new(0.0, 5.0, 8.0)).linear_velocity(Vec3::new(0.0, 0.0, -20.0)).bullet(true),
        );
    }
    w
}

#[derive(Debug, Default)]
struct Window {
    moves: usize,
    begins: usize,
    ends: usize,
    hits: usize,
    rust_allocs: u64,
    b3_allocs: u64,
    b3_bytes_before: i64,
    b3_bytes_after: i64,
    b3_traces: Vec<String>,
}

/// Warm up 120 steps, then run 600 measured steps of
/// step + move events + contact events + 10 closure ray casts.
fn measure(w: &mut World) -> Window {
    for _ in 0..120 {
        w.step(DT, SUBSTEPS);
    }
    let mut out = Window::default();
    let mut checksum = 0.0f64;
    RUST_ALLOCS.store(0, Ordering::Relaxed);
    B3_TRACES.lock().unwrap().clear();
    let b3_before = B3_ALLOCS.load(Ordering::Relaxed);
    out.b3_bytes_before = bytes_allocated();

    B3_TRACE.store(true, Ordering::Relaxed);
    ARMED.with(|a| a.set(true));
    for _ in 0..600 {
        w.step(DT, SUBSTEPS);

        let moves = w.body_events().moves();
        for m in moves {
            checksum += m.position().y as f64 + m.user_data() as f64 * 1e-9;
            if m.fell_asleep() {
                checksum += 1.0;
            }
        }
        out.moves += moves.len();

        let c = w.contact_events();
        out.begins += c.begin().len();
        out.ends += c.end().len();
        for e in c.begin() {
            checksum += e.shape_a().to_bits() as f64 * 1e-20;
        }

        for i in 0..10 {
            let origin = Pos::new((i as f32 - 4.5).into(), 20.0, 0.0);
            w.cast_ray(origin, Vec3::new(0.0, -40.0, 0.0), QueryFilter::default(), |hit| {
                out.hits += 1;
                checksum += hit.point.y as f64;
                hit.fraction // clip to closest
            });
        }
    }
    ARMED.with(|a| a.set(false));
    B3_TRACE.store(false, Ordering::Relaxed);

    std::hint::black_box(checksum);
    out.rust_allocs = RUST_ALLOCS.load(Ordering::Relaxed);
    out.b3_allocs = B3_ALLOCS.load(Ordering::Relaxed) - b3_before;
    out.b3_bytes_after = bytes_allocated();
    out.b3_traces = std::mem::take(&mut *B3_TRACES.lock().unwrap());
    out
}

fn report(name: &str, s: &Window) {
    println!(
        "{name}: 600 steps, {} move events, {} begin / {} end contact events, {} ray-hit callbacks; \
         rust allocs = {}, box3d allocs = {}, box3d bytes held {} -> {}",
        s.moves, s.begins, s.ends, s.hits, s.rust_allocs, s.b3_allocs, s.b3_bytes_before, s.b3_bytes_after
    );
    for t in &s.b3_traces {
        println!("{name}: Box3D allocation of {t}");
    }
}

// One #[test] so no other test in this binary touches Box3D's global
// allocator hooks concurrently.
#[test]
fn no_alloc_steady_state() {
    // SAFETY: the functions forward to the system allocator with the
    // requested alignment, abort instead of unwinding, and are thread-safe.
    unsafe { set_allocator(b3_alloc, b3_free) }.expect("no Box3D memory held yet");

    // Phase 1: the steady state. 300 awake boxes resting in a wall, a
    // swinging pendulum. Both counters must be exactly zero.
    let mut w = build(false);
    let s = measure(&mut w);
    report("steady", &s);
    assert!(s.moves >= 600 * 301, "wall and bob should stay awake, got {} move events", s.moves);
    assert!(s.hits >= 6000, "every ray should hit something, got {}", s.hits);
    assert_eq!(s.rust_allocs, 0, "Rust wrapper allocated during steady-state stepping");
    assert_eq!(s.b3_allocs, 0, "Box3D allocated during steady-state stepping: {:#?}", s.b3_traces);
    drop(w);

    // Phase 2: not a steady state. A heavy bob keeps smashing the wall, so the
    // contact set keeps changing. The wrapper must still never allocate.
    // Box3D may grow internal arrays when a new high-water mark is hit
    // (observed: constraint-graph color arrays, b3AddContactToGraph ->
    // b3Array_Push -> b3GrowAlloc; b3Capacity does not pre-size these). That
    // is amortized doubling growth, so it is bounded and reported here with
    // backtraces instead of being hidden.
    let mut w = build(true);
    let s = measure(&mut w);
    report("impact", &s);
    assert!(s.begins > 0 && s.ends > 0, "the bob should keep making and breaking contacts");
    assert_eq!(s.rust_allocs, 0, "Rust wrapper allocated while stepping");
    assert!(s.b3_allocs <= 8, "Box3D growth should be rare doubling, got {}: {:#?}", s.b3_allocs, s.b3_traces);
    drop(w);

    assert_eq!(bytes_allocated(), 0, "dropping the worlds releases everything");
    assert_eq!(B3_ALLOCS.load(Ordering::Relaxed), B3_FREES.load(Ordering::Relaxed));
    println!(
        "lifetime: {} Box3D allocations, {} bytes, all freed",
        B3_ALLOCS.load(Ordering::Relaxed),
        B3_ALLOC_BYTES.load(Ordering::Relaxed)
    );
}
