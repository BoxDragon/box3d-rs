# Prior work: Rust and Box3D

Surveyed 2026-09-29. Upstream Box3D is at
[`erincatto/box3d@9f998c8`](https://github.com/erincatto/box3d/commit/9f998c862d54c03a633ecea3831937385c78b532)
(2026-09-27). This workspace pins that commit.

## Existing Box3D bindings and ports

| Project | Crates (crates.io) | Kind | Box3D commit tracked | License | Last activity |
|---|---|---|---|---|---|
| [Latias94/boxddd](https://github.com/Latias94/boxddd) ([wolfd fork](https://github.com/wolfd/boxddd)) | [`boxddd`](https://crates.io/crates/boxddd), [`boxddd-sys`](https://crates.io/crates/boxddd-sys) 0.4.0, [`bevy_boxddd`](https://docs.rs/bevy_boxddd) | `-sys` + safe + Bevy 0.19 plugin | `781673be` (2026-07-25, 12 commits behind our pin) | MIT OR Apache-2.0 | release 2026-08-04 |
| [Tebarem/box3d-rs](https://github.com/Tebarem/box3d-rs) ([cark-dev fork](https://github.com/cark-dev/box3d-rs)) | [`box3d`](https://crates.io/crates/box3d) 0.3.1, [`box3d-sys`](https://crates.io/crates/box3d-sys) 0.1.16, `bevy_box3d` | `-sys` + safe + Bevy | `3fc20f5` (2026-07-30, 11 behind) | MIT | 2026-09-19 |
| [larsbrubaker/box3d-rust](https://github.com/larsbrubaker/box3d-rust) | [`box3d-rust`](https://crates.io/crates/box3d-rust) 0.2.1 | pure Rust port | `30c67b5` (2026-08-13, 9 behind) | MIT | 2026-08-24 |

Box2D v3 bindings whose design is relevant (Box3D's C API mirrors Box2D v3):

- [Latias94/boxdd](https://github.com/Latias94/boxdd): [`boxdd`](https://crates.io/crates/boxdd) and `boxdd-sys` 0.6.0. The most mature. Offers owned handles (Drop destroys) and scoped handles, pregenerated bindings, and mint interop. Same author as boxddd.
- [Carroted/box2d_sys](https://github.com/Carroted/box2d_sys): [`box2d-sys`](https://crates.io/crates/box2d-sys) 0.2.1. Raw only, and stale (2024).

Non-Rust bindings, for reference: [box3d-unity](https://github.com/Suvitruf/box3d-unity), [godot-box3d](https://github.com/bearlikelion/godot-box3d), [box3d.js](https://isaac-mason.github.io/box3d.js/), [box3d-wasm](https://github.com/monteslu/box3d-wasm), [Box3D.NET](https://github.com/Miguel249/Box3D.NET).

## Notes per project

**boxddd.** The most complete and most engineered option.
- Vendors the C sources with a manifest of fingerprints and two local patches: an Emscripten timer, and a CCD null pre-solve guard.
- Ships pregenerated bindings, so no libclang is needed. Has single- and double-precision builds, plus mint/glam/nalgebra/cgmath interop and serde.
- Uses owner-scoped ids with fallible `try_*` operations returning `Result`.
- The world is deliberately `!Send`/`!Sync`, and state crosses threads as snapshots.
- It covers debug draw and has a Bevy plugin. It deliberately does not support shape-cast or shape-overlap visitors, task-system callbacks, or contact/material callbacks.
- Rust MSRV is 1.92. It is labelled "experimental 0.x".

**Tebarem/box3d-rs.**
- Owns the `box3d` and `box3d-sys` names on crates.io.
- Generates bindings at build time with bindgen, which needs libclang, and builds the C with CMake.
- The safe layer uses borrow-based handles: bodies borrow the world and shapes borrow bodies.
- Coverage is broad: mover/character, meshes, compounds, debug draw, recording and tasks.
- It has its own math types, not glam. `double-precision` builds the C that way, but the safe API stays f32.

**box3d-rust.** A behavior-matching pure-Rust port with deterministic math. It has no SIMD and no worker threads. Its API mirrors C, with no ergonomic layer yet.

## Should you use an existing binding instead?

**For a quick start, [boxddd](https://github.com/Latias94/boxddd) is the one to use.** It is the most mature binding, is maintained, and is dual-licensed, and it already has glam interop and a Bevy plugin.

This workspace (`box3d-rs`) is worth keeping only if you want what none of the others offer together:

- **Current upstream.** It pins 9f998c8. The others are 9 to 12 commits behind, and fixes are landing fast while Box3D is new.
- **An allocation-free hot path, verified by a test.**
  - Event slices borrow Box3D's buffers directly.
  - Queries use closure trampolines, with no boxing.
  - The destroy queue is pre-sized.
  - Handles are plain 8-byte `Copy` ids and `u64` user data (no registry), which suits an ECS such as hecs.
  - `tests/no_alloc.rs` counts both Box3D's allocator and Rust's global allocator.
- **glam 0.30 as the native math type**, not an interop layer.
- **A `Send` world** that can be driven from a physics thread, plus shape cast/overlap closure queries and custom-filter/pre-solve callbacks.

Its costs are narrower coverage (see "Not wrapped" in README.md) and the name clash below.

**Name clash:** `box3d` and `box3d-sys` on crates.io belong to Tebarem/box3d-rs. This workspace uses the same names as path dependencies, which is fine locally. Before publishing, it would need renaming (for example `box3d-glam` / `box3d-glam-sys`). Otherwise, consume it as a git or path dependency.
