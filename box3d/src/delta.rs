//! Rollback: incremental in-process world snapshots (`b3WorldDelta`, from the
//! kabergstrom/box3d fork) and the world state hash.
//!
//! The tracker belongs to its [`World`] ([`World::enable_delta`]) rather than
//! being a free-standing handle: `World` is `Send`, and a tracker dropped on
//! another thread would detach itself from a world that may be stepping.

use core::ptr::NonNull;

use crate::world::Retained;
use crate::{Error, ShapeId, World, sys};

/// Statistics of the last capture (`b3WorldDeltaStats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeltaStats {
    /// Elements compared by the last capture.
    pub compared: u32,
    /// Elements that changed in the last capture.
    pub changed: u32,
    /// Undo bytes stored by the last capture.
    pub changed_bytes: u32,
    /// Bytes held in all undo logs.
    pub log_bytes: u64,
    /// Bytes held by the mirror (about one full world image).
    pub mirror_bytes: u64,
    /// Changes the candidate set missed (only counted with verification on,
    /// see [`World::set_delta_verify`]). Missed changes are still recorded.
    pub misses: u32,
}

/// A world's rollback tracker: a mirror of the last captured state plus one
/// undo log per captured tick. Created by [`World::enable_delta`], reached
/// through [`World::delta`].
///
/// Ticks are the tracker's own capture count, starting at 0 when tracking
/// starts; restorable ticks are `[oldest_tick, newest_tick]`.
pub struct WorldDelta {
    raw: NonNull<sys::b3WorldDelta>,
    /// Newest captured tick, or the tick last restored to.
    pub(crate) tick: i32,
    max_ticks: u32,
    /// Geometry of shapes destroyed while tracked, kept while a restorable
    /// tick still has the shape (Box3D points at it from the shape).
    graveyard: Vec<Buried>,
}

struct Buried {
    retained: Retained,
    /// Newest captured tick in which the shape still existed.
    died: i32,
}

// SAFETY: the tracker is only reached through its `World` (`&`/`&mut`), and
// `World` is `Send` but not `Sync`.
unsafe impl Send for WorldDelta {}

impl WorldDelta {
    pub fn newest_tick(&self) -> i32 {
        // SAFETY: live tracker; plain getter.
        unsafe { sys::b3WorldDelta_GetNewestTick(self.raw.as_ptr()) }
    }

    pub fn oldest_tick(&self) -> i32 {
        // SAFETY: live tracker; plain getter.
        unsafe { sys::b3WorldDelta_GetOldestTick(self.raw.as_ptr()) }
    }

    /// The rollback window given to [`World::enable_delta`].
    pub fn max_ticks(&self) -> u32 {
        self.max_ticks
    }

    pub fn stats(&self) -> DeltaStats {
        // SAFETY: live tracker; returns a value.
        let s = unsafe { sys::b3WorldDelta_GetStats(self.raw.as_ptr()) };
        DeltaStats {
            compared: s.comparedCount.max(0) as u32,
            changed: s.changedCount.max(0) as u32,
            changed_bytes: s.changedBytes.max(0) as u32,
            log_bytes: s.logBytes.max(0) as u64,
            mirror_bytes: s.mirrorBytes.max(0) as u64,
            misses: s.missCount.max(0) as u32,
        }
    }

    /// Move the geometry of shapes that no longer exist from `resources` to
    /// the graveyard.
    pub(crate) fn bury_dead(&mut self, resources: &mut Vec<Retained>, alive: impl Fn(ShapeId) -> bool) {
        let mut i = 0;
        while i < resources.len() {
            if alive(resources[i].shape) {
                i += 1;
                continue;
            }
            let retained = resources.swap_remove(i);
            self.graveyard.push(Buried { retained, died: self.tick });
        }
    }
}

impl Drop for WorldDelta {
    fn drop(&mut self) {
        // SAFETY: created by b3CreateWorldDelta, destroyed once. `World`
        // drops its tracker before destroying the C world, so the world is
        // alive and this detaches it.
        unsafe { sys::b3DestroyWorldDelta(self.raw.as_ptr()) }
    }
}

/// Rollback.
impl World {
    /// Start tracking for rollback, capturing the current state as tick 0.
    /// `max_ticks` is the window: restorable ticks are
    /// `[newest - max_ticks, newest]`. Restores are valid only in this
    /// process (host pointers such as callback contexts are restored
    /// verbatim).
    ///
    /// Fails if the world already has a tracker or `max_ticks` is 0.
    pub fn enable_delta(&mut self, max_ticks: u32) -> Result<(), Error> {
        if self.delta.is_some() {
            return Err(Error::InvalidArgument("world already has a delta tracker"));
        }
        if max_ticks == 0 || max_ticks > i32::MAX as u32 {
            return Err(Error::InvalidArgument("delta max_ticks"));
        }
        // SAFETY: valid world, never locked outside `step`.
        let raw = unsafe { sys::b3CreateWorldDelta(self.id, max_ticks as i32) };
        let raw = NonNull::new(raw).ok_or(Error::InvalidArgument("world is locked"))?;
        // Every live shape exists at tick 0.
        for r in &mut self.resources {
            r.born = -1;
        }
        self.delta = Some(WorldDelta { raw, tick: 0, max_ticks, graveyard: Vec::new() });
        Ok(())
    }

    /// Stop tracking and free the mirror and logs.
    pub fn disable_delta(&mut self) {
        self.delta = None;
    }

    pub fn delta(&self) -> Option<&WorldDelta> {
        self.delta.as_ref()
    }

    /// Capture the current state as the next tick; call once after each
    /// [`World::step`]. API changes since the last capture belong to this
    /// one. `None` without a tracker.
    pub fn capture_delta(&mut self) -> Option<i32> {
        let delta = self.delta.as_mut()?;
        // SAFETY: live tracker attached to this live, unlocked world.
        let tick = unsafe { sys::b3WorldDelta_Capture(delta.raw.as_ptr()) };
        debug_assert!(tick >= 0, "b3WorldDelta_Capture refused");
        delta.tick = tick;
        let oldest = delta.oldest_tick();
        delta.graveyard.retain(|b| b.died >= oldest);
        Some(tick)
    }

    /// Restore the world to captured `tick`. Captured ticks after it are
    /// discarded and the next capture is `tick + 1`. Body, shape and joint
    /// ids are restored with the id pools, so ids from `tick` are valid
    /// again and ids created later are invalid. The destroy queue is
    /// cleared (the restore undoes its requests anyway), and the custom
    /// filter and pre-solve callbacks are re-registered as currently set.
    ///
    /// Fails without a tracker, outside `[oldest_tick, newest_tick]`, or
    /// while recording.
    pub fn restore_delta(&mut self, tick: i32) -> Result<(), Error> {
        let Some(delta) = self.delta.as_mut() else {
            return Err(Error::InvalidArgument("world has no delta tracker"));
        };
        // SAFETY: live tracker attached to this live, unlocked world. Every
        // mesh / height field a restored shape can point at is held in
        // `resources` or the graveyard (shapes alive at any restorable tick).
        if !unsafe { sys::b3WorldDelta_Restore(delta.raw.as_ptr(), tick) } {
            return Err(Error::RestoreRefused);
        }
        delta.tick = tick;
        self.destroy_queue.clear();
        // Shapes created after `tick` are gone; buried ones alive at `tick`
        // are back.
        self.resources.retain(|r| r.born < tick);
        let graveyard = &mut delta.graveyard;
        let mut i = 0;
        while i < graveyard.len() {
            let b = &graveyard[i];
            if b.retained.born >= tick {
                graveyard.swap_remove(i);
            } else if b.died >= tick {
                self.resources.push(graveyard.swap_remove(i).retained);
            } else {
                i += 1;
            }
        }
        debug_assert!(self.resources.iter().all(|r| self.is_shape_valid(r.shape)));
        self.reregister_callbacks();
        Ok(())
    }

    /// Also scan every element on capture and count changes the candidate
    /// set missed ([`DeltaStats::misses`]). Costs as much as a full
    /// snapshot; for tests.
    pub fn set_delta_verify(&mut self, on: bool) {
        if let Some(delta) = &self.delta {
            // SAFETY: live tracker.
            unsafe { sys::b3WorldDelta_EnableVerify(delta.raw.as_ptr(), on) }
        }
    }

    /// Hash of every body's transform and velocity (`b3World_GetStateHash`,
    /// the per-step hash Box3D recordings store). Two worlds hash equal
    /// exactly when those are bit-identical: compare it across runs or peers
    /// to detect divergence.
    pub fn state_hash(&self) -> u64 {
        // SAFETY: valid world, never locked outside `step`.
        unsafe { sys::b3World_GetStateHash(self.id) }
    }
}
