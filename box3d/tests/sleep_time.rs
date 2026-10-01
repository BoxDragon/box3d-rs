//! Body sleep time: a replica that takes another world's sleep times falls
//! asleep on the same step.
#![allow(clippy::useless_conversion, clippy::unnecessary_cast)]
use box3d::*;

const DT: f32 = 1.0 / 60.0;
const SUBSTEPS: i32 = 4;

/// A box resting on a box resting on the ground: one island of two bodies.
fn stack() -> (World, [BodyId; 2]) {
    let mut w = World::new(&WorldDef::default()).expect("world");
    let g = w.create_body(&BodyDef::fixed(Pos::new(0.0, -0.5, 0.0)));
    w.create_box_shape(g, &ShapeDef::default(), Vec3::new(10.0, 0.5, 10.0));
    let bodies = [0.5, 1.5].map(|y| {
        let b = w.create_body(&BodyDef::dynamic(Pos::new(0.0, y, 0.0)));
        w.create_box_shape(b, &ShapeDef::default(), Vec3::splat(0.5));
        b
    });
    (w, bodies)
}

/// The replica restarts its sleep timers at `late_step` (a peer that adopted
/// the state late). From `sync_step` on it takes the source's sleep times
/// before each step, and both worlds must hash equal afterwards. Returns the
/// steps the replica fell asleep after the source.
fn sleep_step_gap(late_step: usize, sync_step: usize) -> isize {
    let (mut source, src) = stack();
    let (mut replica, rep) = stack();
    let (mut source_sleep, mut replica_sleep) = (None, None);
    for step in 0..600 {
        for i in 0..2 {
            if step == late_step {
                replica.set_body_sleep_time(rep[i], 0.0);
            }
            if step >= sync_step {
                replica.set_body_sleep_time(rep[i], source.body_sleep_time(src[i]));
            }
        }
        source.step(DT, SUBSTEPS);
        replica.step(DT, SUBSTEPS);
        if step >= sync_step {
            assert_eq!(source.state_hash(), replica.state_hash(), "step {step}");
        }
        if source_sleep.is_none() && !source.is_body_awake(src[0]) {
            assert!(!source.is_body_awake(src[1]), "one island");
            source_sleep = Some(step as isize);
        }
        if replica_sleep.is_none() && !replica.is_body_awake(rep[0]) {
            replica_sleep = Some(step as isize);
        }
        if source_sleep.is_some() && replica_sleep.is_some() {
            break;
        }
    }
    let (s, r) = (source_sleep.expect("source sleeps"), replica_sleep.expect("replica sleeps"));
    assert!(s > late_step as isize && r > late_step as isize);
    r - s
}

#[test]
fn late_sleep_timer_delays_sleep_without_sync() {
    assert!(sleep_step_gap(15, usize::MAX) >= 10);
}

#[test]
fn synced_sleep_times_sleep_on_the_same_step() {
    assert_eq!(sleep_step_gap(15, 16), 0);
}

#[test]
fn sync_near_the_threshold_sleeps_on_the_same_step() {
    let threshold_step = (TIME_TO_SLEEP / DT) as usize;
    assert_eq!(sleep_step_gap(15, threshold_step - 3), 0);
    assert_eq!(sleep_step_gap(15, threshold_step - 1), 0);
}

#[test]
fn sleeping_body_ignores_full_time_and_wakes_below_it() {
    let (mut w, b) = stack();
    for _ in 0..600 {
        if !w.is_body_awake(b[0]) {
            break;
        }
        w.step(DT, SUBSTEPS);
    }
    assert!(!w.is_body_awake(b[0]));
    assert!(w.body_sleep_time(b[0]) >= TIME_TO_SLEEP);

    w.set_body_sleep_time(b[0], TIME_TO_SLEEP);
    assert!(!w.is_body_awake(b[0]));

    w.set_body_sleep_time(b[0], 0.1);
    assert!(w.is_body_awake(b[0]) && w.is_body_awake(b[1]), "the island wakes");
    assert_eq!(w.body_sleep_time(b[0]), 0.1);
    assert_eq!(w.body_sleep_time(b[1]), 0.0);

    let fixed = w.create_body(&BodyDef::fixed(Pos::new(20.0, 0.0, 0.0)));
    w.set_body_sleep_time(fixed, 0.2);
    assert_eq!(w.body_sleep_time(fixed), 0.0);
}
