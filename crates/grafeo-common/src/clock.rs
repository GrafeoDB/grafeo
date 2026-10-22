//! The clock: where the crates read the time.
//!
//! Every read of the wall clock or of a monotonic instant goes through this
//! module, so it works on every target. Native targets use [`std::time`]; on
//! wasm32 `std::time::SystemTime::now()` and `Instant::now()` panic, and
//! `web-time` reads the host's clock instead (`Date.now()` and
//! `performance.now()` in a browser).
//!
//! (`no_run`: Miri's isolation has no wall clock; the unit tests check the values.)
//!
//! ```no_run
//! use grafeo_common::clock::{self, Instant};
//!
//! let start = Instant::now();
//! assert!(clock::unix_millis() > 1_577_836_800_000, "after 2020");
//! let _elapsed = start.elapsed();
//! ```

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(target_arch = "wasm32")]
pub use web_time::{Instant, SystemTime, UNIX_EPOCH};

/// Now on the wall clock, in whole milliseconds since the Unix epoch: 0 for a
/// clock set before it.
#[must_use]
pub fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
        })
}

/// Now on the wall clock, in microseconds since the Unix epoch: negative for a
/// clock set before it.
#[must_use]
pub fn unix_micros() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(since) => i64::try_from(since.as_micros()).unwrap_or(i64::MAX),
        Err(before) => {
            i64::try_from(before.duration().as_micros()).map_or(i64::MIN, |micros| -micros)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires wall clock (SystemTime::now) unavailable under Miri isolation"
    )]
    fn the_wall_clock_reads_now() {
        let millis = unix_millis();
        assert!(millis > 1_577_836_800_000, "after 2020: {millis}");
        let micros = unix_micros();
        assert!(
            micros / 1000 >= i64::try_from(millis).unwrap(),
            "not before the earlier read"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "requires a monotonic clock unavailable under Miri isolation"
    )]
    fn an_instant_does_not_go_back() {
        let start = Instant::now();
        assert!(Instant::now() >= start);
    }
}
