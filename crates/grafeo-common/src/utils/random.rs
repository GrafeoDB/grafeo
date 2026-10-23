//! Random numbers for the query functions that draw one per call: `rand()`,
//! and SPARQL's `RAND()`, `UUID()` and `STRUUID()`.
//!
//! One stream per process: SplitMix64 over a process-wide counter, so every
//! call gives a new value, whatever the clock's resolution (the browser's is
//! a millisecond) and however rows are batched, and two threads never draw
//! the same one. The stream starts from a seed taken once per process: the
//! operating system's entropy with the `os-entropy` feature (every native
//! build, and the WASM binding), so two processes do not give the same
//! sequence. Not for keys or secrets: the values are uniform, not
//! unpredictable.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The increment of SplitMix64: the golden ratio as a 64-bit fraction.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// 64 random bits, new on every call.
#[must_use]
pub fn next_u64() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seed = *SEED.get_or_init(seed);
    let step = COUNTER.fetch_add(1, Ordering::Relaxed);
    mix(seed.wrapping_add(step.wrapping_mul(GOLDEN_GAMMA)))
}

/// A double in `[0, 1)`, new on every call: the 53 high bits of
/// [`next_u64`], so every value is exact.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "a value below 2^53 is exact in a double"
)]
pub fn next_f64() -> f64 {
    (next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// The finalizer of SplitMix64 (Steele, Lea and Flood): a bijection that
/// spreads every input bit over the output.
const fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of the process's stream: 64 bits of the operating system's
/// entropy (the browser's crypto in WASM) with the `os-entropy` feature.
/// Without it, or where the system cannot give any, see [`fallback_seed`].
fn seed() -> u64 {
    #[cfg(feature = "os-entropy")]
    if let Ok(seed) = getrandom::u64() {
        return seed;
    }
    fallback_seed()
}

/// The seed without the operating system's entropy: the clock in
/// nanoseconds, the process id where there is one, and the keys of std's
/// `RandomState`, which carry the system's entropy where std has it (not on
/// `wasm32-unknown-unknown`, where two instances started in the same
/// millisecond can share a stream).
fn fallback_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(
        crate::clock::SystemTime::now()
            .duration_since(crate::clock::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos()),
    );
    #[cfg(not(target_arch = "wasm32"))]
    hasher.write_u32(std::process::id());
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_call_draws_a_new_value() {
        let values: HashSet<u64> = (0..100_000).map(|_| next_u64()).collect();
        assert_eq!(values.len(), 100_000);
    }

    #[test]
    fn a_double_is_in_the_unit_interval_and_spread_over_it() {
        let values: Vec<f64> = (0..10_000).map(|_| next_f64()).collect();
        assert!(values.iter().all(|v| (0.0..1.0).contains(v)));
        let below_half = values.iter().filter(|v| **v < 0.5).count();
        assert!(
            (4_500..5_500).contains(&below_half),
            "{below_half} of 10000 below 0.5"
        );
    }

    #[test]
    fn threads_never_draw_the_same_value() {
        let drawn: Vec<Vec<u64>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| (0..10_000).map(|_| next_u64()).collect::<Vec<_>>()))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let all: HashSet<u64> = drawn.iter().flatten().copied().collect();
        assert_eq!(all.len(), 40_000);
    }

    /// Both seeds give a value, and two fallback seeds differ: the clock and
    /// the hasher keys change between calls.
    #[test]
    fn seeding_works_with_and_without_the_systems_entropy() {
        let _ = seed();
        let (first, second) = (fallback_seed(), fallback_seed());
        assert_ne!(first, second);
        #[cfg(feature = "os-entropy")]
        assert_ne!(seed(), seed(), "the system's entropy differs per call");
    }

    #[test]
    fn the_mix_is_a_bijection_on_a_sample() {
        let mixed: HashSet<u64> = (0..100_000u64).map(mix).collect();
        assert_eq!(mixed.len(), 100_000);
    }
}
