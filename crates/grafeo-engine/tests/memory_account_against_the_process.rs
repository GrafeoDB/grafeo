//! The account against the process (#434, #751): what `memory_usage()` reports equals what the
//! process really took, within a stated margin. On Linux the process's private memory
//! (`Private_Clean + Private_Dirty` of `/proc/self/smaps_rollup`) is read before and after a
//! load, and the account must be between [`LOWEST`] and [`HIGHEST`] of the growth. The loads are
//! #434's case, 100,000 nodes with eight integer properties, and 10,000 nodes with a 384-d
//! vector and a vector index.
//!
//! Each load runs in a child process of its own, so no other test's allocations land in the
//! figures and every load starts from a fresh allocator. Every failure message gives the
//! measured ratios, which is what the margin is set from.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --release --test main memory_account_against_the_process::
//! ```

#![cfg(target_os = "linux")]

use grafeo_common::testing::child_process;
use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// The variable naming the load a child runs.
const CHILD_VAR: &str = "GRAFEO_ACCOUNT_AGAINST_THE_PROCESS";

/// The account's lowest and highest share of the process's growth: wide enough for the
/// allocator's own overhead on any Linux (arena split, page size, transparent huge pages), so
/// they catch a broken account (far under the growth, or over it), not the allocator. Measured
/// in WSL, release, 2026-11-07: 0.716 for 100,000 nodes with eight integers, 0.763 for 10,000
/// 384-d vectors with an index.
const LOWEST: f64 = 0.55;
const HIGHEST: f64 = 1.45;

/// The child's exit code when it measured: 2, never 0, which a child that ran nothing also gives.
const MEASURED: i32 = 2;

/// The process's private memory in bytes: `Private_Clean + Private_Dirty` of
/// `/proc/self/smaps_rollup`.
fn private_bytes() -> u64 {
    let text = std::fs::read_to_string("/proc/self/smaps_rollup").expect("smaps_rollup");
    let field = |name: &str| -> u64 {
        text.lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.trim().strip_suffix("kB"))
            .map_or_else(
                || panic!("{name} in smaps_rollup:\n{text}"),
                |kilobytes| kilobytes.trim().parse::<u64>().unwrap() * 1024,
            )
    };
    field("Private_Clean:") + field("Private_Dirty:")
}

/// #434's case: 100,000 nodes with eight integer properties.
fn integer_properties(db: &GrafeoDB) {
    let keys: Vec<PropertyKey> = (0..8).map(|k| PropertyKey::new(format!("p{k}"))).collect();
    for i in 0..100_000_i64 {
        let properties = keys
            .iter()
            .enumerate()
            .map(|(k, key)| (key.clone(), Value::Int64(i * 8 + i64::try_from(k).unwrap())));
        db.create_node_with_props(&["Item"], properties).unwrap();
    }
}

/// 10,000 nodes with a 384-d vector each, and a vector index on them.
#[cfg(feature = "vector-index")]
fn vectors(db: &GrafeoDB) {
    for i in 0..10_000_u32 {
        #[expect(clippy::cast_precision_loss, reason = "small numbers")]
        let vector: Vec<f32> = (0..384)
            .map(|d| ((i * 384 + d) % 9_973) as f32 / 9_973.0)
            .collect();
        db.create_node_with_props(
            &["Doc"],
            [(PropertyKey::new("embedding"), Value::Vector(vector.into()))],
        )
        .unwrap();
    }
    db.create_vector_index("Doc", "embedding", Some(384), None, None, None, None)
        .unwrap();
}

/// The loads, by name.
fn loads() -> Vec<(&'static str, fn(&GrafeoDB))> {
    let mut loads: Vec<(&'static str, fn(&GrafeoDB))> = vec![(
        "100,000 nodes with eight integer properties",
        integer_properties,
    )];
    #[cfg(feature = "vector-index")]
    loads.push(("10,000 nodes with a 384-d vector and an index", vectors));
    loads
}

/// The child: runs load `index` in a fresh in-memory database and prints the account, the
/// process's growth and their ratio, then exits with [`MEASURED`]. A no-op when run directly.
#[test]
fn account_against_the_process_child() {
    let Ok(index) = std::env::var(CHILD_VAR) else {
        return;
    };
    let (_, load) = loads()[index.parse::<usize>().unwrap()];
    let before = private_bytes();
    let db = GrafeoDB::new_in_memory();
    load(&db);
    let account = db.memory_usage().total_bytes as u64;
    let growth = private_bytes().saturating_sub(before);
    println!("ACCOUNT {account} GROWTH {growth}");
    std::process::exit(MEASURED);
}

/// The account of each load is within the margin of the process's growth.
#[test]
fn the_account_follows_the_process() {
    let mut measured = Vec::new();
    let mut wrong = false;
    for (index, (name, _)) in loads().iter().enumerate() {
        let output = child_process::output(
            grafeo_common::child_test!("account_against_the_process_child")
                .env(CHILD_VAR, index.to_string()),
        )
        .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(
            output.status.code(),
            Some(MEASURED),
            "{name}: the child did not measure:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let line = stdout
            .lines()
            .find(|line| line.starts_with("ACCOUNT "))
            .unwrap_or_else(|| panic!("{name}: no figures:\n{stdout}"));
        let figures: Vec<u64> = line
            .split_whitespace()
            .filter_map(|word| word.parse().ok())
            .collect();
        let [account, growth] = figures[..] else {
            panic!("{name}: {line}")
        };
        #[expect(clippy::cast_precision_loss, reason = "a ratio of byte counts")]
        let ratio = account as f64 / growth.max(1) as f64;
        wrong |= !(LOWEST..=HIGHEST).contains(&ratio);
        measured.push(format!(
            "{name}: account {account} bytes, process growth {growth} bytes, ratio {ratio:.3}"
        ));
    }
    assert!(
        !wrong,
        "the account must be {LOWEST} to {HIGHEST} of the process's growth:\n{}",
        measured.join("\n")
    );
    // The figures, for the margin and the docs, also when they pass.
    println!("{}", measured.join("\n"));
}
