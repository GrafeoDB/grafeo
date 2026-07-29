//! Track E — LPG residency accounting + compact/lazy representation experiments.
//!
//! Synthetic workloads (staging DBs under `/data/tmp` only when opened; this
//! harness stays in-process). Measures estimated residency, process RssAnon,
//! point-get latency, and correctness before/after:
//!   A) `shrink_capacities` (capacity waste reclaim)
//!   B) `force_compress_all` + lazy compressed `get` (dictionary-compact strings)

use std::fs;
use std::time::Instant;

use arcstr::ArcStr;
use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::PropertyStorage;

fn rss_anon_kib() -> u64 {
    let status = fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .expect("RssAnon value")
                .parse()
                .expect("parse RssAnon");
            return kib;
        }
    }
    panic!("RssAnon missing from /proc/self/status");
}

fn fill_string_heavy(storage: &PropertyStorage, n: u64, unique_ratio: usize) {
    let key = PropertyKey::new("body");
    let dict: Vec<ArcStr> = (0..unique_ratio)
        .map(|i| ArcStr::from(format!("payload-token-{i:04}-{}", "x".repeat(48))))
        .collect();
    for i in 0..n {
        let s = dict[i as usize % unique_ratio].clone();
        storage.set(NodeId::new(i), key.clone(), Value::String(s));
    }
}

fn point_get_correctness(storage: &PropertyStorage, n: u64, unique_ratio: usize) -> bool {
    let key = PropertyKey::new("body");
    let dict: Vec<String> = (0..unique_ratio)
        .map(|i| format!("payload-token-{i:04}-{}", "x".repeat(48)))
        .collect();
    for i in 0..n {
        let expected = &dict[i as usize % unique_ratio];
        match storage.get(NodeId::new(i), &key) {
            Some(Value::String(s)) if s.as_str() == expected.as_str() => {}
            other => {
                eprintln!("mismatch at {i}: got {other:?}");
                return false;
            }
        }
    }
    true
}

fn bench_point_gets(storage: &PropertyStorage, n: u64, iters: u64) -> f64 {
    let key = PropertyKey::new("body");
    let start = Instant::now();
    let mut hits = 0u64;
    for i in 0..iters {
        if storage.get(NodeId::new(i % n), &key).is_some() {
            hits += 1;
        }
    }
    assert_eq!(hits, iters);
    start.elapsed().as_secs_f64() * 1e9 / iters as f64
}

#[test]
fn experiment_a_shrink_capacities_and_b_lazy_dictionary() {
    const N: u64 = 200_000;
    const UNIQUE: usize = 256;
    const GET_ITERS: u64 = 200_000;

    // Over-reserve then fill so capacity waste is visible.
    let storage = PropertyStorage::new();
    {
        // Pre-touch with a disposable column growth path: insert then rebuild
        // is unnecessary; HashMap grows with inserts. We measure post-fill.
        fill_string_heavy(&storage, N, UNIQUE);
    }

    let baseline = storage.memory_detail();
    let rss_baseline = rss_anon_kib();
    let ns_baseline = bench_point_gets(&storage, N, GET_ITERS);
    assert!(
        point_get_correctness(&storage, N, UNIQUE),
        "baseline correctness"
    );

    println!("=== Track E experiment baseline ===");
    println!(
        "entries={N} unique={UNIQUE} columns={} total_bytes={} map_slots={} decoded={} strings={} waste={} compressed={}",
        baseline.columns.len(),
        baseline.total_bytes,
        baseline.total_map_slot_bytes,
        baseline.total_decoded_payload_bytes,
        baseline.total_string_payload_bytes,
        baseline.total_capacity_waste_bytes,
        baseline.total_compressed_bytes
    );
    if let Some(top) = baseline.columns.first() {
        println!(
            "top_column key={} entries={} cap={} slots={} strings={} waste={}",
            top.key,
            top.entry_count,
            top.map_capacity,
            top.map_slot_bytes,
            top.string_payload_bytes,
            top.capacity_waste_bytes
        );
    }
    println!("rss_anon_kib={rss_baseline} point_get_ns={ns_baseline:.1}");

    // --- Experiment A: shrink capacities ---
    storage.shrink_capacities();
    let after_shrink = storage.memory_detail();
    let rss_shrink = rss_anon_kib();
    let ns_shrink = bench_point_gets(&storage, N, GET_ITERS);
    assert!(
        point_get_correctness(&storage, N, UNIQUE),
        "post-shrink correctness"
    );
    println!("=== Experiment A: shrink_capacities ===");
    println!(
        "total_bytes={} waste={} (Δwaste={}) rss_anon_kib={rss_shrink} (Δrss={}) point_get_ns={ns_shrink:.1}",
        after_shrink.total_bytes,
        after_shrink.total_capacity_waste_bytes,
        after_shrink.total_capacity_waste_bytes as i64
            - baseline.total_capacity_waste_bytes as i64,
        rss_shrink as i64 - rss_baseline as i64
    );
    assert!(
        after_shrink.total_capacity_waste_bytes <= baseline.total_capacity_waste_bytes,
        "shrink should not increase capacity waste"
    );

    // --- Experiment B: force dictionary compress + lazy get ---
    let before_compress = storage.memory_detail();
    let rss_before_compress = rss_anon_kib();
    storage.force_compress_all();
    let after_compress = storage.memory_detail();
    let rss_compress = rss_anon_kib();
    let ns_compress = bench_point_gets(&storage, N, GET_ITERS);
    assert!(
        point_get_correctness(&storage, N, UNIQUE),
        "post-compress lazy-get correctness"
    );
    println!("=== Experiment B: force_compress + lazy get ===");
    println!(
        "before_total={} after_total={} (Δ={}) strings_before={} compressed_after={} rss_kib={rss_compress} (Δ={}) point_get_ns={ns_compress:.1} (was {ns_shrink:.1})",
        before_compress.total_bytes,
        after_compress.total_bytes,
        after_compress.total_bytes as i64 - before_compress.total_bytes as i64,
        before_compress.total_string_payload_bytes,
        after_compress.total_compressed_bytes,
        rss_compress as i64 - rss_before_compress as i64
    );
    assert!(
        after_compress.total_compressed_bytes > 0,
        "dictionary compression should materialize compressed backing"
    );
    assert!(
        after_compress.total_string_payload_bytes < before_compress.total_string_payload_bytes,
        "hot string payloads should drop after compress"
    );

    // Emit JSON-ish summary line for Track E report scraping.
    println!(
        "TRACK_E_SUMMARY shrink_waste_before={} shrink_waste_after={} shrink_rss_delta_kib={} compress_total_before={} compress_total_after={} compress_rss_delta_kib={} get_ns_baseline={:.1} get_ns_shrink={:.1} get_ns_compress={:.1}",
        baseline.total_capacity_waste_bytes,
        after_shrink.total_capacity_waste_bytes,
        rss_shrink as i64 - rss_baseline as i64,
        before_compress.total_bytes,
        after_compress.total_bytes,
        rss_compress as i64 - rss_before_compress as i64,
        ns_baseline,
        ns_shrink,
        ns_compress
    );
}

#[test]
fn accounting_attributes_string_payload_separately_from_slots() {
    let storage = PropertyStorage::new();
    let key = PropertyKey::new("name");
    for i in 0..1_000u64 {
        storage.set(
            NodeId::new(i),
            key.clone(),
            Value::String(ArcStr::from(format!("unique-name-{i:05}"))),
        );
    }
    let detail = storage.memory_detail();
    assert_eq!(detail.columns.len(), 1);
    let col = &detail.columns[0];
    assert!(col.map_slot_bytes > 0);
    assert!(col.string_payload_bytes > 0);
    assert!(col.decoded_payload_bytes >= col.string_payload_bytes);
    // Old map-slot-only accounting undercounts: decoded payloads must be in total.
    assert!(col.total_bytes() > col.map_slot_bytes);
}
