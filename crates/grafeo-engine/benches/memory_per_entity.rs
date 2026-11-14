//! Memory per node, per edge and per property value (#431), to measure the store step by step.
//!
//! The workload is the issue's: 100,000 nodes with one label and three integer properties, and
//! 300,000 edges, written through the direct API into an in-memory database, in three phases
//! (the nodes, then their properties, then the edges). Each phase is measured two ways:
//!
//! - **allocated**: the bytes the process holds after it, less before it, counted by a global
//!   allocator that forwards to the system's (exact, and the same on every platform);
//! - **reported**: what `GrafeoDB::memory_usage()` says, its total and its parts.
//!
//! Each is divided by what the phase wrote: bytes per node, per property value, per edge. The
//! bench prints them as a table next to the issue's targets for the compact store (64 B per
//! node, 96 B per edge, 16 B per integer property), which are shown, not checked.
//!
//! ```bash
//! cargo bench -p grafeo-engine --bench memory_per_entity                      # print the table
//! cargo bench -p grafeo-engine --bench memory_per_entity -- --check           # and fail on a rise
//! cargo bench -p grafeo-engine --bench memory_per_entity -- --update-baseline # write the baseline
//! ```
//!
//! `--check` fails when a count is more than 5% (and half a byte) above
//! `benches/memory_per_entity.baseline.json`. A change that lowers memory updates the baseline
//! in the same commit, so the drop shows in review. `--out <path>` also writes the counts as
//! JSON.

#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this one forwards every call to System"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// The nodes of the workload.
const NODES: usize = 100_000;
/// The edges of each node: 300,000 in all.
const EDGES_PER_NODE: usize = 3;
/// The integer properties of each node.
const PROPERTIES: [&str; 3] = ["id", "age", "score"];
/// How far a count may rise over the baseline before `--check` fails, in percent.
const MAX_RISE_PERCENT: f64 = 5.0;
/// A rise this small (in bytes per entity) never fails: rounding, not memory.
const MIN_RISE_BYTES: f64 = 0.5;

// ── The counting allocator ──────────────────────────────────────────

/// Every byte allocated and every byte freed, since the start of the process: the bytes held
/// are their difference.
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static FREED: AtomicUsize = AtomicUsize::new(0);

/// Forwards to [`System`] and counts the bytes of each call.
struct Counting;

// SAFETY: every method forwards to `System` with the caller's arguments, so `System`'s
// guarantees are this allocator's; the counters are only read for the report.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc`, passed on.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc_zeroed`, passed on.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc`, passed on.
        unsafe { System.dealloc(pointer, layout) };
        FREED.fetch_add(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract for `realloc`, passed on.
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
            FREED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        moved
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The bytes the process holds now.
fn held() -> usize {
    ALLOCATED
        .load(Ordering::Relaxed)
        .wrapping_sub(FREED.load(Ordering::Relaxed))
}

// ── The workload ────────────────────────────────────────────────────

/// What the database holds after a phase: by the allocator, and as it reports it.
struct Measure {
    held: usize,
    usage: grafeo_engine::memory_usage::MemoryUsage,
}

fn measure(db: &GrafeoDB) -> Measure {
    // The report allocates: read it first, so the count after it holds only the database.
    let usage = db.memory_usage();
    Measure {
        held: held(),
        usage,
    }
}

/// The three phases, each measured after it: the nodes, their properties, the edges.
fn run() -> [Measure; 4] {
    let db = GrafeoDB::new_in_memory();
    // The ids' vector is the bench's, not the database's: reserved whole before the first
    // measurement, so no phase's difference holds it (its 8 bytes per node were once counted as
    // the nodes').
    let mut nodes: Vec<NodeId> = Vec::with_capacity(NODES);
    let empty = measure(&db);

    nodes.extend((0..NODES).map(|_| db.create_node(&["Person"]).expect("create a node")));
    assert_eq!(nodes.capacity(), NODES, "the reserved vector never grew");
    let with_nodes = measure(&db);

    for (index, &node) in nodes.iter().enumerate() {
        let index = i64::try_from(index).expect("an index fits an i64");
        for (key, value) in PROPERTIES
            .iter()
            .zip([index, index % 90, (index * 31) % 1000])
        {
            db.set_node_property(node, key, Value::Int64(value))
                .expect("set a property");
        }
    }
    let with_properties = measure(&db);

    for index in 0..NODES {
        for k in 0..EDGES_PER_NODE {
            let target = (index * 7 + k * 13 + 1) % NODES;
            db.create_edge(nodes[index], nodes[target], "KNOWS")
                .expect("create an edge");
        }
    }
    let with_edges = measure(&db);
    drop(nodes);
    [empty, with_nodes, with_properties, with_edges]
}

/// `bytes` per one of `count`, rounded to a tenth of a byte.
fn per(bytes: usize, count: usize) -> f64 {
    let tenths = (bytes * 10 + count / 2) / count;
    f64::from(u32::try_from(tenths).unwrap_or(u32::MAX)) / 10.0
}

/// The counts the bench reports, by name, each in bytes per entity.
fn counts(phases: &[Measure; 4]) -> BTreeMap<&'static str, f64> {
    let [empty, nodes, properties, edges] = phases;
    let values = NODES * PROPERTIES.len();
    let edge_count = NODES * EDGES_PER_NODE;
    let usage = &edges.usage;
    let total = |measure: &Measure| measure.usage.total_bytes;
    BTreeMap::from([
        (
            "allocated per node",
            per(nodes.held.saturating_sub(empty.held), NODES),
        ),
        (
            "allocated per property value",
            per(properties.held.saturating_sub(nodes.held), values),
        ),
        (
            "allocated per edge",
            per(edges.held.saturating_sub(properties.held), edge_count),
        ),
        (
            "reported per node",
            per(total(nodes).saturating_sub(total(empty)), NODES),
        ),
        (
            "reported per property value",
            per(total(properties).saturating_sub(total(nodes)), values),
        ),
        (
            "reported per edge",
            per(total(edges).saturating_sub(total(properties)), edge_count),
        ),
        ("node records", per(usage.store.nodes_bytes, NODES)),
        (
            "node version chains",
            per(usage.mvcc.node_version_chains_bytes, NODES),
        ),
        ("node labels", per(usage.indexes.node_labels_bytes, NODES)),
        ("label index", per(usage.indexes.label_index_bytes, NODES)),
        (
            "node properties",
            per(usage.store.node_properties_bytes, values),
        ),
        ("edge records", per(usage.store.edges_bytes, edge_count)),
        (
            "edge version chains",
            per(usage.mvcc.edge_version_chains_bytes, edge_count),
        ),
        (
            "forward adjacency",
            per(usage.indexes.forward_adjacency_bytes, edge_count),
        ),
        (
            "backward adjacency",
            per(usage.indexes.backward_adjacency_bytes, edge_count),
        ),
    ])
}

/// The unit of a count, as the table prints it.
fn unit(name: &str) -> &'static str {
    if name.contains("property") || name.contains("properties") {
        "B/value"
    } else if name.contains("edge") || name.contains("adjacency") {
        "B/edge"
    } else {
        "B/node"
    }
}

/// The compact store's target for a count (#431), shown beside it.
fn target(name: &str) -> &'static str {
    match name {
        "allocated per node" | "reported per node" => "64",
        "allocated per edge" | "reported per edge" => "96",
        "allocated per property value" | "reported per property value" => "16",
        _ => "",
    }
}

// ── The baseline ────────────────────────────────────────────────────

fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/memory_per_entity.baseline.json")
}

fn to_json(counts: &BTreeMap<&'static str, f64>) -> String {
    let workload = serde_json::json!({
        "nodes": NODES,
        "edges": NODES * EDGES_PER_NODE,
        "properties per node": PROPERTIES.len(),
    });
    let document = serde_json::json!({ "workload": workload, "bytes per entity": counts });
    serde_json::to_string_pretty(&document).expect("the counts serialize") + "\n"
}

/// The counts of the baseline file, or why it cannot be read.
fn read_baseline() -> Result<BTreeMap<String, f64>, String> {
    let path = baseline_path();
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let document: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let counts = document
        .get("bytes per entity")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{}: no \"bytes per entity\" object", path.display()))?;
    counts
        .iter()
        .map(|(name, value)| {
            value
                .as_f64()
                .map(|value| (name.clone(), value))
                .ok_or_else(|| format!("{}: \"{name}\" is not a number", path.display()))
        })
        .collect()
}

/// Each count that rose too far over the baseline, or that the baseline lacks, as a line.
fn rises(counts: &BTreeMap<&'static str, f64>, baseline: &BTreeMap<String, f64>) -> Vec<String> {
    let mut found = Vec::new();
    for (name, &now) in counts {
        match baseline.get(*name) {
            None => found.push(format!("{name}: {now} B, not in the baseline")),
            Some(&before)
                if now - before >= MIN_RISE_BYTES
                    && now > before * (1.0 + MAX_RISE_PERCENT / 100.0) =>
            {
                found.push(format!(
                    "{name}: {now} B, up from {before} B ({:+.1}%)",
                    (now / before - 1.0) * 100.0
                ));
            }
            Some(_) => {}
        }
    }
    found
}

fn table(counts: &BTreeMap<&'static str, f64>, baseline: Option<&BTreeMap<String, f64>>) -> String {
    let mut out =
        String::from("| Count | Bytes | Baseline | Target |\n| --- | --- | --- | --- |\n");
    for (name, value) in counts {
        let before = baseline
            .and_then(|baseline| baseline.get(*name))
            .map_or_else(String::new, |before| format!("{before} {}", unit(name)));
        let target = match target(name) {
            "" => String::new(),
            target => format!("{target} {}", unit(name)),
        };
        let unit = unit(name);
        // Writing to a String cannot fail.
        let _ = writeln!(out, "| {name} | {value} {unit} | {before} | {target} |");
    }
    out
}

fn main() {
    // `cargo bench` passes `--bench`; the bench's own flags follow `--`.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let check = args.iter().any(|arg| arg == "--check");
    let update = args.iter().any(|arg| arg == "--update-baseline");
    let out = args
        .iter()
        .position(|arg| arg == "--out")
        .and_then(|at| args.get(at + 1))
        .map(PathBuf::from);

    let counts = counts(&run());
    let baseline = read_baseline();
    let table = table(&counts, baseline.as_ref().ok());
    println!(
        "Memory per entity: {NODES} nodes, {} properties each, {} edges\n\n{table}",
        PROPERTIES.len(),
        NODES * EDGES_PER_NODE
    );
    if let Ok(summary) = std::env::var("GITHUB_STEP_SUMMARY") {
        let text = format!("## Memory per entity (#431)\n\n{table}\n");
        if let Err(error) = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&summary)
            .and_then(|mut file| std::io::Write::write_all(&mut file, text.as_bytes()))
        {
            eprintln!("could not write the step summary {summary}: {error}");
        }
    }
    let json = to_json(&counts);
    if let Some(out) = out {
        std::fs::write(&out, &json).unwrap_or_else(|error| panic!("{}: {error}", out.display()));
    }
    if update {
        let path = baseline_path();
        std::fs::write(&path, &json).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        println!("Baseline written to {}", path.display());
    }
    if check {
        let baseline = baseline.unwrap_or_else(|error| {
            eprintln!("no baseline to check against: {error}");
            std::process::exit(2)
        });
        let rises = rises(&counts, &baseline);
        if !rises.is_empty() {
            eprintln!(
                "Memory per entity rose more than {MAX_RISE_PERCENT}% over the baseline:\n  {}\n\
                 If the rise is intended, run with --update-baseline and commit the baseline.",
                rises.join("\n  ")
            );
            std::process::exit(1);
        }
        println!("No count rose more than {MAX_RISE_PERCENT}% over the baseline.");
    }
}
