//! A database over 4 GiB round trips through checkpoints and reopens (#428,
//! #392).
//!
//! 0.5.x stored the offset of each block of a section in a `u32`: past 4 GiB
//! the offsets wrapped, `close()` still reported success, and the next open
//! failed with `block 0 CRC mismatch` (#392). The chunked format of 0.6 places
//! every chunk at a `u64` file offset. This test writes a database whose LPG
//! section alone holds about 4.6 GiB: large text and bytes values on 1,490 of
//! 190,388 nodes, with their edges and a named graph, and a vector and a text
//! index (when the build has them), whose sections a checkpoint writes after
//! the LPG section, so past 4 GiB. It checkpoints the database with
//! `close()`, reads every chunk of the file back against its checksum,
//! reopens it read-write and compares every node, edge and index with what
//! was written. Then it changes a few values and closes again: the second
//! image goes into free pages (copy-on-write), which apart from a few at the
//! front lie after the first image, from about 4.6 to 9.2 GiB, and a
//! read-only reopen checks everything once more.
//!
//! It holds about 5 GiB in memory, needs about 15 GiB of free disk and
//! writes about 14 GiB (the WAL, then two images), so it is ignored in
//! normal runs; it takes about a minute. Run it in release mode before a
//! release:
//!
//! ```bash
//! cargo test --release -p grafeo-engine --features full,arrow-export \
//!     --test large_database -- --ignored --nocapture
//! ```
//!
//! It writes into a temporary directory under `GRAFEO_LARGE_TEST_DIR` (the
//! system's temporary directory when unset) and refuses to start when that
//! drive has less than three times the database size free. With
//! `--nocapture` it prints the time, the file size and the process's working
//! set at each step (on Windows and Linux), and where each section of each
//! image lies.

#![cfg(all(feature = "lpg", feature = "grafeo-file", feature = "wal"))]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use grafeo_common::storage::SectionSource;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::file::v3::ImageReader;

const PEOPLE: [&str; 5] = ["Alix", "Gus", "Vincent", "Mia", "Jules"];
const CITIES: [&str; 4] = ["Amsterdam", "Berlin", "Paris", "Prague"];

/// Nodes of the default graph: three row groups of the node table.
const NODES: usize = 190_388;
/// Only nodes below this id (the first two row groups) carry a large value:
/// the third row group holds small values only, so all of its chunks lie
/// past 4 GiB.
const LARGE_BELOW: usize = 2 * 65_536;
/// Every 88th node, from node 3 on, carries a large value.
const LARGE_STRIDE: usize = 88;
/// The sizes of the large values, in turn: below the 1 MiB chunk cap (several
/// in a chunk) and above it (a chunk of their own).
const LARGE_SIZES: [usize; 4] = [
    600 * 1024 + 3,
    (1 << 20) - 19,
    3 * (1 << 20) + 88,
    8 * (1 << 20) + 19,
];
/// The last node with a large value, which the second round gives another.
const LAST_LARGE: usize = (LARGE_BELOW - 1 - 3) / LARGE_STRIDE * LARGE_STRIDE + 3;
/// The KNOWS edge the second round deletes.
const DELETED_EDGE: usize = 88;
/// Cities of the named graph `trips`.
const STOPS: usize = 88;
/// What the database holds besides its large values, at most.
const SMALL_DATA: u64 = 256 << 20;

// --- What the database holds ------------------------------------------------------

/// Which state of the database a check expects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Round {
    /// As written before the first `close()`.
    Written,
    /// After [`change`]: node [`LAST_LARGE`] has another large value, KNOWS
    /// edge [`DELETED_EDGE`] is gone and Mia joined with a photo.
    Changed,
}

/// The ids of what [`write`] and [`change`] created.
struct Ids {
    /// The nodes of the default graph, node `i` at `i`.
    nodes: Vec<NodeId>,
    /// `knows[k]` goes from node `k` to node `k + 1`.
    knows: Vec<EdgeId>,
    /// The cities of the named graph `trips`.
    stops: Vec<NodeId>,
    /// `routes[j]` goes from stop `j` to stop `j + 1`.
    routes: Vec<EdgeId>,
    /// The node [`change`] adds.
    added: Option<NodeId>,
}

/// `len` bytes of xorshift output seeded by `seed`: no two large values, and
/// no two pages of one value, are alike, so a chunk read from the wrong place
/// cannot compare equal.
fn noise(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut bytes = vec![0u8; len];
    for word in bytes.chunks_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        word.copy_from_slice(&state.to_le_bytes()[..word.len()]);
    }
    bytes
}

/// `len` ASCII letters from `seed`, starting with `header`.
fn text(header: &str, seed: u64, len: usize) -> String {
    let mut bytes = noise(seed, len);
    for byte in &mut bytes {
        *byte = b'a' + *byte % 26;
    }
    bytes[..header.len()].copy_from_slice(header.as_bytes());
    String::from_utf8(bytes).expect("ASCII letters")
}

/// The large value of node `i` in `round`, if it has one: four `bio` texts,
/// then four `photo`s of bytes, and so on, of the [`LARGE_SIZES`] in turn.
fn large_value(i: usize, round: Round) -> Option<(&'static str, Value)> {
    if i >= LARGE_BELOW || i % LARGE_STRIDE != 3 {
        return None;
    }
    let k = i / LARGE_STRIDE;
    let seed = u64::try_from(i).unwrap();
    let (seed, len) = if round == Round::Changed && i == LAST_LARGE {
        (seed + 88_000_000, LARGE_SIZES[3])
    } else {
        (seed, LARGE_SIZES[k % 4])
    };
    Some(if (k / 4).is_multiple_of(2) {
        let header = format!("{} {i}, {}: ", PEOPLE[i % 5], CITIES[i % 4]);
        ("bio", Value::from(text(&header, seed, len)))
    } else {
        ("photo", Value::Bytes(noise(seed, len).into()))
    })
}

/// The bytes of the large values [`write`] writes.
fn large_bytes() -> u64 {
    (3..LARGE_BELOW)
        .step_by(LARGE_STRIDE)
        .map(|i| u64::try_from(LARGE_SIZES[i / LARGE_STRIDE % 4]).unwrap())
        .sum()
}

/// Eight distinct components for node `i` (below 88^3).
fn embedding(i: usize) -> Value {
    let part = |x: usize| f32::from(u16::try_from(x % 88).unwrap());
    Value::Vector(
        vec![
            part(i),
            part(i / 88),
            part(i / 7_744),
            part(i % 19),
            part(i % 3),
            3.0,
            19.0,
            88.0,
        ]
        .into(),
    )
}

/// Node `i` of the default graph in `round`: every fourth a City with a name,
/// the others Persons with a name, an age and, in the first two row groups,
/// every 88th a large value; in the third row group every third Person has
/// an embedding and every nineteenth a motto.
fn node(i: usize, round: Round) -> (&'static str, Vec<(&'static str, Value)>) {
    let mut properties = Vec::new();
    let label = if i.is_multiple_of(4) {
        properties.push(("name", Value::from(format!("{} {i}", CITIES[i / 4 % 4]))));
        "City"
    } else {
        properties.push(("name", Value::from(format!("{} {i}", PEOPLE[i % 5]))));
        properties.push(("age", Value::Int64(i64::try_from(i % 88).unwrap())));
        if i >= LARGE_BELOW && i.is_multiple_of(3) {
            properties.push(("embedding", embedding(i)));
        }
        if i >= LARGE_BELOW && i.is_multiple_of(19) {
            let motto = format!(
                "{} cycles from {} to {}",
                PEOPLE[i % 5],
                CITIES[i % 4],
                CITIES[i / 4 % 4]
            );
            properties.push(("motto", Value::from(motto)));
        }
        "Person"
    };
    properties.extend(large_value(i, round));
    (label, properties)
}

/// The node [`change`] adds.
fn added_node() -> (&'static str, Vec<(&'static str, Value)>) {
    (
        "Person",
        vec![
            ("name", Value::from("Mia, added")),
            (
                "photo",
                Value::Bytes(noise(3_019_088, LARGE_SIZES[2]).into()),
            ),
        ],
    )
}

/// The properties of KNOWS edge `k`: every third has a `since` year.
fn knows_properties(k: usize) -> Vec<(&'static str, Value)> {
    if k.is_multiple_of(3) {
        vec![("since", Value::Int64(1988 + i64::try_from(k % 19).unwrap()))]
    } else {
        Vec::new()
    }
}

/// The name of stop `j` of the named graph.
fn stop_name(j: usize) -> Value {
    Value::from(format!("{} stop {j}", CITIES[j % 4]))
}

/// The length of route `j` of the named graph.
fn route_km(j: usize) -> Value {
    Value::Int64(3 + 19 * i64::try_from(j).unwrap())
}

/// Writes the database of [`Round::Written`] and its indexes.
fn write(db: &GrafeoDB) -> Ids {
    let mut nodes = Vec::with_capacity(NODES);
    for i in 0..NODES {
        let (label, properties) = node(i, Round::Written);
        let id = db
            .create_node_with_props(&[label], properties)
            .unwrap_or_else(|error| panic!("node {i}: {error}"));
        nodes.push(id);
    }
    let knows = (0..NODES - 1)
        .map(|k| {
            db.create_edge_with_props(nodes[k], nodes[k + 1], "KNOWS", knows_properties(k))
                .unwrap_or_else(|error| panic!("KNOWS edge {k}: {error}"))
        })
        .collect();

    assert!(db.create_graph("trips").unwrap(), "trips is new");
    let trips = db.graph("trips").unwrap();
    let stops: Vec<NodeId> = (0..STOPS)
        .map(|j| {
            trips
                .create_node_with_props(&["City"], [("name", stop_name(j))])
                .unwrap()
        })
        .collect();
    let routes = (0..STOPS - 1)
        .map(|j| {
            trips
                .create_edge_with_props(stops[j], stops[j + 1], "ROUTE", [("km", route_km(j))])
                .unwrap()
        })
        .collect();

    #[cfg(feature = "vector-index")]
    db.create_vector_index(
        "Person",
        "embedding",
        Some(8),
        Some("euclidean"),
        None,
        None,
        None,
    )
    .unwrap();
    #[cfg(feature = "text-index")]
    db.create_text_index("Person", "motto").unwrap();

    Ids {
        nodes,
        knows,
        stops,
        routes,
        added: None,
    }
}

/// The changes of [`Round::Changed`].
fn change(db: &GrafeoDB, ids: &mut Ids) {
    let (key, value) = large_value(LAST_LARGE, Round::Changed).expect("a large value");
    db.set_node_property(ids.nodes[LAST_LARGE], key, value)
        .unwrap();
    assert!(
        db.delete_edge(ids.knows[DELETED_EDGE]).unwrap(),
        "KNOWS edge {DELETED_EDGE} existed"
    );
    let (label, properties) = added_node();
    ids.added = Some(db.create_node_with_props(&[label], properties).unwrap());
}

// --- Checks -------------------------------------------------------------------------

/// The kind and length of a text or bytes value, or the value itself.
fn describe(value: &Value) -> String {
    match value {
        Value::String(text) => format!("a text of {} bytes", text.len()),
        Value::Bytes(bytes) => format!("{} bytes", bytes.len()),
        other => format!("{other:?}"),
    }
}

/// Compares properties value by value. A large value that differs is
/// described by its length and the offset of its first different byte, not
/// printed.
fn assert_properties<'a>(
    found: impl Iterator<Item = (&'a PropertyKey, &'a Value)>,
    expected: Vec<(&'static str, Value)>,
    what: &str,
) {
    let found: BTreeMap<&str, &Value> = found
        .filter(|(_, value)| !value.is_null())
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    let expected: BTreeMap<&str, Value> = expected.into_iter().collect();
    assert_eq!(
        found.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>(),
        "{what}: property keys"
    );
    for (key, value) in &expected {
        let got = found[key];
        if got == value {
            continue;
        }
        let first_difference = match (got, value) {
            (Value::String(a), Value::String(b)) => Some((a.as_bytes(), b.as_bytes())),
            (Value::Bytes(a), Value::Bytes(b)) => Some((&a[..], &b[..])),
            _ => None,
        }
        .map_or_else(String::new, |(a, b)| {
            let at = a.iter().zip(b).position(|(x, y)| x != y);
            format!(", first different byte at {at:?}")
        });
        panic!(
            "{what}, property {key}: found {}, expected {}{first_difference}",
            describe(got),
            describe(value)
        );
    }
}

/// Checks the label and properties of a node.
fn assert_node(
    found: Option<grafeo_core::graph::lpg::Node>,
    (label, properties): (&'static str, Vec<(&'static str, Value)>),
    what: &str,
) {
    let found = found.unwrap_or_else(|| panic!("{what} is missing"));
    let labels: Vec<&str> = found.labels.iter().map(|label| label.as_str()).collect();
    assert_eq!(labels, [label], "{what}: labels");
    assert_properties(found.properties.iter(), properties, what);
}

/// Checks the endpoints, type and properties of an edge.
fn assert_edge(
    found: Option<grafeo_core::graph::lpg::Edge>,
    (src, dst, edge_type): (NodeId, NodeId, &str),
    properties: Vec<(&'static str, Value)>,
    what: &str,
) {
    let found = found.unwrap_or_else(|| panic!("{what} is missing"));
    assert_eq!(
        (found.src, found.dst, found.edge_type.as_str()),
        (src, dst, edge_type),
        "{what}: endpoints and type"
    );
    assert_properties(found.properties.iter(), properties, what);
}

/// Checks the counts and every node and edge of both graphs against
/// `round`, the large values byte for byte.
fn check(db: &GrafeoDB, ids: &Ids, round: Round, what: &str) {
    let changed = usize::from(round == Round::Changed);
    assert_eq!(db.node_count(), NODES + changed, "{what}: nodes");
    assert_eq!(db.edge_count(), NODES - 1 - changed, "{what}: edges");
    for (i, &id) in ids.nodes.iter().enumerate() {
        assert_node(
            db.get_node(id),
            node(i, round),
            &format!("{what}: node {i}"),
        );
    }
    if round == Round::Changed {
        let added = ids.added.expect("the added node");
        assert_node(db.get_node(added), added_node(), &format!("{what}: Mia"));
    }
    for (k, &id) in ids.knows.iter().enumerate() {
        let edge = db.get_edge(id);
        let what = format!("{what}: KNOWS edge {k}");
        if round == Round::Changed && k == DELETED_EDGE {
            assert!(edge.is_none(), "{what} was deleted");
            continue;
        }
        let ends = (ids.nodes[k], ids.nodes[k + 1], "KNOWS");
        assert_edge(edge, ends, knows_properties(k), &what);
    }

    let trips = db
        .graph("trips")
        .unwrap_or_else(|error| panic!("{what}: the named graph: {error}"));
    for (j, &id) in ids.stops.iter().enumerate() {
        let found = trips.get_node(id).unwrap();
        let expected = ("City", vec![("name", stop_name(j))]);
        assert_node(found, expected, &format!("{what}: stop {j}"));
    }
    for (j, &id) in ids.routes.iter().enumerate() {
        let found = trips.get_edge(id).unwrap();
        let ends = (ids.stops[j], ids.stops[j + 1], "ROUTE");
        let properties = vec![("km", route_km(j))];
        assert_edge(found, ends, properties, &format!("{what}: route {j}"));
    }
}

/// What the indexes of a database hold, in the builds that have them.
#[derive(PartialEq)]
struct IndexState {
    /// The HNSW graph of the vector index: a rebuild draws other levels, so
    /// an equal topology after a reopen was read from the section.
    #[cfg(feature = "vector-index")]
    topology: Topology,
    /// What the text index finds for "Prague".
    #[cfg(feature = "text-index")]
    mottos: Vec<(NodeId, u64)>,
}

#[cfg(feature = "vector-index")]
type Topology = (Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>);

/// The HNSW graph of the vector index of `db`.
#[cfg(feature = "vector-index")]
fn topology(db: &GrafeoDB) -> Topology {
    use grafeo_core::index::vector::VectorIndexKind;

    let index = db
        .store()
        .get_vector_index("Person", "embedding")
        .expect("the vector index");
    match &*index {
        VectorIndexKind::Hnsw(hnsw) => hnsw.snapshot_topology(),
        VectorIndexKind::Quantized(_) => panic!("the index was created without quantization"),
    }
}

/// Every match of "Prague" in the mottos, by id, with the bits of its score.
#[cfg(feature = "text-index")]
fn prague_mottos(matches: Vec<(NodeId, f64)>) -> Vec<(NodeId, u64)> {
    let mut matches: Vec<(NodeId, u64)> = matches
        .into_iter()
        .map(|(id, score)| (id, score.to_bits()))
        .collect();
    matches.sort_unstable();
    matches
}

/// The matches of the text index of `db`.
#[cfg(feature = "text-index")]
fn text_index_matches(db: &GrafeoDB) -> Vec<(NodeId, u64)> {
    prague_mottos(
        db.text_search("Person", "motto", "Prague", 100_000, None)
            .unwrap(),
    )
}

/// The matches of the text index decoded from the section of the active
/// image itself. When the section does not decode, a reopen builds the index
/// from the data instead, with the same matches: only decoding the section
/// shows that it holds the index.
#[cfg(feature = "text-index")]
fn text_section_matches(db: &GrafeoDB) -> Vec<(NodeId, u64)> {
    use std::sync::Arc;

    use grafeo_common::storage::{Section, SectionType};
    use grafeo_core::index::text::{BM25Config, InvertedIndex, TextIndexSection};

    let shell = Arc::new(parking_lot::RwLock::new(InvertedIndex::new(
        BM25Config::default(),
    )));
    db.file_manager()
        .expect("a database file")
        .read_image(|image| {
            let source = image
                .section_source(SectionType::TextIndex)
                .expect("a text index section");
            TextIndexSection::new(vec![("Person:motto".to_string(), Arc::clone(&shell))])
                .read_from(&*source)
        })
        .unwrap_or_else(|error| panic!("the text index section decodes: {error}"));
    let matches = shell.read().search("Prague", 100_000);
    assert!(!matches.is_empty(), "the decoded text index finds Prague");
    prague_mottos(matches)
}

// --- The file -----------------------------------------------------------------------

/// Where the chunks of one section lie in an image.
#[derive(Debug, Default)]
struct Placement {
    chunks: usize,
    bytes: u64,
    /// The lowest and the highest offset of a chunk with bytes.
    first: Option<u64>,
    last: Option<u64>,
}

/// Reads the directory of the active image of the file at `path` and every
/// chunk it lists, each against its checksum (#392 failed here, with a CRC
/// mismatch), and returns where each section's chunks lie, by section name.
fn image_layout(path: &Path, what: &str) -> BTreeMap<String, Placement> {
    let manager = GrafeoFileManager::open_read_only(path, None)
        .unwrap_or_else(|error| panic!("{what}: the file opens: {error}"));
    let stats = manager.image_stats().unwrap();
    let root = manager.active_header().root;
    // The manager's shared lock keeps writers out while this handle reads.
    let mut file = File::open(path).unwrap();
    let reader = ImageReader::open(&mut file, root, None)
        .unwrap_or_else(|error| panic!("{what}: the directory reads: {error}"));

    let mut layout: BTreeMap<String, Placement> = BTreeMap::new();
    let mut section_types = Vec::new();
    for entry in reader.entries() {
        let placement = layout
            .entry(format!("{:?}", entry.section_type))
            .or_default();
        placement.chunks += 1;
        placement.bytes += entry.length;
        if entry.length > 0 {
            placement.first = Some(
                placement
                    .first
                    .map_or(entry.offset, |at| at.min(entry.offset)),
            );
            placement.last = Some(
                placement
                    .last
                    .map_or(entry.offset, |at| at.max(entry.offset)),
            );
        }
        if !section_types.contains(&entry.section_type) {
            section_types.push(entry.section_type);
        }
    }
    for section_type in section_types {
        let section = reader.section(section_type).expect("a listed section");
        for index in 0..section.chunks().len() {
            section.fetch(index).unwrap_or_else(|error| {
                panic!("{what}: chunk {index} of section {section_type:?} reads back: {error}")
            });
        }
    }

    println!(
        "[large_database] {what}: {} chunks in {} directory blocks, {} pages",
        stats.chunks, stats.directory_blocks, stats.pages
    );
    let at = |offset: Option<u64>| offset.map_or_else(|| "-".to_string(), gib);
    for (name, placement) in &layout {
        println!(
            "[large_database]   {name:<16} {:>6} chunks {:>10}, from {} to {}",
            placement.chunks,
            gib(placement.bytes),
            at(placement.first),
            at(placement.last)
        );
    }
    layout
}

/// Asserts that the LPG section alone holds more than 4 GiB and that it and
/// the index sections written after it have chunks past 4 GiB.
fn assert_past_4_gib(layout: &BTreeMap<String, Placement>, what: &str) {
    let limit = u64::from(u32::MAX);
    let lpg = &layout["LpgStore"];
    assert!(
        lpg.bytes > limit,
        "{what}: the LPG section alone holds more than 4 GiB: {layout:?}"
    );
    let mut past = vec!["LpgStore"];
    if cfg!(feature = "vector-index") {
        past.push("VectorStore");
    }
    if cfg!(feature = "text-index") {
        past.push("TextIndex");
    }
    for name in past {
        let last = layout.get(name).and_then(|placement| placement.last);
        assert!(
            last.is_some_and(|offset| offset > limit),
            "{what}: the {name} section has chunks past 4 GiB: {layout:?}"
        );
    }
}

// --- The machine --------------------------------------------------------------------

/// `bytes` in GiB with two decimals.
fn gib(bytes: u64) -> String {
    format!(
        "{}.{:02} GiB",
        bytes >> 30,
        ((bytes & ((1 << 30) - 1)) * 100) >> 30
    )
}

/// The bytes free for this user on the drive of `dir`.
#[cfg(windows)]
fn free_bytes(dir: &Path) -> Result<u64, String> {
    let dir = dir.to_str().ok_or("the path is not UTF-8")?;
    let script = format!(
        "(New-Object System.IO.DriveInfo '{}').AvailableFreeSpace",
        dir.replace('\'', "''")
    );
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .map_err(|error| format!("powershell: {error}"))?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim()
        .parse()
        .map_err(|error| format!("powershell printed {text:?}: {error}"))
}

/// The bytes free for this user on the drive of `dir`.
#[cfg(unix)]
fn free_bytes(dir: &Path) -> Result<u64, String> {
    let output = Command::new("df")
        .arg("-Pk")
        .arg(dir)
        .output()
        .map_err(|error| format!("df: {error}"))?;
    let text = String::from_utf8_lossy(&output.stdout);
    let available = text
        .lines()
        .nth(1)
        .and_then(|line| line.split_whitespace().nth(3))
        .ok_or_else(|| format!("df printed {text:?}"))?;
    available
        .parse::<u64>()
        .map(|kib| kib * 1024)
        .map_err(|error| format!("df printed {text:?}: {error}"))
}

#[cfg(not(any(unix, windows)))]
fn free_bytes(_dir: &Path) -> Result<u64, String> {
    Err("no way to ask for the free space on this platform".to_string())
}

/// The working set of this process and its peak so far, in bytes, from
/// `Get-Process` (no `unsafe` code, no new dependency).
#[cfg(windows)]
fn memory() -> Option<(u64, u64)> {
    let script = format!(
        "Get-Process -Id {} | ForEach-Object {{ $_.WorkingSet64; $_.PeakWorkingSet64 }}",
        std::process::id()
    );
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let mut numbers = text.split_whitespace().map(str::parse::<u64>);
    Some((numbers.next()?.ok()?, numbers.next()?.ok()?))
}

/// The resident set of this process and its peak so far, in bytes.
#[cfg(target_os = "linux")]
fn memory() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let field = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| {
                rest.trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse::<u64>()
                    .ok()
            })
            .map(|kib| kib * 1024)
    };
    Some((field("VmRSS:")?, field("VmHWM:")?))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn memory() -> Option<(u64, u64)> {
    None
}

/// Prints the time since `started`, the size of the file at `path` and the
/// working set after `step`.
fn report(started: Instant, path: &Path, step: &str) {
    let file = std::fs::metadata(path).map_or_else(
        |_| "no file".to_string(),
        |metadata| format!("file {}", gib(metadata.len())),
    );
    let memory = memory().map_or_else(
        || "working set not available on this platform".to_string(),
        |(now, peak)| format!("working set {}, peak {}", gib(now), gib(peak)),
    );
    println!(
        "[large_database] {:>7.1} s  {step}: {file}, {memory}",
        started.elapsed().as_secs_f64()
    );
}

// --- The test -----------------------------------------------------------------------

/// A database whose LPG section alone holds more than 4 GiB is written with
/// `close()`, every chunk reads back against its checksum, and a reopen holds
/// every node, edge and index as written; a second checkpoint puts its image
/// after the first, and a read-only reopen holds the changes.
#[test]
#[ignore = "writes more than 4 GiB; run before releases with --release"]
fn a_database_over_4_gib_round_trips_through_checkpoints_and_reopens() {
    let large = large_bytes();
    assert!(
        large > u64::from(u32::MAX) + SMALL_DATA,
        "the large values alone pass 4 GiB with room: {large} bytes"
    );
    let base =
        std::env::var_os("GRAFEO_LARGE_TEST_DIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let needed = 3 * (large + SMALL_DATA);
    let free = free_bytes(&base).unwrap_or_else(|error| {
        panic!(
            "refusing to run: cannot tell the free space in {}: {error}",
            base.display()
        )
    });
    assert!(
        free >= needed,
        "refusing to run: {} free in {}, {} needed (three times the database); point \
         GRAFEO_LARGE_TEST_DIR at a directory on a drive with room",
        gib(free),
        base.display(),
        gib(needed)
    );
    let dir = tempfile::tempdir_in(&base).unwrap();
    let path = dir.path().join("large.grafeo");
    let started = Instant::now();
    let step = |what: &str| report(started, &path, what);
    step(&format!("start, {} of large values", gib(large)));

    let (mut ids, written) = {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let ids = write(&db);
        step("written");
        let written = IndexState {
            #[cfg(feature = "vector-index")]
            topology: topology(&db),
            #[cfg(feature = "text-index")]
            mottos: text_index_matches(&db),
        };
        db.close()
            .expect("the checkpoint of more than 4 GiB succeeds");
        step("closed");
        (ids, written)
    };
    assert!(
        !dir.path().join("large.grafeo.wal").exists(),
        "the checkpoint removed the sidecar WAL: the reopen reads the file"
    );
    let first = image_layout(&path, "first image");
    assert_past_4_gib(&first, "first image");
    step("first image read back");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap_or_else(|error| {
            panic!(
                "the database over 4 GiB reopens (#392 failed here, with a CRC mismatch): {error}"
            )
        });
        step("reopened");
        check(&db, &ids, Round::Written, "reopened");
        let reopened = IndexState {
            #[cfg(feature = "vector-index")]
            topology: topology(&db),
            #[cfg(feature = "text-index")]
            mottos: text_section_matches(&db),
        };
        assert!(
            reopened == written,
            "reopened: the vector index has the topology written (read from its section past 4 \
             GiB, not rebuilt) and the text index section finds what the index found"
        );
        #[cfg(feature = "text-index")]
        assert_eq!(
            text_index_matches(&db),
            written.mottos,
            "reopened: the text index"
        );
        step("checked");
        change(&db, &mut ids);
        db.close().expect("the second checkpoint succeeds");
        step("closed again");
    }
    let second = image_layout(&path, "second image");
    assert_past_4_gib(&second, "second image");
    let first_end = first.values().filter_map(|placement| placement.last).max();
    assert!(
        second["LpgStore"].last > first_end,
        "the second image goes after the first: {first:?} then {second:?}"
    );
    step("second image read back");

    let db = GrafeoDB::open_read_only(&path).unwrap_or_else(|error| {
        panic!("the database reopens after the second checkpoint: {error}")
    });
    step("reopened read-only");
    check(&db, &ids, Round::Changed, "after the second checkpoint");
    let reopened = IndexState {
        #[cfg(feature = "vector-index")]
        topology: topology(&db),
        #[cfg(feature = "text-index")]
        mottos: text_section_matches(&db),
    };
    assert!(
        reopened == written,
        "after the second checkpoint: the indexes are the ones written"
    );
    drop(db);
    step("checked again");
}
