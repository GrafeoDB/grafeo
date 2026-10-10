//! Databases written in small chunks reopen with everything they held (#555,
//! #392).
//!
//! The tests shrink the chunk caps of their thread with `with_chunk_caps`, so
//! the paths that cut a section into many chunks, chain directory blocks and
//! reuse free pages run on small data, and reopen with the default caps. A
//! section reads the caps of the thread that builds it: `close()` and
//! `wal_checkpoint()` build and write the sections on the calling thread, so
//! the checkpoints of these tests run with their caps. (No test here starts
//! the checkpoint timer, whose thread would use the default caps.)
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test chunked_sections
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "grafeo-file",
    feature = "wal"
))]

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;

use grafeo_common::storage::ChunkCaps;
use grafeo_common::storage::value_codec::encode_value;
use grafeo_common::testing::chunk_caps::with_chunk_caps;
use grafeo_common::types::{
    Date, Duration, EdgeId, NodeId, PropertyKey, Time, Timestamp, Value, ZonedDatetime,
};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::v3::directory::ENTRIES_PER_BLOCK;
use proptest::prelude::*;
use proptest::sample::Index;

/// Three rows and 1 KiB per chunk: a few nodes already fill several chunks.
const TINY: ChunkCaps = ChunkCaps {
    max_rows: 3,
    max_bytes: 1024,
};

const PEOPLE: [&str; 5] = ["Alix", "Gus", "Vincent", "Mia", "Jules"];
const CITIES: [&str; 4] = ["Amsterdam", "Berlin", "Paris", "Prague"];

// --- What a database holds ------------------------------------------------------

/// Every node and edge of `db` with its labels or type and its properties,
/// graph by graph (the default graph, then the named graphs by name), and the
/// RDF triples of every graph, one line each.
///
/// A property line holds the value's lossless encoding next to its debug
/// form, so floats compare by their bits, times and zoned datetimes with
/// their offsets, and counters whatever the order of their replicas. A
/// property whose value is null is left out: it does not exist, and 0.6 does
/// not write it (a 0.5.x file can hold one, which a store without `temporal`
/// lists after a 0.5.x load).
fn dump(db: &GrafeoDB) -> Vec<String> {
    let mut lines = Vec::new();
    let store = db.store();
    dump_graph(&mut lines, "default graph", &store);
    let mut names = db.list_graphs();
    names.sort();
    for name in names {
        let graph = store
            .graph(&name)
            .unwrap_or_else(|| panic!("the listed graph {name} exists"));
        dump_graph(&mut lines, &format!("graph {name}"), &graph);
    }
    #[cfg(feature = "triple-store")]
    dump_triples(&mut lines, db);
    lines
}

fn dump_graph(lines: &mut Vec<String>, title: &str, store: &LpgStore) {
    lines.push(title.to_string());
    for id in store.all_node_ids() {
        let Some(node) = store.get_node(id) else {
            continue;
        };
        let mut labels: Vec<&str> = node.labels.iter().map(|label| label.as_str()).collect();
        labels.sort_unstable();
        lines.push(format!("  node {} {labels:?}", id.as_u64()));
        push_properties(lines, node.properties.iter());
    }
    for raw in 0..store.next_edge_id() {
        let Some(edge) = store.get_edge(EdgeId::new(raw)) else {
            continue;
        };
        lines.push(format!(
            "  edge {raw}: {} -[{}]-> {}",
            edge.src.as_u64(),
            edge.edge_type,
            edge.dst.as_u64()
        ));
        push_properties(lines, edge.properties.iter());
    }
}

fn push_properties<'a>(
    lines: &mut Vec<String>,
    properties: impl Iterator<Item = (&'a PropertyKey, &'a Value)>,
) {
    let mut sorted: Vec<(&PropertyKey, &Value)> =
        properties.filter(|(_, value)| !value.is_null()).collect();
    sorted.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
    for (key, value) in sorted {
        lines.push(format!(
            "    {} = {} [{}]",
            key.as_str(),
            readable(value),
            encoded(value)
        ));
    }
}

/// `value`'s debug form, with the replicas of a counter in name order.
fn readable(value: &Value) -> String {
    let sorted = |counts: &HashMap<String, u64>| -> BTreeMap<String, u64> {
        counts
            .iter()
            .map(|(replica, count)| (replica.clone(), *count))
            .collect()
    };
    match value {
        Value::GCounter(counts) => format!("GCounter({:?})", sorted(counts)),
        Value::OnCounter { pos, neg } => format!(
            "OnCounter {{ pos: {:?}, neg: {:?} }}",
            sorted(pos),
            sorted(neg)
        ),
        other => format!("{other:?}"),
    }
}

/// The lossless encoding of `value` (the value codec of the column chunks),
/// in hex.
fn encoded(value: &Value) -> String {
    let mut bytes = Vec::new();
    encode_value(value, &mut bytes).unwrap_or_else(|error| panic!("{value:?} encodes: {error}"));
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(hex, "{byte:02x}").unwrap();
    }
    hex
}

#[cfg(feature = "triple-store")]
fn dump_triples(lines: &mut Vec<String>, db: &GrafeoDB) {
    let rdf = db.rdf_store();
    push_triples(lines, "rdf default graph", rdf);
    let mut names = rdf.graph_names();
    names.sort();
    for name in names {
        let graph = rdf
            .graph(&name)
            .unwrap_or_else(|| panic!("the listed RDF graph {name} exists"));
        push_triples(lines, &format!("rdf graph {name}"), &graph);
    }
}

#[cfg(feature = "triple-store")]
fn push_triples(lines: &mut Vec<String>, title: &str, store: &grafeo_core::graph::rdf::RdfStore) {
    lines.push(title.to_string());
    let mut triples: Vec<String> = store
        .triples()
        .iter()
        .map(|triple| format!("  {triple}"))
        .collect();
    triples.sort();
    lines.extend(triples);
}

/// The first difference between two dumps, after the lines before it (cut to
/// 160 characters), or `None` when they are the same.
fn difference(expected: &[String], found: &[String]) -> Option<String> {
    let at = expected
        .iter()
        .zip(found)
        .position(|(expected, found)| expected != found)
        .or_else(|| (expected.len() != found.len()).then(|| expected.len().min(found.len())))?;
    let nothing = "(nothing)".to_string();
    Some(format!(
        "{} lines expected, {} found; the first difference is line {at}, after\n{}\n\
         expected: {}\n   found: {}",
        expected.len(),
        found.len(),
        expected[at.saturating_sub(3)..at]
            .iter()
            .map(|line| line.chars().take(160).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n"),
        expected.get(at).unwrap_or(&nothing),
        found.get(at).unwrap_or(&nothing),
    ))
}

fn assert_same(expected: &[String], found: &[String], what: &str) {
    if let Some(difference) = difference(expected, found) {
        panic!("{what}: {difference}");
    }
}

/// How many chunks each section of the active image of `db`'s file has, by
/// section name.
#[cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "vector-index",
    feature = "text-index"
))]
fn chunk_counts(db: &GrafeoDB) -> BTreeMap<String, usize> {
    use grafeo_common::storage::SectionType;

    let fm = db.file_manager().expect("a database file");
    fm.read_image(|image| {
        Ok((0..=u8::MAX)
            .filter_map(SectionType::from_u8)
            .filter_map(|section_type| {
                image
                    .section_source(section_type)
                    .map(|source| (format!("{section_type:?}"), source.chunks().len()))
            })
            .collect())
    })
    .unwrap()
}

// --- A large image ----------------------------------------------------------------

/// Writes `count` nodes into the default graph and a tenth of that into each
/// of three named graphs, as [`write_graph`] does.
fn populate(db: &GrafeoDB, count: usize) {
    write_graph(db, count);
    for name in ["museums", "trips", "work"] {
        assert!(db.create_graph(name).unwrap(), "{name} is new");
        db.set_current_graph(Some(name)).unwrap();
        write_graph(db, count / 10);
    }
    db.set_current_graph(None).unwrap();
}

/// Writes `count` nodes into the graph `db` works in: every fourth a City
/// with a name, a population and a last-seen time, the others Persons with a
/// name, an age, a height, a birth date, a last-seen time and, on every other
/// one, tags; node 3 gets a bio of 5 KiB. Each node KNOWS the next one (every
/// third edge with a `since` year). Then every seventh node is deleted with
/// its edges.
fn write_graph(db: &GrafeoDB, count: usize) {
    let mut nodes = Vec::with_capacity(count);
    for i in 0..count {
        let n = i64::try_from(i).unwrap();
        let mut properties = vec![(
            "seen",
            Value::Timestamp(Timestamp::from_micros(1_696_500_000_123_457 + 19 * n)),
        )];
        let label = if i % 4 == 0 {
            properties.push(("name", Value::from(format!("{} {i}", CITIES[i / 4 % 4]))));
            properties.push(("population", Value::Int64(88 * n)));
            "City"
        } else {
            properties.push(("name", Value::from(format!("{} {i}", PEOPLE[i % 5]))));
            properties.push(("age", Value::Int64(n % 88)));
            properties.push(("height", Value::Float64(1.5 + (n % 88) as f64 / 100.0)));
            properties.push((
                "born",
                Value::Date(Date::from_days(-3 * i32::try_from(i).unwrap())),
            ));
            if i % 2 == 0 {
                properties.push((
                    "tags",
                    Value::List(Arc::from(vec![
                        Value::from("jazz"),
                        Value::from(CITIES[i % 4]),
                    ])),
                ));
            }
            "Person"
        };
        if i == 3 {
            properties.push(("bio", Value::from("Amsterdam ".repeat(512))));
        }
        nodes.push(db.create_node_with_props(&[label], properties).unwrap());
    }
    // `knows[k]` goes from node k to node k + 1.
    let mut knows = Vec::with_capacity(count);
    for k in 1..count {
        let properties = if k % 3 == 0 {
            vec![("since", Value::Int64(1988 + i64::try_from(k % 19).unwrap()))]
        } else {
            Vec::new()
        };
        knows.push(
            db.create_edge_with_props(nodes[k - 1], nodes[k], "KNOWS", properties)
                .unwrap(),
        );
    }
    for i in (6..count).step_by(7) {
        assert!(db.delete_edge(knows[i - 1]).unwrap());
        if let Some(&next) = knows.get(i) {
            assert!(db.delete_edge(next).unwrap());
        }
        assert!(db.delete_node(nodes[i]).unwrap());
    }
}

/// An image of some 5,000 nodes in chunks of three rows has more chunks than
/// one directory block lists, so its directory is a chain of blocks, and it
/// reopens with every node and edge as it was written.
#[test]
fn a_large_multi_chunk_image_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.grafeo");
    let expected = with_chunk_caps(TINY, || {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        populate(&db, 5_000);
        let before = dump(&db);
        // The final checkpoint runs on this thread, with the tiny caps.
        db.close().unwrap();
        before
    });
    assert!(
        expected.len() > 30_000,
        "the dump lists the nodes, edges and properties: {} lines",
        expected.len()
    );

    let db = GrafeoDB::open_read_only(&path).unwrap();
    let stats = db.file_manager().unwrap().image_stats().unwrap();
    assert!(
        stats.chunks > ENTRIES_PER_BLOCK && stats.directory_blocks >= 2,
        "more chunks than one directory block lists ({ENTRIES_PER_BLOCK}): {stats:?}"
    );
    assert_same(&expected, &dump(&db), "reopened");
}

/// Each checkpoint writes a new image next to the active one, into the pages
/// of the image before it (which nothing reaches any more), and the file is
/// cut after the new image: ten checkpoints of an unchanged database in
/// hundreds of chunks keep the file at about two images.
#[test]
fn repeated_checkpoints_of_a_multi_chunk_image_reuse_space() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reuse.grafeo");
    with_chunk_caps(TINY, || {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        for i in 0..600 {
            db.create_node_with_props(
                &["Person"],
                [
                    ("name", Value::from(format!("{} {i}", PEOPLE[i % 5]))),
                    ("age", Value::Int64(i64::try_from(i % 88).unwrap())),
                    ("city", Value::from(CITIES[i % 4])),
                ],
            )
            .unwrap();
        }
        let fm = Arc::clone(db.file_manager().unwrap());
        let sizes: Vec<u64> = (0..10)
            .map(|_| {
                db.wal_checkpoint().unwrap();
                fm.file_size().unwrap()
            })
            .collect();
        let stats = fm.image_stats().unwrap();
        assert!(
            stats.chunks > 600,
            "the image is cut into hundreds of chunks: {stats:?}"
        );
        assert!(
            sizes[9] <= 2 * sizes[1],
            "the tenth checkpoint reuses the pages of the images before: file sizes {sizes:?}"
        );
        db.close().unwrap();
    });
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(db.node_count(), 600);
}

/// An empty database and one whose only content is an empty named graph are
/// written as metadata chunks alone, and both reopen: the named graph exists
/// and takes writes.
#[test]
fn an_empty_database_and_an_empty_named_graph_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.grafeo");
    let trips = dir.path().join("trips.grafeo");
    with_chunk_caps(TINY, || {
        GrafeoDB::with_config(Config::persistent(&empty))
            .unwrap()
            .close()
            .unwrap();
        let db = GrafeoDB::with_config(Config::persistent(&trips)).unwrap();
        db.execute("CREATE GRAPH trips").unwrap();
        db.close().unwrap();
    });

    let db = GrafeoDB::open(&empty).unwrap();
    assert_eq!((db.node_count(), db.edge_count()), (0, 0));
    assert_eq!(db.list_graphs(), Vec::<String>::new());
    db.close().unwrap();

    let db = GrafeoDB::open(&trips).unwrap();
    assert_eq!(db.list_graphs(), ["trips"]);
    {
        let graph = db.graph("trips").unwrap();
        assert_eq!(
            graph.execute("MATCH (n) RETURN count(n)").unwrap().rows(),
            [vec![Value::Int64(0)]],
            "the named graph is empty"
        );
        graph.execute("INSERT (:City {name: 'Prague'})").unwrap();
    }
    db.close().unwrap();
    let db = GrafeoDB::open(&trips).unwrap();
    assert_eq!(
        db.graph("trips")
            .unwrap()
            .execute("MATCH (c:City) RETURN c.name")
            .unwrap()
            .rows(),
        [vec![Value::from("Prague")]],
        "the named graph took a write after the reopen"
    );
}

/// The 0.5.x block format wrote timestamps and zoned datetimes to the
/// millisecond and counters as null; a chunked image keeps them exactly.
#[test]
fn sub_millisecond_timestamps_and_counters_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exact.grafeo");
    let instant = Timestamp::from_micros(1_696_500_000_123_457);
    let replicas = |counts: &[(&str, u64)]| -> Arc<HashMap<String, u64>> {
        Arc::new(
            counts
                .iter()
                .map(|(replica, count)| ((*replica).to_string(), *count))
                .collect(),
        )
    };
    let values = [
        ("seen", Value::Timestamp(instant)),
        (
            "seen_in_amsterdam",
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, 7200)),
        ),
        (
            "visits",
            Value::GCounter(replicas(&[("Alix", 3), ("Gus", 19)])),
        ),
        (
            "balance",
            Value::OnCounter {
                pos: replicas(&[("Mia", 88)]),
                neg: replicas(&[("Jules", 3)]),
            },
        ),
    ];
    let alix = {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let alix = db.create_node(&["Person"]).unwrap();
        for (key, value) in &values {
            db.set_node_property(alix, key, value.clone()).unwrap();
        }
        db.close().unwrap();
        alix
    };

    let db = GrafeoDB::open(&path).unwrap();
    let node = db.get_node(alix).expect("Alix is back");
    for (key, value) in &values {
        let back = node.properties.get(&PropertyKey::new(*key));
        assert_eq!(
            back.map(encoded),
            Some(encoded(value)),
            "{key}: {} came back as {:?}",
            readable(value),
            back.map(readable)
        );
    }
}

/// The next ids of every graph are kept: after a reopen, new nodes and edges
/// get ids above those of deleted ones (0.5.x handed out the ids of nodes
/// deleted at the end again).
#[test]
fn ids_of_deleted_nodes_and_edges_are_not_reused_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ids.grafeo");
    let (people, last_edge, cities) = {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let people: Vec<NodeId> = (0..5)
            .map(|_| db.create_node(&["Person"]).unwrap())
            .collect();
        db.create_edge(people[0], people[1], "KNOWS").unwrap();
        let last_edge = db.create_edge(people[1], people[2], "KNOWS").unwrap();
        assert!(db.delete_edge(last_edge).unwrap());
        assert!(db.delete_node(people[3]).unwrap());
        assert!(db.delete_node(people[4]).unwrap());
        db.create_graph("trips").unwrap();
        db.set_current_graph(Some("trips")).unwrap();
        let cities: Vec<NodeId> = (0..3).map(|_| db.create_node(&["City"]).unwrap()).collect();
        assert!(db.delete_node(cities[2]).unwrap());
        db.set_current_graph(None).unwrap();
        db.close().unwrap();
        (people, last_edge, cities)
    };

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.create_node(&["Person"]).unwrap(),
        NodeId::new(people[4].as_u64() + 1),
        "the ids of the deleted nodes {:?} stay retired",
        &people[3..]
    );
    assert_eq!(
        db.create_edge(people[0], people[2], "KNOWS").unwrap(),
        EdgeId::new(last_edge.as_u64() + 1),
        "the id of the deleted edge {last_edge:?} stays retired"
    );
    db.set_current_graph(Some("trips")).unwrap();
    assert_eq!(
        db.create_node(&["City"]).unwrap(),
        NodeId::new(cities[2].as_u64() + 1),
        "the named graph keeps its own next id"
    );
}

#[cfg(feature = "temporal")]
#[test]
fn point_in_time_reads_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.grafeo");
    let (alix, epochs) = {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let alix = db.create_node(&["Person"]).unwrap();
        let epochs: Vec<_> = CITIES[..3]
            .iter()
            .map(|city| {
                db.set_node_property(alix, "city", Value::from(*city))
                    .unwrap();
                db.current_epoch()
            })
            .collect();
        assert!(
            epochs.windows(2).all(|pair| pair[0] < pair[1]),
            "three commits, three epochs: {epochs:?}"
        );
        with_chunk_caps(TINY, || db.close()).unwrap();
        (alix, epochs)
    };

    let db = GrafeoDB::open(&path).unwrap();
    for (epoch, city) in epochs.iter().zip(CITIES) {
        assert_eq!(
            db.get_node_property_at_epoch(alix, "city", *epoch),
            Some(Value::from(city)),
            "the city at {epoch:?}"
        );
    }
}

// --- Every kind of section --------------------------------------------------------

/// One database with every kind of section, written in small chunks: the LPG
/// store with a named graph and (with `temporal`) property history, RDF
/// triples in the default and a named graph with non-ASCII literals and
/// their ring, a vector and a text index, a constraint in the catalog, and a
/// delete after `compact()`. It reopens with all of it, and once more after a
/// checkpoint with the default caps.
#[cfg(all(
    feature = "sparql",
    feature = "ring-index",
    feature = "vector-index",
    feature = "text-index"
))]
#[test]
fn every_section_kind_in_one_database_survives_a_reopen_in_small_chunks() {
    fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
        db.execute(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"))
            .rows()
            .to_vec()
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("every.grafeo");
    #[cfg_attr(
        not(feature = "temporal"),
        expect(
            unused_variables,
            reason = "only temporal builds keep property history"
        )
    )]
    let (vincent, epochs) = with_chunk_caps(TINY, || {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.execute("CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE")
            .unwrap();
        db.execute(
            "INSERT (:Person {name: 'Alix', email: 'alix@example.org'})-[:KNOWS {since: 1988}]->\
             (:Person {name: 'Gus', email: 'gus@example.org'}), \
             (:Person {name: 'Mia', email: 'mia@example.org'})",
        )
        .unwrap();
        db.create_graph("trips").unwrap();
        db.graph("trips")
            .unwrap()
            .execute("INSERT (:City {name: 'Paris'})-[:ROUTE {km: 1030}]->(:City {name: 'Prague'})")
            .unwrap();
        db.execute_sparql(
            "INSERT DATA { <http://example.org/alix> <http://example.org/knows> \
             <http://example.org/gus> . <http://example.org/prague> <http://example.org/name> \
             \"Hlavní město Praha\"@cs . GRAPH <http://example.org/trips> { \
             <http://example.org/paris> <http://example.org/route> \"Paříž → Praha, 1030 km\" } }",
        )
        .unwrap();
        db.rdf_store().rebuild_ring();
        // Mia is written before `compact()` and deleted after it.
        db.compact().unwrap();
        db.execute("MATCH (p:Person {name: 'Mia'}) DELETE p")
            .unwrap();
        db.execute(
            "INSERT (:Person {name: 'Vincent', email: 'vincent@example.org', \
             bio: 'jazz in Amsterdam', embedding: vector([3.0, 19.0, 88.0])}), \
             (:Person {name: 'Jules', email: 'jules@example.org', \
             bio: 'cycling in Berlin', embedding: vector([88.0, 19.0, 3.0])})",
        )
        .unwrap();
        db.create_vector_index("Person", "embedding", None, None, None, None, None)
            .unwrap();
        db.create_text_index("Person", "bio").unwrap();
        let vincent = match &rows(&db, "MATCH (p:Person {name: 'Vincent'}) RETURN id(p)")[0][0] {
            Value::Int64(id) => NodeId::new(u64::try_from(*id).unwrap()),
            other => panic!("an id: {other:?}"),
        };
        let epochs: Vec<_> = ["Amsterdam", "Berlin", "Prague"]
            .into_iter()
            .map(|city| {
                db.set_node_property(vincent, "city", Value::from(city))
                    .unwrap();
                db.current_epoch()
            })
            .collect();
        db.close().unwrap();
        (vincent, epochs)
    });

    let check = |db: &GrafeoDB, what: &str| {
        assert_eq!(
            rows(db, "MATCH (p:Person) RETURN p.name ORDER BY p.name"),
            [["Alix"], ["Gus"], ["Jules"], ["Vincent"]].map(|[name]| vec![Value::from(name)]),
            "{what}: Mia, deleted after compact(), stays deleted"
        );
        assert_eq!(
            rows(
                db,
                "MATCH (a:Person)-[k:KNOWS]->(b:Person) RETURN a.name, k.since, b.name"
            ),
            [vec![
                Value::from("Alix"),
                Value::Int64(1988),
                Value::from("Gus")
            ]],
            "{what}: the edge"
        );
        assert_eq!(
            db.graph("trips")
                .unwrap()
                .execute("MATCH (a)-[r:ROUTE]->(b) RETURN a.name, r.km, b.name")
                .unwrap()
                .rows(),
            [vec![
                Value::from("Paris"),
                Value::Int64(1030),
                Value::from("Prague")
            ]],
            "{what}: the named graph"
        );
        let rdf = db.rdf_store();
        let triples: Vec<String> = rdf.triples().iter().map(ToString::to_string).collect();
        assert_eq!(
            triples.len(),
            2,
            "{what}: the default graph's triples: {triples:?}"
        );
        assert!(
            triples
                .iter()
                .any(|triple| triple.contains("\"Hlavní město Praha\"@cs")),
            "{what}: the non-ASCII literal: {triples:?}"
        );
        let named: Vec<String> = rdf
            .graph("http://example.org/trips")
            .unwrap_or_else(|| panic!("{what}: the RDF graph"))
            .triples()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert!(
            named.len() == 1 && named[0].contains("\"Paříž → Praha, 1030 km\""),
            "{what}: the named RDF graph: {named:?}"
        );
        assert_eq!(
            rdf.ring().map(|ring| ring.len()),
            Some(2),
            "{what}: the ring"
        );

        let nearest = db
            .vector_search("Person", "embedding", &[3.0, 19.0, 88.0], 1, None, None)
            .unwrap_or_else(|error| panic!("{what}: vector search: {error}"));
        assert_eq!(
            nearest
                .into_iter()
                .map(|(node, _)| node)
                .collect::<Vec<_>>(),
            [vincent],
            "{what}: vector search"
        );
        let matches = db
            .text_search("Person", "bio", "cycling", 3)
            .unwrap_or_else(|error| panic!("{what}: text search: {error}"));
        assert_eq!(matches.len(), 1, "{what}: text search: {matches:?}");
        assert_eq!(
            rows(db, "SHOW CONSTRAINTS")
                .into_iter()
                .map(|row| row[0].clone())
                .collect::<Vec<_>>(),
            [Value::from("person_email")],
            "{what}: the constraint"
        );
        #[cfg(feature = "temporal")]
        for (epoch, city) in epochs.iter().zip(["Amsterdam", "Berlin", "Prague"]) {
            assert_eq!(
                db.get_node_property_at_epoch(vincent, "city", *epoch),
                Some(Value::from(city)),
                "{what}: Vincent's city at {epoch:?}"
            );
        }
    };

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let counts = chunk_counts(&db);
    for section in [
        "Catalog",
        "LpgStore",
        "RdfStore",
        "VectorStore",
        "TextIndex",
        "RdfRing",
    ] {
        let chunks = counts.get(section).copied().unwrap_or(0);
        // Every section is a metadata chunk and its data (the catalog's
        // records: the constraint and the index definitions).
        assert!(
            chunks >= 2,
            "the {section} section is in the image in 2 chunks or more: {counts:?}"
        );
    }
    check(&db, "written in small chunks");
    db.close().unwrap();

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    check(&db, "checkpointed again with the default caps");
    db.close().unwrap();
}

/// A database written by 0.5.x and migrated with the tiny caps (the migration
/// and the close write their images on the opening thread) reopens with what
/// a read-only open of the 0.5.x files finds, and its LPG section is in more
/// chunks than a migration with the default caps writes.
#[cfg(all(
    feature = "triple-store",
    feature = "sparql",
    feature = "vector-index",
    feature = "text-index"
))]
#[test]
fn a_released_database_migrated_in_small_chunks_holds_what_the_release_wrote() {
    use std::path::Path;

    fn copy(from: &Path, to: &Path) {
        if from.is_dir() {
            std::fs::create_dir_all(to).unwrap();
            for entry in std::fs::read_dir(from).unwrap() {
                let entry = entry.unwrap();
                copy(&entry.path(), &to.join(entry.file_name()));
            }
        } else {
            std::fs::copy(from, to).unwrap();
        }
    }

    /// Migrates the database at `path` with `caps`, and returns the chunk
    /// counts of the image its close writes.
    fn migrate(path: &Path, caps: ChunkCaps, what: &str) -> BTreeMap<String, usize> {
        with_chunk_caps(caps, || {
            let db =
                GrafeoDB::open(path).unwrap_or_else(|error| panic!("{what}: migration: {error}"));
            db.close().unwrap();
        });
        let db = GrafeoDB::open_read_only(path).unwrap();
        chunk_counts(&db)
    }

    let released = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released");
    for version in ["0.5.43", "0.5.44"] {
        for name in ["closed.grafeo", "unflushed.grafeo", "directory"] {
            let what = format!("{version}/{name}");
            // A copy of the database and, for `unflushed.grafeo`, its sidecar
            // WAL, in a directory of its own.
            let copy_fixture = || {
                let dir = tempfile::tempdir().unwrap();
                for entry in std::fs::read_dir(released.join(version)).unwrap() {
                    let entry = entry.unwrap();
                    if entry.file_name().to_string_lossy().starts_with(name) {
                        copy(&entry.path(), &dir.path().join(entry.file_name()));
                    }
                }
                dir
            };
            let dir = copy_fixture();
            let path = dir.path().join(name);
            let expected = {
                let db = GrafeoDB::open_read_only(&path)
                    .unwrap_or_else(|error| panic!("{what}: read-only: {error}"));
                let lines = dump(&db);
                db.close().unwrap();
                lines
            };

            let small = migrate(&path, TINY, &what);
            let db = GrafeoDB::open_read_only(&path).unwrap();
            assert_same(&expected, &dump(&db), &what);

            let default_dir = copy_fixture();
            let default = migrate(&default_dir.path().join(name), ChunkCaps::DEFAULT, &what);
            assert!(
                small.get("LpgStore") > default.get("LpgStore"),
                "{what}: the migration wrote the LPG section in small chunks: {small:?}, with \
                 the default caps {default:?}"
            );
        }
    }
}

// --- Random graphs ----------------------------------------------------------------

const LABELS: [&str; 3] = ["Person", "City", "Museum"];
const EDGE_TYPES: [&str; 2] = ["KNOWS", "LIVES_IN"];
/// The keys of generated map values.
const KEYS: [&str; 5] = ["name", "age", "score", "seen", "notes"];
/// Words strings are built from, repeated up to 200 times: the empty string,
/// non-ASCII text and strings longer than a chunk's byte cap.
const WORDS: [&str; 4] = ["Amsterdam", "Hlavní město Praha ", "", "3 19 88 "];
/// The names of the named graphs a generated database may have.
const GRAPH_NAMES: [&str; 2] = ["museums", "trips"];

/// The nodes, edges and deletions of one generated graph. Edge endpoints and
/// deletions are indexes into the nodes (and edges) the graph ends up with.
#[derive(Debug, Clone)]
struct GraphSpec {
    nodes: Vec<(Vec<&'static str>, Vec<(&'static str, Value)>)>,
    edges: Vec<(Index, Index, &'static str, Vec<(&'static str, Value)>)>,
    deleted_edges: Vec<Index>,
    deleted_nodes: Vec<Index>,
}

/// A generated database: the default graph, up to two named graphs, and the
/// caps it is written with.
#[derive(Debug, Clone)]
struct ArbitraryGraph {
    caps: ChunkCaps,
    graphs: Vec<GraphSpec>,
}

/// Bits of floats an encoding may lose: NaNs with a payload and a sign,
/// negative zero, the infinities and a subnormal.
const SPECIAL_F64_BITS: [u64; 6] = [
    0x7FF8_0000_0000_0058,
    0xFFF0_0000_0000_0013,
    0x8000_0000_0000_0000,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x0000_0000_0000_0003,
];

/// [`SPECIAL_F64_BITS`] for the `f32` of vectors.
const SPECIAL_F32_BITS: [u32; 6] = [
    0x7FC0_0058,
    0xFF80_0013,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x0000_0003,
];

/// The bits of a float: any bits, or (one in four) bits from
/// [`SPECIAL_F64_BITS`], which random bits hardly ever give.
fn float_bits() -> impl Strategy<Value = u64> {
    prop_oneof![
        3 => any::<u64>(),
        1 => prop::sample::select(SPECIAL_F64_BITS.to_vec()),
    ]
}

/// A value of any kind but a list, map, path or counter, with floats from any
/// bits.
fn arbitrary_leaf() -> impl Strategy<Value = Value> {
    let offset = -64_800i32..=64_800;
    prop_oneof![
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::Int64),
        float_bits().prop_map(|bits| Value::Float64(f64::from_bits(bits))),
        arbitrary_string(),
        prop::collection::vec(any::<u8>(), 0..40).prop_map(|bytes| Value::Bytes(Arc::from(bytes))),
        any::<i64>().prop_map(|micros| Value::Timestamp(Timestamp::from_micros(micros))),
        any::<i32>().prop_map(|days| Value::Date(Date::from_days(days))),
        (
            0u64..86_400_000_000_000,
            proptest::option::of(offset.clone())
        )
            .prop_map(|(nanos, offset)| {
                let time = Time::from_nanos(nanos).expect("a time of day");
                Value::Time(offset.map_or(time, |offset| time.with_offset(offset)))
            }),
        (any::<i64>(), any::<i64>(), any::<i64>())
            .prop_map(|(months, days, nanos)| Value::Duration(Duration::new(months, days, nanos))),
        (any::<i64>(), offset).prop_map(|(micros, offset)| Value::ZonedDatetime(
            ZonedDatetime::from_timestamp_offset(Timestamp::from_micros(micros), offset)
        )),
        prop::collection::vec(
            prop_oneof![
                any::<u32>(),
                prop::sample::select(SPECIAL_F32_BITS.to_vec())
            ],
            1..5
        )
        .prop_map(|bits| Value::Vector(bits.into_iter().map(f32::from_bits).collect())),
    ]
}

/// A counter with up to three replicas.
fn arbitrary_counter() -> impl Strategy<Value = Value> {
    let replicas = || {
        prop::collection::hash_map(prop::sample::select(PEOPLE.to_vec()), any::<u64>(), 0..3)
            .prop_map(|counts| {
                Arc::new(
                    counts
                        .into_iter()
                        .map(|(replica, count)| (replica.to_string(), count))
                        .collect::<HashMap<String, u64>>(),
                )
            })
    };
    prop_oneof![
        replicas().prop_map(Value::GCounter),
        (replicas(), replicas()).prop_map(|(pos, neg)| Value::OnCounter { pos, neg }),
    ]
}

/// A property value of any kind: leaves, lists, maps and paths of them (with
/// nulls inside), and counters. Counters stay at the top, where [`readable`]
/// orders their replicas.
fn arbitrary_value() -> impl Strategy<Value = Value> {
    let nested = arbitrary_leaf().prop_recursive(2, 16, 4, |inner| {
        let item = prop_oneof![Just(Value::Null), inner.clone()];
        prop_oneof![
            prop::collection::vec(item.clone(), 0..4)
                .prop_map(|items| Value::List(items.into_iter().collect())),
            prop::collection::btree_map(
                prop::sample::select(KEYS.to_vec()).prop_map(PropertyKey::new),
                item,
                0..4
            )
            .prop_map(|map| Value::Map(Arc::new(map))),
            prop::collection::vec(inner, 1..3).prop_map(|nodes| {
                let edges = vec![Value::Null; nodes.len() - 1];
                Value::Path {
                    nodes: nodes.into_iter().collect(),
                    edges: edges.into_iter().collect(),
                }
            }),
        ]
    });
    prop_oneof![3 => arbitrary_leaf(), 5 => nested, 1 => arbitrary_counter()]
}

/// A string built from [`WORDS`].
fn arbitrary_string() -> impl Strategy<Value = Value> {
    (0..WORDS.len(), 0usize..200).prop_map(|(word, times)| Value::from(WORDS[word].repeat(times)))
}

/// Some of the properties `name`, `age`, `score`, `flag` and `embedding`,
/// each always of one kind (a string, an integer, a float, a boolean, a
/// vector of three dimensions), so their chunks take the typed codecs at any
/// caps, and `notes` and `seen`, of any kind.
fn arbitrary_properties() -> impl Strategy<Value = Vec<(&'static str, Value)>> {
    let typed = (
        proptest::option::of(arbitrary_string()),
        proptest::option::of(any::<i64>().prop_map(Value::Int64)),
        proptest::option::of(float_bits().prop_map(|bits| Value::Float64(f64::from_bits(bits)))),
        proptest::option::of(any::<bool>().prop_map(Value::Bool)),
        proptest::option::of(
            prop::collection::vec(
                prop_oneof![
                    3 => any::<u32>(),
                    1 => prop::sample::select(SPECIAL_F32_BITS.to_vec())
                ],
                3,
            )
            .prop_map(|bits| Value::Vector(bits.into_iter().map(f32::from_bits).collect())),
        ),
    );
    let mixed = (
        proptest::option::of(arbitrary_value()),
        proptest::option::of(arbitrary_value()),
    );
    (typed, mixed).prop_map(|((name, age, score, flag, embedding), (notes, seen))| {
        [
            ("name", name),
            ("age", age),
            ("score", score),
            ("flag", flag),
            ("embedding", embedding),
            ("notes", notes),
            ("seen", seen),
        ]
        .into_iter()
        .filter_map(|(key, value)| value.map(|value| (key, value)))
        .collect()
    })
}

fn arbitrary_graph_spec() -> impl Strategy<Value = GraphSpec> {
    (
        prop::collection::vec(
            (
                prop::sample::subsequence(LABELS.to_vec(), 0..=3),
                arbitrary_properties(),
            ),
            0..=60,
        ),
        prop::collection::vec(
            (
                any::<Index>(),
                any::<Index>(),
                prop::sample::select(EDGE_TYPES.to_vec()),
                arbitrary_properties(),
            ),
            0..=80,
        ),
        prop::collection::vec(any::<Index>(), 0..=8),
        prop::collection::vec(any::<Index>(), 0..=8),
    )
        .prop_map(|(nodes, edges, deleted_edges, deleted_nodes)| GraphSpec {
            nodes,
            edges,
            deleted_edges,
            deleted_nodes,
        })
}

fn arbitrary_graph() -> impl Strategy<Value = ArbitraryGraph> {
    (
        prop::sample::select(vec![1u32, 3, 64]),
        prop::sample::select(vec![64u32, 1024]),
        prop::collection::vec(arbitrary_graph_spec(), 1..=3),
    )
        .prop_map(|(max_rows, max_bytes, graphs)| ArbitraryGraph {
            caps: ChunkCaps {
                max_rows,
                max_bytes,
            },
            graphs,
        })
}

/// Writes `spec` into the graph `db` works in.
fn write_spec(db: &GrafeoDB, spec: &GraphSpec) {
    let nodes: Vec<NodeId> = spec
        .nodes
        .iter()
        .map(|(labels, properties)| {
            db.create_node_with_props(labels, properties.iter().cloned())
                .unwrap()
        })
        .collect();
    if nodes.is_empty() {
        return;
    }
    let edges: Vec<(EdgeId, usize, usize)> = spec
        .edges
        .iter()
        .map(|(src, dst, edge_type, properties)| {
            let (src, dst) = (src.index(nodes.len()), dst.index(nodes.len()));
            let edge = db
                .create_edge_with_props(
                    nodes[src],
                    nodes[dst],
                    edge_type,
                    properties.iter().cloned(),
                )
                .unwrap();
            (edge, src, dst)
        })
        .collect();
    if !edges.is_empty() {
        for index in &spec.deleted_edges {
            db.delete_edge(edges[index.index(edges.len())].0).unwrap();
        }
    }
    for index in &spec.deleted_nodes {
        let node = index.index(nodes.len());
        for &(edge, src, dst) in &edges {
            if src == node || dst == node {
                db.delete_edge(edge).unwrap();
            }
        }
        db.delete_node(nodes[node]).unwrap();
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// A generated database written with generated caps reopens with every
    /// node, edge, label and property as written, and so does an in-memory
    /// copy of the reopened database.
    #[test]
    fn random_graphs_round_trip(graph in arbitrary_graph()) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("random.grafeo");
        let expected = with_chunk_caps(graph.caps, || {
            let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
            for (index, spec) in graph.graphs.iter().enumerate() {
                if let Some(name) = index.checked_sub(1).map(|named| GRAPH_NAMES[named]) {
                    db.create_graph(name).unwrap();
                    db.set_current_graph(Some(name)).unwrap();
                }
                write_spec(&db, spec);
            }
            db.set_current_graph(None).unwrap();
            let before = dump(&db);
            db.close().unwrap();
            before
        });

        let db = GrafeoDB::open_read_only(&path).unwrap();
        prop_assert_eq!(difference(&expected, &dump(&db)), None, "reopened");
        let copy = db.to_memory().unwrap();
        prop_assert_eq!(difference(&expected, &dump(&copy)), None, "in-memory copy");
    }
}

// --- A crash during a checkpoint in small chunks ---------------------------------

#[cfg(feature = "testing-crash-injection")]
mod crash_during_close {
    use std::collections::HashMap;
    use std::path::Path;

    use grafeo_common::testing::chunk_caps::with_chunk_caps;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_engine::{Config, GrafeoDB};

    use super::{PEOPLE, TINY};

    /// The crash points of `close()` with the WAL turned off, in order (the
    /// `WAL_DISABLED_CLOSE_POINTS` of `crash_injection_single_file.rs`).
    const CLOSE_POINTS: [&str; 8] = [
        "flush:before_serialize",
        "flush:after_rotate",
        "checkpoint:after_chunks",
        "checkpoint:after_data_sync",
        "checkpoint:after_header",
        "checkpoint:before_trim",
        "flush:after_write",
        "close:before_remove_sidecar_wal",
    ];

    /// The first of [`CLOSE_POINTS`] (1-based) at which the new image's database
    /// header is on disk: `checkpoint:after_header`.
    const HEADER_WRITTEN: u64 = 5;

    const CHILD_POINT_VAR: &str = "GRAFEO_CHUNKED_CRASH_POINT";
    const CHILD_PATH_VAR: &str = "GRAFEO_CHUNKED_CRASH_PATH";
    /// Exit code of a child whose `close()` crashed.
    const CRASHED: i32 = 3;

    /// A persistent configuration without a WAL: a reopen sees exactly the image
    /// on disk.
    fn wal_disabled_config(path: &Path) -> Config {
        Config::persistent(path).without_wal()
    }

    /// Adds Persons `first` to `first + count - 1`, each with a name, in one
    /// transaction.
    fn add_people(db: &GrafeoDB, first: usize, count: usize) {
        let properties = (first..first + count)
            .map(|i| {
                HashMap::from([(
                    PropertyKey::new("name"),
                    Value::from(format!("{} {i}", PEOPLE[i % 5])),
                )])
            })
            .collect();
        db.batch_create_nodes_with_props("Person", properties)
            .unwrap();
    }

    /// A crash at each point of a `close()` whose checkpoint writes hundreds of
    /// chunks leaves a file that opens the image before it until the new header
    /// is written, and the new one from then on, never anything else. A
    /// checkpoint after the reopen succeeds, and after a crash between the chunks
    /// and the header it writes into the pages the cut-off image used: the file
    /// does not grow.
    #[test]
    fn a_crash_after_the_chunks_reopens_the_previous_image() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = dir.path().join("fixture.grafeo");
        with_chunk_caps(TINY, || {
            let db = GrafeoDB::with_config(wal_disabled_config(&fixture)).unwrap();
            add_people(&db, 0, 300);
            db.close().unwrap();
        });

        let points = u64::try_from(CLOSE_POINTS.len()).unwrap();
        for point in 1..=points + 1 {
            let name = usize::try_from(point - 1)
                .ok()
                .and_then(|index| CLOSE_POINTS.get(index))
                .copied()
                .unwrap_or("no crash");
            let path = dir.path().join(format!("crash-{point}.grafeo"));
            std::fs::copy(&fixture, &path).unwrap();
            assert_eq!(
                close_in_child(point, &path),
                point > points,
                "{name}: close() without a WAL has {points} crash points"
            );
            let crashed_size = std::fs::metadata(&path).unwrap().len();

            let expected = if point < HEADER_WRITTEN { 300 } else { 600 };
            let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
            assert_eq!(
                db.node_count(),
                expected,
                "{name}: the image the reopen finds"
            );
            with_chunk_caps(TINY, || db.wal_checkpoint())
                .unwrap_or_else(|error| panic!("{name}: a checkpoint after the reopen: {error}"));
            if matches!(
                name,
                "checkpoint:after_chunks" | "checkpoint:after_data_sync"
            ) {
                let size = db.file_manager().unwrap().file_size().unwrap();
                assert!(
                    size <= crashed_size,
                    "{name}: the checkpoint reuses the cut-off image's pages: {size} bytes \
                     after it, {crashed_size} after the crash"
                );
            }
            db.close().unwrap();
            let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
            assert_eq!(db.node_count(), expected, "{name}: after the checkpoint");
            db.close().unwrap();
        }
    }

    /// Reopens the WAL-less database at `path` in a child process, adds 300
    /// Persons and crashes at `crash_point` inside a `close()` with the tiny
    /// caps. Returns whether the close completed.
    fn close_in_child(crash_point: u64, path: &Path) -> bool {
        let status = grafeo_common::testing::child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "crash_during_close::closes_in_small_chunks_in_a_child",
                    "--nocapture",
                ])
                .env(CHILD_POINT_VAR, crash_point.to_string())
                .env(CHILD_PATH_VAR, path),
        )
        .unwrap();
        match status.code() {
            Some(0) => true,
            Some(CRASHED) => false,
            other => panic!("crash point {crash_point}: the child failed with {other:?}"),
        }
    }

    /// Child-process entry of [`close_in_child`]; a no-op when run directly.
    #[test]
    fn closes_in_small_chunks_in_a_child() {
        let (Ok(point), Some(path)) = (
            std::env::var(CHILD_POINT_VAR),
            std::env::var_os(CHILD_PATH_VAR),
        ) else {
            return;
        };
        let db = GrafeoDB::with_config(wal_disabled_config(Path::new(&path))).unwrap();
        add_people(&db, 300, 300);
        let target = std::panic::AssertUnwindSafe(&db);
        let result = with_chunk_caps(TINY, || {
            with_crash_at(point.parse().unwrap(), move || target.close())
        });
        // Exit without running destructors, like a crash.
        match result {
            CrashResult::Completed(closed) => {
                closed.unwrap();
                std::process::exit(0);
            }
            _ => std::process::exit(CRASHED),
        }
    }
}

// --- Chunks moved within an image -------------------------------------------------

#[cfg(feature = "encryption")]
mod moved_chunks {
    use std::io::{Seek, SeekFrom, Write};
    use std::path::Path;
    use std::sync::Arc;

    use grafeo_common::encryption::{KeyChain, PageEncryptor, random_nonce};
    use grafeo_common::storage::{
        ChunkKind, ChunkMeta, ChunkNamespace, SectionSource, SectionType,
    };
    use grafeo_common::testing::chunk_caps::with_chunk_caps;
    use grafeo_common::types::Value;
    use grafeo_engine::config::EncryptionConfig;
    use grafeo_engine::{Config, GrafeoDB};
    use grafeo_storage::file::v3::ImageReader;
    use grafeo_storage::file::v3::alloc::PageRun;
    use grafeo_storage::file::v3::directory::{DirectoryEntry, encode_blocks};
    use grafeo_storage::file::v3::header::{DbHeaderV3, FileHeaderV3, PAGE_SIZE, active_header};

    use super::TINY;

    fn with_key(config: Config, chain: &Arc<KeyChain>) -> Config {
        config.with_encryption(EncryptionConfig::new(Arc::clone(chain)))
    }

    /// Writes 30 Persons whose names have one length ("Alix 00" to "Alix
    /// 29") with the tiny caps, so the LPG section holds name chunks of one
    /// stored length.
    fn write_thirty_people(config: Config) {
        with_chunk_caps(TINY, || {
            let db = GrafeoDB::with_config(config).unwrap();
            for i in 0..30 {
                db.create_node_with_props(
                    &["Person"],
                    [("name", Value::from(format!("Alix {i:02}")))],
                )
                .unwrap();
            }
            db.close().unwrap();
        });
    }

    /// The key that encrypts the chunks and directory of the file at `path`:
    /// derived from "grafeo-container" and the database id.
    fn container_cipher(path: &Path, chain: &KeyChain) -> PageEncryptor {
        let bytes = std::fs::read(path).unwrap();
        let database_id = FileHeaderV3::decode(&bytes[..4096]).unwrap().database_id;
        chain.encryptor_for("grafeo-container", &database_id.to_le_bytes())
    }

    /// The active image of the file at `path`, read through `ImageReader`
    /// (decrypting with `cipher`) from a copy of the file: the slot of its
    /// database header, the header, its directory entries and the pages of
    /// its directory blocks in chain order.
    fn active_image(
        path: &Path,
        cipher: Option<&PageEncryptor>,
    ) -> (u8, DbHeaderV3, Vec<DirectoryEntry>, Vec<PageRun>) {
        let bytes = std::fs::read(path).unwrap();
        let slots = [
            DbHeaderV3::decode(&bytes[4096..8192]),
            DbHeaderV3::decode(&bytes[8192..12288]),
        ];
        let (slot, header) = active_header(slots).unwrap().expect("a database");
        let copy = path.with_extension("copy");
        std::fs::copy(path, &copy).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&copy)
            .unwrap();
        let reader = ImageReader::open(&mut file, header.root, cipher).unwrap();
        let (entries, runs) = (reader.entries().to_vec(), reader.directory_runs().to_vec());
        (slot, header, entries, runs)
    }

    /// Two Column chunks of the LPG section with one stored length and
    /// different bytes.
    fn twins(path: &Path, entries: &[DirectoryEntry]) -> (DirectoryEntry, DirectoryEntry) {
        let bytes = std::fs::read(path).unwrap();
        let stored = |entry: &DirectoryEntry| {
            let start = usize::try_from(entry.offset).unwrap();
            bytes[start..start + usize::try_from(entry.length).unwrap()].to_vec()
        };
        let columns: Vec<&DirectoryEntry> = entries
            .iter()
            .filter(|entry| {
                entry.section_type == SectionType::LpgStore && entry.meta.kind == ChunkKind::Column
            })
            .collect();
        for (at, first) in columns.iter().enumerate() {
            for second in &columns[at + 1..] {
                if first.length == second.length && stored(first) != stored(second) {
                    return (**first, **second);
                }
            }
        }
        panic!("no two Column chunks of the LPG section share a stored length: {columns:?}");
    }

    /// Swaps the stored bytes of `first` and `second` (of one length) in the
    /// file at `path`; the directory stays as it is.
    fn swap_bytes(path: &Path, first: &DirectoryEntry, second: &DirectoryEntry) {
        let mut bytes = std::fs::read(path).unwrap();
        let length = usize::try_from(first.length).unwrap();
        let (a, b) = (
            usize::try_from(first.offset).unwrap(),
            usize::try_from(second.offset).unwrap(),
        );
        let first_bytes = bytes[a..a + length].to_vec();
        bytes.copy_within(b..b + length, a);
        bytes[b..b + length].copy_from_slice(&first_bytes);
        std::fs::write(path, bytes).unwrap();
    }

    /// The error of an open that must fail.
    fn error_of(result: grafeo_common::utils::error::Result<GrafeoDB>) -> String {
        match result {
            Ok(_) => panic!("the open succeeded, expected an error"),
            Err(error) => error.to_string(),
        }
    }

    /// Two chunks of one stored length swapped in the file fail the open of a
    /// plain and of an encrypted database, never giving one chunk's rows to
    /// the other: each directory entry holds the checksum of its chunk's
    /// stored bytes, checked before a chunk is decrypted (and in an encrypted
    /// file the directory itself is encrypted, so the checksums cannot be
    /// changed without the key).
    #[test]
    fn swapped_chunks_fail_to_open() {
        let dir = tempfile::tempdir().unwrap();
        let chain = Arc::new(KeyChain::new([19; 32]));
        let plain = dir.path().join("plain.grafeo");
        let secret = dir.path().join("secret.grafeo");
        write_thirty_people(Config::persistent(&plain));
        write_thirty_people(with_key(Config::persistent(&secret), &chain));
        let cipher = container_cipher(&secret, &chain);

        for (path, cipher, config) in [
            (&plain, None, Config::read_only(&plain)),
            (
                &secret,
                Some(&cipher),
                with_key(Config::read_only(&secret), &chain),
            ),
        ] {
            let (_, _, entries, _) = active_image(path, cipher);
            let (first, second) = twins(path, &entries);
            swap_bytes(path, &first, &second);
            let error = error_of(GrafeoDB::with_config(config));
            assert!(
                error.contains("chunk of section LpgStore at offset")
                    && error.contains("fails its checksum"),
                "{}: {error}",
                path.display()
            );
        }
    }

    /// Rewrites the directory of the encrypted file at `path` to list
    /// `entries`, encrypted with `cipher` into the pages `runs` of the old
    /// directory, and points the header in `slot` at it. The associated data
    /// of a directory block is "grafeo-directory:" and its offset (pinned by
    /// grafeo-storage's cipher tests).
    fn rewrite_directory(
        path: &Path,
        cipher: &PageEncryptor,
        (slot, header): (u8, &DbHeaderV3),
        entries: &[DirectoryEntry],
        runs: &[PageRun],
    ) {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        // `encode_blocks` places the last block first.
        let mut offsets: Vec<u64> = runs.iter().map(|run| run.first * PAGE_SIZE).collect();
        let (root, blocks) = encode_blocks(entries, |_| {
            Ok(offsets.pop().expect("a page run per block"))
        })
        .unwrap();
        for (offset, plain) in blocks {
            let aad = format!("grafeo-directory:{offset}");
            let stored = cipher
                .encrypt(&plain, &random_nonce(), aad.as_bytes())
                .unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&stored).unwrap();
        }
        let next = DbHeaderV3 {
            root,
            ..header.clone()
        };
        file.seek(SeekFrom::Start(PAGE_SIZE * (1 + u64::from(slot))))
            .unwrap();
        file.write_all(&next.encode()).unwrap();
        file.sync_all().unwrap();
    }

    /// A chunk of an encrypted file moved to another chunk's place, with the
    /// checksum that place's directory entry then holds, fails to decrypt:
    /// its associated data binds it to the section, kind, namespace, graph,
    /// column and first row it was written for, so it never gives its rows to another
    /// place. (Only a writer holding the key can change the directory this
    /// way.)
    #[test]
    fn a_chunk_moved_to_another_place_fails_to_decrypt() {
        let dir = tempfile::tempdir().unwrap();
        let chain = Arc::new(KeyChain::new([19; 32]));
        let path = dir.path().join("secret.grafeo");
        write_thirty_people(with_key(Config::persistent(&path), &chain));
        let cipher = container_cipher(&path, &chain);
        let (slot, header, entries, runs) = active_image(&path, Some(&cipher));

        // The directory rewritten as it was: the database still opens, so
        // the failure below comes from the moved chunks.
        rewrite_directory(&path, &cipher, (slot, &header), &entries, &runs);
        let header = active_image(&path, Some(&cipher)).1;
        let db = GrafeoDB::with_config(with_key(Config::read_only(&path), &chain)).unwrap();
        assert_eq!(
            db.node_count(),
            30,
            "the rewritten directory reads as before"
        );
        db.close().unwrap();
        drop(db);

        let (first, second) = twins(&path, &entries);
        let moved: Vec<DirectoryEntry> = entries
            .iter()
            .map(|entry| {
                let mut entry = *entry;
                if entry == first {
                    (entry.offset, entry.crc) = (second.offset, second.crc);
                } else if entry == second {
                    (entry.offset, entry.crc) = (first.offset, first.crc);
                }
                entry
            })
            .collect();
        rewrite_directory(&path, &cipher, (slot, &header), &moved, &runs);
        let error = error_of(GrafeoDB::with_config(with_key(
            Config::read_only(&path),
            &chain,
        )));
        assert!(
            error.contains("chunk of section LpgStore at offset")
                && error.contains("decryption failed"),
            "{error}"
        );
    }

    /// The namespace is part of a chunk's associated data: a node property
    /// chunk whose directory entry is rewritten into the edge property
    /// namespace (same offset, checksum, graph, column and rows) fails to
    /// decrypt, so a chunk never passes for one of another namespace whose
    /// column id is the same. (Only a writer holding the key can change the
    /// directory this way.)
    #[test]
    fn a_chunk_moved_to_another_namespace_fails_to_decrypt() {
        let dir = tempfile::tempdir().unwrap();
        let chain = Arc::new(KeyChain::new([19; 32]));
        let path = dir.path().join("secret.grafeo");
        write_thirty_people(with_key(Config::persistent(&path), &chain));
        let cipher = container_cipher(&path, &chain);
        let (slot, header, entries, runs) = active_image(&path, Some(&cipher));
        let node = *entries
            .iter()
            .find(|entry| {
                entry.section_type == SectionType::LpgStore
                    && entry.meta.kind == ChunkKind::Column
                    && entry.meta.namespace == ChunkNamespace::NodeProperties
            })
            .expect("a node property chunk");
        // Fetches the LPG chunk `meta` describes through the active directory.
        let fetch = |meta: ChunkMeta| {
            let mut file = std::fs::File::open(&path).unwrap();
            let root = active_image(&path, Some(&cipher)).1.root;
            let reader = ImageReader::open(&mut file, root, Some(&cipher)).unwrap();
            let section = reader.section(SectionType::LpgStore).unwrap();
            let index = section
                .chunks()
                .iter()
                .position(|chunk| *chunk == meta)
                .expect("the chunk is listed");
            section.fetch(index).map(|_| ())
        };
        fetch(node.meta).expect("the chunk decrypts in its own namespace");

        let edge_meta = node.meta.in_namespace(ChunkNamespace::EdgeProperties);
        let moved: Vec<DirectoryEntry> = entries
            .iter()
            .map(|entry| {
                if *entry == node {
                    DirectoryEntry {
                        meta: edge_meta,
                        ..*entry
                    }
                } else {
                    *entry
                }
            })
            .collect();
        rewrite_directory(&path, &cipher, (slot, &header), &moved, &runs);
        let error = fetch(edge_meta).unwrap_err().to_string();
        assert!(error.contains("decryption failed"), "{error}");
    }
}
