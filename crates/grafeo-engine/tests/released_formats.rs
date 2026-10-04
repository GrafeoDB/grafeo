//! Databases written by released versions open with today's code (#427).
//!
//! `scripts/released_fixtures.py` wrote every fixture under `fixtures/released/` with a
//! release from PyPI: one database, changed in two sessions, in three layouts. These tests
//! open a copy of each and check what it holds. Where a fixture holds less than was
//! written, its expectation says why: the release did not write it, or today's code does
//! not read it yet (and the issue that will).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test released_formats
//! ```

#![cfg(all(
    feature = "grafeo-file",
    feature = "wal",
    feature = "triple-store",
    feature = "sparql",
    feature = "vector-index",
    feature = "text-index"
))]

use std::collections::BTreeMap;
use std::path::Path;

use grafeo_common::types::{Date, Duration, NodeId, PropertyKey, Value, ZonedDatetime};
use grafeo_engine::GrafeoDB;

/// How the release left the database on disk.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// `closed.grafeo`: closed cleanly, so everything is in the file.
    Closed,
    /// `unflushed.grafeo`: the second session is only in the sidecar WAL, because the
    /// process exited without `close()`.
    Unflushed,
    /// `directory/`: a WAL-directory database, closed cleanly.
    Directory,
}

#[derive(Clone, Copy)]
struct Fixture {
    version: &'static str,
    layout: Layout,
}

const FIXTURES: [Fixture; 6] = [
    Fixture {
        version: "0.5.43",
        layout: Layout::Closed,
    },
    Fixture {
        version: "0.5.43",
        layout: Layout::Unflushed,
    },
    Fixture {
        version: "0.5.43",
        layout: Layout::Directory,
    },
    Fixture {
        version: "0.5.44",
        layout: Layout::Closed,
    },
    Fixture {
        version: "0.5.44",
        layout: Layout::Unflushed,
    },
    Fixture {
        version: "0.5.44",
        layout: Layout::Directory,
    },
];

impl Fixture {
    fn file_name(self) -> &'static str {
        match self.layout {
            Layout::Closed => "closed.grafeo",
            Layout::Unflushed => "unflushed.grafeo",
            Layout::Directory => "directory",
        }
    }

    fn name(self) -> String {
        format!("{}/{}", self.version, self.file_name())
    }

    /// Opens a copy, so the committed files stay as the release wrote them.
    fn open(self) -> (tempfile::TempDir, GrafeoDB) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/released")
            .join(self.version);
        let dir = tempfile::tempdir().unwrap();
        // The database and, for `unflushed.grafeo`, its sidecar WAL directory.
        for entry in std::fs::read_dir(&source).unwrap() {
            let entry = entry.unwrap();
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with(self.file_name())
            {
                copy(&entry.path(), &dir.path().join(entry.file_name()));
            }
        }
        let db = GrafeoDB::open(dir.path().join(self.file_name()))
            .unwrap_or_else(|error| panic!("{}: {error}", self.name()));
        (dir, db)
    }

    /// 0.5.43 did not log a switch back to the default graph, so its WAL directory
    /// replays every change after the first switch to a named graph into that graph (as
    /// 0.5.43 itself does): the default graph holds the first session up to that switch.
    fn replays_into_a_named_graph(self) -> bool {
        self.version == "0.5.43" && self.layout == Layout::Directory
    }

    /// Whether the first session's schema is only in WAL records, which carry no default
    /// values, parent types or edge endpoints (#517 adds them to the WAL).
    fn first_session_schema_in_wal(self) -> bool {
        self.layout == Layout::Directory
    }

    /// Whether the second session's schema is only in WAL records.
    fn second_session_schema_in_wal(self) -> bool {
        self.layout != Layout::Closed
    }

    /// The constraint names `SHOW CONSTRAINTS` lists. 0.5.43 files store no constraint
    /// names (0.5.44 added them, #420), but its WAL records do.
    fn constraint_names(self) -> &'static [&'static str] {
        match (self.version, self.layout) {
            ("0.5.43", Layout::Closed) => &[],
            ("0.5.43", Layout::Unflushed) => &["museum_name"],
            ("0.5.43", Layout::Directory) => &["Person_not_null", "museum_name", "person_email"],
            _ => &["Person_name_not_null", "museum_name", "person_email"],
        }
    }

    /// The indexes `SHOW INDEXES` lists (it leaves out vector indexes). 0.5.43 files hold
    /// no index definitions, and today's code does not replay the index records of a WAL
    /// (#401), so an index created in a WAL is missing.
    fn index_names(self) -> &'static [&'static str] {
        match (self.version, self.layout) {
            ("0.5.44", Layout::Closed) => &["doc_content", "museum_name_index", "person_name"],
            ("0.5.44", Layout::Unflushed) => &["doc_content", "person_name"],
            _ => &[],
        }
    }

    /// Whether the vector and text indexes of the first session are there. 0.5.43 wrote
    /// them to the file but lost them on reopen, so the checkpoint at the end of its second
    /// session wrote the file without them; in a WAL directory they are only index records
    /// (#401).
    fn has_search_indexes(self) -> bool {
        match self.layout {
            Layout::Closed => self.version != "0.5.43",
            Layout::Unflushed => true,
            Layout::Directory => false,
        }
    }
}

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

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

fn strings(values: &[&str]) -> Value {
    Value::List(values.iter().map(|v| Value::from(*v)).collect())
}

/// The first column of `query`, sorted, as strings.
fn names(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut names: Vec<String> = rows(db, query)
        .into_iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("{query}: expected a name, got {other:?}"),
        })
        .collect();
    names.sort();
    names
}

#[test]
fn the_default_graph_holds_what_was_written() {
    for fixture in FIXTURES {
        let (_dir, db) = fixture.open();
        let name = fixture.name();
        let complete = !fixture.replays_into_a_named_graph();

        let (labels, edges, people) = if complete {
            (
                vec![
                    vec![strings(&["City"]), Value::Int64(2)],
                    vec![strings(&["Document"]), Value::Int64(3)],
                    vec![strings(&["Employee", "Manager", "Person"]), Value::Int64(1)],
                    vec![strings(&["Museum"]), Value::Int64(1)],
                    vec![strings(&["Person"]), Value::Int64(2)],
                ],
                vec![
                    vec![Value::from("IN_CITY"), Value::Int64(1)],
                    vec![Value::from("KNOWS"), Value::Int64(1)],
                    vec![Value::from("LIVES_IN"), Value::Int64(1)],
                ],
                ["Alix", "Gus", "Mia"],
            )
        } else {
            (
                vec![
                    vec![strings(&["City"]), Value::Int64(2)],
                    vec![strings(&["Document"]), Value::Int64(2)],
                    vec![strings(&["Employee", "Person"]), Value::Int64(1)],
                    vec![strings(&["Person"]), Value::Int64(2)],
                ],
                vec![
                    vec![Value::from("KNOWS"), Value::Int64(1)],
                    vec![Value::from("LIVES_IN"), Value::Int64(1)],
                ],
                ["Alix", "Gus", "Vincent"],
            )
        };
        assert_eq!(
            rows(
                &db,
                "MATCH (n) RETURN labels(n) AS l, count(*) AS c ORDER BY l"
            ),
            labels,
            "{name}: nodes"
        );
        assert_eq!(
            rows(
                &db,
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t"
            ),
            edges,
            "{name}: edges"
        );
        assert_eq!(
            names(&db, "MATCH (p:Person) RETURN p.name"),
            people,
            "{name}: people"
        );

        // Every value type, and the second session's update and removal.
        let (age, score) = if complete {
            (Value::Int64(31), Value::Null)
        } else {
            (Value::Int64(30), Value::Int64(-3))
        };
        let address = BTreeMap::from([
            (PropertyKey::new("city"), Value::from("Amsterdam")),
            (PropertyKey::new("number"), Value::Int64(19)),
        ]);
        assert_eq!(
            rows(
                &db,
                "MATCH (p:Person {name: 'Alix'}) RETURN p.email, p.age, p.score, p.height, \
                 p.active, p.tags, p.address, p.born, p.seen, p.stay, p.embedding"
            ),
            [vec![
                Value::from("alix@example.org"),
                age,
                score,
                Value::Float64(1.88),
                Value::Bool(true),
                strings(&["amsterdam", "jazz"]),
                Value::Map(address.into()),
                Value::Date(Date::parse("1994-03-19").unwrap()),
                Value::ZonedDatetime(ZonedDatetime::parse("2024-03-19T08:30:00+01:00").unwrap()),
                Value::Duration(Duration::parse("P3D").unwrap()),
                Value::Vector(vec![3.0, 19.0, 88.0].into()),
            ]],
            "{name}: Alix"
        );
    }
}

#[test]
fn named_graphs_hold_what_was_written() {
    for fixture in FIXTURES {
        let (_dir, db) = fixture.open();
        let name = fixture.name();
        let mut graphs = db.list_graphs();
        graphs.sort();
        assert_eq!(graphs, ["museums", "trips"], "{name}: graphs");

        let museums = db.graph("museums").unwrap();
        assert_eq!(
            museums
                .execute("MATCH (n) RETURN labels(n), n.name")
                .unwrap()
                .rows(),
            [vec![strings(&["Museum"]), Value::from("Louvre")]],
            "{name}: museums"
        );

        if fixture.replays_into_a_named_graph() {
            continue;
        }
        let trips = db.graph("trips").unwrap();
        assert_eq!(
            trips
                .execute("MATCH (c) RETURN labels(c), c.name, c.country ORDER BY c.name")
                .unwrap()
                .rows(),
            [
                vec![strings(&["City"]), Value::from("Paris"), Value::from("FR")],
                vec![strings(&["City"]), Value::from("Prague"), Value::from("CZ")],
            ],
            "{name}: trips nodes"
        );
        assert_eq!(
            trips
                .execute("MATCH ()-[r]->() RETURN type(r), r.km")
                .unwrap()
                .rows(),
            [vec![Value::from("ROUTE"), Value::Int64(1030)]],
            "{name}: trips edges"
        );
    }
}

#[test]
fn rdf_triples_hold_what_was_written() {
    let iri = |name: &str| Value::from(format!("http://example.org/{name}"));
    for fixture in FIXTURES {
        let (_dir, db) = fixture.open();
        assert_eq!(
            db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o } ORDER BY ?s ?p ?o")
                .unwrap()
                .rows(),
            [
                vec![iri("alix"), iri("knows"), iri("gus")],
                vec![iri("gus"), iri("knows"), iri("mia")],
            ],
            "{}: triples",
            fixture.name()
        );
    }
}

#[test]
fn constraints_hold() {
    for fixture in FIXTURES {
        let (_dir, db) = fixture.open();
        let name = fixture.name();
        let error = |query: &str| match db.execute(query) {
            Ok(result) => panic!("{name}: {query} succeeded: {:?}", result.rows()),
            Err(error) => error.to_string(),
        };
        let duplicate = error("INSERT (:Person {name: 'Jules', email: 'alix@example.org'})");
        assert!(duplicate.contains("UNIQUE"), "{name}: {duplicate}");
        let unnamed = error("INSERT (:Person {email: 'jules@example.org'})");
        assert!(unnamed.contains("NOT NULL"), "{name}: {unnamed}");
        if !fixture.replays_into_a_named_graph() {
            let museum = error("INSERT (:Museum {name: 'Rijksmuseum'})");
            assert!(museum.contains("UNIQUE"), "{name}: {museum}");
        }
        assert_eq!(
            names(&db, "SHOW CONSTRAINTS"),
            fixture.constraint_names(),
            "{name}: constraint names"
        );
    }
}

#[test]
fn the_schema_holds_what_was_written() {
    for fixture in FIXTURES {
        let (_dir, db) = fixture.open();
        let name = fixture.name();
        assert_eq!(
            names(&db, "SHOW NODE TYPES"),
            ["Capital", "City", "Museum", "Person"],
            "{name}: node types"
        );
        assert_eq!(
            names(&db, "SHOW GRAPH TYPES"),
            ["culture", "travel"],
            "{name}: graph types"
        );

        let first = !fixture.first_session_schema_in_wal();
        let second = !fixture.second_session_schema_in_wal();
        let column = |query: &str, type_name: &str, columns: std::ops::Range<usize>| {
            rows(&db, query)
                .into_iter()
                .find(|row| row[0] == Value::from(type_name))
                .map_or_else(
                    || panic!("{name}: no {type_name} in {query}"),
                    |row| row[columns].to_vec(),
                )
        };
        let kept = |kept: bool, values: &[&str]| -> Vec<Value> {
            values
                .iter()
                .map(|value| Value::from(if kept { *value } else { "" }))
                .collect()
        };
        assert_eq!(
            column("SHOW NODE TYPES", "Capital", 3..4),
            kept(first, &["City"]),
            "{name}: parent of Capital"
        );
        assert_eq!(
            column("SHOW EDGE TYPES", "ROUTE", 2..4),
            kept(first, &["City", "City"]),
            "{name}: endpoints of ROUTE"
        );
        assert_eq!(
            column("SHOW EDGE TYPES", "IN_CITY", 2..4),
            kept(second, &["Museum", "City"]),
            "{name}: endpoints of IN_CITY"
        );

        // Default values apply to new nodes.
        db.execute("INSERT (:City {name: 'Barcelona'})").unwrap();
        assert_eq!(
            rows(&db, "MATCH (c:City {name: 'Barcelona'}) RETURN c.country"),
            [vec![if first {
                Value::from("NL")
            } else {
                Value::Null
            }]],
            "{name}: default of City.country"
        );
        db.execute("INSERT (:Museum {name: 'Stedelijk'})").unwrap();
        assert_eq!(
            rows(&db, "MATCH (m:Museum {name: 'Stedelijk'}) RETURN m.open"),
            [vec![if second {
                Value::Bool(true)
            } else {
                Value::Null
            }]],
            "{name}: default of Museum.open"
        );
    }
}

#[test]
fn indexes_hold_what_was_written() {
    let as_value = |node: NodeId| Value::Int64(i64::try_from(node.as_u64()).unwrap());
    for fixture in FIXTURES {
        let (_dir, db) = fixture.open();
        let name = fixture.name();
        assert_eq!(
            names(&db, "SHOW INDEXES"),
            fixture.index_names(),
            "{name}: indexes"
        );
        let vector = db.vector_search("Document", "embedding", &[1.0, 0.0, 0.0], 1, None, None);
        let text = db.text_search("Document", "content", "canals", 3);
        if !fixture.has_search_indexes() {
            assert!(vector.is_err(), "{name}: a vector index");
            assert!(text.is_err(), "{name}: a text index");
            continue;
        }
        let id = |title: &str| {
            let found = rows(
                &db,
                &format!("MATCH (d:Document {{title: '{title}'}}) RETURN id(d)"),
            );
            assert_eq!(found.len(), 1, "{name}: documents titled {title}");
            found[0][0].clone()
        };
        let nearest: Vec<Value> = vector
            .unwrap_or_else(|error| panic!("{name}: vector search: {error}"))
            .into_iter()
            .map(|(node, _)| as_value(node))
            .collect();
        assert_eq!(nearest, [id("Canals")], "{name}: vector search");
        let mut matches: Vec<Value> = text
            .unwrap_or_else(|error| panic!("{name}: text search: {error}"))
            .into_iter()
            .map(|(node, _)| as_value(node))
            .collect();
        let mut expected = vec![id("Canals"), id("Bridges")];
        matches.sort_by_key(Value::as_int64);
        expected.sort_by_key(Value::as_int64);
        assert_eq!(matches, expected, "{name}: text search");
    }
}
