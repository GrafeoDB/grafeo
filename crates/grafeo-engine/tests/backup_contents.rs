//! What a backup chain restores: exactly the committed state of each
//! epoch, with what the database holds besides its nodes (#267, #362).
//!
//! Each test takes a full backup, changes the database, takes an
//! incremental backup, and restores to both epochs: a rolled-back
//! transaction between the backups is left out, the writes of a named graph
//! are kept, the property, vector and text indexes and the stored
//! configuration come back, and so does a SPARQL named-graph MODIFY.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test backup_contents
//! ```

#![cfg(all(feature = "wal", feature = "grafeo-file", feature = "gql"))]

use std::path::Path;

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::{Config, GrafeoDB};

fn names(db: &GrafeoDB, graph: Option<&str>) -> Vec<Value> {
    let session = db.session();
    if let Some(graph) = graph {
        session.use_graph(graph);
    }
    session
        .execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

fn strings(values: &[&str]) -> Vec<Value> {
    values.iter().map(|value| Value::from(*value)).collect()
}

/// Simple RDF literals, as SPARQL results hold them.
fn literals(values: &[&str]) -> Vec<Value> {
    values
        .iter()
        .map(|value| {
            Value::RdfTerm(std::sync::Arc::new(
                grafeo_common::types::rdf::Term::literal(*value),
            ))
        })
        .collect()
}

/// Restores `backups` to `epoch` in a new directory, opens the result with
/// `open` and runs `check` on it, then once more on a read-only open.
fn restored(
    backups: &Path,
    epoch: EpochId,
    open: impl Fn(&Path) -> GrafeoDB,
    check: impl Fn(&GrafeoDB, &str),
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(backups, epoch, &path).unwrap();
    let db = open(&path);
    check(&db, "restored");
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::open_read_only(&path).unwrap();
    check(&db, "restored, read-only");
    db.close().unwrap();
}

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::open(path).unwrap()
}

/// A transaction rolled back between the backups is in neither: the
/// incremental restores the commits around it.
#[test]
fn an_incremental_backup_leaves_out_a_rolled_back_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let backups = dir.path().join("backups");
    let db = open(&dir.path().join("db.grafeo"));
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let full = db.backup_full(&backups).unwrap().end_epoch;
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.rollback().unwrap();
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    let incremental = db.backup_incremental(&backups).unwrap().end_epoch;
    db.close().unwrap();

    restored(&backups, full, open, |db, what| {
        assert_eq!(names(db, None), strings(&["Alix"]), "{what}, full");
    });
    restored(&backups, incremental, open, |db, what| {
        assert_eq!(
            names(db, None),
            strings(&["Alix", "Jules"]),
            "{what}, incremental"
        );
    });
}

/// The writes of a named graph between the backups, the graph's creation
/// among them, come back in that graph, and the default graph keeps its own.
#[test]
fn an_incremental_backup_carries_named_graph_writes() {
    let dir = tempfile::tempdir().unwrap();
    let backups = dir.path().join("backups");
    let db = open(&dir.path().join("db.grafeo"));
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let full = db.backup_full(&backups).unwrap().end_epoch;
    db.execute("CREATE GRAPH trips").unwrap();
    let session = db.session();
    session.use_graph("trips");
    session
        .execute("INSERT (:Person {name: 'Mia'})-[:VISITED]->(:City {name: 'Prague'})")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let incremental = db.backup_incremental(&backups).unwrap().end_epoch;
    db.close().unwrap();

    restored(&backups, full, open, |db, what| {
        assert_eq!(db.list_graphs(), Vec::<String>::new(), "{what}, full");
    });
    restored(&backups, incremental, open, |db, what| {
        assert_eq!(names(db, None), strings(&["Alix", "Gus"]), "{what}");
        assert_eq!(db.list_graphs(), ["trips"], "{what}");
        assert_eq!(names(db, Some("trips")), strings(&["Mia"]), "{what}");
        let session = db.session();
        session.use_graph("trips");
        assert_eq!(
            session
                .execute("MATCH (p)-[:VISITED]->(c) RETURN p.name, c.name")
                .unwrap()
                .rows(),
            [[Value::from("Mia"), Value::from("Prague")]],
            "{what}"
        );
    });
}

/// A property index made before the full backup and one made after it come
/// back with a restore to each epoch, and the lookups use them (#638); so do
/// a vector and a text index (#374), with the nodes written after them.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn a_restore_keeps_property_vector_and_text_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let backups = dir.path().join("backups");
    let db = open(&dir.path().join("db.grafeo"));
    db.execute(
        "INSERT (:Doc {title: 'Canals', body: 'boats on the canals of Amsterdam', \
         emb: vector([1.0, 0.0, 0.0])})",
    )
    .unwrap();
    db.execute("CREATE INDEX doc_title FOR (d:Doc) ON (d.title)")
        .unwrap();
    db.create_vector_index("Doc", "emb", Some(3), Some("euclidean"), None, None, None)
        .unwrap();
    db.create_text_index("Doc", "body").unwrap();
    let full = db.backup_full(&backups).unwrap().end_epoch;
    db.execute(
        "INSERT (:Doc {title: 'Bridges', body: 'bridges over the river in Prague', \
         emb: vector([0.0, 1.0, 0.0])})",
    )
    .unwrap();
    db.execute("CREATE INDEX doc_year FOR (d:Doc) ON (d.year)")
        .unwrap();
    db.execute("MATCH (d:Doc {title: 'Bridges'}) SET d.year = 2026")
        .unwrap();
    let incremental = db.backup_incremental(&backups).unwrap().end_epoch;
    db.close().unwrap();

    let check = |db: &GrafeoDB, titles: &[&str], what: &str| {
        assert!(db.has_property_index("title"), "{what}: the title index");
        let plan = db
            .execute("EXPLAIN MATCH (d:Doc {title: 'Canals'}) RETURN d.body")
            .unwrap();
        let Value::String(plan) = &plan.rows()[0][0] else {
            panic!("{what}: a plan");
        };
        assert!(plan.contains("[index: title]"), "{what}: {plan}");
        let title = |id: grafeo_common::types::NodeId| {
            db.get_node(id)
                .unwrap()
                .and_then(|node| node.get_property("title").cloned())
                .unwrap()
        };
        let nearest: Vec<Value> = db
            .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], titles.len(), None, None)
            .unwrap()
            .into_iter()
            .map(|(id, _)| title(id))
            .collect();
        let mut expected = strings(titles);
        expected.reverse();
        assert_eq!(nearest, expected, "{what}: vector search, nearest first");
        let found: Vec<Value> = db
            .text_search("Doc", "body", "canals", 5, None)
            .unwrap()
            .into_iter()
            .map(|(id, _)| title(id))
            .collect();
        assert_eq!(found, strings(&["Canals"]), "{what}: text search");
    };
    restored(&backups, full, open, |db, what| {
        check(db, &["Canals"], &format!("{what}, full"));
        assert!(!db.has_property_index("year"), "{what}, full");
    });
    restored(&backups, incremental, open, |db, what| {
        check(db, &["Canals", "Bridges"], &format!("{what}, incremental"));
        assert!(
            db.has_property_index("year"),
            "{what}: the index made after the full backup"
        );
        assert_eq!(
            db.execute("MATCH (d:Doc {year: 2026}) RETURN d.title")
                .unwrap()
                .rows(),
            [[Value::from("Bridges")]],
            "{what}"
        );
    });
}

/// A restored backup keeps the configuration its database was created with
/// (grafeo-server#67): an open without settings applies the stored memory
/// limit.
#[test]
fn a_restore_keeps_the_stored_configuration() {
    let dir = tempfile::tempdir().unwrap();
    let backups = dir.path().join("backups");
    let limit = 88 * 1024 * 1024;
    let db = GrafeoDB::with_config(
        Config::persistent(dir.path().join("db.grafeo")).with_memory_limit(limit),
    )
    .unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let full = db.backup_full(&backups).unwrap().end_epoch;
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let incremental = db.backup_incremental(&backups).unwrap().end_epoch;
    db.close().unwrap();

    for (epoch, people) in [(full, &["Alix"][..]), (incremental, &["Alix", "Gus"][..])] {
        restored(&backups, epoch, open, |db, what| {
            assert_eq!(db.memory_limit(), Some(limit), "{what}");
            assert_eq!(names(db, None), strings(people), "{what}");
        });
    }
}

/// A SPARQL MODIFY of a named graph between the backups (#367) restores in
/// that graph: the old value is gone, the new one is there once, and the
/// default graph is untouched.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn a_named_graph_modify_restores_in_its_graph() {
    use grafeo_engine::GraphModel;

    let open_rdf = |path: &Path| {
        GrafeoDB::with_config(Config::persistent(path).with_graph_model(GraphModel::Rdf)).unwrap()
    };
    let statuses = |db: &GrafeoDB, graph: Option<&str>| {
        let query = match graph {
            Some(graph) => format!(
                "SELECT ?s WHERE {{ GRAPH <{graph}> {{ <http://ex.org/alix> <http://ex.org/status> ?s }} }} ORDER BY ?s"
            ),
            None => {
                "SELECT ?s WHERE { <http://ex.org/alix> <http://ex.org/status> ?s } ORDER BY ?s"
                    .to_string()
            }
        };
        db.execute_sparql(&query)
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[0].clone())
            .collect::<Vec<Value>>()
    };
    let dir = tempfile::tempdir().unwrap();
    let backups = dir.path().join("backups");
    let db = open_rdf(&dir.path().join("db.grafeo"));
    db.execute_sparql(
        r#"INSERT DATA {
               <http://ex.org/alix> <http://ex.org/status> "default" .
               GRAPH <http://ex.org/g> { <http://ex.org/alix> <http://ex.org/status> "active" . }
           }"#,
    )
    .unwrap();
    let full = db.backup_full(&backups).unwrap().end_epoch;
    db.execute_sparql(
        r#"WITH <http://ex.org/g>
           DELETE { <http://ex.org/alix> <http://ex.org/status> ?s }
           INSERT { <http://ex.org/alix> <http://ex.org/status> "archived" }
           WHERE  { <http://ex.org/alix> <http://ex.org/status> ?s }"#,
    )
    .unwrap();
    let incremental = db.backup_incremental(&backups).unwrap().end_epoch;
    db.close().unwrap();

    restored(&backups, full, open_rdf, |db, what| {
        assert_eq!(
            statuses(db, Some("http://ex.org/g")),
            literals(&["active"]),
            "{what}, full"
        );
    });
    restored(&backups, incremental, open_rdf, |db, what| {
        assert_eq!(
            statuses(db, Some("http://ex.org/g")),
            literals(&["archived"]),
            "{what}, incremental"
        );
        assert_eq!(
            statuses(db, None),
            literals(&["default"]),
            "{what}, the default graph"
        );
    });
}

/// The backup chain survives a close: an incremental backup taken after
/// the database was closed and opened again (as grafeo-server does on a
/// restart, and the CLI on every command) holds the commits since the full
/// backup, and the restore to its epoch has them.
#[test]
fn an_incremental_backup_works_after_a_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let backups = dir.path().join("backups");
    let db = open(&path);
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let full = db.backup_full(&backups).unwrap().end_epoch;
    db.close().unwrap();
    drop(db);

    let db = open(&path);
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.close().unwrap();
    drop(db);

    let db = open(&path);
    let incremental = db.backup_incremental(&backups).unwrap().end_epoch;
    db.close().unwrap();

    restored(&backups, full, open, |db, what| {
        assert_eq!(names(db, None), strings(&["Alix"]), "{what}, full");
    });
    restored(&backups, incremental, open, |db, what| {
        assert_eq!(
            names(db, None),
            strings(&["Alix", "Gus"]),
            "{what}, incremental"
        );
    });
}
