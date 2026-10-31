//! The native CSV loader (0.6.1, `.superpowers/sdd/r061/loader.md`):
//! `GrafeoDB::load_nodes_csv` and `load_edges_csv`, and the procedures
//! `grafeo.load_nodes_csv` / `grafeo.load_edges_csv` in GQL and Cypher.
//!
//! - typed values per column kind (String, Int64, Float64, Bool, Date,
//!   Timestamp, Skip), every column a string without a column list;
//! - nulls: an empty field and a null marker give no property;
//! - CSV rules: quoted fields with the delimiter, escaped quotes and line
//!   breaks inside, a BOM, `\r\n`, another delimiter, no header;
//! - a header-only file, a missing file, a bad value (row and column named);
//! - edges: ends found by key, with an index on the key and without, a
//!   missing end (an error naming the value, or skipped), edge properties;
//! - a repeated `UNIQUE` key; one bad row loads nothing (one transaction),
//!   also inside a session's transaction that rolls back;
//! - the report: rows, created, skipped.
//!
//! The files are written by the test into its own temporary directory.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test csv_loader
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::loader::{
    Column, ColumnKind, EdgeLoad, EndKey, LoadReport, MissingEnd, NodeLoad,
};

/// A file named `name` with `content`, in `dir`.
fn file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    path
}

fn column(name: &str, kind: ColumnKind) -> Column {
    Column {
        name: name.to_string(),
        kind,
    }
}

fn nodes(labels: &[&str], key: Option<&str>, columns: Vec<Column>) -> NodeLoad {
    NodeLoad {
        labels: labels.iter().map(ToString::to_string).collect(),
        key: key.map(ToString::to_string),
        columns,
        ..NodeLoad::default()
    }
}

fn end(label: &str, property: &str, column: &str) -> EndKey {
    EndKey {
        label: label.to_string(),
        property: property.to_string(),
        column: column.to_string(),
    }
}

fn edges(edge_type: &str, source: EndKey, target: EndKey, columns: Vec<Column>) -> EdgeLoad {
    EdgeLoad {
        edge_type: edge_type.to_string(),
        source,
        target,
        columns,
        ..EdgeLoad::default()
    }
}

fn report(rows: u64, created: u64, skipped: u64) -> LoadReport {
    LoadReport {
        rows,
        created,
        skipped,
    }
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("`{query}`: {error}"))
        .rows()
        .to_vec()
}

fn count(db: &GrafeoDB, query: &str) -> i64 {
    match rows(db, query)[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("`{query}`: {other:?}"),
    }
}

const PEOPLE: &str = "id,name,score,active,born,seen,junk\n\
                      1,Alix,9.5,true,1990-05-01,2024-01-15T10:30:00Z,x\n\
                      2,Gus,7,false,1985-12-24,2024-02-01T08:00:00Z,y\n\
                      3,Vincent,8.25,TRUE,2001-02-03,2024-03-10T23:59:59Z,z\n";

fn people_columns() -> Vec<Column> {
    vec![
        column("id", ColumnKind::Int64),
        column("name", ColumnKind::String),
        column("score", ColumnKind::Float64),
        column("active", ColumnKind::Bool),
        column("born", ColumnKind::Date),
        column("seen", ColumnKind::Timestamp),
        column("junk", ColumnKind::Skip),
    ]
}

/// Each column kind gives its value type; a Skip column is not stored.
#[test]
fn typed_values_per_kind() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "people.csv", PEOPLE);
    let db = GrafeoDB::new_in_memory();
    let loaded = db
        .load_nodes_csv(&path, &nodes(&["Person"], Some("id"), people_columns()))
        .unwrap();
    assert_eq!(loaded, report(3, 3, 0));
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person) RETURN n.id, n.name, n.score, n.active, \
             n.born = DATE '1990-05-01', toString(n.seen) STARTS WITH '2024-01-15T10:30', \
             n.junk IS NULL ORDER BY n.id LIMIT 1"
        ),
        [[
            Value::Int64(1),
            Value::from("Alix"),
            Value::Float64(9.5),
            Value::Bool(true),
            Value::Bool(true),
            Value::Bool(true),
            Value::Bool(true),
        ]]
    );
    // An integer text in a Float64 column is a float; TRUE is a Bool.
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person) WHERE n.id > 1 RETURN n.score, n.active ORDER BY n.id"
        ),
        [
            [Value::Float64(7.0), Value::Bool(false)],
            [Value::Float64(8.25), Value::Bool(true)]
        ]
    );
}

/// Without a column list every header column is a string, as LOAD CSV.
#[test]
fn every_column_a_string_without_a_column_list() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "people.csv", PEOPLE);
    let db = GrafeoDB::new_in_memory();
    db.load_nodes_csv(&path, &nodes(&["Person"], None, vec![]))
        .unwrap();
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person {name: 'Gus'}) RETURN n.id, n.score, n.junk"
        ),
        [[Value::from("2"), Value::from("7"), Value::from("y")]]
    );
}

/// An empty field, and a field equal to a null marker, give no property.
#[test]
fn empty_fields_and_null_markers_give_no_property() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(
        dir.path(),
        "people.csv",
        "id,name,score\n1,,9.5\n2,Gus,NULL\n",
    );
    let db = GrafeoDB::new_in_memory();
    let mut spec = nodes(
        &["Person"],
        Some("id"),
        vec![
            column("id", ColumnKind::Int64),
            column("name", ColumnKind::String),
            column("score", ColumnKind::Float64),
        ],
    );
    spec.nulls = vec!["NULL".to_string()];
    assert_eq!(db.load_nodes_csv(&path, &spec).unwrap(), report(2, 2, 0));
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person) RETURN n.id, n.name IS NULL, n.score IS NULL ORDER BY n.id"
        ),
        [
            [Value::Int64(1), Value::Bool(true), Value::Bool(false)],
            [Value::Int64(2), Value::Bool(false), Value::Bool(true)]
        ]
    );
}

/// Quoted fields hold the delimiter, escaped quotes and line breaks; a BOM
/// and `\r\n` line ends are read as in LOAD CSV.
#[test]
fn quoted_fields_bom_and_crlf() {
    let dir = tempfile::tempdir().unwrap();
    let content = "\u{feff}id,text\r\n1,\"Amsterdam, Berlin\"\r\n2,\"say \"\"hi\"\"\"\r\n3,\"two\r\nlines\"\r\n";
    let path = file(dir.path(), "texts.csv", content);
    let db = GrafeoDB::new_in_memory();
    let spec = nodes(
        &["Text"],
        Some("id"),
        vec![
            column("id", ColumnKind::Int64),
            column("text", ColumnKind::String),
        ],
    );
    assert_eq!(db.load_nodes_csv(&path, &spec).unwrap(), report(3, 3, 0));
    assert_eq!(
        rows(&db, "MATCH (n:Text) RETURN n.text ORDER BY n.id"),
        [
            [Value::from("Amsterdam, Berlin")],
            [Value::from("say \"hi\"")],
            [Value::from("two\r\nlines")]
        ]
    );
}

/// Another delimiter, and a file without a header: the columns are named by
/// the column list, in file order.
#[test]
fn another_delimiter_and_no_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(
        dir.path(),
        "cities.csv",
        "1;Amsterdam;921000\n2;Prague;1357000\n",
    );
    let db = GrafeoDB::new_in_memory();
    let mut spec = nodes(
        &["City"],
        Some("id"),
        vec![
            column("id", ColumnKind::Int64),
            column("name", ColumnKind::String),
            column("pop", ColumnKind::Int64),
        ],
    );
    spec.delimiter = b';';
    spec.header = false;
    assert_eq!(db.load_nodes_csv(&path, &spec).unwrap(), report(2, 2, 0));
    assert_eq!(
        rows(&db, "MATCH (c:City) RETURN c.name, c.pop ORDER BY c.id"),
        [
            [Value::from("Amsterdam"), Value::Int64(921_000)],
            [Value::from("Prague"), Value::Int64(1_357_000)]
        ]
    );
}

/// A header-only file loads nothing and reports zero rows.
#[test]
fn a_header_only_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "empty.csv", "id,name\n");
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        db.load_nodes_csv(&path, &nodes(&["Person"], None, vec![]))
            .unwrap(),
        report(0, 0, 0)
    );
    assert_eq!(count(&db, "MATCH (n) RETURN count(n)"), 0);
}

/// A missing file is an error naming the path.
#[test]
fn a_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nowhere.csv");
    let db = GrafeoDB::new_in_memory();
    let error = db
        .load_nodes_csv(&path, &nodes(&["Person"], None, vec![]))
        .unwrap_err()
        .to_string();
    assert!(error.contains("nowhere.csv"), "{error}");
}

/// A value its column's kind cannot read is an error naming the row and
/// the column, and nothing of the file is loaded (one transaction).
#[test]
fn a_bad_value_names_row_and_column_and_loads_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(
        dir.path(),
        "people.csv",
        "id,name,score\n1,Alix,9.5\n2,Gus,lots\n3,Vincent,8\n",
    );
    let db = GrafeoDB::new_in_memory();
    let spec = nodes(
        &["Person"],
        Some("id"),
        vec![
            column("id", ColumnKind::Int64),
            column("name", ColumnKind::String),
            column("score", ColumnKind::Float64),
        ],
    );
    let error = db.load_nodes_csv(&path, &spec).unwrap_err().to_string();
    assert!(
        error.contains("score") && error.contains("lots") && error.contains('3'),
        "the column, the value and the line (3, counting the header): {error}"
    );
    assert_eq!(count(&db, "MATCH (n) RETURN count(n)"), 0);
}

/// A key repeated under a UNIQUE constraint fails the load, and loads
/// nothing.
#[test]
fn a_repeated_unique_key_loads_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "people.csv", "id,name\n1,Alix\n2,Gus\n1,Mia\n");
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.id IS UNIQUE")
        .unwrap();
    let spec = nodes(
        &["Person"],
        Some("id"),
        vec![
            column("id", ColumnKind::Int64),
            column("name", ColumnKind::String),
        ],
    );
    assert!(db.load_nodes_csv(&path, &spec).is_err());
    assert_eq!(count(&db, "MATCH (n) RETURN count(n)"), 0);
}

/// The key column is unique within the file, without a constraint: a
/// repeated key is an error naming the value and both lines, and loads
/// nothing.
#[test]
fn a_repeated_key_in_the_file_loads_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "people.csv", "id,name\n1,Alix\n2,Gus\n1,Mia\n");
    let db = GrafeoDB::new_in_memory();
    let spec = nodes(
        &["Person"],
        Some("id"),
        vec![
            column("id", ColumnKind::Int64),
            column("name", ColumnKind::String),
        ],
    );
    let error = db.load_nodes_csv(&path, &spec).unwrap_err().to_string();
    assert!(
        error.contains('1') && error.contains('2') && error.contains('4'),
        "the value 1 and its lines 2 and 4: {error}"
    );
    assert_eq!(count(&db, "MATCH (n) RETURN count(n)"), 0);
}

/// A row without a key is an error naming its line, and loads nothing.
#[test]
fn a_null_key_names_its_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "people.csv", "id,name\n1,Alix\n,Gus\n3,Mia\n");
    let db = GrafeoDB::new_in_memory();
    let spec = nodes(
        &["Person"],
        Some("id"),
        vec![
            column("id", ColumnKind::Int64),
            column("name", ColumnKind::String),
        ],
    );
    let error = db.load_nodes_csv(&path, &spec).unwrap_err().to_string();
    assert!(
        error.contains("id") && error.contains('3'),
        "the key column and line 3: {error}"
    );
    assert_eq!(count(&db, "MATCH (n) RETURN count(n)"), 0);
}

/// A load inside a session's transaction rolls back with it.
#[test]
fn a_load_in_a_transaction_rolls_back_with_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(dir.path(), "people.csv", PEOPLE);
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .load_nodes_csv(&path, &nodes(&["Person"], Some("id"), people_columns()))
        .unwrap();
    assert_eq!(
        session
            .execute("MATCH (n:Person) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(3)
    );
    session.rollback().unwrap();
    assert_eq!(count(&db, "MATCH (n) RETURN count(n)"), 0);
}

const KNOWS: &str = "src,dst,since\n1,2,2019\n2,3,2021\n3,1,2020\n";

fn knows(missing: MissingEnd) -> EdgeLoad {
    let mut spec = edges(
        "KNOWS",
        end("Person", "id", "src"),
        end("Person", "id", "dst"),
        vec![
            column("src", ColumnKind::Int64),
            column("dst", ColumnKind::Int64),
            column("since", ColumnKind::Int64),
        ],
    );
    spec.missing = missing;
    spec
}

/// People, then KNOWS edges found by `id`, with an index on `id` and
/// without: the same edges, with their property, and the key columns are
/// not stored on the edge.
#[test]
fn edges_by_key_with_and_without_an_index() {
    for index in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let people = file(dir.path(), "people.csv", PEOPLE);
        let knows_path = file(dir.path(), "knows.csv", KNOWS);
        let db = GrafeoDB::new_in_memory();
        if index {
            db.create_property_index("id").unwrap();
        }
        db.load_nodes_csv(&people, &nodes(&["Person"], Some("id"), people_columns()))
            .unwrap();
        assert_eq!(
            db.load_edges_csv(&knows_path, &knows(MissingEnd::Error))
                .unwrap(),
            report(3, 3, 0),
            "index {index}"
        );
        assert_eq!(
            rows(
                &db,
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) \
                 RETURN a.name, b.name, r.since, r.src IS NULL ORDER BY r.since"
            ),
            [
                [
                    Value::from("Alix"),
                    Value::from("Gus"),
                    Value::Int64(2019),
                    Value::Bool(true)
                ],
                [
                    Value::from("Vincent"),
                    Value::from("Alix"),
                    Value::Int64(2020),
                    Value::Bool(true)
                ],
                [
                    Value::from("Gus"),
                    Value::from("Vincent"),
                    Value::Int64(2021),
                    Value::Bool(true)
                ]
            ],
            "index {index}"
        );
    }
}

/// An end whose key no node has: an error naming the value that loads no
/// edge, or, with skip, the row is skipped and reported.
#[test]
fn a_missing_end_is_an_error_or_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let people = file(dir.path(), "people.csv", PEOPLE);
    let knows_path = file(
        dir.path(),
        "knows.csv",
        "src,dst,since\n1,2,2019\n2,77,2021\n3,1,2020\n",
    );
    let db = GrafeoDB::new_in_memory();
    db.load_nodes_csv(&people, &nodes(&["Person"], Some("id"), people_columns()))
        .unwrap();
    let error = db
        .load_edges_csv(&knows_path, &knows(MissingEnd::Error))
        .unwrap_err()
        .to_string();
    assert!(error.contains("77"), "the missing key: {error}");
    assert_eq!(count(&db, "MATCH ()-[r:KNOWS]->() RETURN count(r)"), 0);
    assert_eq!(
        db.load_edges_csv(&knows_path, &knows(MissingEnd::Skip))
            .unwrap(),
        report(3, 2, 1)
    );
    assert_eq!(count(&db, "MATCH ()-[r:KNOWS]->() RETURN count(r)"), 2);
}

/// An end key of another type than the stored key does not match: the
/// file's `src` read as text finds no Person whose `id` is an Int64.
#[test]
fn an_end_key_of_another_type_finds_no_node() {
    let dir = tempfile::tempdir().unwrap();
    let people = file(dir.path(), "people.csv", PEOPLE);
    let knows_path = file(dir.path(), "knows.csv", KNOWS);
    let db = GrafeoDB::new_in_memory();
    db.load_nodes_csv(&people, &nodes(&["Person"], Some("id"), people_columns()))
        .unwrap();
    let mut spec = knows(MissingEnd::Skip);
    spec.columns = vec![column("since", ColumnKind::Int64)];
    assert_eq!(
        db.load_edges_csv(&knows_path, &spec).unwrap(),
        report(3, 0, 3),
        "the keys read as strings"
    );
}

/// The procedures, in GQL and Cypher: the same nodes and edges, and the
/// report as YIELD columns; type names in any case.
#[test]
fn the_procedures_in_gql_and_cypher() {
    for language in ["gql", "cypher"] {
        let dir = tempfile::tempdir().unwrap();
        let people = file(dir.path(), "people.csv", PEOPLE);
        let knows_path = file(dir.path(), "knows.csv", KNOWS);
        let slash = |p: &Path| p.to_string_lossy().replace('\\', "/");
        let db = GrafeoDB::new_in_memory();
        let loaded = db
            .execute_language(
                &format!(
                    "CALL grafeo.load_nodes_csv({{path: '{}', labels: ['Person'], key: 'id', \
                     types: {{id: 'Int64', score: 'FLOAT64', active: 'bool', born: 'Date', \
                     seen: 'TIMESTAMP', junk: 'skip'}}}}) YIELD rows, created, skipped \
                     RETURN rows, created, skipped",
                    slash(&people)
                ),
                language,
                None,
            )
            .unwrap_or_else(|error| panic!("{language}: {error}"));
        assert_eq!(
            loaded.rows(),
            [[Value::Int64(3), Value::Int64(3), Value::Int64(0)]],
            "{language}"
        );
        let linked = db
            .execute_language(
                &format!(
                    "CALL grafeo.load_edges_csv({{path: '{}', type: 'KNOWS', \
                     source: {{label: 'Person', property: 'id', column: 'src'}}, \
                     target: {{label: 'Person', property: 'id', column: 'dst'}}, \
                     types: {{src: 'INT64', dst: 'INT64', since: 'INT64'}}}}) \
                     YIELD rows, created, skipped RETURN rows, created, skipped",
                    slash(&knows_path)
                ),
                language,
                None,
            )
            .unwrap_or_else(|error| panic!("{language}: {error}"));
        assert_eq!(
            linked.rows(),
            [[Value::Int64(3), Value::Int64(3), Value::Int64(0)]],
            "{language}"
        );
        assert_eq!(
            count(
                &db,
                "MATCH (:Person)-[r:KNOWS]->(:Person) WHERE r.since >= 2019 RETURN count(r)"
            ),
            3,
            "{language}"
        );
    }
}
