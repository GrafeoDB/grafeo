//! `import::import_nodes`: every row of a CSV or JSON Lines file becomes a
//! node, and every column or key a property under the name the file gives
//! it. The bindings and the CLI built their own `LOAD DATA` statements with
//! sanitized names, so a header with a space, a BOM, a quoted header holding
//! the delimiter or a JSONL key that first appears on a later line lost its
//! values without an error.

#![cfg(feature = "gql")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_common::utils::error::Error;
use grafeo_engine::GrafeoDB;
use grafeo_engine::import::{ImportFormat, import_nodes};

const CSV: ImportFormat = ImportFormat::Csv {
    headers: true,
    delimiter: None,
};

const RAW_CSV: ImportFormat = ImportFormat::Csv {
    headers: false,
    delimiter: None,
};

fn file(dir: &tempfile::TempDir, name: &str, content: &str) -> PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, content).unwrap();
    path
}

/// The properties of every node labeled `label`, as text, sorted.
fn properties(db: &GrafeoDB, label: &str) -> Vec<BTreeMap<String, String>> {
    let label = label.replace('`', "``");
    let result = db
        .session()
        .execute(&format!("MATCH (n:`{label}`) RETURN properties(n) AS p"))
        .unwrap();
    let mut nodes: Vec<BTreeMap<String, String>> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::Map(map) => map
                .iter()
                .map(|(key, value)| (key.as_str().to_string(), text(value)))
                .collect(),
            other => panic!("properties() returned {other:?}"),
        })
        .collect();
    nodes.sort();
    nodes
}

fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(value) => format!("int {value}"),
        Value::Float64(value) => format!("float {value}"),
        other => format!("{other:?}"),
    }
}

fn node(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

fn import(db: &GrafeoDB, path: &Path, label: &str, format: ImportFormat) -> u64 {
    import_nodes(&db.session(), path, label, format).unwrap()
}

#[test]
fn a_header_with_a_space_keeps_its_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(
        &dir,
        "people.csv",
        "name,home city\nAlix,Amsterdam\nGus,Berlin\n",
    );
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Person", CSV), 2);
    assert_eq!(
        properties(&db, "Person"),
        [
            node(&[("home city", "Amsterdam"), ("name", "Alix")]),
            node(&[("home city", "Berlin"), ("name", "Gus")]),
        ]
    );
}

#[test]
fn punctuation_in_headers_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(
        &dir,
        "contacts.csv",
        "e-mail,geo.lat,1st\nmia@example.com,52.37,yes\n",
    );
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Contact", CSV), 1);
    assert_eq!(
        properties(&db, "Contact"),
        [node(&[
            ("1st", "yes"),
            ("e-mail", "mia@example.com"),
            ("geo.lat", "52.37")
        ])]
    );
}

/// Excel writes a byte order mark before the first header.
#[test]
fn a_byte_order_mark_does_not_rename_the_first_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "excel.csv", "\u{feff}id,city\n3,Paris\n");
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Row", CSV), 1);
    assert_eq!(
        properties(&db, "Row"),
        [node(&[("city", "Paris"), ("id", "3")])]
    );
}

#[test]
fn a_quoted_header_holding_the_delimiter_is_one_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "quoted.csv", "name,\"Amsterdam, NL\"\nJules,19\n");
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Row", CSV), 1);
    assert_eq!(
        properties(&db, "Row"),
        [node(&[("Amsterdam, NL", "19"), ("name", "Jules")])]
    );
}

#[test]
fn an_empty_field_sets_no_property() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "gaps.csv", "name,city\nVincent,\n");
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Row", CSV), 1);
    assert_eq!(properties(&db, "Row"), [node(&[("name", "Vincent")])]);
}

#[test]
fn a_delimiter_other_than_a_comma() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "semicolons.csv", "name;city\nButch;Prague\n");
    let db = GrafeoDB::new_in_memory();
    let format = ImportFormat::Csv {
        headers: true,
        delimiter: Some(';'),
    };
    assert_eq!(import(&db, &path, "Row", format), 1);
    assert_eq!(
        properties(&db, "Row"),
        [node(&[("city", "Prague"), ("name", "Butch")])]
    );
}

/// As DuckDB names them: `column0`, `column1`, as many as the widest record
/// has; a shorter record leaves its last columns unset.
#[test]
fn a_csv_without_headers_names_its_columns_by_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "raw.csv", "Alix,Amsterdam\nGus,Berlin,3\n\nMia\n");
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Row", RAW_CSV), 3);
    assert_eq!(
        properties(&db, "Row"),
        [
            node(&[("column0", "Alix"), ("column1", "Amsterdam")]),
            node(&[("column0", "Gus"), ("column1", "Berlin"), ("column2", "3")]),
            node(&[("column0", "Mia")]),
        ]
    );
}

#[test]
fn a_csv_without_headers_and_with_a_delimiter() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "tabs.tsv", "Django\tBarcelona\n");
    let db = GrafeoDB::new_in_memory();
    let format = ImportFormat::Csv {
        headers: false,
        delimiter: Some('\t'),
    };
    assert_eq!(import(&db, &path, "Row", format), 1);
    assert_eq!(
        properties(&db, "Row"),
        [node(&[("column0", "Django"), ("column1", "Barcelona")])]
    );
}

#[test]
fn a_csv_with_only_headers_creates_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let headers_only = file(&dir, "headers.csv", "name,city\n");
    let empty = file(&dir, "empty.csv", "");
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &headers_only, "Row", CSV), 0);
    assert_eq!(import(&db, &empty, "Row", RAW_CSV), 0);
    let nodes = properties(&db, "Row");
    assert!(nodes.is_empty(), "{nodes:?}");
}

#[cfg(feature = "jsonl-import")]
#[test]
fn every_jsonl_key_becomes_a_property_with_its_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(
        &dir,
        "events.jsonl",
        "{\"id\": 1}\n\n{\"id\": 2, \"geo.lat\": 52.5, \"home city\": \"Berlin\"}\n",
    );
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Event", ImportFormat::Jsonl), 2);
    assert_eq!(
        properties(&db, "Event"),
        [
            node(&[
                ("geo.lat", "float 52.5"),
                ("home city", "Berlin"),
                ("id", "int 2")
            ]),
            node(&[("id", "int 1")]),
        ]
    );
}

/// The label is used as given, in backticks.
#[test]
fn a_label_with_a_space_or_a_backtick_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "people.csv", "name\nHans\n");
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Film Fan", CSV), 1);
    assert_eq!(import(&db, &path, "odd`label", CSV), 1);
    assert_eq!(properties(&db, "Film Fan"), [node(&[("name", "Hans")])]);
    assert_eq!(properties(&db, "odd`label"), [node(&[("name", "Hans")])]);
}

/// The count is the nodes this import created, whatever the label held.
#[test]
fn the_count_is_what_the_import_created() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "people.csv", "name\nShosanna\nBeatrix\n");
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("INSERT (:Person {name: 'Marcus'})")
        .unwrap();
    assert_eq!(import(&db, &path, "Person", CSV), 2);
    assert_eq!(import(&db, &path, "Person", CSV), 2);
    assert_eq!(properties(&db, "Person").len(), 5);
}

#[test]
fn a_path_with_a_quote_and_a_backslash_is_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "O'Brien.csv", "name\nLucas\n");
    let db = GrafeoDB::new_in_memory();
    assert_eq!(import(&db, &path, "Row", CSV), 1);
}

#[test]
fn a_rolled_back_transaction_keeps_none_of_the_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "people.csv", "name\nHarm\nMaxence\n");
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    assert_eq!(import_nodes(&session, &path, "Person", CSV).unwrap(), 2);
    session.rollback().unwrap();
    let nodes = properties(&db, "Person");
    assert!(nodes.is_empty(), "{nodes:?}");
}

#[test]
fn a_missing_file_is_an_io_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::new_in_memory();
    let error = import_nodes(&db.session(), &dir.path().join("nope.csv"), "Row", CSV).unwrap_err();
    assert!(
        matches!(&error, Error::Io(io) if io.kind() == std::io::ErrorKind::NotFound),
        "{error}"
    );
}

#[test]
fn an_empty_label_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir, "people.csv", "name\nJules\n");
    let db = GrafeoDB::new_in_memory();
    let error = import_nodes(&db.session(), &path, "", CSV).unwrap_err();
    assert!(matches!(error, Error::InvalidValue(_)), "{error}");
    let nodes = properties(&db, "Row");
    assert!(nodes.is_empty(), "{nodes:?}");
}
