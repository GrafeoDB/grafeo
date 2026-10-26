//! Tests for LOAD DATA edge cases and CSV/JSONL coverage gaps.
//!
//! Targets: load_data.rs (61.77%), parser.rs LOAD DATA syntax
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test coverage_load_data
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use std::io::Write;

fn temp_file(name: &str, content: &str) -> String {
    let dir = std::env::temp_dir().join("grafeo_test_load");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path.to_string_lossy().replace('\\', "/")
}

// ---------------------------------------------------------------------------
// CSV with BOM (exercises BOM-stripping in open_text)
// ---------------------------------------------------------------------------

#[test]
fn test_csv_with_utf8_bom() {
    let csv = "\u{feff}name,score\nAlix,95\nGus,88";
    let path = temp_file("bom_test.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name ORDER BY row.name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(r.rows()[1][0], Value::String("Gus".into()));
}

// ---------------------------------------------------------------------------
// CSV without headers (exercises List branch)
// ---------------------------------------------------------------------------

#[test]
fn test_csv_without_headers_returns_list() {
    let csv = "Alix,30\nGus,25\n";
    let path = temp_file("no_headers.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV AS row RETURN row[0] AS name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

// ---------------------------------------------------------------------------
// CSV with Windows line endings (\r\n)
// ---------------------------------------------------------------------------

#[test]
fn test_csv_windows_line_endings() {
    let csv = "name,city\r\nAlix,Amsterdam\r\nGus,Berlin\r\n";
    let path = temp_file("crlf.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS n ORDER BY row.name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

// ---------------------------------------------------------------------------
// CSV with quoted fields containing commas and newlines
// ---------------------------------------------------------------------------

#[test]
fn test_csv_quoted_commas() {
    let csv = "name,bio\n\"Alix\",\"likes cats, dogs\"\n\"Gus\",\"quiet\"\n";
    let path = temp_file("quoted.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row.bio AS bio ORDER BY row.name"
        ))
        .unwrap();
    assert_eq!(r.rows()[0][0], Value::String("likes cats, dogs".into()));
}

#[test]
fn csv_quoted_line_breaks_keep_exact_rows_and_values() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("multiline.csv");
    std::fs::write(
        &file,
        "name,bio\r\nAlix,\"first\r\n\r\nsaid \"\"hi\"\"\"\r\nGus,quiet",
    )
    .unwrap();
    let path = file.to_string_lossy().replace('\\', "/");
    let db = GrafeoDB::new_in_memory();
    let check = |language: &str, query: &str| {
        for _ in 0..2 {
            let result = db.execute_language(query, language, None).unwrap();
            assert_eq!(
                result.rows(),
                &[
                    vec![
                        Value::String("Alix".into()),
                        Value::String("first\r\n\r\nsaid \"hi\"".into())
                    ],
                    vec![Value::String("Gus".into()), Value::String("quiet".into())],
                ],
                "{language}"
            );
        }
    };
    check(
        "gql",
        &format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row \
         RETURN row.name, row.bio ORDER BY row.name"
        ),
    );
    #[cfg(feature = "cypher")]
    check(
        "cypher",
        &format!(
            "LOAD CSV WITH HEADERS FROM '{path}' AS row \
         RETURN row.name, row.bio ORDER BY row.name"
        ),
    );
}

// ---------------------------------------------------------------------------
// CSV with empty file (headers only, no data rows)
// ---------------------------------------------------------------------------

#[test]
fn test_csv_headers_only() {
    let csv = "name,age\n";
    let path = temp_file("empty_data.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 0);
}

// ---------------------------------------------------------------------------
// CSV with pipe delimiter
// ---------------------------------------------------------------------------

#[test]
fn test_csv_pipe_delimiter() {
    let csv = "name|score\nAlix|100\nGus|85\n";
    let path = temp_file("pipe.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row FIELDTERMINATOR '|' RETURN row.score AS s ORDER BY row.name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2);
    assert_eq!(r.rows()[0][0], Value::String("100".into()));
}

// ---------------------------------------------------------------------------
// CSV with semicolon delimiter
// ---------------------------------------------------------------------------

#[test]
fn test_csv_semicolon_delimiter() {
    let csv = "name;city\nAlix;Amsterdam\nGus;Berlin\n";
    let path = temp_file("semicolon.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row FIELDTERMINATOR ';' RETURN row.city AS c ORDER BY row.name"
        ))
        .unwrap();
    assert_eq!(r.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(r.rows()[1][0], Value::String("Berlin".into()));
}

// ---------------------------------------------------------------------------
// JSONL with blank lines interspersed
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "jsonl-import")]
fn test_jsonl_blank_lines() {
    let jsonl = "{\"name\":\"Alix\"}\n\n{\"name\":\"Gus\"}\n\n";
    let path = temp_file("blanks.jsonl", jsonl);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT JSONL AS row RETURN row.name AS name ORDER BY row.name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2);
}

// ---------------------------------------------------------------------------
// JSONL with nested objects
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "jsonl-import")]
fn test_jsonl_nested_objects() {
    let jsonl = "{\"name\":\"Alix\",\"address\":{\"city\":\"Amsterdam\",\"zip\":\"1011\"}}\n";
    let path = temp_file("nested.jsonl", jsonl);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT JSONL AS row RETURN row.name AS name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

// ---------------------------------------------------------------------------
// JSONL with arrays
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "jsonl-import")]
fn test_jsonl_arrays() {
    let jsonl = "{\"name\":\"Alix\",\"tags\":[\"rust\",\"graph\"]}\n";
    let path = temp_file("arrays.jsonl", jsonl);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT JSONL AS row RETURN row.name AS name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 1);
}

// ---------------------------------------------------------------------------
// JSONL with null and boolean values
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "jsonl-import")]
fn test_jsonl_null_and_bool() {
    let jsonl = "{\"name\":\"Alix\",\"active\":true,\"note\":null}\n{\"name\":\"Gus\",\"active\":false,\"note\":\"hello\"}\n";
    let path = temp_file("null_bool.jsonl", jsonl);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT JSONL AS row RETURN row.active AS a, row.note AS n ORDER BY row.name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2);
    assert_eq!(r.rows()[0][0], Value::Bool(true));
    assert_eq!(r.rows()[0][1], Value::Null);
    assert_eq!(r.rows()[1][0], Value::Bool(false));
}

// ---------------------------------------------------------------------------
// File not found error
// ---------------------------------------------------------------------------

#[test]
fn test_load_data_file_not_found() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session.execute(
        "LOAD DATA FROM '/does/not/exist/data.csv' FORMAT CSV WITH HEADERS AS row RETURN row",
    );
    assert!(r.is_err());
}

// ---------------------------------------------------------------------------
// CSV + INSERT (exercises the operator in mutation context)
// ---------------------------------------------------------------------------

#[test]
fn test_csv_insert_nodes() {
    let csv = "name,age\nVincent,40\nJules,35\n";
    let path = temp_file("insert_nodes.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row INSERT (:Person {{name: row.name}})"
        ))
        .unwrap();
    let r = session
        .execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap();
    assert_eq!(r.rows().len(), 2);
    assert_eq!(r.rows()[0][0], Value::String("Jules".into()));
    assert_eq!(r.rows()[1][0], Value::String("Vincent".into()));
}

// ---------------------------------------------------------------------------
// NDJSON alias (should work same as JSONL)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "jsonl-import")]
fn test_ndjson_alias() {
    let jsonl = "{\"name\":\"Mia\"}\n{\"name\":\"Butch\"}\n";
    let path = temp_file("ndjson.jsonl", jsonl);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT NDJSON AS row RETURN row.name AS name ORDER BY row.name"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2);
}

// ---------------------------------------------------------------------------
// Malformed CSV input (T3-08)
// ---------------------------------------------------------------------------

#[test]
fn test_csv_fewer_columns_than_header() {
    // Row has fewer columns than header: should still parse (missing columns become empty)
    let csv = "name,score,grade\nAlix,95\nGus,88,A";
    let path = temp_file("fewer_cols.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2, "both rows should be returned");
}

#[test]
fn test_csv_more_columns_than_header() {
    // Row has more columns than header: extra columns should be handled gracefully
    let csv = "name,score\nAlix,95,extra_value\nGus,88";
    let path = temp_file("more_cols.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 2, "both rows should be returned");
}

#[test]
fn test_csv_empty_file() {
    let csv = "";
    let path = temp_file("empty.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 0, "empty CSV should return no rows");
}

#[test]
#[cfg(feature = "jsonl-import")]
fn test_jsonl_invalid_json_line() {
    let jsonl = "{\"name\":\"Alix\"}\n{invalid json}\n{\"name\":\"Gus\"}";
    let path = temp_file("invalid.jsonl", jsonl);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session.execute(&format!(
        "LOAD DATA FROM '{path}' FORMAT JSONL AS row RETURN row.name AS name"
    ));
    // Invalid JSON line should cause an error
    assert!(r.is_err(), "invalid JSON line should produce an error");
    let err = r.unwrap_err().to_string();
    assert!(
        err.contains("JSON") || err.contains("parse"),
        "error should mention JSON parsing, got: {err}"
    );
}

#[test]
fn test_csv_only_newlines() {
    let csv = "\n\n\n";
    let path = temp_file("only_newlines.csv", csv);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row"
        ))
        .unwrap();
    assert_eq!(r.rows().len(), 0, "newlines-only CSV should return no rows");
}
