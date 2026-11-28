//! A SPARQL join that looks a pattern up per row holds its rows until it knows how many there
//! are, up to an eighth of the pattern's triples however many that is, and chooses once: the rows
//! look up at or under that limit, and all of them go to the hash join past it. Either way the
//! rows are those of the hash join, once each, in an inner join and in an `OPTIONAL`; the rows
//! held count against the query memory limit, and PROFILE shows which way the join went.

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::collections::HashSet;
use std::fmt::Write as _;

use grafeo_engine::{Config, GrafeoDB, RdfFormat};

const EX: &str = "http://example.org/";

/// Rows of the left side: Alix's hub reaches this many subjects.
const ROWS: usize = 2_000;

/// Alix's hub reaches `ROWS` subjects; `ex:p` holds `triples` triples, one on each subject `y_i`
/// for which `holds(i)` is true (`i < ROWS`), the rest on subjects no row reaches.
fn text(triples: usize, holds: impl Fn(usize) -> bool) -> String {
    let mut text = String::new();
    for i in 0..ROWS {
        writeln!(text, "<{EX}alix> <{EX}hub> <{EX}y_{i}> .").unwrap();
    }
    let mut written = 0;
    for i in 0..ROWS {
        if holds(i) {
            writeln!(text, "<{EX}y_{i}> <{EX}p> <{EX}z_{i}> .").unwrap();
            written += 1;
        }
    }
    for i in 0..triples - written {
        writeln!(text, "<{EX}other_{i}> <{EX}p> <{EX}z_{i}> .").unwrap();
    }
    text
}

fn load(text: &str) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.load_rdf_text(text, RdfFormat::NTriples, None).unwrap();
    db
}

fn store(triples: usize, holds: impl Fn(usize) -> bool) -> GrafeoDB {
    load(&text(triples, holds))
}

/// The hub's rows joined with `body`.
fn query(body: &str) -> String {
    format!(
        "PREFIX ex: <{EX}> SELECT ?y ?z WHERE {{ VALUES ?x {{ ex:alix }} ?x ex:hub ?y . {body} }}"
    )
}

/// The `(y, z)` pairs of `body`, each row once, and the rows it scanned.
fn pairs(db: &GrafeoDB, body: &str) -> (Vec<(String, String)>, u64) {
    let query = query(body);
    let result = db
        .execute_sparql(&query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let rows: Vec<(String, String)> = result
        .rows()
        .iter()
        .map(|row| (format!("{}", row[0]), format!("{}", row[1])))
        .collect();
    (rows, result.rows_scanned.expect("rows_scanned"))
}

/// `rows` are those of the hash join of the hub's rows with `pattern`: the pattern in a
/// sub-select, which the join never looks up.
#[track_caller]
fn same_as_the_hash_join(db: &GrafeoDB, rows: &[(String, String)], pattern: &str) {
    let (hashed, _) = pairs(db, &format!("{{ SELECT ?y ?z WHERE {{ {pattern} }} }}"));
    let mut rows = rows.to_vec();
    let mut hashed = hashed;
    rows.sort();
    hashed.sort();
    assert_eq!(rows, hashed, "the rows of the hash join");
}

/// Each of `rows` once, and `expected` of them with a `z`.
#[track_caller]
fn each_row_once(rows: &[(String, String)], expected_with_z: usize) {
    assert_eq!(rows.len(), ROWS, "one row per subject the hub reaches");
    let distinct: HashSet<&String> = rows.iter().map(|(y, _)| y).collect();
    assert_eq!(distinct.len(), ROWS, "no subject twice");
    // The number after the last `_` of a term (`y_7`, `z_7`).
    let number = |term: &str| {
        term.rsplit_once('_')
            .map(|(_, n)| n.trim_end_matches('>').to_string())
    };
    let with_z = rows
        .iter()
        .filter(|(y, z)| z.contains("z_") && number(y) == number(z))
        .count();
    assert_eq!(with_z, expected_with_z, "each z with its own y");
}

/// The PROFILE line of the join of the hub's rows with `body`.
fn lookup_line(db: &GrafeoDB, body: &str) -> String {
    let query = format!("EXPLAIN ANALYZE {}", query(body));
    let result = db
        .execute_sparql(&query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let text = format!("{}", result.rows()[0][0]).replace("\\n", "\n");
    // The join of the hub's rows with `body` is the outermost lookup.
    text.lines()
        .map(str::trim_start)
        .find(|line| line.starts_with("RdfTripleLookup"))
        .unwrap_or_else(|| panic!("a lookup: {text}"))
        .to_string()
}

#[test]
fn more_than_a_thousand_rows_look_up_a_large_pattern() {
    // 2,000 rows against 30,000 triples: an eighth is 3,750, so every row looks up.
    let db = store(30_000, |_| true);
    let (rows, scanned) = pairs(&db, "?y ex:p ?z");
    each_row_once(&rows, ROWS);
    same_as_the_hash_join(&db, &rows, "?y ex:p ?z");
    assert!(
        scanned < 10_000,
        "the rows look ex:p up instead of hashing its 30,000 triples: {scanned} rows scanned"
    );
}

#[test]
fn past_an_eighth_of_the_pattern_the_rows_are_hashed() {
    // 2,000 rows against 8,000 triples: an eighth is 1,000, so the join hashes every row.
    let db = store(8_000, |_| true);
    let (rows, scanned) = pairs(&db, "?y ex:p ?z");
    each_row_once(&rows, ROWS);
    same_as_the_hash_join(&db, &rows, "?y ex:p ?z");
    assert!(
        (8_000..12_000).contains(&scanned),
        "the hub's 2,000 triples and the pattern's 8,000 once each: {scanned} rows scanned"
    );
}

#[test]
fn rows_past_the_limit_in_the_first_chunk_are_hashed() {
    // The first 1,024 subjects (the first chunk of rows) hold 4 triples each: 4,096, an eighth
    // of which is 512. Looking that chunk up and then hashing would read every triple twice.
    let mut text = String::new();
    for i in 0..ROWS {
        writeln!(text, "<{EX}alix> <{EX}hub> <{EX}y_{i}> .").unwrap();
    }
    for i in 0..1_024 {
        for k in 0..4 {
            writeln!(text, "<{EX}y_{i}> <{EX}p> <{EX}z_{}> .", i * 4 + k).unwrap();
        }
    }
    let db = load(&text);
    let (rows, scanned) = pairs(&db, "?y ex:p ?z");
    assert_eq!(rows.len(), 4_096);
    let distinct: HashSet<&(String, String)> = rows.iter().collect();
    assert_eq!(distinct.len(), 4_096, "each pair once");
    same_as_the_hash_join(&db, &rows, "?y ex:p ?z");
    assert!(
        scanned < 8_000,
        "the hub's 2,000 triples and the pattern's 4,096 once each: {scanned} rows scanned"
    );
}

#[test]
fn rows_past_the_limit_after_the_first_chunk_are_all_hashed() {
    // 2,000 rows against 10,000 triples (5 on each subject): an eighth is 1,250, past the first
    // chunk of 1,024 rows. Looking that chunk up before the second one passes the limit would
    // read its 5,120 triples twice.
    let mut text = String::new();
    for i in 0..ROWS {
        writeln!(text, "<{EX}alix> <{EX}hub> <{EX}y_{i}> .").unwrap();
        for k in 0..5 {
            writeln!(text, "<{EX}y_{i}> <{EX}p> <{EX}z_{}> .", i * 5 + k).unwrap();
        }
    }
    let db = load(&text);
    let (rows, scanned) = pairs(&db, "?y ex:p ?z");
    assert_eq!(rows.len(), 10_000);
    let distinct: HashSet<&(String, String)> = rows.iter().collect();
    assert_eq!(distinct.len(), 10_000, "each pair once");
    same_as_the_hash_join(&db, &rows, "?y ex:p ?z");
    assert!(
        scanned < 14_000,
        "the hub's 2,000 triples and the pattern's 10,000 once each: {scanned} rows scanned"
    );
}

#[test]
fn an_optional_pattern_keeps_every_row_either_way() {
    // Half the subjects hold the pattern: hashed against 8,000 triples, looked up against 30,000.
    let db = store(8_000, |i| i % 2 == 0);
    let (rows, _) = pairs(&db, "OPTIONAL { ?y ex:p ?z }");
    each_row_once(&rows, ROWS / 2);
    let db = store(30_000, |i| i % 2 == 0);
    let (rows, scanned) = pairs(&db, "OPTIONAL { ?y ex:p ?z }");
    each_row_once(&rows, ROWS / 2);
    assert!(scanned < 10_000, "{scanned} rows scanned");
}

#[test]
fn a_pattern_with_no_triples_joins_nothing() {
    // `ex:none` holds no triple: an estimate of 0 still gives the join a limit.
    let db = store(8_000, |_| true);
    let (rows, _) = pairs(&db, "?y ex:none ?z");
    assert!(rows.is_empty(), "{} rows", rows.len());
    let (rows, _) = pairs(&db, "OPTIONAL { ?y ex:none ?z }");
    each_row_once(&rows, 0);
}

#[test]
fn the_rows_held_before_the_choice_count_against_the_query_memory_limit() {
    // 2,000 rows held while the join chooses (an eighth of 30,000 is 3,750), under 16 KiB.
    let text = text(30_000, |_| true);
    let db = GrafeoDB::with_config(Config::in_memory().with_query_memory_limit(16 * 1024)).unwrap();
    db.load_rdf_text(&text, RdfFormat::NTriples, None).unwrap();
    let error = db
        .execute_sparql(&query("?y ex:p ?z"))
        .expect_err("the rows held pass the limit")
        .to_string();
    assert!(error.starts_with("GRAFEO-S001"), "{error}");
    assert!(error.contains("query memory limit"), "{error}");
    // Under a limit that holds them, the same query answers.
    let db = GrafeoDB::with_config(Config::in_memory().with_query_memory_limit(64 << 20)).unwrap();
    db.load_rdf_text(&text, RdfFormat::NTriples, None).unwrap();
    let (rows, _) = pairs(&db, "?y ex:p ?z");
    each_row_once(&rows, ROWS);
}

#[test]
fn profile_shows_which_way_the_join_went_and_the_rows_it_held() {
    let looked_up = lookup_line(&store(30_000, |_| true), "?y ex:p ?z");
    assert!(looked_up.contains("2000 rows looked up"), "{looked_up}");
    let hashed = lookup_line(&store(8_000, |_| true), "?y ex:p ?z");
    assert!(hashed.contains("hashed after 1024 rows"), "{hashed}");
}
