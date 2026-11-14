//! SPARQL `ORDER BY` under the query memory limit (0.6.3 sweep, finding 1).
//!
//! `ORDER BY ... LIMIT n` (`OFFSET m`) keeps at most `m + n` rows and a chunk, so it runs under a
//! limit smaller than its input, and answers as a full sort followed by the slice, ties in input
//! order. A full sort past the limit (a top-k of every row too) refuses with `GRAFEO-S001` naming
//! the query memory limit and its setting, with or without a spill path: the SPARQL sort does not
//! spill yet. The order is SPARQL 1.1's (section 15.1) and does not change.
//!
//! The oracle is the same query on the same data without a limit.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_order_under_limit::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store", feature = "spill"))]

use grafeo_core::graph::rdf::{Term, Triple};
use grafeo_engine::{Config, GrafeoDB};

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;

const EX: &str = "http://ex.org/";
const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
const PREFIXES: &str = "PREFIX ex: <http://ex.org/> ";

fn iri(local: &str) -> Term {
    Term::iri(format!("{EX}{local}"))
}

fn integer(n: i64) -> Term {
    Term::typed_literal(n.to_string(), format!("{XSD}integer"))
}

/// `count` subjects `ex:s<i>`, each with `ex:v` an integer whose order is not the subjects'.
fn numbers(count: i64) -> Vec<Triple> {
    (0..count)
        .map(|i| {
            Triple::new(
                iri(&format!("s{i}")),
                iri("v"),
                integer((i * 7_919) % count),
            )
        })
        .collect()
}

/// The objects of every SPARQL order-key kind, by subject number: blank nodes, IRIs, plain,
/// language-tagged and `xsd:string` literals, integers, decimals and doubles (numerics of three
/// types in one column), dates, date-times and booleans, many of them equal (ties).
fn object_of(i: i64) -> Term {
    let n = i % 97;
    match i % 12 {
        0 => Term::blank(format!("b{n}")),
        1 => iri(&format!("o{n}")),
        2 => Term::literal(format!("text {n}")),
        3 => Term::lang_literal(format!("text {n}"), "en"),
        4 => Term::lang_literal(format!("text {n}"), "fr"),
        5 => Term::typed_literal(format!("text {n}"), format!("{XSD}string")),
        6 => integer(n - 40),
        7 => Term::typed_literal(format!("{}.5", n - 40), format!("{XSD}decimal")),
        8 => Term::typed_literal(format!("{}e0", n - 40), format!("{XSD}double")),
        9 => Term::typed_literal(
            format!("2026-{:02}-{:02}", n % 12 + 1, n % 28 + 1),
            format!("{XSD}date"),
        ),
        10 => Term::typed_literal(
            format!("2026-11-{:02}T{:02}:00:00Z", n % 28 + 1, n % 24),
            format!("{XSD}dateTime"),
        ),
        _ => Term::typed_literal(
            if n % 2 == 0 { "true" } else { "false" },
            format!("{XSD}boolean"),
        ),
    }
}

/// `count` subjects with `ex:v` of every kind, a small `ex:k` (ties), and `ex:w` on a third of
/// them only (unbound for the rest, through OPTIONAL).
fn every_kind(count: i64) -> Vec<Triple> {
    let mut triples = Vec::new();
    for i in 0..count {
        let subject = iri(&format!("s{i}"));
        triples.push(Triple::new(subject.clone(), iri("v"), object_of(i)));
        triples.push(Triple::new(subject.clone(), iri("k"), integer(i % 4)));
        if i % 3 == 0 {
            triples.push(Triple::new(subject, iri("w"), integer((i * 31) % 50)));
        }
    }
    triples
}

/// A database holding `triples`, under `limit` (a query memory limit) and with a spill path when
/// `spill` holds one.
fn database(triples: &[Triple], limit: Option<usize>, spill: Option<&std::path::Path>) -> GrafeoDB {
    let mut config = Config::in_memory();
    if let Some(limit) = limit {
        config = config.with_query_memory_limit(limit);
    }
    if let Some(path) = spill {
        config = config.with_spill_path(path);
    }
    let db = GrafeoDB::with_config(config).unwrap();
    db.batch_insert_rdf(triples.iter().cloned()).unwrap();
    db
}

/// The rows of `query`, in their order, as text.
fn ordered(db: &GrafeoDB, query: &str) -> Result<Vec<String>, String> {
    db.execute_sparql(&format!("{PREFIXES}{query}"))
        .map(|result| result.rows().iter().map(|row| format!("{row:?}")).collect())
        .map_err(|error| format!("{} {error}", error.error_code().as_str()))
}

fn unlimited(triples: &[Triple], query: &str) -> Vec<String> {
    let db = database(triples, None, None);
    ordered(&db, query).unwrap_or_else(|error| panic!("{query}, unlimited: {error}"))
}

/// `result` is the S001 refusal naming the query memory limit and its setting.
fn assert_refused(result: Result<Vec<String>, String>, what: &str) {
    let error = result.expect_err(what);
    assert!(error.starts_with("GRAFEO-S001"), "{what}: {error}");
    assert!(
        error.contains("query memory limit"),
        "{what}: names the bound: {error}"
    );
    assert!(
        error.contains("with_query_memory_limit"),
        "{what}: names the setting: {error}"
    );
}

/// The sweep's three cases: the two top-k answer as the unlimited run; the full sort over the
/// limit refuses naming it, with and without a spill path (it cannot spill).
#[test]
fn the_sweeps_cases_under_the_limit() {
    let small = numbers(3_000);
    let query = "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY DESC(?o) LIMIT 2";
    assert_eq!(
        ordered(&database(&small, Some(64 * KIB), None), query),
        Ok(unlimited(&small, query)),
        "a top-k of 2 over 3,000 triples under 64 KiB"
    );

    let large = numbers(20_000);
    let query = "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY ?o LIMIT 3";
    assert_eq!(
        ordered(&database(&large, Some(MIB), None), query),
        Ok(unlimited(&large, query)),
        "a top-k of 3 over 20,000 triples under 1 MiB"
    );

    let query = "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY ?o";
    assert_eq!(
        unlimited(&large, query).len(),
        20_000,
        "without a limit the full sort answers"
    );
    assert_refused(
        ordered(&database(&large, Some(MIB), None), query),
        "a full sort of 20,000 triples under 1 MiB",
    );
    let dir = tempfile::tempdir().unwrap();
    assert_refused(
        ordered(&database(&large, Some(MIB), Some(dir.path())), query),
        "a full sort of 20,000 triples under 1 MiB, with a spill path",
    );
}

/// The orders of the key-kind class test: one key of every kind both ways, an unbound key, an
/// expression key, and two keys with ties on the first (an expression of few values, then the
/// mixed kinds descending).
const ORDERS: &[&str] = &[
    "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY ?o",
    "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY DESC(?o)",
    "SELECT ?s ?w WHERE { ?s ex:v ?o OPTIONAL { ?s ex:w ?w } } ORDER BY ?w",
    "SELECT ?s ?w WHERE { ?s ex:v ?o OPTIONAL { ?s ex:w ?w } } ORDER BY DESC(?w)",
    "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY STR(?o)",
    "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY STRLEN(STR(?o)) DESC(?o)",
    "SELECT ?s ?k WHERE { ?s ex:k ?k } ORDER BY ?k",
];

/// The limits the key-kind class test chooses from, smallest first.
const CLASS_LIMITS: [usize; 9] = [
    256 * KIB,
    320 * KIB,
    384 * KIB,
    448 * KIB,
    512 * KIB,
    640 * KIB,
    768 * KIB,
    MIB,
    2 * MIB,
];

/// The top-k orders every key kind as the full sort does. Each order runs under a limit of its
/// own from [`CLASS_LIMITS`]: the smallest at least a quarter above the smallest its pattern fits
/// in without `ORDER BY` (an `OPTIONAL` needs more than a single pattern), so the top-k has room
/// for its rows and a chunk. There, with no spill path, the full sort must be past the limit
/// (refused), so the order's top 5 after 17 rows, which must equal the unlimited sort's slice,
/// ties included, cannot come from a full sort that happened to fit.
#[test]
fn a_top_k_orders_every_key_kind_as_the_full_sort() {
    let triples = every_kind(12_000);
    let capped: Vec<(usize, GrafeoDB)> = CLASS_LIMITS
        .iter()
        .map(|&limit| (limit, database(&triples, Some(limit), None)))
        .collect();
    let mut wrong = Vec::new();
    for order in ORDERS {
        let full = unlimited(&triples, order);
        let pattern = order.split(" ORDER BY").next().unwrap();
        let Some(fits) = capped
            .iter()
            .find(|(_, db)| ordered(db, pattern).is_ok())
            .map(|(limit, _)| *limit)
        else {
            wrong.push(format!(
                "{order}: the pattern alone fits under none of {CLASS_LIMITS:?}"
            ));
            continue;
        };
        let Some((limit, db)) = capped.iter().find(|(limit, _)| *limit >= fits + fits / 4) else {
            wrong.push(format!(
                "{order}: no limit of {CLASS_LIMITS:?} is a quarter above {fits}"
            ));
            continue;
        };
        match ordered(db, order) {
            Err(error) if error.starts_with("GRAFEO-S001") => {}
            other => wrong.push(format!(
                "{order}: the full sort is not past {limit} bytes, where the pattern fits: {:?}",
                other.map(|rows| rows.len())
            )),
        }
        let query = format!("{order} LIMIT 5 OFFSET 17");
        let want: Vec<String> = full.iter().skip(17).take(5).cloned().collect();
        match ordered(db, &query) {
            Ok(got) if got == want => {}
            other => wrong.push(format!(
                "{query} under {limit} bytes: {other:?}, want {want:?}"
            )),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The orders of the top-k test, each with the unlimited full sort as its oracle: one total
/// order (`?s` breaks every tie), one of ties only (`ex:k` has four values, so a slice is
/// decided by input order alone, which a heap that does not keep it gets wrong), and a
/// descending one over the mixed kinds.
const TOP_K_ORDERS: &[&str] = &[
    "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY ?o ?s",
    "SELECT ?s ?k WHERE { ?s ex:k ?k } ORDER BY ?k",
    "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY DESC(?o)",
];

/// The top-k equals a full sort and the slice for `k` (offset plus limit) of 0, 1 and a few,
/// with and without `OFFSET`, under a limit far below the input and with no spill path (a
/// top-k needs none). A `k` of the row count or more keeps every row, a full sort: under the
/// same limit it refuses as the full sort does, offsets at and past the row count included; with
/// no limit it answers as the slice.
#[test]
fn a_top_k_equals_the_full_sort_and_the_slice() {
    let triples = every_kind(6_000);
    let capped = database(&triples, Some(128 * KIB), None);
    let dir = tempfile::tempdir().unwrap();
    let capped_with_a_spill_path = database(&triples, Some(128 * KIB), Some(dir.path()));
    let free = database(&triples, None, None);
    let mut wrong = Vec::new();
    for order in TOP_K_ORDERS {
        let full = unlimited(&triples, order);
        let rows = full.len();
        assert_eq!(rows, 6_000, "{order}: one row per subject");
        // The oracle of ties is the input order: two databases loaded alike give the same one.
        assert_eq!(
            unlimited(&triples, order),
            full,
            "{order}: the unlimited order is one"
        );
        let slice = |limit: usize, offset: usize| -> Vec<String> {
            full.iter().skip(offset).take(limit).cloned().collect()
        };
        for (limit, offset) in [(0, 0), (0, 17), (1, 0), (5, 0), (5, 17)] {
            let query = format!("{order} LIMIT {limit} OFFSET {offset}");
            match ordered(&capped, &query) {
                Ok(got) if got == slice(limit, offset) => {}
                Ok(got) => wrong.push(format!(
                    "{query}: {} rows, want {}; first difference at {:?}",
                    got.len(),
                    limit,
                    got.iter()
                        .zip(&slice(limit, offset))
                        .position(|(a, b)| a != b)
                )),
                Err(error) => wrong.push(format!("{query}: {error}")),
            }
        }
        for (limit, offset) in [
            (1, rows - 1),
            (3, rows),
            (3, rows + 5),
            (rows, 0),
            (rows + 10, 0),
            (rows, 5),
        ] {
            let query = format!("{order} LIMIT {limit} OFFSET {offset}");
            for db in [&capped, &capped_with_a_spill_path] {
                match ordered(db, &query) {
                    Err(error)
                        if error.starts_with("GRAFEO-S001")
                            && error.contains("query memory limit") => {}
                    other => wrong.push(format!(
                        "{query} under the limit: {:?}, want S001",
                        other.map(|rows| rows.len())
                    )),
                }
            }
            if ordered(&free, &query).as_ref() != Ok(&slice(limit, offset)) {
                wrong.push(format!("{query} without a limit: not the slice"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A full sort past the limit refuses with `GRAFEO-S001` naming the query memory limit and its
/// setting, with or without a spill path: the SPARQL sort does not spill yet. GQL's sort of as
/// many rows without a spill path is the reference refusal.
#[test]
fn a_full_sort_past_the_limit_refuses() {
    let triples = numbers(20_000);
    let query = "SELECT ?s ?o WHERE { ?s ex:v ?o } ORDER BY ?o";
    let db = database(&triples, Some(MIB), None);
    #[cfg(feature = "gql")]
    {
        let gql = db
            .execute("UNWIND range(1, 200000) AS i RETURN i ORDER BY i")
            .expect_err("GQL, no spill path: refused");
        assert_eq!(gql.error_code().as_str(), "GRAFEO-S001", "{gql}");
        assert!(gql.to_string().contains("query memory limit"), "GQL: {gql}");
    }
    assert_refused(ordered(&db, query), "no spill path");

    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_query_memory_limit(MIB)
            .with_spill_path(dir.path())
            .with_unlimited_spill(),
    )
    .unwrap();
    db.batch_insert_rdf(triples.iter().cloned()).unwrap();
    assert_refused(ordered(&db, query), "a spill path with room");
    let higher = database(&triples, Some(512 * MIB), None);
    assert_eq!(
        ordered(&higher, query).map(|rows| rows.len()),
        Ok(20_000),
        "a higher limit lets the same sort through"
    );
}
