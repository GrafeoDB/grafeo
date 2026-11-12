//! The per-query memory limit bounds every operator that materializes its
//! input (#674), not only sorts and groupings: a distinct's seen rows, a set
//! operation, a hash join's build side, `collect`, a write's eager buffer, a
//! stored procedure's result, the pull sort and aggregate of the languages
//! that run pull plans, and what their values hold on the heap. Past the
//! limit an operator that cannot spill fails with `GRAFEO-S001` naming the
//! query memory limit, a writing statement changes nothing (in an explicit
//! transaction too), and a higher limit lets the same statement through.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main memory_query_limit_operators::
//! ```

#![cfg(all(feature = "gql", feature = "spill"))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::Error;
use grafeo_engine::{Config, GrafeoDB};

const MIB: usize = 1024 * 1024;

/// `result` is the S001 refusal naming the query memory limit and its setting.
fn assert_refused<T: std::fmt::Debug>(result: Result<T, Error>) {
    let error = result.expect_err("refused at the query memory limit");
    let text = error.to_string();
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{text}");
    assert!(
        text.contains("query memory limit"),
        "names the bound: {text}"
    );
    assert!(
        text.contains("with_query_memory_limit"),
        "names the setting: {text}"
    );
}

/// A database whose statements may hold `limit` bytes of operator memory.
fn capped(limit: usize) -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_query_memory_limit(limit)).unwrap()
}

fn count(db: &GrafeoDB, label: &str) -> usize {
    db.execute(&format!("MATCH (n:{label}) RETURN count(n) AS n"))
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .and_then(|n| usize::try_from(n).ok())
        .unwrap()
}

/// GQL statements whose materializing operator cannot spill: a distinct (the
/// push distinct of the pipeline) and a set operation (a pull operator).
const GQL: &[&str] = &[
    "UNWIND range(1, 200000) AS i RETURN DISTINCT i",
    "UNWIND range(1, 200000) AS i RETURN i EXCEPT UNWIND range(1, 10) AS i RETURN i",
    "UNWIND range(1, 200000) AS i RETURN i INTERSECT UNWIND range(1, 200000) AS i RETURN i",
];

#[test]
fn gql_distinct_and_set_operations_refuse_at_the_query_limit() {
    let small = capped(MIB);
    let large = capped(512 * MIB);
    for query in GQL {
        assert_refused(small.execute(query));
        assert!(!large.execute(query).unwrap().rows().is_empty(), "{query}");
    }
}

/// A writing statement refused at the query limit by its distinct writes
/// nothing.
#[test]
fn a_write_refused_by_its_distinct_changes_nothing() {
    let db = capped(MIB);
    assert_refused(db.execute("UNWIND range(1, 200000) AS i WITH DISTINCT i INSERT (:S {i: i})"));
    assert_eq!(count(&db, "S"), 0);
}

/// Cypher runs pull plans: its sort, aggregate and distinct are the pull
/// operators, bounded as the push ones are.
#[cfg(feature = "cypher")]
#[test]
fn cypher_pull_operators_refuse_at_the_query_limit() {
    let small = capped(MIB);
    let large = capped(512 * MIB);
    for query in [
        "UNWIND range(1, 200000) AS i RETURN i ORDER BY i DESC",
        "UNWIND range(1, 200000) AS i RETURN i % 100000 AS k, count(*) AS n",
        "UNWIND range(1, 200000) AS i RETURN DISTINCT i",
    ] {
        assert_refused(small.execute_cypher(query));
        assert!(
            !large.execute_cypher(query).unwrap().rows().is_empty(),
            "{query}"
        );
    }
    assert_refused(
        small.execute_cypher("UNWIND range(1, 200000) AS i WITH DISTINCT i CREATE (:C {i: i})"),
    );
    assert_eq!(count(&small, "C"), 0);
}

/// A refused statement gives back what its operators held to the database's
/// memory limit: nothing stays granted to execution buffers.
#[test]
fn a_refused_statement_gives_its_memory_back() {
    use grafeo_common::memory::buffer::MemoryRegion;

    let db = capped(MIB);
    let execution = |db: &GrafeoDB| {
        db.buffer_manager()
            .stats()
            .region_usage(MemoryRegion::ExecutionBuffers)
    };
    let before = execution(&db);
    for query in GQL {
        assert_refused(db.execute(query));
        assert_eq!(execution(&db), before, "{query}");
    }
    let rows = db
        .execute("UNWIND range(1, 1000) AS i RETURN DISTINCT i")
        .unwrap();
    assert_eq!(rows.rows().len(), 1000);
    assert_eq!(execution(&db), before);
}

/// `COUNT(DISTINCT)` counts its DISTINCT sets, globally and per group.
#[test]
fn count_distinct_refuses_at_the_query_limit() {
    let small = capped(MIB);
    let large = capped(512 * MIB);
    for (query, rows) in [
        (
            "UNWIND range(1, 200000) AS i RETURN count(DISTINCT i) AS n",
            1,
        ),
        (
            "UNWIND range(1, 200000) AS i RETURN i % 4 AS k, count(DISTINCT i) AS n",
            4,
        ),
    ] {
        assert_refused(small.execute(query));
        assert_eq!(large.execute(query).unwrap().rows().len(), rows, "{query}");
    }
}

/// A MERGE over many rows keys the label's nodes by their values: that table
/// counts, and a MERGE refused at the query limit creates nothing.
#[test]
fn a_merge_refused_by_its_keys_changes_nothing() {
    for (limit, refused) in [(MIB, true), (512 * MIB, false)] {
        let db = capped(limit);
        db.execute("UNWIND range(1, 50000) AS i INSERT (:M {id: i})")
            .unwrap();
        let merge = db.execute("UNWIND range(1, 60000) AS i MERGE (:M {id: i})");
        if refused {
            assert_refused(merge);
            assert_eq!(count(&db, "M"), 50000);
        } else {
            merge.unwrap();
            assert_eq!(count(&db, "M"), 60000);
        }
    }
}

/// A database of 10,000 people who each know someone, in RDF.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn rdf_people(limit: usize) -> GrafeoDB {
    use std::fmt::Write;

    let db = capped(limit);
    let mut data = String::from("PREFIX : <urn:> INSERT DATA {");
    for i in 0..10_000 {
        write!(data, " :p{i} :t :person . :p{i} :knows :p{} .", i + 1).unwrap();
    }
    data.push_str(" }");
    db.execute_sparql(&data).unwrap();
    db
}

/// The SPARQL operators that keep their input: an `EXISTS` mark's subquery
/// rows (by lookup, and by a check over them) and CONSTRUCT's seen triples.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_exists_and_construct_refuse_at_the_query_limit() {
    let small = rdf_people(128 * 1024);
    let large = rdf_people(512 * MIB);
    for (query, route) in [
        (
            "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :person \
             FILTER(EXISTS { ?s :knows ?o } || ?s = :nobody) }",
            Some("ExistsMark"),
        ),
        (
            "PREFIX : <urn:> SELECT ?s WHERE { ?s :t ?kind \
             FILTER EXISTS { ?s :knows ?o FILTER(?kind = :person) } }",
            Some("ExistsRowCheck"),
        ),
        (
            "PREFIX : <urn:> CONSTRUCT { ?s :known ?o } WHERE { ?s :knows ?o }",
            None,
        ),
    ] {
        if let Some(route) = route {
            let profile = large.execute_sparql(&format!("PROFILE {query}")).unwrap();
            assert!(
                format!("{:?}", profile.rows()).contains(route),
                "{query} runs as {route}: {:?}",
                profile.rows()
            );
        }
        assert_refused(small.execute_sparql(query));
        assert_eq!(
            large.execute_sparql(query).unwrap().rows().len(),
            10_000,
            "{query}"
        );
    }
}

/// A built-in procedure's result, kept whole until its rows are read.
#[cfg(feature = "algos")]
#[test]
fn a_procedure_result_refuses_at_the_query_limit() {
    let query = "CALL grafeo.pagerank() YIELD node_id, score RETURN node_id, score";
    for (limit, refused) in [(MIB, true), (512 * MIB, false)] {
        let db = capped(limit);
        db.execute("UNWIND range(1, 20000) AS i INSERT (:P {i: i})")
            .unwrap();
        if refused {
            assert_refused(db.execute(query));
        } else {
            assert_eq!(db.execute(query).unwrap().rows().len(), 20_000);
        }
    }
}

/// What a string of 4,000 `x` adds to each of 2,000 computed values: 8 MB of
/// strings in all, past a 1 MiB cap that the same rows without them fit in.
fn long() -> String {
    "x".repeat(4000)
}

/// A sort or a distinct counts what its values hold on the heap, not only a
/// slot per value: 2,000 computed strings of 4 KB pass a 1 MiB cap, and run
/// at a large one. Cypher's ORDER BY is the pull sort; GQL's DISTINCT is the
/// push distinct.
#[test]
fn computed_long_strings_count_against_the_query_limit() {
    let small = capped(MIB);
    let large = capped(512 * MIB);
    let distinct = format!(
        "UNWIND range(1, 2000) AS i RETURN DISTINCT toString(i) || '{}' AS s",
        long()
    );
    assert_refused(small.execute(&distinct));
    assert_eq!(large.execute(&distinct).unwrap().rows().len(), 2000);
    #[cfg(feature = "cypher")]
    {
        let sort = format!(
            "UNWIND range(1, 2000) AS i WITH toString(i) + '{}' AS s RETURN s ORDER BY s",
            long()
        );
        assert_refused(small.execute_cypher(&sort));
        assert_eq!(large.execute_cypher(&sort).unwrap().rows().len(), 2000);
    }
}

/// A stored procedure's result is kept whole while the caller reads it: it
/// counts against the calling statement's limit.
#[test]
fn a_stored_procedure_result_refuses_at_the_query_limit() {
    for (limit, refused) in [(MIB, true), (512 * MIB, false)] {
        let db = capped(limit);
        db.execute(
            "CREATE PROCEDURE many() RETURNS (n INTEGER) AS \
             { UNWIND range(1, 200000) AS i RETURN i AS n }",
        )
        .unwrap();
        let call = db.execute("CALL many() YIELD n RETURN n");
        if refused {
            assert_refused(call);
        } else {
            assert_eq!(call.unwrap().rows().len(), 200_000);
        }
    }
}

/// A hash join keeps its build side: 50,000 rows a side by value pass a
/// 1 MiB cap, and join at a large one.
#[test]
fn a_hash_join_build_side_refuses_at_the_query_limit() {
    let query = "MATCH (a:A), (b:B) WHERE a.k = b.k RETURN count(*) AS n";
    for (limit, refused) in [(MIB, true), (512 * MIB, false)] {
        let db = capped(limit);
        db.execute("UNWIND range(1, 50000) AS i INSERT (:A {k: i}), (:B {k: i})")
            .unwrap();
        let joined = db.execute(query);
        if refused {
            assert_refused(joined);
        } else {
            assert_eq!(joined.unwrap().rows()[0][0].as_int64(), Some(50_000));
        }
    }
}

/// `collect` keeps every value of its group.
#[test]
fn collect_refuses_at_the_query_limit() {
    let query = "UNWIND range(1, 200000) AS i RETURN collect(i) AS xs";
    assert_refused(capped(MIB).execute(query));
    let rows = capped(512 * MIB).execute(query).unwrap();
    match &rows.rows()[0][0] {
        Value::List(values) => assert_eq!(values.len(), 200_000),
        other => panic!("a list: {other:?}"),
    }
}

/// A write above an expand reads its input whole first (an eager buffer, a
/// step of the physical plan that `EXPLAIN` does not show): that buffer
/// counts, and a write it refuses changes nothing. The same read without the
/// write, which buffers nothing, fits.
#[test]
fn the_eager_buffer_of_a_write_refuses_at_the_query_limit() {
    let write = "MATCH (a:A)-[:R]->(b:B) SET b.seen = true";
    for (limit, refused) in [(MIB, true), (512 * MIB, false)] {
        let db = capped(limit);
        db.execute("UNWIND range(1, 50000) AS i INSERT (:A {k: i})-[:R]->(:B {k: i})")
            .unwrap();
        let read = db
            .execute("MATCH (a:A)-[:R]->(b:B) RETURN count(*) AS n")
            .unwrap();
        assert_eq!(read.rows()[0][0].as_int64(), Some(50_000), "the read fits");
        let seen = "MATCH (b:B) WHERE b.seen = true RETURN count(b) AS n";
        if refused {
            assert_refused(db.execute(write));
            assert_eq!(db.execute(seen).unwrap().rows()[0][0].as_int64(), Some(0));
        } else {
            db.execute(write).unwrap();
            assert_eq!(
                db.execute(seen).unwrap().rows()[0][0].as_int64(),
                Some(50_000)
            );
        }
    }
}

/// A refusal inside an explicit transaction undoes that statement only: the
/// transaction stays usable, and what it wrote before is kept and commits.
#[test]
fn a_refusal_inside_a_transaction_undoes_only_its_statement() {
    let db = capped(MIB);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    assert_refused(
        session.execute("UNWIND range(1, 200000) AS i WITH DISTINCT i INSERT (:Person {i: i})"),
    );
    let in_transaction = session
        .execute("MATCH (p:Person) RETURN count(p) AS n")
        .unwrap();
    assert_eq!(in_transaction.rows()[0][0].as_int64(), Some(1));
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.commit().unwrap();
    assert_eq!(count(&db, "Person"), 2);
}

/// `EXPLAIN` plans a statement without running it, so a statement the limit
/// refuses explains; `PROFILE` runs it, and is refused.
#[test]
fn explain_of_a_refused_statement_succeeds_and_profile_is_refused() {
    let db = capped(MIB);
    let refused = "UNWIND range(1, 200000) AS i RETURN DISTINCT i";
    assert_refused(db.execute(refused));
    let explained = db.execute(&format!("EXPLAIN {refused}")).unwrap();
    assert_ne!(explained.rows().len(), 0, "EXPLAIN gives the plan");
    assert_refused(db.execute(&format!("PROFILE {refused}")));
}

/// A strict expected failure of a grouping that should spill: `body` fails
/// with `GRAFEO-S001` today (the behavior `spec` names is not there yet,
/// `issue` tracks it). If it passes, the test fails so the mark is removed;
/// a failure of another kind (a timeout, a wrong answer) fails it too.
fn xfail(spec: &str, issue: &str, body: impl FnOnce() -> Result<(), Error>) {
    match body() {
        Ok(()) => panic!("XPASS: {spec} ({issue}) holds now; remove the xfail"),
        Err(error) => assert_eq!(
            error.error_code().as_str(),
            "GRAFEO-S001",
            "{spec} ({issue}) fails for another reason: {error}"
        ),
    }
}

/// The groupings of 100,000 groups over 200,000 rows, past a 1 MiB cap.
const GROUPINGS: &[&str] = &[
    "UNWIND range(1, 200000) AS i RETURN i % 100000 AS k, count(*) AS n",
    "UNWIND range(1, 200000) AS i RETURN i % 100000 AS k, count(*) AS n ORDER BY k",
    "UNWIND range(1, 200000) AS i RETURN i % 100000 AS k, count(*) AS n LIMIT 5",
    "UNWIND range(1, 200000) AS i RETURN i % 100000 AS k, count(DISTINCT i) AS n",
    "UNWIND range(1, 200000) AS i RETURN i % 100000 AS k, collect(i) AS xs",
];

/// A grouping's answer in a form two runs compare by: its rows in order for
/// an ORDER BY, else sorted, with each collected list sorted; for a LIMIT
/// without an ORDER BY, which groups come is open, so only how many and
/// their counts.
fn answer(query: &str, rows: &[Vec<Value>]) -> Vec<String> {
    let row_text = |row: &Vec<Value>| -> String {
        row.iter()
            .map(|value| match value {
                Value::List(items) => {
                    let mut items: Vec<String> = items.iter().map(|v| format!("{v:?}")).collect();
                    items.sort();
                    format!("{items:?}")
                }
                other => format!("{other:?}"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    if query.contains("LIMIT") && !query.contains("ORDER BY") {
        return rows.iter().map(|row| format!("{:?}", row[1])).collect();
    }
    let mut text: Vec<String> = rows.iter().map(row_text).collect();
    if !query.contains("ORDER BY") {
        text.sort();
    }
    text
}

/// A grouping past the query limit with a spill path spills and answers as in
/// memory; without room to spill it is refused naming the query limit; a
/// higher limit gives the same answer. The spilling arm is a strict expected
/// failure: a grouping counts its spilled groups and refuses at once, until
/// spilled groups merge at the end instead of being loaded again.
#[test]
fn a_grouping_past_the_query_limit_spills_or_names_the_limit() {
    for query in GROUPINGS {
        let want = answer(
            query,
            GrafeoDB::new_in_memory().execute(query).unwrap().rows(),
        );

        let higher = GrafeoDB::with_config(
            Config::in_memory()
                .with_query_memory_limit(512 * MIB)
                .with_spill_limit(0),
        )
        .unwrap();
        assert_eq!(
            answer(query, higher.execute(query).unwrap().rows()),
            want,
            "a higher limit, in memory: {query}"
        );

        let no_spill = GrafeoDB::with_config(
            Config::in_memory()
                .with_query_memory_limit(MIB)
                .with_spill_limit(0),
        )
        .unwrap();
        assert_refused(no_spill.execute(query));

        let spill = tempfile::tempdir().unwrap();
        let spilling = GrafeoDB::with_config(
            Config::in_memory()
                .with_query_memory_limit(MIB)
                .with_spill_path(spill.path())
                .with_query_timeout(std::time::Duration::from_secs(20)),
        )
        .unwrap();
        xfail(
            "docs/architecture/memory: at the query memory limit a grouping spills and answers as in memory",
            "#674",
            || {
                let rows = spilling.execute(query)?;
                assert_eq!(answer(query, rows.rows()), want, "spilled: {query}");
                Ok(())
            },
        );
    }
}
