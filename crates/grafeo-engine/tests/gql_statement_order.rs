//! GQL statements in any order the standard allows (#483).
//!
//! ISO/IEC 39075:2024 composes a query of simple statements: a
//! `<simple linear query statement>` is a sequence of `<simple query
//! statement>`s (`<match statement>`, `<let statement>`, `<for statement>`,
//! `<filter statement>`, `<order by and page statement>`, `<call query
//! statement>`) in any order, ending in a `<primitive result statement>`; a
//! `<linear data-modifying statement>` mixes them with `<simple data-modifying
//! statement>`s (`<insert statement>`, `<set statement>`, `<remove
//! statement>`, `<delete statement>`, `<call data-modifying procedure
//! statement>`) and may end without a result. Each statement reads the rows the
//! ones before it leave: an `ORDER BY` and `LIMIT` in the middle cut the rows
//! the statements after them see, and a `WHERE` or `FILTER` between `MATCH`
//! statements filters the rows so far. Grafeo's `WITH` (an extension) is one
//! more such statement.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test gql_statement_order
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix (33), Gus (19), Vincent (88) and Mia (38), `:Person` with an `age`;
/// Alix and Mia live in Amsterdam, Gus in Berlin, Vincent in Paris; Alix
/// knows Gus and Mia, Gus knows Vincent.
fn town() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix', age: 33}), (gus:Person {name: 'Gus', age: 19}), \
         (vincent:Person {name: 'Vincent', age: 88}), (mia:Person {name: 'Mia', age: 38}), \
         (amsterdam:City {name: 'Amsterdam'}), (berlin:City {name: 'Berlin'}), \
         (paris:City {name: 'Paris'}), \
         (alix)-[:LIVES_IN]->(amsterdam), (mia)-[:LIVES_IN]->(amsterdam), \
         (gus)-[:LIVES_IN]->(berlin), (vincent)-[:LIVES_IN]->(paris), \
         (alix)-[:KNOWS]->(gus), (gus)-[:KNOWS]->(vincent), (alix)-[:KNOWS]->(mia)",
    )
    .unwrap();
    db
}

/// The rows of `query` in the order it returns them.
fn ordered(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("`{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// The rows of `query`, sorted (for a query without an `ORDER BY`).
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    let mut rows = ordered(db, query);
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

fn text(value: &str) -> Value {
    Value::from(value)
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

// ---------------------------------------------------------------------------
// <order by and page statement> between other statements
// ---------------------------------------------------------------------------

#[test]
fn an_order_by_and_limit_before_a_match_cut_the_rows_it_reads() {
    let db = town();
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person) ORDER BY p.age DESC LIMIT 2 \
             MATCH (p)-[:LIVES_IN]->(c:City) RETURN p.name, c.name ORDER BY p.name",
        ),
        [
            vec![text("Mia"), text("Amsterdam")],
            vec![text("Vincent"), text("Paris")],
        ],
        "the two oldest people, then their cities"
    );
}

#[test]
fn a_limit_after_with_cuts_the_rows_before_the_next_match() {
    let db = town();
    // The two youngest are Gus and Alix, who know three people between them:
    // a LIMIT 2 applied after the second MATCH would leave two rows.
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person) WITH p ORDER BY p.age LIMIT 2 \
             MATCH (p)-[:KNOWS]->(f) RETURN p.name, f.name",
        ),
        [
            vec![text("Alix"), text("Gus")],
            vec![text("Alix"), text("Mia")],
            vec![text("Gus"), text("Vincent")],
        ]
    );
    assert_eq!(
        rows(&db, "MATCH (p:Person) WITH p LIMIT 1 RETURN count(*)"),
        [vec![int(1)]]
    );
}

#[test]
fn an_offset_between_statements_skips_rows() {
    let db = town();
    // By age: Gus 19, Alix 33, Mia 38, Vincent 88.
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person) ORDER BY p.age OFFSET 1 LIMIT 2 RETURN p.name ORDER BY p.name",
        ),
        [vec![text("Alix")], vec![text("Mia")]]
    );
    assert_eq!(
        ordered(&db, "MATCH (p:Person) SKIP 3 RETURN count(*)"),
        [vec![int(1)]],
        "SKIP is a synonym of OFFSET"
    );
}

#[test]
fn an_aggregate_after_a_page_statement_reads_only_the_kept_rows() {
    let db = town();
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person) ORDER BY p.age DESC LIMIT 3 RETURN sum(p.age) AS total",
        ),
        [vec![int(88 + 38 + 33)]]
    );
}

#[test]
fn a_write_after_a_limit_writes_only_the_rows_it_keeps() {
    let db = town();
    db.execute("MATCH (p:Person) ORDER BY p.age LIMIT 2 SET p.young = true")
        .unwrap();
    assert_eq!(
        rows(&db, "MATCH (p:Person) WHERE p.young = true RETURN p.name"),
        [vec![text("Alix")], vec![text("Gus")]]
    );
}

#[test]
fn a_write_before_a_limit_writes_every_row() {
    let db = town();
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person) SET p.seen = 3 LIMIT 1 RETURN count(*)"
        ),
        [vec![int(1)]],
        "the LIMIT cuts the rows after the SET"
    );
    assert_eq!(
        ordered(&db, "MATCH (p:Person) WHERE p.seen = 3 RETURN count(*)"),
        [vec![int(4)]],
        "the SET before the LIMIT wrote every row"
    );
}

#[test]
fn two_page_statements_apply_in_turn() {
    let db = town();
    // The three oldest (Vincent, Mia, Alix), then the youngest of them.
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person) ORDER BY p.age DESC LIMIT 3 ORDER BY p.age LIMIT 1 RETURN p.name",
        ),
        [vec![text("Alix")]]
    );
}

// ---------------------------------------------------------------------------
// WHERE and FILTER between MATCH statements
// ---------------------------------------------------------------------------

#[test]
fn a_where_between_match_statements_filters_the_rows_so_far() {
    let db = town();
    let expected = [
        vec![text("Alix"), text("Amsterdam")],
        vec![text("Mia"), text("Amsterdam")],
        vec![text("Vincent"), text("Paris")],
    ];
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person) WHERE a.age > 30 MATCH (a)-[:LIVES_IN]->(c) RETURN a.name, c.name",
        ),
        expected
    );
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person) FILTER a.age > 30 MATCH (a)-[:LIVES_IN]->(c) RETURN a.name, c.name",
        ),
        expected
    );
    // A MATCH that shares no variable with the rows so far: a cross product
    // of the filtered rows.
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person) WHERE a.age > 80 MATCH (c:City) RETURN a.name, c.name",
        ),
        [
            vec![text("Vincent"), text("Amsterdam")],
            vec![text("Vincent"), text("Berlin")],
            vec![text("Vincent"), text("Paris")],
        ]
    );
}

#[test]
fn a_where_before_an_optional_match_keeps_its_rows_without_a_match() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person) WHERE a.age > 30 \
             OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name",
        ),
        [
            vec![text("Alix"), text("Gus")],
            vec![text("Alix"), text("Mia")],
            vec![text("Mia"), Value::Null],
            vec![text("Vincent"), Value::Null],
        ]
    );
    // The WHERE of the OPTIONAL MATCH belongs to it: a person without a match
    // keeps a row of nulls.
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person) WHERE a.age > 30 \
             OPTIONAL MATCH (a)-[:KNOWS]->(b) WHERE b.age > 35 RETURN a.name, b.name",
        ),
        [
            vec![text("Alix"), text("Mia")],
            vec![text("Mia"), Value::Null],
            vec![text("Vincent"), Value::Null],
        ]
    );
    // A FILTER there is a statement of its own: it filters every row, the
    // rows without a match included.
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person) WHERE a.age > 30 \
             OPTIONAL MATCH (a)-[:KNOWS]->(b) FILTER b.age > 35 RETURN a.name, b.name",
        ),
        [vec![text("Alix"), text("Mia")]]
    );
}

#[test]
fn a_where_on_a_match_after_with_reads_both() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person) WITH a MATCH (b:Person) \
             WHERE a.age < b.age AND a.name = 'Mia' RETURN a.name, b.name",
        ),
        [vec![text("Mia"), text("Vincent")]]
    );
}

#[test]
fn a_let_after_a_where_or_filter_reads_the_filtered_rows() {
    let db = town();
    for keyword in ["WHERE", "FILTER"] {
        assert_eq!(
            rows(
                &db,
                &format!(
                    "MATCH (p:Person)-[:LIVES_IN]->(c:City) {keyword} c.name IN ['Amsterdam'] \
                     LET x = p.name RETURN x"
                ),
            ),
            [vec![text("Alix")], vec![text("Mia")]],
            "{keyword}"
        );
    }
}

#[test]
fn a_call_after_a_where_runs_for_the_filtered_rows() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person) WHERE a.age > 30 \
             CALL (a) { MATCH (a)-[:LIVES_IN]->(c) RETURN c.name AS city } RETURN a.name, city",
        ),
        [
            vec![text("Alix"), text("Amsterdam")],
            vec![text("Mia"), text("Amsterdam")],
            vec![text("Vincent"), text("Paris")],
        ]
    );
}

#[test]
fn a_where_reads_only_the_variables_bound_before_it() {
    let db = town();
    let error = db
        .execute("MATCH (a:Person) WHERE b.age > 3 MATCH (b:Person) RETURN a.name")
        .expect_err("`b` is bound only after the WHERE");
    assert!(
        error.to_string().contains("Undefined variable 'b'"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// Data-modifying statements after WITH, SET, LET and FOR
// ---------------------------------------------------------------------------

#[test]
fn a_set_after_with_writes_the_rows_of_the_with() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person) WITH p WHERE p.age > 30 SET p.senior = true RETURN p.name, p.senior",
        ),
        [
            vec![text("Alix"), Value::Bool(true)],
            vec![text("Mia"), Value::Bool(true)],
            vec![text("Vincent"), Value::Bool(true)],
        ]
    );
    assert_eq!(
        rows(&db, "MATCH (p:Person) WHERE p.senior IS NULL RETURN p.name"),
        [vec![text("Gus")]]
    );
}

#[test]
fn a_delete_after_with_deletes_the_rows_of_the_with() {
    let db = town();
    db.execute("MATCH (p:Person) WITH p WHERE p.name = 'Gus' DETACH DELETE p")
        .unwrap();
    assert_eq!(
        rows(&db, "MATCH (p:Person) RETURN p.name"),
        [vec![text("Alix")], vec![text("Mia")], vec![text("Vincent")]]
    );
    assert_eq!(
        rows(&db, "MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name"),
        [vec![text("Alix"), text("Mia")]],
        "DETACH DELETE removed the edges of Gus"
    );
}

#[test]
fn a_remove_after_with_removes_from_the_rows_of_the_with() {
    let db = town();
    db.execute("MATCH (p:Person) WITH p WHERE p.age < 30 REMOVE p.age")
        .unwrap();
    assert_eq!(
        rows(&db, "MATCH (p:Person) WHERE p.age IS NULL RETURN p.name"),
        [vec![text("Gus")]]
    );
}

#[test]
fn an_insert_after_with_inserts_once_per_row_of_the_with() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person) WITH p WHERE p.age > 35 \
             INSERT (p)-[:VISITED]->(:City {name: 'Prague'}) RETURN p.name",
        ),
        [vec![text("Mia")], vec![text("Vincent")]]
    );
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person)-[:VISITED]->(c:City) RETURN p.name, c.name",
        ),
        [
            vec![text("Mia"), text("Prague")],
            vec![text("Vincent"), text("Prague")],
        ]
    );
}

#[test]
fn a_call_after_with_runs_for_each_row_of_the_with() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person) WITH p WHERE p.age < 30 \
             CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS friend } RETURN p.name, friend",
        ),
        [vec![text("Gus"), text("Vincent")]]
    );
}

#[test]
fn a_call_after_a_set_sees_the_whole_set() {
    let db = town();
    // The subquery counts every person the SET wrote, not only the ones
    // before the current row.
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person) SET p.score = 3 \
             CALL () { MATCH (q:Person) WHERE q.score = 3 RETURN count(q) AS done } \
             RETURN p.name, done",
        ),
        [
            vec![text("Alix"), int(4)],
            vec![text("Gus"), int(4)],
            vec![text("Mia"), int(4)],
            vec![text("Vincent"), int(4)],
        ]
    );
}

#[test]
fn a_filter_after_a_set_reads_the_written_values() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person) SET p.band = p.age - 30 FILTER p.band > 0 RETURN p.name, p.band",
        ),
        [
            vec![text("Alix"), int(3)],
            vec![text("Mia"), int(8)],
            vec![text("Vincent"), int(58)],
        ]
    );
}

#[test]
fn a_delete_after_a_set_and_after_let_and_for() {
    let db = town();
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person {name: 'Gus'}) SET p.gone = true DETACH DELETE p RETURN count(*)",
        ),
        [vec![int(1)]]
    );
    db.execute("MATCH (p:Person) LET years = p.age FILTER years > 80 DETACH DELETE p")
        .unwrap();
    db.execute("MATCH (p:Person) WITH p FOR x IN [3] FILTER p.age = 38 DETACH DELETE p")
        .unwrap();
    assert_eq!(
        rows(&db, "MATCH (p:Person) RETURN p.name"),
        [vec![text("Alix")]]
    );
}

#[test]
fn remove_and_set_apply_in_the_order_written() {
    let db = town();
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person {name: 'Alix'}) REMOVE p.age SET p.age = 19 RETURN p.age",
        ),
        [vec![int(19)]]
    );
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person {name: 'Gus'}) SET p.age = 3 REMOVE p.age RETURN p.age",
        ),
        [vec![Value::Null]]
    );
}

/// A WHERE or FILTER right after INSERT, CREATE, MERGE or DELETE is an
/// error, not a filter of the rows after the write: it used to filter the
/// rows before the write, and a silent switch would change what the write
/// touches (`DETACH DELETE p FILTER p.age > 30` would delete every person).
/// A WHERE right after SET is an error too. Nothing is written.
#[test]
fn a_filter_right_after_a_write_is_rejected_and_writes_nothing() {
    let db = town();
    for (query, message) in [
        (
            "MATCH (p:Person) DETACH DELETE p WHERE p.age > 30",
            "a WHERE or FILTER cannot follow INSERT, CREATE, MERGE or DELETE",
        ),
        (
            "MATCH (p:Person) DETACH DELETE p FILTER p.age > 30",
            "a WHERE or FILTER cannot follow INSERT, CREATE, MERGE or DELETE",
        ),
        (
            "MATCH (p:Person) INSERT (p)-[:VISITED]->(:City {name: 'Prague'}) FILTER p.age > 30",
            "a WHERE or FILTER cannot follow INSERT, CREATE, MERGE or DELETE",
        ),
        (
            "MATCH (p:Person) SET p.flag = 3 WHERE p.age > 30",
            "a WHERE cannot follow SET or REMOVE",
        ),
    ] {
        let error = db.execute(query).expect_err(query);
        assert!(error.to_string().contains(message), "`{query}`: {error}");
    }
    assert_eq!(
        ordered(&db, "MATCH (p:Person) RETURN count(*)"),
        [vec![int(4)]]
    );
    assert_eq!(
        ordered(&db, "MATCH (c:City {name: 'Prague'}) RETURN count(*)"),
        [vec![int(0)]]
    );
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person) WHERE p.flag IS NOT NULL RETURN count(*)"
        ),
        [vec![int(0)]]
    );
}

/// A WITH between makes the order plain: its WHERE filters the rows after
/// the write, which touches every row before it.
#[test]
fn a_with_after_a_write_filters_the_rows_after_it() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person) INSERT (p)-[:VISITED]->(:City {name: 'Prague'}) \
             WITH p WHERE p.age > 35 RETURN p.name",
        ),
        [vec![text("Mia")], vec![text("Vincent")]]
    );
    assert_eq!(
        ordered(&db, "MATCH (:Person)-[:VISITED]->(c:City) RETURN count(c)"),
        [vec![int(4)]],
        "the INSERT ran for every person"
    );
}

/// The WHERE of a MATCH after a write filters the rows of that MATCH, after
/// the write (it used to be moved before the write).
#[test]
fn a_where_on_a_match_after_a_write_filters_after_the_write() {
    let db = town();
    assert_eq!(
        ordered(
            &db,
            "MATCH (p:Person) INSERT (:Tag {owner: p.name}) \
             MATCH (t:Tag) WHERE p.age > 80 RETURN count(*)",
        ),
        [vec![int(4)]],
        "Vincent's row, with each of the four tags"
    );
    assert_eq!(
        ordered(&db, "MATCH (t:Tag) RETURN count(*)"),
        [vec![int(4)]],
        "a tag for every person"
    );
}

#[test]
fn a_match_after_an_insert_sees_the_inserted_node() {
    let db = town();
    assert_eq!(
        rows(
            &db,
            "INSERT (j:Person {name: 'Jules', age: 3}) WITH j \
             MATCH (q:Person) WHERE q.age < 20 RETURN q.name",
        ),
        [vec![text("Gus")], vec![text("Jules")]]
    );
    assert_eq!(
        ordered(
            &db,
            "MATCH (a:Person {name: 'Alix'}) INSERT (a)-[:VISITED]->(c:City {name: 'Prague'}) \
             SET c.year = 2019 RETURN c.name, c.year",
        ),
        [vec![text("Prague"), int(2019)]]
    );
}

// ---------------------------------------------------------------------------
// A statement that starts with LET, FILTER or FOR
// ---------------------------------------------------------------------------

#[test]
fn a_statement_may_start_with_let_or_filter() {
    let db = town();
    assert_eq!(ordered(&db, "LET x = 3 RETURN x"), [vec![int(3)]]);
    assert_eq!(
        ordered(&db, "LET x = 19 FILTER x > 3 RETURN x + 88"),
        [vec![int(107)]]
    );
    assert!(
        ordered(&db, "FILTER 3 > 19 RETURN 3").is_empty(),
        "a FILTER that fails removes the one row"
    );
    assert_eq!(
        ordered(
            &db,
            "LET who = 'Alix' MATCH (p:Person {name: who}) RETURN p.age"
        ),
        [vec![int(33)]]
    );
    assert_eq!(
        ordered(&db, "RETURN 3 AS x NEXT LET y = x + 19 RETURN y"),
        [vec![int(22)]]
    );
}

#[test]
fn a_with_may_follow_a_for() {
    let db = town();
    assert_eq!(
        rows(&db, "FOR x IN [3, 19] WITH x WHERE x > 3 RETURN x"),
        [vec![int(19)]]
    );
}
