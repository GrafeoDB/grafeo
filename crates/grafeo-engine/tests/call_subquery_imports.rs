//! The variables a `CALL` subquery imports stay in scope for its whole body:
//! a `WITH` in it that leaves one out does not end its scope (openCypher:
//! "a subsequent WITH within the subquery cannot descope an imported
//! variable"; in GQL they are fields of the body's incoming working record).
//! This holds for every form of import: a scope clause (`CALL (a) { ... }`),
//! Cypher's importing `WITH`, GQL's implicit import of the whole outer row,
//! nested and `OPTIONAL` calls, and each part of a `UNION` in the body. After
//! an aggregating `WITH` a count over no match is still one row, with the
//! import (#545). A variable the body binds itself still goes out of scope
//! as anywhere else.
//!
//! The data is the spec dataset `entity_kinds`: node and edge IDs overlap and
//! both carry `w`, so an import read as an edge shows an edge's `w` (1 to 8)
//! instead of the node's (100 and up).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test call_subquery_imports
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    Gql,
    Cypher,
}

use Language::{Cypher, Gql};

/// Alix (30, w 100), Gus (25, w 101), Vincent (40, w 103), Jules (35) and Mia
/// (28); Amsterdam (w 104), Berlin and Paris (w 105). KNOWS (w 1 to 5):
/// Alix->Gus, Gus->Vincent, Vincent->Alix, Jules->Mia, Alix->Jules. LIVES_IN
/// (w 6 to 8): Alix->Amsterdam, Gus->Berlin, Mia->Paris.
fn entity_kinds() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix', age: 30, w: 100}), \
         (gus:Person {name: 'Gus', age: 25, w: 101}), \
         (vincent:Person {name: 'Vincent', age: 40, w: 103}), \
         (jules:Person {name: 'Jules', age: 35}), (mia:Person {name: 'Mia', age: 28}), \
         (ams:City {name: 'Amsterdam', w: 104}), (ber:City {name: 'Berlin'}), \
         (par:City {name: 'Paris', w: 105}), \
         (alix)-[:KNOWS {w: 1}]->(gus), (gus)-[:KNOWS {w: 2}]->(vincent), \
         (vincent)-[:KNOWS {w: 3}]->(alix), (jules)-[:KNOWS {w: 4}]->(mia), \
         (alix)-[:KNOWS {w: 5}]->(jules), (alix)-[:LIVES_IN {w: 6}]->(ams), \
         (gus)-[:LIVES_IN {w: 7}]->(ber), (mia)-[:LIVES_IN {w: 8}]->(par)",
    )
    .unwrap();
    db
}

fn execute(
    session: &Session,
    language: Language,
    query: &str,
) -> grafeo_common::utils::error::Result<Vec<Vec<Value>>> {
    let result = match language {
        Gql => session.execute(query),
        Cypher => session.execute_cypher(query),
    };
    result.map(|result| result.rows().to_vec())
}

/// The rows of `query` in `language` on a new `entity_kinds` database.
fn rows(language: Language, query: &str) -> Vec<Vec<Value>> {
    let db = entity_kinds();
    execute(&db.session(), language, query)
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
}

/// The error of `query` in `language` on a new `entity_kinds` database.
fn error(language: Language, query: &str) -> String {
    let db = entity_kinds();
    match execute(&db.session(), language, query) {
        Ok(rows) => panic!("{language:?} `{query}` returned {rows:?}, expected an error"),
        Err(error) => error.to_string(),
    }
}

fn s(text: &str) -> Value {
    Value::String(text.into())
}

fn i(value: i64) -> Value {
    Value::Int64(value)
}

/// Each query, in each of its languages, returns `expected`.
fn assert_rows(cases: &[(&[Language], &str)], expected: &[Vec<Value>]) {
    for (languages, query) in cases {
        for language in *languages {
            assert_eq!(rows(*language, query), expected, "{language:?} `{query}`");
        }
    }
}

const BOTH: &[Language] = &[Gql, Cypher];

#[test]
fn an_import_stays_visible_after_a_with_that_leaves_it_out() {
    assert_rows(
        &[
            (
                BOTH,
                "MATCH (a:Person {name: 'Alix'}) \
                 CALL (a) { WITH 3 AS x RETURN a.name AS n, a.w AS w, x } RETURN n, w, x",
            ),
            // Cypher's importing WITH
            (
                &[Cypher],
                "MATCH (a:Person {name: 'Alix'}) \
                 CALL { WITH a WITH 3 AS x RETURN a.name AS n, a.w AS w, x } RETURN n, w, x",
            ),
            // GQL imports the whole outer row without a scope clause
            (
                &[Gql],
                "MATCH (a:Person {name: 'Alix'}) \
                 CALL { WITH 3 AS x RETURN a.name AS n, a.w AS w, x } RETURN n, w, x",
            ),
            // Two WITHs in a row
            (
                BOTH,
                "MATCH (a:Person {name: 'Alix'}) \
                 CALL (a) { WITH 1 AS y WITH y + 2 AS x RETURN a.name AS n, a.w AS w, x } \
                 RETURN n, w, x",
            ),
        ],
        &[vec![s("Alix"), i(100), i(3)]],
    );
}

/// A count over no match is one row, with the import: Mia knows nobody.
#[test]
fn an_aggregating_with_keeps_the_import_and_its_one_row() {
    assert_rows(
        &[
            (
                BOTH,
                "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k \
                 RETURN a.name AS n, a.w AS w, k } RETURN n, w, k ORDER BY n",
            ),
            (
                &[Cypher],
                "MATCH (a:Person) CALL { WITH a MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k \
                 RETURN a.name AS n, a.w AS w, k } RETURN n, w, k ORDER BY n",
            ),
            (
                &[Gql],
                "MATCH (a:Person) CALL { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k \
                 RETURN a.name AS n, a.w AS w, k } RETURN n, w, k ORDER BY n",
            ),
        ],
        &[
            vec![s("Alix"), i(100), i(2)],
            vec![s("Gus"), i(101), i(1)],
            vec![s("Jules"), Value::Null, i(1)],
            vec![s("Mia"), Value::Null, i(0)],
            vec![s("Vincent"), i(103), i(1)],
        ],
    );
}

/// An imported value and an imported edge come back as what they are after
/// an aggregating WITH: a number to add to, an edge with its type and `w`
/// (not the node with the same ID).
#[test]
fn an_aggregating_with_keeps_imported_values_and_edges() {
    assert_rows(
        &[(
            BOTH,
            "UNWIND [3, 19] AS v CALL (v) { MATCH (p:Person) WITH count(p) AS k \
             RETURN v + k AS s, v * 2 AS d } RETURN s, d ORDER BY s",
        )],
        &[vec![i(8), i(6)], vec![i(24), i(38)]],
    );
    assert_rows(
        &[
            (
                BOTH,
                "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) \
                 CALL (r) { MATCH (c:City) WITH count(c) AS k \
                 RETURN type(r) AS t, r.w AS w, k } RETURN t, w, k",
            ),
            (
                &[Cypher],
                "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) \
                 CALL { WITH r MATCH (c:City) WITH count(c) AS k \
                 RETURN type(r) AS t, r.w AS w, k } RETURN t, w, k",
            ),
        ],
        &[vec![s("KNOWS"), i(1), i(3)]],
    );
    // The edge matches as an edge after the WITH.
    assert_rows(
        &[(
            BOTH,
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) \
             CALL (r) { MATCH (c:City) WITH count(c) AS k \
             MATCH (x)-[r]->(y) RETURN x.name AS x, y.name AS y, k } RETURN x, y, k",
        )],
        &[vec![s("Alix"), s("Gus"), i(3)]],
    );
}

/// A group key of the aggregation is its own: the import beside it stays.
#[test]
fn an_aggregation_with_its_own_group_key_keeps_the_import() {
    assert_rows(
        &[(
            BOTH,
            "MATCH (a:Person {name: 'Alix'}) CALL (a) { MATCH (a)-[:KNOWS]->(b) \
             WITH b.age > 30 AS older, count(*) AS k RETURN a.name AS n, older, k } \
             RETURN n, older, k ORDER BY older",
        )],
        &[
            vec![s("Alix"), Value::Bool(false), i(1)],
            vec![s("Alix"), Value::Bool(true), i(1)],
        ],
    );
}

#[test]
fn a_with_that_renames_an_import_keeps_the_import_too() {
    assert_rows(
        &[
            (
                BOTH,
                "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH a AS p RETURN p.name AS m, a.w AS w } \
                 RETURN m, w",
            ),
            (
                &[Cypher],
                "MATCH (a:Person {name: 'Alix'}) \
                 CALL { WITH a WITH a AS p RETURN p.name AS m, a.w AS w } RETURN m, w",
            ),
        ],
        &[vec![s("Alix"), i(100)]],
    );
}

/// A WITH that names only a variable of the body: the import stays too.
#[test]
fn a_with_that_drops_the_import_on_purpose_keeps_it() {
    assert_rows(
        &[(
            BOTH,
            "MATCH (a:Person {name: 'Alix'}) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH b \
             RETURN a.name AS n, b.name AS f } RETURN n, f ORDER BY f",
        )],
        &[vec![s("Alix"), s("Gus")], vec![s("Alix"), s("Jules")]],
    );
}

/// A variable the body binds itself goes out of scope after a WITH that
/// leaves it out, as anywhere else.
#[test]
fn a_with_ends_the_scope_of_a_variable_of_the_body() {
    for (languages, query) in [
        (
            BOTH,
            "MATCH (a:Person {name: 'Alix'}) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH 1 AS x \
             RETURN b.name AS n } RETURN n",
        ),
        (
            BOTH,
            "MATCH (a:Person {name: 'Alix'}) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k \
             RETURN b.name AS n } RETURN n",
        ),
        (
            &[Cypher][..],
            "MATCH (a:Person {name: 'Alix'}) CALL { WITH a MATCH (a)-[:KNOWS]->(b) WITH a \
             RETURN b.name AS n } RETURN n",
        ),
    ] {
        for language in languages {
            let message = error(*language, query);
            assert!(
                message.contains("Undefined variable 'b'"),
                "{language:?} `{query}`: {message}"
            );
        }
    }
}

/// A variable of the outer query that the subquery does not import stays
/// invisible after a WITH too: only imports come back.
#[test]
fn a_variable_that_is_not_imported_stays_invisible_after_a_with() {
    for (languages, query) in [
        (
            BOTH,
            "MATCH (a:Person {name: 'Alix'}), (c:City {name: 'Paris'}) \
             CALL (a) { WITH 1 AS x RETURN c.name AS m } RETURN m",
        ),
        // The nested call imports a, not c, from the outer body.
        (
            BOTH,
            "MATCH (a:Person {name: 'Alix'}), (c:City {name: 'Paris'}) \
             CALL (a, c) { CALL (a) { WITH 2 AS y RETURN c.name AS m } RETURN m } RETURN m",
        ),
        (
            &[Cypher][..],
            "MATCH (a:Person {name: 'Alix'}), (c:City {name: 'Paris'}) \
             CALL (a, c) { WITH 1 AS x CALL (a) { WITH 2 AS y RETURN c.name AS m } RETURN m } \
             RETURN m",
        ),
    ] {
        for language in languages {
            let message = error(*language, query);
            assert!(
                message.contains("Undefined variable 'c'"),
                "{language:?} `{query}`: {message}"
            );
        }
    }
}

#[test]
fn the_where_of_a_with_reads_an_import() {
    assert_rows(
        &[(
            BOTH,
            "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH b WHERE b.age > a.age \
             RETURN b.name AS older } RETURN a.name AS a, older ORDER BY a",
        )],
        &[vec![s("Alix"), s("Jules")], vec![s("Gus"), s("Vincent")]],
    );
    // A condition on both the aggregate and the import: Mia knows nobody,
    // Vincent is older than 35.
    assert_rows(
        &[(
            BOTH,
            "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k \
             WHERE k = 0 OR a.age > 35 RETURN a.name AS n, k } RETURN n, k ORDER BY n",
        )],
        &[vec![s("Mia"), i(0)], vec![s("Vincent"), i(1)]],
    );
}

/// ORDER BY and LIMIT after a WITH (Cypher; a GQL WITH takes neither): the
/// oldest person each one knows.
#[test]
fn an_import_after_a_with_that_orders_and_cuts() {
    assert_rows(
        &[(
            &[Cypher],
            "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH b ORDER BY b.age DESC \
             LIMIT 1 RETURN a.name AS n, b.name AS f } RETURN n, f ORDER BY n",
        )],
        &[
            vec![s("Alix"), s("Jules")],
            vec![s("Gus"), s("Vincent")],
            vec![s("Jules"), s("Mia")],
            vec![s("Vincent"), s("Alix")],
        ],
    );
}

/// A WITH DISTINCT: one row for each person who knows someone.
#[test]
fn an_import_after_a_distinct_with() {
    assert_rows(
        &[(
            BOTH,
            "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH DISTINCT 1 AS one \
             RETURN a.name AS n, one } RETURN n, one ORDER BY n",
        )],
        &[
            vec![s("Alix"), i(1)],
            vec![s("Gus"), i(1)],
            vec![s("Jules"), i(1)],
            vec![s("Vincent"), i(1)],
        ],
    );
}

/// A MATCH and an UNWIND after the WITH read the import.
#[test]
fn clauses_after_a_with_read_the_import() {
    assert_rows(
        &[(
            BOTH,
            "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 1 AS x \
             MATCH (a)-[:LIVES_IN]->(c) UNWIND [a.w, x] AS v RETURN c.name AS c, v } \
             RETURN c, v ORDER BY v",
        )],
        &[vec![s("Amsterdam"), i(1)], vec![s("Amsterdam"), i(100)]],
    );
}

#[test]
fn nested_calls_keep_their_imports_after_a_with() {
    // A WITH in the nested body and one in the outer body after it.
    assert_rows(
        &[(
            BOTH,
            "MATCH (a:Person) CALL (a) { \
             CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k RETURN k } \
             WITH k + 1 AS s RETURN a.name AS n, s } RETURN n, s ORDER BY n",
        )],
        &[
            vec![s("Alix"), i(3)],
            vec![s("Gus"), i(2)],
            vec![s("Jules"), i(2)],
            vec![s("Mia"), i(1)],
            vec![s("Vincent"), i(2)],
        ],
    );
    assert_rows(
        &[
            (
                BOTH,
                "MATCH (a:Person {name: 'Alix'}) CALL (a) { \
                 CALL (a) { WITH 2 AS y RETURN a.w + y AS s } WITH s RETURN a.w + s AS t } \
                 RETURN t",
            ),
            // GQL: the nested call imports the outer body's whole row
            (
                &[Gql],
                "MATCH (a:Person {name: 'Alix'}) CALL { \
                 CALL { WITH 2 AS y RETURN a.w + y AS s } WITH s RETURN a.w + s AS t } \
                 RETURN t",
            ),
            (
                &[Cypher],
                "MATCH (a:Person {name: 'Alix'}) CALL { WITH a \
                 CALL { WITH a WITH 2 AS y RETURN a.w + y AS s } WITH s RETURN a.w + s AS t } \
                 RETURN t",
            ),
        ],
        &[vec![i(202)]],
    );
    // A nested call after a WITH (Cypher; a GQL WITH takes no CALL after it)
    // imports from the outer body's row, which still holds the import.
    assert_rows(
        &[
            (
                &[Cypher],
                "MATCH (a:Person {name: 'Alix'}) CALL { WITH a WITH 1 AS x \
                 CALL { WITH a, x WITH 2 AS y RETURN a.w + x + y AS s } RETURN s } RETURN s",
            ),
            (
                &[Cypher],
                "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 1 AS x \
                 CALL (a, x) { WITH 2 AS y RETURN a.w + x + y AS s } RETURN s } RETURN s",
            ),
        ],
        &[vec![i(103)]],
    );
}

/// An OPTIONAL CALL keeps a row without inner rows; the inner rows it has
/// read the import after a WITH.
#[test]
fn an_optional_call_reads_the_import_after_a_with() {
    assert_rows(
        &[(
            &[Gql],
            "MATCH (a:Person) OPTIONAL CALL (a) { MATCH (a)-[:LIVES_IN]->(c) WITH c \
             RETURN a.name || ' in ' || c.name AS s } RETURN a.name AS a, s ORDER BY a",
        )],
        &[
            vec![s("Alix"), s("Alix in Amsterdam")],
            vec![s("Gus"), s("Gus in Berlin")],
            vec![s("Jules"), Value::Null],
            vec![s("Mia"), s("Mia in Paris")],
            vec![s("Vincent"), Value::Null],
        ],
    );
    assert_rows(
        &[(
            &[Gql],
            "MATCH (a:Person) OPTIONAL CALL (a) { MATCH (a)-[:LIVES_IN]->(c) WITH count(c) AS k \
             RETURN a.name AS n, k } RETURN n, k ORDER BY n",
        )],
        &[
            vec![s("Alix"), i(1)],
            vec![s("Gus"), i(1)],
            vec![s("Jules"), i(0)],
            vec![s("Mia"), i(1)],
            vec![s("Vincent"), i(0)],
        ],
    );
}

/// Each part of a UNION in the body keeps its own imports.
#[test]
fn each_part_of_a_union_keeps_its_imports() {
    assert_rows(
        &[
            (
                BOTH,
                "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 1 AS x RETURN a.name AS n, x \
                 UNION ALL WITH 2 AS x RETURN a.name AS n, x } RETURN n, x ORDER BY x",
            ),
            (
                &[Cypher],
                "MATCH (a:Person {name: 'Alix'}) CALL { WITH a WITH 1 AS x RETURN a.name AS n, x \
                 UNION ALL WITH a WITH 2 AS x RETURN a.name AS n, x } RETURN n, x ORDER BY x",
            ),
        ],
        &[vec![s("Alix"), i(1)], vec![s("Alix"), i(2)]],
    );
    // The second part imports nothing: its first WITH names no variable.
    let message = error(
        Cypher,
        "MATCH (a:Person {name: 'Alix'}) CALL { WITH a WITH 1 AS x RETURN a.name AS n \
         UNION ALL WITH 2 AS x RETURN a.name AS n } RETURN n",
    );
    assert!(message.contains("Undefined variable 'a'"), "{message}");
}

/// After a WITH the import is the outer node itself: an EXISTS subquery
/// matches from it, and SET writes it (not the edge with the same ID).
#[test]
fn the_import_after_a_with_is_the_outer_node() {
    assert_rows(
        &[(
            BOTH,
            "MATCH (a:Person) CALL (a) { WITH 1 AS x \
             RETURN EXISTS { MATCH (a)-[:LIVES_IN]->(:City) } AS housed } \
             RETURN a.name AS n, housed ORDER BY n",
        )],
        &[
            vec![s("Alix"), Value::Bool(true)],
            vec![s("Gus"), Value::Bool(true)],
            vec![s("Jules"), Value::Bool(false)],
            vec![s("Mia"), Value::Bool(true)],
            vec![s("Vincent"), Value::Bool(false)],
        ],
    );
    // A GQL WITH takes no SET after it.
    for (language, query) in [
        (
            Cypher,
            "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k \
             SET a.friends = k RETURN k } RETURN count(*) AS people",
        ),
        (
            Cypher,
            "MATCH (a:Person) CALL { WITH a MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k \
             SET a.friends = k RETURN k } RETURN count(*) AS people",
        ),
    ] {
        let db = entity_kinds();
        let session = db.session();
        let people = execute(&session, language, query)
            .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"));
        assert_eq!(people, [vec![i(5)]], "{language:?} `{query}`");
        let friends = execute(
            &session,
            Gql,
            "MATCH (p:Person) RETURN p.name AS n, p.friends AS f ORDER BY n",
        )
        .unwrap();
        assert_eq!(
            friends,
            [
                vec![s("Alix"), i(2)],
                vec![s("Gus"), i(1)],
                vec![s("Jules"), i(1)],
                vec![s("Mia"), i(0)],
                vec![s("Vincent"), i(1)],
            ],
            "{language:?} `{query}`"
        );
        let edges = execute(
            &session,
            Gql,
            "MATCH ()-[r]->() WHERE r.friends IS NOT NULL RETURN count(r)",
        )
        .unwrap();
        assert_eq!(edges, [vec![i(0)]], "{language:?} `{query}` wrote an edge");
    }
}

/// A unit subquery (no RETURN) writes through the import after a WITH: the
/// MERGE starts at each person (a GQL WITH takes no SET after it), the SET
/// writes each person.
#[test]
fn a_unit_subquery_writes_through_the_import_after_a_with() {
    for (language, query, check) in [
        (
            Gql,
            "MATCH (a:Person) CALL (a) { WITH 19 AS x MERGE (a)-[:HAS]->(t:Tag {x: x}) } \
             RETURN count(*) AS people",
            "MATCH (:Person)-[:HAS]->(t:Tag {x: 19}) RETURN count(t)",
        ),
        (
            Cypher,
            "MATCH (a:Person) CALL { WITH a WITH 19 AS x SET a.x = x } RETURN count(*) AS people",
            "MATCH (p:Person {x: 19}) RETURN count(p)",
        ),
        (
            Cypher,
            "MATCH (a:Person) CALL (a) { WITH 19 AS x SET a.x = x } RETURN count(*) AS people",
            "MATCH (p:Person {x: 19}) RETURN count(p)",
        ),
    ] {
        let db = entity_kinds();
        let session = db.session();
        let people = execute(&session, language, query)
            .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"));
        assert_eq!(people, [vec![i(5)]], "{language:?} `{query}`");
        let written = execute(&session, Gql, check).unwrap();
        assert_eq!(written, [vec![i(5)]], "{language:?} `{query}`: {check}");
    }
}

/// A WITH that binds the import's name to another value: from there on the
/// name is a variable of the body with that value, which a later WITH that
/// leaves it out ends the scope of (the import does not come back).
#[test]
fn a_with_that_rebinds_an_import_name_shadows_it() {
    assert_rows(
        &[
            (
                BOTH,
                "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 3 AS a RETURN a AS v } RETURN v",
            ),
            (
                BOTH,
                "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 3 AS a WITH a, 1 AS x \
                 RETURN a AS v } RETURN v",
            ),
        ],
        &[vec![i(3)]],
    );
    for query in [
        "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 3 AS a WITH 1 AS x RETURN a AS v } \
         RETURN v",
        "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 3 AS a WITH count(*) AS x \
         RETURN a AS v } RETURN v",
    ] {
        for language in BOTH {
            let message = error(*language, query);
            assert!(
                message.contains("Undefined variable 'a'"),
                "{language:?} `{query}`: {message}"
            );
        }
    }
}
