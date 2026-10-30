//! A subquery in an AND chain whose other conjuncts are false for a row does
//! not run for that row: the lifted Apply skips it and gives what an empty
//! run gives (0 for COUNT, false for EXISTS, null for VALUE). The rows are
//! those of running it for every row:
//!
//! - COUNT and VALUE in a rejected branch, in WHERE and in RETURN, and the
//!   same subquery outside the chain still gives its real value per row;
//! - a null conjunct does not reject the row: `NOT (n.missing = 1 AND
//!   EXISTS {...})` still runs the subquery;
//! - two subqueries guarded by one chain;
//! - an uncorrelated EXISTS in a guarded OR branch.
//!
//! In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test guarded_subqueries
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Six P nodes, `id` 1 to 6; `flag` true on 1, 2 and 3. R edges: 1 has
/// three (to targets of `x` 10, 20, 30), 2 one (`x` 5), 4 two (`x` 7, 8);
/// S edges: 1 and 4 two each. One Q node with `k: 1`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 6) AS i INSERT (:P {id: i, flag: i <= 3})")
        .unwrap();
    for (id, xs) in [(1, vec![10, 20, 30]), (2, vec![5]), (4, vec![7, 8])] {
        for x in xs {
            db.execute(&format!(
                "MATCH (p:P {{id: {id}}}) INSERT (p)-[:R]->(:T {{x: {x}}})"
            ))
            .unwrap();
        }
    }
    for id in [1, 4] {
        db.execute(&format!(
            "MATCH (p:P {{id: {id}}}) INSERT (p)-[:S]->(:U), (p)-[:S]->(:U)"
        ))
        .unwrap();
    }
    db.execute("INSERT (:Q {k: 1})").unwrap();
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

fn ids(values: &[i64]) -> Vec<Vec<Value>> {
    values.iter().map(|v| vec![Value::Int64(*v)]).collect()
}

/// COUNT in a branch whose flag is false: the rows are those of running it.
#[test]
fn count_in_a_rejected_branch() {
    let db = database();
    for language in LANGUAGES {
        let query = "MATCH (n:P) WHERE (n.flag AND COUNT { MATCH (n)-[:R]->() } >= 1) OR n.id = 6 \
                     RETURN n.id AS id ORDER BY id";
        assert_eq!(rows(&db, language, query), ids(&[1, 2, 6]), "{language}");
        let query = "MATCH (n:P) WHERE NOT (n.flag AND COUNT { MATCH (n)-[:R]->() } = 0) \
                     RETURN n.id AS id ORDER BY id";
        assert_eq!(
            rows(&db, language, query),
            ids(&[1, 2, 4, 5, 6]),
            "{language}"
        );
    }
}

/// A guarded COUNT in RETURN, beside the same COUNT unguarded: the guarded
/// one is false where the flag is, the unguarded one is the real count on
/// every row (the skipped 0 must not stand in for it).
#[test]
fn a_guarded_count_beside_the_same_count_unguarded() {
    let db = database();
    for language in LANGUAGES {
        let query = "MATCH (n:P) RETURN n.id AS id, \
                     (n.flag AND COUNT { MATCH (n)-[:R]->() } > 0) AS many, \
                     COUNT { MATCH (n)-[:R]->() } AS c ORDER BY id";
        let got = rows(&db, language, query);
        let expected: Vec<Vec<Value>> = [
            (1, true, 3),
            (2, true, 1),
            (3, false, 0),
            (4, false, 2),
            (5, false, 0),
            (6, false, 0),
        ]
        .iter()
        .map(|(id, many, c)| vec![Value::Int64(*id), Value::Bool(*many), Value::Int64(*c)])
        .collect();
        assert_eq!(got, expected, "{language}");
    }
}

/// VALUE in a rejected branch (GQL; Cypher has no VALUE subquery): the
/// rows are those of running it, and the same VALUE outside the chain gives
/// the real value on every row.
#[test]
fn value_in_a_rejected_branch() {
    let db = database();
    let language = "gql";
    let value = "VALUE { MATCH (n)-[:R]->(t) RETURN t.x ORDER BY t.x LIMIT 1 }";
    let query = format!(
        "MATCH (n:P) WHERE (n.flag AND {value} = 10) OR n.id = 4 RETURN n.id AS id ORDER BY id"
    );
    assert_eq!(rows(&db, language, &query), ids(&[1, 4]), "{language}");
    let query = format!(
        "MATCH (n:P) RETURN n.id AS id, (n.flag AND {value} > 1) AS a, {value} AS v ORDER BY id"
    );
    let got: Vec<Value> = rows(&db, language, &query)
        .into_iter()
        .map(|row| row[2].clone())
        .collect();
    assert_eq!(
        got,
        [
            Value::Int64(10),
            Value::Int64(5),
            Value::Null,
            Value::Int64(7),
            Value::Null,
            Value::Null
        ],
        "{language}: the unguarded VALUE on every row"
    );
}

/// A null conjunct does not reject the row: `null AND EXISTS` is null when
/// the subquery matches and false when it does not, so NOT of it keeps the
/// rows without a match. The subquery has to run.
#[test]
fn a_null_conjunct_still_runs_the_subquery() {
    let db = database();
    for language in LANGUAGES {
        let query = "MATCH (n:P) WHERE NOT (n.missing = 1 AND EXISTS { MATCH (n)-[:R]->() }) \
                     RETURN n.id AS id ORDER BY id";
        assert_eq!(rows(&db, language, query), ids(&[3, 5, 6]), "{language}");
    }
}

/// Two subqueries guarded by one chain.
#[test]
fn two_subqueries_in_one_guarded_chain() {
    let db = database();
    for language in LANGUAGES {
        let query = "MATCH (n:P) WHERE (n.flag AND EXISTS { MATCH (n)-[:R]->() } \
                     AND COUNT { MATCH (n)-[:S]->() } > 1) OR n.id = 5 \
                     RETURN n.id AS id ORDER BY id";
        assert_eq!(rows(&db, language, query), ids(&[1, 5]), "{language}");
    }
}

/// An uncorrelated EXISTS in a guarded OR branch gives the rows of the same
/// query with the EXISTS's value written in.
#[test]
fn an_uncorrelated_exists_in_a_guarded_branch() {
    let db = database();
    for language in LANGUAGES {
        let query = "MATCH (n:P) WHERE (n.flag AND EXISTS { MATCH (m:Q) WHERE m.k = 1 }) OR n.id = 6 \
                     RETURN n.id AS id ORDER BY id";
        assert_eq!(rows(&db, language, query), ids(&[1, 2, 3, 6]), "{language}");
        let query = "MATCH (n:P) WHERE (n.flag AND EXISTS { MATCH (m:Q) WHERE m.k = 2 }) OR n.id = 6 \
                     RETURN n.id AS id ORDER BY id";
        assert_eq!(rows(&db, language, query), ids(&[6]), "{language}");
    }
}
