//! Grouping keys come back as their values beside any aggregate (#589).
//!
//! `GROUP BY` with an aggregate whose argument is not a grouping key (a
//! property, an expression, a `CASE`) returned `0` as the keys and merged the
//! groups: Microsoft Fabric's documented multi-column grouping example gave
//! two rows `[0, 0, ...]` and `[0, null, ...]` for five groups, and SQL/PGQ's
//! `SUM(CASE ...)` one row. Such an argument is computed by a projection below
//! the aggregate, and that projection copied the columns it passed through
//! (the grouping keys) as node IDs, which turned every string into node 0.
//! Without that aggregate (`count(*)` alone) the groups were right.
//!
//! Fabric's example is quoted from
//! <https://learn.microsoft.com/en-us/fabric/graph/gql-language-guide>
//! (MIT, Copyright (c) Microsoft Corporation).
//!
//! These tests run the issue's queries, Fabric's example and the same groups
//! in GQL, Cypher and SQL/PGQ, check the value types of the keys, and check
//! that adding an aggregate over a non-key never changes the keys or the
//! groups, for keys of every type and over many rows. The spec test
//! `tests/spec/rosetta/group_keys_beside_aggregates.gtest` runs the same
//! queries as text in every binding.
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --all-features --test group_by_keys_beside_aggregates
//! ```

use grafeo_common::types::{Date, Value};
use grafeo_engine::GrafeoDB;

/// Six people in the shape of Fabric's social network sample: Alix (female,
/// Chrome), Gus (male, Firefox), Vincent (male, Chrome), Mia (female,
/// Firefox), Jules (male, Chrome) and Butch (male, no browser), so the
/// (gender, browser) groups are five, one of them with a null key.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (:Person {name: 'Alix', gender: 'female', browserUsed: 'Chrome', birthday: 1988, creationDate: 2019, id: 3}), \
                    (:Person {name: 'Gus', gender: 'male', browserUsed: 'Firefox', birthday: 1919, creationDate: 2003, id: 19}), \
                    (:Person {name: 'Vincent', gender: 'male', browserUsed: 'Chrome', birthday: 1983, creationDate: 2088, id: 88}), \
                    (:Person {name: 'Mia', gender: 'female', browserUsed: 'Firefox', birthday: 1993, creationDate: 2033, id: 33}), \
                    (:Person {name: 'Jules', gender: 'male', browserUsed: 'Chrome', birthday: 1988, creationDate: 2038, id: 38}), \
                    (:Person {name: 'Butch', gender: 'male', birthday: 1938, creationDate: 2008, id: 8})",
        )
        .unwrap();
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.session()
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
        .rows()
        .to_vec()
}

/// The rows sorted by their text, so they compare whatever the group order.
fn sorted(mut rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

fn text(value: &str) -> Value {
    Value::String(value.into())
}

/// One expected row of (gender, browser, count, average birthday, first
/// creation date, highest id).
fn group(
    gender: &str,
    browser: Option<&str>,
    count: i64,
    average: f64,
    first: i64,
    highest: i64,
) -> Vec<Value> {
    vec![
        text(gender),
        browser.map_or(Value::Null, text),
        Value::Int64(count),
        Value::Float64(average),
        Value::Int64(first),
        Value::Int64(highest),
    ]
}

/// The five (gender, browser) groups of [`people`], in the order of Fabric's
/// example: the average birthday descending.
fn five_groups() -> Vec<Vec<Value>> {
    vec![
        group("female", Some("Firefox"), 1, 1993.0, 2033, 33),
        group("female", Some("Chrome"), 1, 1988.0, 2019, 3),
        group("male", Some("Chrome"), 2, 1985.5, 2038, 88),
        group("male", None, 1, 1938.0, 2008, 8),
        group("male", Some("Firefox"), 1, 1919.0, 2003, 19),
    ]
}

/// The same (gender, browser) groups with the same aggregates in every
/// language this build has: GQL's `LET` and `GROUP BY` (the issue's form),
/// GQL's grouping by property, Cypher's implicit grouping (also after a
/// `WITH` that names the keys), and SQL/PGQ's `GROUP BY` over `COLUMNS`.
fn grouped_queries() -> Vec<(&'static str, &'static str)> {
    let mut queries = vec![
        (
            "gql",
            "MATCH (p:Person) LET gender = p.gender LET browser = p.browserUsed \
             RETURN gender, browser, count(*) AS n, avg(p.birthday) AS a, \
             min(p.creationDate) AS f, max(p.id) AS h GROUP BY gender, browser",
        ),
        (
            "gql",
            "MATCH (p:Person) LET gender = p.gender, browser = p.browserUsed \
             RETURN gender, browser, count(*) AS n, avg(p.birthday) AS a, \
             min(p.creationDate) AS f, max(p.id) AS h GROUP BY gender, browser",
        ),
        (
            "gql",
            "MATCH (p:Person) RETURN p.gender AS gender, p.browserUsed AS browser, \
             count(*) AS n, avg(p.birthday) AS a, min(p.creationDate) AS f, max(p.id) AS h \
             GROUP BY p.gender, p.browserUsed",
        ),
    ];
    if cfg!(feature = "cypher") {
        queries.extend([
            (
                "cypher",
                "MATCH (p:Person) RETURN p.gender AS gender, p.browserUsed AS browser, \
                 count(*) AS n, avg(p.birthday) AS a, min(p.creationDate) AS f, max(p.id) AS h",
            ),
            (
                "cypher",
                "MATCH (p:Person) WITH p, p.gender AS gender, p.browserUsed AS browser \
                 RETURN gender, browser, count(*) AS n, avg(p.birthday) AS a, \
                 min(p.creationDate) AS f, max(p.id) AS h",
            ),
        ]);
    }
    if cfg!(feature = "sql-pgq") {
        queries.push((
            "sql",
            "SELECT gender, browser, COUNT(*) AS n, AVG(birthday) AS a, MIN(created) AS f, \
             MAX(id) AS h FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.gender AS gender, \
             p.browserUsed AS browser, p.birthday AS birthday, p.creationDate AS created, \
             p.id AS id)) GROUP BY gender, browser",
        ));
    }
    queries
}

// ============================================================================
// The issue's queries
// ============================================================================

/// The issue's query: an average over a property beside two `LET` keys gives
/// the five groups with their keys (it gave `[[0, 0, 3, ...], [0, null, 1,
/// ...]]`), the null browser one of them.
#[test]
fn an_average_over_a_property_keeps_the_keys_and_the_groups() {
    let db = people();
    let result = db
        .session()
        .execute(
            "MATCH (p:Person) LET gender = p.gender LET browser = p.browserUsed \
             RETURN gender, browser, count(*) AS n, avg(p.birthday) AS a GROUP BY gender, browser",
        )
        .unwrap();
    assert_eq!(result.columns, ["gender", "browser", "n", "a"]);
    let expected: Vec<Vec<Value>> = five_groups()
        .into_iter()
        .map(|row| row[..4].to_vec())
        .collect();
    assert_eq!(sorted(result.rows().to_vec()), sorted(expected));
}

/// `min` and `max` over properties broke the keys the same way as `avg`.
#[test]
fn min_and_max_over_properties_keep_the_keys_and_the_groups() {
    let db = people();
    for (aggregate, column) in [("min(p.creationDate)", 4), ("max(p.id)", 5)] {
        let query = format!(
            "MATCH (p:Person) LET gender = p.gender LET browser = p.browserUsed \
             RETURN gender, browser, count(*) AS n, {aggregate} AS x GROUP BY gender, browser"
        );
        let expected: Vec<Vec<Value>> = five_groups()
            .into_iter()
            .map(|row| {
                vec![
                    row[0].clone(),
                    row[1].clone(),
                    row[2].clone(),
                    row[column].clone(),
                ]
            })
            .collect();
        assert_eq!(
            sorted(rows(&db, "gql", &query)),
            sorted(expected),
            "{aggregate}"
        );
    }
}

/// Microsoft Fabric's "multi-column grouping" example, word for word, gives
/// the five groups in the order of their average birthday.
#[test]
fn fabrics_multi_column_grouping_example_gives_every_group_in_order() {
    let db = people();
    let result = rows(
        &db,
        "gql",
        "MATCH (p:Person)
         LET gender = p.gender
         LET browser = p.browserUsed
         RETURN gender,
                browser,
                count(*) AS person_count,
                avg(p.birthday) AS avg_birth_year,
                min(p.creationDate) AS first_joined,
                max(p.id) AS highest_id
         GROUP BY gender, browser
         ORDER BY avg_birth_year DESC
         LIMIT 10",
    );
    assert_eq!(result, five_groups());
}

/// GQL, Cypher and SQL/PGQ give the same five groups, with string keys, a
/// null key, an integer count and minimum, and a float average.
#[test]
fn the_groups_are_the_same_in_every_language() {
    let db = people();
    for (language, query) in grouped_queries() {
        assert_eq!(
            sorted(rows(&db, language, query)),
            sorted(five_groups()),
            "{language}: {query}"
        );
    }
}

/// An aggregate over an expression (`sum(p.id + 1)`) keeps the keys.
#[test]
fn an_aggregate_over_an_expression_keeps_the_keys() {
    let db = people();
    let mut queries = vec![(
        "gql",
        "MATCH (p:Person) LET gender = p.gender LET browser = p.browserUsed \
         RETURN gender, browser, sum(p.id + 1) AS s GROUP BY gender, browser",
    )];
    if cfg!(feature = "cypher") {
        queries.extend([
            (
                "cypher",
                "MATCH (p:Person) RETURN p.gender AS gender, p.browserUsed AS browser, \
                 sum(p.id + 1) AS s",
            ),
            (
                "cypher",
                "MATCH (p:Person) WITH p, p.gender AS gender, p.browserUsed AS browser \
                 RETURN gender, browser, sum(p.id + 1) AS s",
            ),
        ]);
    }
    if cfg!(feature = "sql-pgq") {
        queries.push((
            "sql",
            "SELECT gender, browser, SUM(id + 1) AS s FROM GRAPH_TABLE (MATCH (p:Person) \
             COLUMNS (p.gender AS gender, p.browserUsed AS browser, p.id AS id)) \
             GROUP BY gender, browser",
        ));
    }
    let expected = vec![
        vec![text("female"), text("Chrome"), Value::Int64(4)],
        vec![text("female"), text("Firefox"), Value::Int64(34)],
        vec![text("male"), text("Chrome"), Value::Int64(128)],
        vec![text("male"), text("Firefox"), Value::Int64(20)],
        vec![text("male"), Value::Null, Value::Int64(9)],
    ];
    for (language, query) in queries {
        assert_eq!(
            sorted(rows(&db, language, query)),
            sorted(expected.clone()),
            "{language}: {query}"
        );
    }
}

/// `SUM(CASE ...)` and `COUNT(DISTINCT CASE ...)` keep the keys: the issue's
/// SQL/PGQ form merged every group into one row with the key `0`. Two
/// genders; the women use Chrome once and two browsers since 1950, the men
/// Chrome twice and one browser since 1950.
#[test]
fn aggregates_over_a_case_keep_the_keys() {
    let db = people();
    let mut queries = vec![(
        "gql",
        "MATCH (p:Person) LET gender = p.gender RETURN gender, count(*) AS n, \
         sum(CASE WHEN p.browserUsed = 'Chrome' THEN 1 ELSE 0 END) AS chrome, \
         count(DISTINCT CASE WHEN p.birthday > 1950 THEN p.browserUsed END) AS recent \
         GROUP BY gender",
    )];
    if cfg!(feature = "cypher") {
        queries.extend([
            (
                "cypher",
                "MATCH (p:Person) RETURN p.gender AS gender, count(*) AS n, \
                 sum(CASE WHEN p.browserUsed = 'Chrome' THEN 1 ELSE 0 END) AS chrome, \
                 count(DISTINCT CASE WHEN p.birthday > 1950 THEN p.browserUsed END) AS recent",
            ),
            (
                "cypher",
                "MATCH (p:Person) WITH p, p.gender AS gender RETURN gender, count(*) AS n, \
                 sum(CASE WHEN p.browserUsed = 'Chrome' THEN 1 ELSE 0 END) AS chrome, \
                 count(DISTINCT CASE WHEN p.birthday > 1950 THEN p.browserUsed END) AS recent",
            ),
        ]);
    }
    if cfg!(feature = "sql-pgq") {
        queries.push((
            "sql",
            "SELECT gender, COUNT(*) AS n, \
             SUM(CASE WHEN browser = 'Chrome' THEN 1 ELSE 0 END) AS chrome, \
             COUNT(DISTINCT CASE WHEN birthday > 1950 THEN browser END) AS recent \
             FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.gender AS gender, \
             p.browserUsed AS browser, p.birthday AS birthday)) GROUP BY gender",
        ));
    }
    let expected = vec![
        vec![
            text("female"),
            Value::Int64(2),
            Value::Int64(1),
            Value::Int64(2),
        ],
        vec![
            text("male"),
            Value::Int64(4),
            Value::Int64(2),
            Value::Int64(1),
        ],
    ];
    for (language, query) in queries {
        assert_eq!(
            sorted(rows(&db, language, query)),
            sorted(expected.clone()),
            "{language}: {query}"
        );
    }
}

// ============================================================================
// An aggregate over a non-key never changes the groups
// ============================================================================

/// The aggregates added to a grouping: over a property, an expression and a
/// `CASE`, each computed by a projection below the aggregate.
const MORE_AGGREGATES: &str = "avg(p.birthday) AS a, min(p.creationDate) AS f, \
     sum(p.id + 1) AS s, sum(CASE WHEN p.browserUsed = 'Chrome' THEN 1 ELSE 0 END) AS c";

/// A key of each type, as `(expression, groups, a key it gives)`: a boolean,
/// an integer, a float, a string, a null among strings, a date, a list, a map
/// and the node itself.
fn keys() -> Vec<(&'static str, usize, Option<Value>)> {
    vec![
        ("p.birthday > 1950", 2, Some(Value::Bool(true))),
        ("p.id % 2", 2, Some(Value::Int64(1))),
        ("p.id % 2 + 0.5", 2, Some(Value::Float64(1.5))),
        ("p.gender", 2, Some(text("female"))),
        ("p.browserUsed", 3, Some(Value::Null)),
        (
            "date({year: 2020, month: 1, day: p.id % 2 + 1})",
            2,
            Some(Value::Date(Date::from_ymd(2020, 1, 2).unwrap())),
        ),
        (
            "[p.gender, p.browserUsed]",
            5,
            Some(Value::List(vec![text("male"), Value::Null].into())),
        ),
        ("{g: p.gender}", 2, None),
        ("p", 6, None),
    ]
}

/// The first `width` columns of every row, sorted.
fn leading(rows: Vec<Vec<Value>>, width: usize) -> Vec<Vec<Value>> {
    sorted(rows.into_iter().map(|row| row[..width].to_vec()).collect())
}

/// For a key of every type, the keys and counts of a grouping are the same
/// with aggregates over a property, an expression and a `CASE` beside them
/// as with `count(*)` alone (which was always right), in GQL and Cypher.
#[test]
fn more_aggregates_never_change_the_keys_or_the_groups() {
    let db = people();
    for (key, groups, value) in keys() {
        let mut pairs = vec![(
            "gql",
            format!("MATCH (p:Person) LET k = {key} RETURN k, count(*) AS n GROUP BY k"),
            format!(
                "MATCH (p:Person) LET k = {key} RETURN k, count(*) AS n, {MORE_AGGREGATES} \
                 GROUP BY k"
            ),
        )];
        if cfg!(feature = "cypher") {
            pairs.push((
                "cypher",
                format!("MATCH (p:Person) WITH p, {key} AS k RETURN k, count(*) AS n"),
                format!(
                    "MATCH (p:Person) WITH p, {key} AS k RETURN k, count(*) AS n, \
                     {MORE_AGGREGATES}"
                ),
            ));
        }
        for (language, alone, beside) in pairs {
            let expected = leading(rows(&db, language, &alone), 2);
            assert_eq!(expected.len(), groups, "{language}: {alone}");
            if let Some(value) = &value {
                assert!(
                    expected.iter().any(|row| &row[0] == value),
                    "{language}: {alone}: no key {value:?} in {expected:?}"
                );
            }
            assert_eq!(
                leading(rows(&db, language, &beside), 2),
                expected,
                "{language}: {beside}"
            );
        }
    }
}

/// The same in SQL/PGQ, for keys a `COLUMNS` item computes.
#[cfg(feature = "sql-pgq")]
#[test]
fn more_sql_pgq_aggregates_never_change_the_keys_or_the_groups() {
    let db = people();
    for (key, groups, value) in [
        ("p.birthday > 1950", 2, Value::Bool(true)),
        ("p.id % 2", 2, Value::Int64(1)),
        ("p.gender", 2, text("female")),
        ("p.browserUsed", 3, Value::Null),
    ] {
        let table = format!(
            "GRAPH_TABLE (MATCH (p:Person) COLUMNS ({key} AS k, p.birthday AS birthday, \
             p.creationDate AS created, p.id AS id, p.browserUsed AS browser))"
        );
        let alone = format!("SELECT k, COUNT(*) AS n FROM {table} GROUP BY k");
        let beside = format!(
            "SELECT k, COUNT(*) AS n, AVG(birthday) AS a, MIN(created) AS f, SUM(id + 1) AS s, \
             SUM(CASE WHEN browser = 'Chrome' THEN 1 ELSE 0 END) AS c FROM {table} GROUP BY k"
        );
        let expected = leading(rows(&db, "sql", &alone), 2);
        assert_eq!(expected.len(), groups, "{alone}");
        assert!(
            expected.iter().any(|row| row[0] == value),
            "{alone}: no key {value:?} in {expected:?}"
        );
        assert_eq!(leading(rows(&db, "sql", &beside), 2), expected, "{beside}");
    }
}

/// Over 5,000 people (several chunks), the six (gender, browser) groups keep
/// their keys, sizes, averages, first creation dates and highest ids in every
/// language.
#[test]
fn many_rows_keep_their_groups() {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "UNWIND range(0, 4999) AS i INSERT (:Person {\
             gender: CASE WHEN i % 2 = 0 THEN 'female' ELSE 'male' END, \
             browserUsed: CASE i % 3 WHEN 0 THEN 'Chrome' WHEN 1 THEN 'Firefox' ELSE 'Safari' END, \
             birthday: i, creationDate: 4999 - i, id: i})",
        )
        .unwrap();
    // Person i is in the group of i % 6: the ids i, i + 6, ... below 5,000.
    let expected: Vec<Vec<Value>> = (0..6_i32)
        .map(|residue| {
            let gender = if residue % 2 == 0 { "female" } else { "male" };
            let browser = ["Chrome", "Firefox", "Safari"][usize::try_from(residue % 3).unwrap()];
            let count = (4999 - residue) / 6 + 1;
            let highest = residue + 6 * (count - 1);
            group(
                gender,
                Some(browser),
                i64::from(count),
                f64::from(residue + highest) / 2.0,
                i64::from(4999 - highest),
                i64::from(highest),
            )
        })
        .collect();
    for (language, query) in grouped_queries() {
        assert_eq!(
            sorted(rows(&db, language, query)),
            sorted(expected.clone()),
            "{language}: {query}"
        );
    }
}
