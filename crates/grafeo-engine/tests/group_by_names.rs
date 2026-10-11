//! The names a grouping query reads: a GQL `GROUP BY` on an alias of the
//! RETURN list, a GQL RETURN item that is not a grouping key, and in SQL/PGQ
//! the table alias of `GRAPH_TABLE` and the SELECT alias of a computed
//! grouping key.
//!
//! In ISO GQL a grouping element is a name: `<grouping element> ::= <binding
//! variable reference>` (ISO/IEC 39075:2024 <group by clause>). Grafeo also
//! takes an expression over the incoming variables (`GROUP BY c.name`), as
//! Microsoft Fabric does, whose documented example groups by an alias of the
//! RETURN list and an expression at once:
//! <https://learn.microsoft.com/en-us/fabric/graph/gql-expressions>, "Aggregation
//! across rows" (MIT, Copyright (c) Microsoft Corporation).
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --all-features --test group_by_names
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Six people in three cities, in the shape of Fabric's social network
/// sample: Amsterdam (id 3) has Alix (1988), Vincent (1983) and Butch
/// (1939); Berlin (id 19) has Gus (1919) and Jules (1988); Paris (id 88) has
/// Mia (1993). Alix and Mia are female, the others male.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (amsterdam:City {id: 3, name: 'Amsterdam'}), \
                    (berlin:City {id: 19, name: 'Berlin'}), \
                    (paris:City {id: 88, name: 'Paris'}), \
                    (:Person {name: 'Alix', gender: 'female', birthday: 1988})-[:isLocatedIn]->(amsterdam), \
                    (:Person {name: 'Vincent', gender: 'male', birthday: 1983})-[:isLocatedIn]->(amsterdam), \
                    (:Person {name: 'Butch', gender: 'male', birthday: 1939})-[:isLocatedIn]->(amsterdam), \
                    (:Person {name: 'Gus', gender: 'male', birthday: 1919})-[:isLocatedIn]->(berlin), \
                    (:Person {name: 'Jules', gender: 'male', birthday: 1988})-[:isLocatedIn]->(berlin), \
                    (:Person {name: 'Mia', gender: 'female', birthday: 1993})-[:isLocatedIn]->(paris)",
        )
        .unwrap();
    db
}

/// The columns and rows of `query`, the rows sorted by their text so they
/// compare whatever the group order.
fn table(db: &GrafeoDB, language: &str, query: &str) -> (Vec<String>, Vec<Vec<Value>>) {
    let result = db
        .session()
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
    let mut rows = result.rows().to_vec();
    rows.sort_by_key(|row| format!("{row:?}"));
    (result.columns.clone(), rows)
}

/// The error message of `query`, which must fail.
fn error(db: &GrafeoDB, language: &str, query: &str) -> String {
    match db.session().execute_language(query, language, None) {
        Ok(result) => panic!("{language}: {query}: expected an error, got {result:?}"),
        Err(error) => error.to_string(),
    }
}

fn text(value: &str) -> Value {
    Value::String(value.into())
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

// ============================================================================
// GQL: GROUP BY an alias of the RETURN list
// ============================================================================

#[test]
fn group_by_names_an_alias_of_a_computed_return_item() {
    let db = people();
    let (columns, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person) RETURN p.birthday % 2 AS odd, count(*) AS n GROUP BY odd",
    );
    assert_eq!(columns, ["odd", "n"]);
    // Even birthdays: Alix and Jules (1988); odd: the other four.
    assert_eq!(rows, [[int(0), int(2)], [int(1), int(4)]]);
}

#[test]
fn fabrics_example_groups_by_an_alias_and_an_expression() {
    let db = people();
    let (columns, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person)-[:isLocatedIn]->(c:City) \
         RETURN c.id AS cityId, c.name, count(*) AS population, avg(p.birthday) AS average_birth_year \
         GROUP BY cityId, c.name",
    );
    assert_eq!(
        columns,
        ["cityId", "c.name", "population", "average_birth_year"]
    );
    assert_eq!(
        rows,
        [
            vec![int(19), text("Berlin"), int(2), Value::Float64(1953.5)],
            vec![int(3), text("Amsterdam"), int(3), Value::Float64(1970.0)],
            vec![int(88), text("Paris"), int(1), Value::Float64(1993.0)],
        ]
    );
}

#[test]
fn an_alias_grouping_key_is_the_same_group_as_its_expression() {
    // GROUP BY cityId and GROUP BY c.id make the same groups, with an
    // aggregate over a non-key and without one.
    let db = people();
    for returned in ["count(*) AS n", "max(p.birthday) AS n"] {
        let by_alias = table(
            &db,
            "gql",
            &format!(
                "MATCH (p:Person)-[:isLocatedIn]->(c:City) RETURN c.id AS cityId, {returned} GROUP BY cityId"
            ),
        );
        let by_expression = table(
            &db,
            "gql",
            &format!(
                "MATCH (p:Person)-[:isLocatedIn]->(c:City) RETURN c.id AS cityId, {returned} GROUP BY c.id"
            ),
        );
        assert_eq!(by_alias, by_expression, "{returned}");
        assert_eq!(by_alias.1.len(), 3, "{returned}: one row per city");
    }
}

#[test]
fn an_alias_grouping_key_works_with_having_and_order_by() {
    // Grafeo's GQL HAVING follows the whole RETURN statement, its ORDER BY
    // included.
    let db = people();
    let result = db
        .session()
        .execute(
            "MATCH (p:Person)-[:isLocatedIn]->(c:City) \
             RETURN c.name AS city, count(*) AS population \
             GROUP BY city ORDER BY city DESC HAVING count(*) > 1 AND city <> 'Prague'",
        )
        .unwrap();
    assert_eq!(
        result.rows(),
        [[text("Berlin"), int(2)], [text("Amsterdam"), int(3)]]
    );
}

#[test]
fn an_alias_grouping_key_need_not_be_the_first_item() {
    let db = people();
    let (columns, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person) RETURN count(*) AS n, p.gender AS gender GROUP BY gender",
    );
    assert_eq!(columns, ["n", "gender"]);
    assert_eq!(rows, [[int(2), text("female")], [int(4), text("male")]]);
}

#[test]
fn an_alias_of_a_node_groups_by_the_node() {
    let db = people();
    let (_, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person)-[:isLocatedIn]->(c:City) RETURN c AS town, count(*) AS n GROUP BY town",
    );
    let mut counts: Vec<&Value> = rows.iter().map(|row| &row[1]).collect();
    counts.sort_by_key(|value| format!("{value:?}"));
    assert_eq!(counts, [&int(1), &int(2), &int(3)], "{rows:?}");
}

#[test]
fn a_let_variable_still_groups_as_before() {
    // Fabric's how-to guides bind the keys with LET first: the name is then
    // both an incoming variable and the alias of a RETURN item that returns
    // that variable.
    let db = people();
    let (columns, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person)-[:isLocatedIn]->(c:City) LET cityName = c.name \
         RETURN cityName, count(*) AS population GROUP BY cityName",
    );
    assert_eq!(columns, ["cityName", "population"]);
    assert_eq!(
        rows,
        [
            [text("Amsterdam"), int(3)],
            [text("Berlin"), int(2)],
            [text("Paris"), int(1)]
        ]
    );
}

#[test]
fn group_by_an_alias_after_a_procedure_call() {
    let db = people();
    let (columns, rows) = table(
        &db,
        "gql",
        "CALL db.labels() YIELD label RETURN size(label) AS letters, count(*) AS n GROUP BY letters",
    );
    assert_eq!(columns, ["letters", "n"]);
    // City has four letters, Person six.
    assert_eq!(rows, [[int(4), int(1)], [int(6), int(1)]]);
}

#[test]
fn group_by_an_aggregate_alias_says_it_is_an_aggregate() {
    let db = people();
    let message = error(
        &db,
        "gql",
        "MATCH (p:Person) RETURN p.gender AS gender, count(*) AS n GROUP BY gender, n",
    );
    assert!(
        message.contains("'n'") && message.contains("aggregate"),
        "{message}"
    );
}

#[test]
fn an_alias_inside_a_grouping_expression_says_to_name_it_alone() {
    let db = people();
    let message = error(
        &db,
        "gql",
        "MATCH (p:Person) RETURN p.birthday % 2 AS odd, count(*) AS n GROUP BY odd + 1",
    );
    assert!(
        message.contains("'odd'") && message.contains("GROUP BY odd"),
        "{message}"
    );
}

#[test]
fn an_alias_that_shadows_an_incoming_variable_is_ambiguous() {
    // `p` names the incoming node and the alias of `p.gender`: grouping by
    // the node and grouping by the gender make different groups.
    let db = people();
    let message = error(
        &db,
        "gql",
        "MATCH (p:Person) RETURN p.gender AS p, count(*) AS n GROUP BY p",
    );
    assert!(
        message.contains("'p'") && message.contains("ambiguous"),
        "{message}"
    );
}

// ============================================================================
// GQL: a RETURN item of a grouping query that is not a grouping key
// ============================================================================

#[test]
fn a_return_item_of_another_variable_than_the_keys_says_why() {
    let db = people();
    let message = error(
        &db,
        "gql",
        "MATCH (p:Person)-[:isLocatedIn]->(c:City) RETURN p.name AS name, count(*) AS n GROUP BY c",
    );
    assert!(
        message.contains("p.name")
            && message.contains("GROUP BY")
            && !message.contains("Undefined"),
        "{message}"
    );
}

#[test]
fn a_constant_return_item_is_returned_for_every_group() {
    // A constant has one value per group, whatever the keys (it failed with
    // "Undefined variable '3'").
    let db = people();
    let (columns, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person) RETURN 3 AS three, p.gender AS gender, count(*) AS n GROUP BY gender",
    );
    assert_eq!(columns, ["three", "gender", "n"]);
    assert_eq!(
        rows,
        [
            [int(3), text("female"), int(2)],
            [int(3), text("male"), int(4)]
        ]
    );
}

#[test]
fn a_property_of_a_grouped_node_is_returned_per_group() {
    // A property of a grouped node has one value per group, as in Neo4j's
    // Cypher 25 GROUP BY and SQL's functionally dependent columns (it failed
    // with "Undefined variable 'c.name'").
    let db = people();
    let expected = [
        [text("Amsterdam"), int(3)],
        [text("Berlin"), int(2)],
        [text("Paris"), int(1)],
    ];
    for query in [
        "MATCH (p:Person)-[:isLocatedIn]->(c:City) RETURN c.name AS city, count(*) AS n GROUP BY c",
        "MATCH (p:Person)-[:isLocatedIn]->(c:City) LET town = c \
         RETURN town.name AS city, count(*) AS n GROUP BY town",
        "MATCH (p:Person)-[:isLocatedIn]->(c:City) RETURN c AS town, c.name AS city, \
         count(*) AS n GROUP BY town",
    ] {
        let (columns, rows) = table(&db, "gql", query);
        let mut rows: Vec<Vec<Value>> = rows
            .into_iter()
            .map(|row| row[row.len() - 2..].to_vec())
            .collect();
        rows.sort_by_key(|row| format!("{row:?}"));
        assert_eq!(columns[columns.len() - 2..], ["city", "n"], "{query}");
        assert_eq!(rows, expected, "{query}");
    }
    // One row per person: the names are the groups' names.
    let (_, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person) RETURN p.name AS name, count(*) AS n GROUP BY p",
    );
    let names: Vec<&Value> = rows.iter().map(|row| &row[0]).collect();
    assert_eq!(
        names,
        ["Alix", "Butch", "Gus", "Jules", "Mia", "Vincent"]
            .map(text)
            .iter()
            .collect::<Vec<_>>()
    );
    assert!(rows.iter().all(|row| row[1] == int(1)), "{rows:?}");
}

#[test]
fn an_expression_over_the_keys_is_returned_per_group() {
    let db = people();
    let (columns, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person)-[:isLocatedIn]->(c:City) \
         RETURN upper(c.name) AS town, c.name, count(*) * 100 + c.id AS code GROUP BY c",
    );
    // An unaliased item is named after its text.
    assert_eq!(columns, ["town", "c.name", "code"]);
    assert_eq!(
        rows,
        [
            [text("AMSTERDAM"), text("Amsterdam"), int(303)],
            [text("BERLIN"), text("Berlin"), int(219)],
            [text("PARIS"), text("Paris"), int(188)]
        ]
    );
    let (_, rows) = table(
        &db,
        "gql",
        "MATCH (p:Person) RETURN upper(p.gender) AS g, count(*) AS n GROUP BY p.gender",
    );
    assert_eq!(rows, [[text("FEMALE"), int(2)], [text("MALE"), int(4)]]);
}

#[test]
fn an_item_over_a_grouped_node_works_with_order_by_and_having() {
    let db = people();
    let result = db
        .session()
        .execute(
            "MATCH (p:Person)-[:isLocatedIn]->(c:City) \
             RETURN c.name AS city, count(*) AS n GROUP BY c \
             ORDER BY city DESC HAVING city <> 'Berlin'",
        )
        .unwrap();
    assert_eq!(
        result.rows(),
        [[text("Paris"), int(1)], [text("Amsterdam"), int(3)]]
    );
}

#[test]
fn an_aggregate_beside_a_non_key_says_why() {
    // `p.birthday` has a value per person, not per gender: the item was null
    // in every group.
    let db = people();
    let message = error(
        &db,
        "gql",
        "MATCH (p:Person) RETURN p.gender AS gender, count(*) + p.birthday AS n GROUP BY gender",
    );
    assert!(
        message.contains("reads p") && message.contains("GROUP BY"),
        "{message}"
    );
}

// ============================================================================
// SQL/PGQ: the GRAPH_TABLE alias and the SELECT alias of a computed key
// ============================================================================
//
// A column of GRAPH_TABLE (... COLUMNS (x AS c)) AS g is `c` and `g.c` alike
// (ISO/IEC 9075-2 <column reference>; ISO/IEC 9075-16 <graph table>), in the
// select list, in GROUP BY, in HAVING, in ORDER BY and in an aggregate's
// argument.

#[cfg(feature = "sql-pgq")]
mod sql_pgq {
    use super::*;

    const GENDERS: &str = "FROM GRAPH_TABLE (MATCH (p:Person) \
         COLUMNS (p.gender AS gender, p.birthday AS birthday)) AS g";

    #[test]
    fn a_qualified_column_groups_like_its_name() {
        let db = people();
        for (select, group_by) in [
            ("g.gender", "g.gender"),
            ("g.gender", "gender"),
            ("gender", "g.gender"),
            ("gender", "gender"),
        ] {
            let query = format!("SELECT {select}, COUNT(*) AS m {GENDERS} GROUP BY {group_by}");
            let (columns, rows) = table(&db, "sql", &query);
            assert_eq!(columns, ["gender", "m"], "{query}");
            assert_eq!(
                rows,
                [[text("female"), int(2)], [text("male"), int(4)]],
                "{query}"
            );
        }
    }

    #[test]
    fn a_qualified_column_is_an_aggregate_argument() {
        let db = people();
        let query = format!(
            "SELECT g.gender, AVG(g.birthday) AS average, MIN(birthday) AS first \
             {GENDERS} GROUP BY g.gender"
        );
        let (columns, rows) = table(&db, "sql", &query);
        assert_eq!(columns, ["gender", "average", "first"]);
        assert_eq!(
            rows,
            [
                vec![text("female"), Value::Float64(1990.5), int(1988)],
                vec![text("male"), Value::Float64(1957.25), int(1919)],
            ]
        );
    }

    #[test]
    fn a_qualified_column_works_in_having_and_order_by() {
        let db = people();
        let query = format!(
            "SELECT g.gender, COUNT(*) AS m {GENDERS} GROUP BY g.gender \
             HAVING g.gender <> 'nobody' AND COUNT(*) > 1 ORDER BY g.gender DESC"
        );
        let result = db.session().execute_sql(&query).unwrap();
        assert_eq!(
            result.rows(),
            [[text("male"), int(4)], [text("female"), int(2)]]
        );
    }

    #[test]
    fn having_reads_the_grouping_key_of_a_column() {
        // HAVING read the column as the graph property it came from, which no
        // longer exists after the grouping: every group was left out.
        let db = people();
        for having in ["gender = 'male'", "g.gender = 'male'"] {
            let query =
                format!("SELECT gender, COUNT(*) AS n {GENDERS} GROUP BY gender HAVING {having}");
            let (_, rows) = table(&db, "sql", &query);
            assert_eq!(rows, [[text("male"), int(4)]], "{query}");
        }
    }

    #[test]
    fn unaliased_items_of_a_grouping_query_are_named_after_their_text() {
        // They failed with "Undefined variable 'result'".
        let db = people();
        let (columns, rows) = table(&db, "sql", &format!("SELECT COUNT(*) {GENDERS}"));
        assert_eq!(columns, ["count(*)"]);
        assert_eq!(rows, [[int(6)]]);
        let (columns, rows) = table(
            &db,
            "sql",
            &format!("SELECT UPPER(gender), MAX(g.birthday) {GENDERS} GROUP BY gender"),
        );
        assert_eq!(columns, ["UPPER(gender)", "max(birthday)"]);
        assert_eq!(
            rows,
            [[text("FEMALE"), int(1993)], [text("MALE"), int(1988)]]
        );
    }

    #[test]
    fn the_select_alias_of_a_computed_key_names_its_column() {
        let db = people();
        let query = format!("SELECT UPPER(gender) AS g, COUNT(*) AS n {GENDERS} GROUP BY gender");
        let (columns, rows) = table(&db, "sql", &query);
        assert_eq!(columns, ["g", "n"]);
        assert_eq!(rows, [[text("FEMALE"), int(2)], [text("MALE"), int(4)]]);
    }

    #[test]
    fn a_computed_key_of_a_qualified_column_is_grouped() {
        let db = people();
        let query = format!(
            "SELECT g.birthday % 2 AS odd, COUNT(*) AS n {GENDERS} GROUP BY g.birthday % 2 \
             ORDER BY odd"
        );
        let result = db.session().execute_sql(&query).unwrap();
        assert_eq!(result.columns, ["odd", "n"]);
        assert_eq!(result.rows(), [[int(0), int(2)], [int(1), int(4)]]);
    }

    #[test]
    fn an_aggregate_in_columns_is_refused_with_the_place_it_belongs() {
        let db = people();
        for columns in ["COUNT(*) AS n", "p.name AS name, MAX(p.birthday) AS last"] {
            let query =
                format!("SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS ({columns})) AS g");
            let message = error(&db, "sql", &query);
            assert!(
                message.contains("COLUMNS") && message.contains("SELECT"),
                "{query}: {message}"
            );
        }
    }
}
