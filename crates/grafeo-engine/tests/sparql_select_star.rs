//! `SELECT *` gives the query's own variables (SPARQL 1.1 sec 18.2.1, its in-scope variables),
//! never the names the translation adds: the middle node of a sequence path, the predicate of a
//! negated property set, a blank node. Each case is checked against the same query with its
//! variables written out.

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

const PREFIX: &str = "PREFIX ex: <http://example.org/> ";

/// Alix reaches Amsterdam over two middle nodes, Gus over one.
fn store() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&format!(
        "{PREFIX}INSERT DATA {{
            ex:alix ex:knows ex:vincent . ex:vincent ex:lives ex:amsterdam .
            ex:alix ex:knows ex:mia . ex:mia ex:lives ex:amsterdam .
            ex:gus ex:knows ex:jules . ex:jules ex:lives ex:paris .
            ex:alix ex:likes ex:gus .
        }}"
    ))
    .unwrap();
    db
}

fn run(db: &GrafeoDB, query: &str) -> QueryResult {
    db.execute_sparql(&format!("{PREFIX}{query}"))
        .unwrap_or_else(|error| panic!("{query}: {error}"))
}

/// The columns and the sorted rows of `query`.
fn answer(db: &GrafeoDB, query: &str) -> (Vec<String>, Vec<String>) {
    let result = run(db, query);
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    (result.columns.clone(), rows)
}

/// `SELECT *` over `body` against `SELECT <variables>` over it, listed in the order `*` gives
/// them (SPARQL leaves that order open): the same variables and rows.
#[track_caller]
fn same_as_written_out(db: &GrafeoDB, modifier: &str, body: &str, variables: &[&str]) {
    let query = format!("SELECT {modifier} * WHERE {{ {body} }}");
    let star = answer(db, &query);
    let mut columns = star.0.clone();
    columns.sort();
    let mut expected: Vec<&str> = variables.to_vec();
    expected.sort_unstable();
    assert_eq!(columns, expected, "{query}");
    let listed = star
        .0
        .iter()
        .map(|name| format!("?{name}"))
        .collect::<Vec<_>>()
        .join(" ");
    let written = answer(
        db,
        &format!("SELECT {modifier} {listed} WHERE {{ {body} }}"),
    );
    assert_eq!(star, written, "{query}");
}

#[test]
fn a_sequence_path_gives_its_ends_alone() {
    let db = store();
    same_as_written_out(&db, "", "ex:alix ex:knows/ex:lives ?x", &["x"]);
    same_as_written_out(&db, "", "?s ex:knows/ex:lives ?x", &["s", "x"]);
    // An inverse sequence (W3C pp09).
    same_as_written_out(&db, "", "?x ^ex:lives/^ex:knows ex:alix", &["x"]);
    assert_eq!(
        run(&db, "SELECT * WHERE { ex:alix ex:knows/ex:lives ?x }")
            .rows()
            .len(),
        2,
        "one solution per middle node"
    );
}

#[test]
fn distinct_reads_the_query_variables_only() {
    let db = store();
    // Two middle nodes reach Amsterdam: one distinct solution, not one per middle node.
    same_as_written_out(&db, "DISTINCT", "ex:alix ex:knows/ex:lives ?x", &["x"]);
    assert_eq!(
        run(
            &db,
            "SELECT DISTINCT * WHERE { ex:alix ex:knows/ex:lives ?x }"
        )
        .rows()
        .len(),
        1
    );
}

#[test]
fn a_negated_property_set_gives_no_predicate() {
    let db = store();
    same_as_written_out(&db, "", "ex:alix !(ex:knows) ?x", &["x"]);
    same_as_written_out(&db, "", "ex:alix !(ex:knows|^ex:likes) ?x", &["x"]);
}

#[test]
fn a_blank_node_is_no_variable() {
    let db = store();
    same_as_written_out(&db, "", "?s ex:knows [ ex:lives ?city ]", &["s", "city"]);
    same_as_written_out(
        &db,
        "",
        "?s ex:knows _:m . _:m ex:lives ?city",
        &["s", "city"],
    );
}

#[test]
fn a_pattern_of_hidden_names_alone_gives_its_rows_with_no_column() {
    let db = store();
    let result = run(&db, "SELECT * WHERE { [] ex:lives [] }");
    assert!(result.columns.is_empty(), "{:?}", result.columns);
    assert_eq!(result.rows().len(), 3);
    // One solution with no variable, however many triples match.
    let result = run(&db, "SELECT DISTINCT * WHERE { [] ex:lives [] }");
    assert!(result.columns.is_empty(), "{:?}", result.columns);
    assert_eq!(result.rows().len(), 1);
}

#[test]
fn the_other_variables_keep_their_columns() {
    let db = store();
    // No hidden name: the columns as before.
    same_as_written_out(
        &db,
        "",
        "?s ex:knows ?m . ?m ex:lives ?city",
        &["s", "m", "city"],
    );
    // A path beside an ORDER BY and an OPTIONAL.
    let ordered = answer(
        &db,
        "SELECT * WHERE { ?s ex:knows/ex:lives ?x OPTIONAL { ?s ex:likes ?l } } ORDER BY ?x",
    );
    assert_eq!(ordered.0, ["s", "x", "l"]);
    assert_eq!(ordered.1.len(), 3);
}

#[test]
fn a_subquery_gives_its_own_variables() {
    let db = store();
    let (columns, rows) = answer(
        &db,
        "SELECT * WHERE { { SELECT * WHERE { ex:alix ex:knows/ex:lives ?x } } }",
    );
    assert_eq!(columns, ["x"]);
    assert_eq!(rows.len(), 2);
}
