//! A nested `GRAPH ?g` on a join path joins within each graph (0.6.2): the join pairs a triple
//! only with triples of its own graph, so no operator produces more rows than the answer has.
//! After EG11 put the scope's graph check above the join, the join paired every subject across
//! all graphs and the check dropped the cross-graph pairs: graphs x graphs rows, 47x slower at
//! 100 graphs. The test counts the rows each operator produces through PROFILE, so it fails on the
//! shape at once, whatever a machine's timing; the rows themselves are checked too.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_graph_join_rows::
//! ```

#![cfg(feature = "sparql")]

use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// `graphs` named graphs, each with `subjects` subjects holding one `<urn:p>` and one `<urn:q>`.
fn named_graphs(graphs: usize, subjects: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for g in 0..graphs {
        let triples: Vec<String> = (0..subjects)
            .map(|s| format!("<urn:s{s}> <urn:p> {s} . <urn:s{s}> <urn:q> <urn:v{g}> ."))
            .collect();
        session
            .execute_sparql(&format!(
                "INSERT DATA {{ GRAPH <urn:g{g}> {{ {} }} }}",
                triples.join(" ")
            ))
            .unwrap();
    }
    db
}

/// The rows each operator of `query`'s PROFILE produced.
fn operator_rows(session: &Session, query: &str) -> Vec<(String, u64)> {
    let result = session
        .execute_sparql(&format!("PROFILE {query}"))
        .unwrap_or_else(|error| panic!("PROFILE {query}: {error}"));
    let text = result.rows()[0][0].as_str().unwrap().to_string();
    text.lines()
        .filter_map(|line| {
            let (operator, rest) = line.split_once("  rows=")?;
            let rows: String = rest.chars().take_while(char::is_ascii_digit).collect();
            Some((operator.trim().to_string(), rows.parse().ok()?))
        })
        .collect()
}

#[test]
fn a_nested_graph_scope_joins_within_each_graph() {
    for (graphs, subjects) in [(3, 1), (10, 20), (40, 5)] {
        let db = named_graphs(graphs, subjects);
        let session = db.session();
        for query in [
            "SELECT ?g ?s ?x WHERE { GRAPH ?g { ?s <urn:p> ?o . GRAPH ?g { ?s <urn:q> ?x } } }",
            "SELECT ?g ?s ?x WHERE { GRAPH ?g { ?s <urn:p> ?o . ?s <urn:q> ?x } }",
        ] {
            let answer = session.execute_sparql(query).unwrap().rows().len();
            assert_eq!(answer, graphs * subjects, "{graphs} graphs: `{query}`");
            let operators = operator_rows(&session, query);
            assert!(!operators.is_empty(), "no operator rows read from PROFILE");
            for (operator, rows) in &operators {
                assert!(
                    *rows <= answer as u64,
                    "{graphs} graphs: `{operator}` produced {rows} rows for an answer of {answer} \
                     (a join across graphs)\n{operators:?}"
                );
            }
        }
    }
}

/// The answer pairs each subject with its own graph's value: no cross-graph pair survives.
#[test]
fn every_pair_comes_from_one_graph() {
    let db = named_graphs(5, 4);
    let session = db.session();
    let result = session
        .execute_sparql(
            "SELECT ?g ?x WHERE { GRAPH ?g { ?s <urn:p> ?o . GRAPH ?g { ?s <urn:q> ?x } } }",
        )
        .unwrap();
    for row in result.rows() {
        let (graph, value) = (format!("{:?}", row[0]), format!("{:?}", row[1]));
        let index = graph
            .split("urn:g")
            .nth(1)
            .unwrap()
            .trim_end_matches(|c: char| !c.is_ascii_digit());
        assert!(
            value.contains(&format!("urn:v{index}")),
            "{graph} with {value}"
        );
    }
}
