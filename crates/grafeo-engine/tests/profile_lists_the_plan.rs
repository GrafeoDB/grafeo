//! `PROFILE` lists the operators of the plan the query runs, as `EXPLAIN`
//! shows them: every scan, every expand of a subquery and the filter that
//! holds it, and the top line counts the rows the query returns. On a
//! selector shaped like a code graph's (an OR of labels, `n.active` and three
//! EXISTS subqueries, one negated), PROFILE showed only the expand of the
//! last subquery and a scan EXPLAIN does not have. In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test profile_lists_the_plan
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

/// Each code element has `:Graph` and its kind. Alix and Butch match the
/// selector: Gus has no outgoing USES or CONTAINS, Vincent nothing coming in,
/// Hans is deprecated, Mia is no Method or File.
const SETUP: &str = "INSERT \
    (alix:Graph:Method {name: 'Alix', active: true}), \
    (gus:Graph:Method {name: 'Gus', active: true}), \
    (vincent:Graph:File {name: 'Vincent', active: true}), \
    (butch:Graph:File {name: 'Butch', active: true}), \
    (django:Graph:Package {name: 'Django', active: true}), \
    (hans:Graph:Method {name: 'Hans', active: true}), \
    (jules:Graph:Test {name: 'Jules', active: true}), \
    (mia:Graph:Method {name: 'Mia', active: false}), \
    (alix)-[:USES]->(gus), (vincent)-[:CONTAINS]->(alix), \
    (butch)-[:CONTAINS]->(alix), (django)-[:CONTAINS]->(butch), \
    (hans)-[:USES]->(gus), (jules)-[:COVERS]->(hans), \
    (gus)-[:DEPRECATED_BY]->(mia), (hans)-[:DEPRECATED_BY]->(mia)";

const SELECTOR: &str = "MATCH (n:Graph) WHERE (n:Method OR n:File) AND n.active \
    AND EXISTS { MATCH (n)-[:USES|CONTAINS]->() } \
    AND EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() } \
    AND NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() } \
    RETURN n.name AS name ORDER BY name";

fn session() -> Session {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute(SETUP).expect("setup");
    session
}

fn run(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// The text of `EXPLAIN query` or `PROFILE query`.
fn plan(session: &Session, language: Language, keyword: &str, query: &str) -> String {
    let rows = run(session, language, &format!("{keyword} {query}"));
    let Value::String(text) = &rows[0][0] else {
        panic!("{language:?} `{keyword} {query}` returned {rows:?}");
    };
    text.to_string()
}

/// The operator lines that scan nodes (an operator's arguments may name a
/// scan of a subquery, so only the operator name counts).
fn scan_lines(text: &str) -> usize {
    text.lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("Scan") || line.starts_with("NodeScan"))
        .count()
}

#[test]
fn the_selector_returns_alix_and_butch() {
    for language in LANGUAGES {
        let rows = run(&session(), language, SELECTOR);
        assert_eq!(
            rows,
            vec![vec![Value::from("Alix")], vec![Value::from("Butch")]],
            "{language:?}"
        );
    }
}

/// The top line of PROFILE counts the rows the query returns.
#[test]
fn profile_counts_the_rows_of_the_selector() {
    for language in LANGUAGES {
        let text = plan(&session(), language, "PROFILE", SELECTOR);
        assert!(
            text.lines().next().unwrap_or_default().contains("rows=2"),
            "{language:?} the top line of\n{text}"
        );
    }
}

/// Every edge type a subquery expands shows in PROFILE, as in EXPLAIN.
#[test]
fn profile_shows_every_subquery_expand() {
    for language in LANGUAGES {
        let session = session();
        let explain = plan(&session, language, "EXPLAIN", SELECTOR);
        let profile = plan(&session, language, "PROFILE", SELECTOR);
        for edge_type in ["USES", "CONTAINS", "COVERS", "DEPRECATED_BY"] {
            assert!(
                explain.contains(edge_type),
                "{language:?} EXPLAIN without {edge_type}:\n{explain}"
            );
            assert!(
                profile.contains(edge_type),
                "{language:?} PROFILE without {edge_type}:\n{profile}\nEXPLAIN:\n{explain}"
            );
        }
    }
}

/// PROFILE scans as often as EXPLAIN does: no scan the plan does not have.
#[test]
fn profile_has_the_scans_of_explain() {
    for language in LANGUAGES {
        let session = session();
        let explain = plan(&session, language, "EXPLAIN", SELECTOR);
        let profile = plan(&session, language, "PROFILE", SELECTOR);
        assert_eq!(
            scan_lines(&profile),
            scan_lines(&explain),
            "{language:?} PROFILE:\n{profile}\nEXPLAIN:\n{explain}"
        );
    }
}

/// A single EXISTS: its expand and the filter holding it show in PROFILE.
#[test]
fn profile_shows_a_single_exists() {
    let query = "MATCH (n:Method) WHERE EXISTS { MATCH (n)-[:USES]->() } RETURN n.name AS name";
    for language in LANGUAGES {
        let session = session();
        let profile = plan(&session, language, "PROFILE", query);
        assert!(profile.contains("USES"), "{language:?} PROFILE:\n{profile}");
        assert!(
            profile.lines().any(|line| line.contains("Filter")),
            "{language:?} PROFILE without the filter:\n{profile}"
        );
        assert!(
            profile
                .lines()
                .next()
                .unwrap_or_default()
                .contains("rows=2"),
            "{language:?} the top line of\n{profile}"
        );
    }
}

/// Two EXISTS, one negated: both expands show in PROFILE.
#[test]
fn profile_shows_two_exists() {
    let query = "MATCH (n:Method) WHERE EXISTS { MATCH (n)-[:USES]->() } \
        AND NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() } RETURN n.name AS name";
    for language in LANGUAGES {
        let session = session();
        assert_eq!(
            run(&session, language, query),
            vec![vec![Value::from("Alix")]],
            "{language:?}"
        );
        let profile = plan(&session, language, "PROFILE", query);
        for edge_type in ["USES", "DEPRECATED_BY"] {
            assert!(
                profile.contains(edge_type),
                "{language:?} PROFILE without {edge_type}:\n{profile}"
            );
        }
        assert!(
            profile
                .lines()
                .next()
                .unwrap_or_default()
                .contains("rows=1"),
            "{language:?} the top line of\n{profile}"
        );
    }
}

/// A code-graph selector as Deriva writes it: `n.active` and an OR of
/// labels, each with its own EXISTS (one negated). Alix (an active Method,
/// not deprecated), and Vincent and Butch (Files that contain something).
const DERIVA_SELECTOR: &str = concat!(
    "MATCH (n:Graph) WHERE n.active ",
    "AND ((n:Method AND NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() }) ",
    "OR (n:File AND EXISTS { MATCH (n)-[:CONTAINS]->() })) ",
    "RETURN n.name AS name ORDER BY name"
);

/// The operator lines of a plan text: each line up to its statistics or
/// annotations, with its indentation (the tree's shape).
fn operator_lines(text: &str) -> Vec<String> {
    text.lines()
        .take_while(|line| !line.trim().is_empty())
        .map(|line| {
            let line = line.split("  rows=").next().unwrap_or(line);
            line.trim_end_matches(" [label-first]").to_string()
        })
        .collect()
}

/// The indentation of each operator line, the shape of the tree.
fn shape(lines: &[String]) -> Vec<usize> {
    lines
        .iter()
        .map(|line| line.len() - line.trim_start().len())
        .collect()
}

/// The filter lines of a plan, each with its predicate.
fn filters(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| line.trim_start().to_string())
        .filter(|line| line.starts_with("Filter"))
        .collect()
}

#[test]
fn the_deriva_selector_returns_its_nodes() {
    for language in LANGUAGES {
        let rows = run(&session(), language, DERIVA_SELECTOR);
        assert_eq!(
            rows,
            ["Alix", "Butch", "Vincent"].map(|name| vec![Value::from(name)]),
            "{language:?}"
        );
    }
}

/// PROFILE of Deriva's selector has the tree EXPLAIN shows: as many
/// operators, the same nesting, every filter with its predicate (the outer
/// one holding the EXISTS included), and the result on top with its rows.
#[test]
fn profile_lists_the_whole_plan_of_the_deriva_selector() {
    for language in LANGUAGES {
        let session = session();
        let explain = operator_lines(&plan(&session, language, "EXPLAIN", DERIVA_SELECTOR));
        let profile_text = plan(&session, language, "PROFILE", DERIVA_SELECTOR);
        let profile = operator_lines(&profile_text);
        let explained = explain.join("\n");
        assert_eq!(
            shape(&profile),
            shape(&explain),
            "{language:?} PROFILE:\n{profile_text}\nEXPLAIN:\n{explained}"
        );
        assert_eq!(
            filters(&profile),
            filters(&explain),
            "{language:?} the filters of PROFILE:\n{profile_text}"
        );
        assert!(
            filters(&profile)
                .iter()
                .any(|line| line.contains("EXISTS {")),
            "{language:?} no filter holds the EXISTS:\n{profile_text}"
        );
        assert!(
            profile_text
                .lines()
                .next()
                .unwrap_or_default()
                .contains("rows=3"),
            "{language:?} the top line of\n{profile_text}"
        );
    }
}

/// The same for the selector of three EXISTS: PROFILE has EXPLAIN's tree,
/// its filters (the one holding the subqueries included) and its rows.
#[test]
fn profile_lists_the_whole_plan_of_the_selector() {
    for language in LANGUAGES {
        let session = session();
        let explain = operator_lines(&plan(&session, language, "EXPLAIN", SELECTOR));
        let profile_text = plan(&session, language, "PROFILE", SELECTOR);
        let profile = operator_lines(&profile_text);
        let explained = explain.join("\n");
        assert_eq!(
            shape(&profile),
            shape(&explain),
            "{language:?} PROFILE:\n{profile_text}\nEXPLAIN:\n{explained}"
        );
        assert_eq!(
            filters(&profile),
            filters(&explain),
            "{language:?} the filters of PROFILE:\n{profile_text}\nEXPLAIN:\n{explained}"
        );
    }
}
