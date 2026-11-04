//! A label expression holds wherever a node pattern is written (ISO/IEC 39075:2024 16.8).
//!
//! `(x:A&B)`, `(x:A|B)`, `(x:!A)`, `(x:%)`, `(x:(A))` and `(x IS A)` were kept on a lone node and on
//! later nodes of a path, and dropped on a path's first node: a read gave every source with the edge,
//! and a write changed them. Each scenario is compared with a table of what the expression holds for,
//! checked against the lone pattern by `a_node_alone`, under every index and with the rules on and
//! off. `16.8b_label_expressions_in_patterns.gtest` runs the same table on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main label_expressions_in_paths::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::PlannerRules;
use grafeo_engine::session::Session;

/// Five nodes, one per kind of label set, each with an edge to the hub and an edge from it.
const GRAPH: &str = "INSERT (a:A {name: 'Alix', k: 1}), (g:B {name: 'Gus', k: 1}), (v:A:B {name: 'Vincent', k: 1}), (j:C {name: 'Jules', k: 1}), (m {name: 'Mia', k: 1}), (t:T {name: 'Amsterdam', tk: 1}), (a)-[:R]->(t), (g)-[:R]->(t), (v)-[:R]->(t), (j)-[:R]->(t), (m)-[:R]->(t), (t)-[:S]->(a), (t)-[:S]->(g), (t)-[:S]->(v), (t)-[:S]->(j), (t)-[:S]->(m)";

const HUB: &str = "Amsterdam";

/// Every node but the hub, sorted.
const SOURCES: [&str; 5] = ["Alix", "Gus", "Jules", "Mia", "Vincent"];

/// Each form as written after the variable, and the nodes it holds for, sorted.
const FORMS: &[(&str, &[&str])] = &[
    (":A", &["Alix", "Vincent"]),
    (":A:B", &["Vincent"]),
    (" IS A", &["Alix", "Vincent"]),
    (" IS A&B", &["Vincent"]),
    (":A&B", &["Vincent"]),
    (":A|B", &["Alix", "Gus", "Vincent"]),
    (":!A", &["Gus", "Jules", "Mia"]),
    (":%", &["Alix", "Gus", "Jules", "Vincent"]),
    (":(A)", &["Alix", "Vincent"]),
    (":Z&A", &[]),
    (" IS A|B", &["Alix", "Gus", "Vincent"]),
    (" IS !A", &["Gus", "Jules", "Mia"]),
    (" IS %", &["Alix", "Gus", "Jules", "Vincent"]),
    (" IS (A)", &["Alix", "Vincent"]),
    (":!%", &["Mia"]),
    (":A&!B", &["Alix"]),
    (":!A&!B", &["Jules", "Mia"]),
    (":(A|B)&!C", &["Alix", "Gus", "Vincent"]),
    (":(A&B)|C", &["Jules", "Vincent"]),
    (":!(A&B)", &["Alix", "Gus", "Jules", "Mia"]),
    (":Z", &[]),
    (":!Z", &["Alix", "Gus", "Jules", "Mia", "Vincent"]),
];

/// What a form of a scenario gives.
#[derive(Clone, Copy, Debug)]
enum Want {
    /// One row for each node the expression holds for.
    Holders,
    /// One row for each other source.
    Complement,
    /// The same, or one null row when it holds for none.
    Optional,
    /// The hub when it holds for some node.
    HubIfAny,
    /// The hub when it holds for none.
    HubIfNone,
    /// How many nodes it holds for.
    Count,
    /// Every source, with the hub for those it holds for.
    Kept,
}

impl Want {
    fn rows(self, holders: &[&str]) -> Vec<String> {
        let mut rows: Vec<String> = match self {
            Self::Holders => holders.iter().map(ToString::to_string).collect(),
            Self::Complement => SOURCES
                .iter()
                .filter(|name| !holders.contains(name))
                .map(ToString::to_string)
                .collect(),
            Self::Optional if holders.is_empty() => vec!["null".to_string()],
            Self::Optional => holders.iter().map(ToString::to_string).collect(),
            Self::HubIfAny if holders.is_empty() => Vec::new(),
            Self::HubIfNone if !holders.is_empty() => Vec::new(),
            Self::HubIfAny | Self::HubIfNone => vec![HUB.to_string()],
            Self::Count => vec![holders.len().to_string()],
            Self::Kept => SOURCES
                .iter()
                .map(|name| {
                    let far = if holders.contains(name) { HUB } else { "null" };
                    format!("{name}|{far}")
                })
                .collect(),
        };
        rows.sort();
        rows
    }
}

/// The index settings the expression must not depend on.
#[derive(Clone, Copy, Debug)]
enum Setup {
    NoIndex,
    OnLabel,
    KeyWide,
    OnHub,
}

const SETUPS: [Setup; 4] = [Setup::NoIndex, Setup::OnLabel, Setup::KeyWide, Setup::OnHub];

fn graph(setup: Setup) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute(GRAPH).unwrap();
    match setup {
        Setup::NoIndex => {}
        Setup::OnLabel => {
            session
                .execute("CREATE INDEX on_a FOR (n:A) ON (n.k)")
                .unwrap();
        }
        Setup::KeyWide => db.create_property_index("k").unwrap(),
        Setup::OnHub => {
            session
                .execute("CREATE INDEX on_hub FOR (n:T) ON (n.tk)")
                .unwrap();
        }
    }
    db
}

fn cell(value: &Value) -> String {
    match value {
        Value::String(text) => text.as_str().to_string(),
        Value::Null => "null".to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::List(items) => {
            let mut items: Vec<String> = items.iter().map(cell).collect();
            items.sort();
            format!("[{}]", items.join(","))
        }
        other => format!("{other:?}"),
    }
}

/// The rows, each its cells joined, sorted.
fn rows_of(session: &Session, query: &str) -> Vec<String> {
    let result = session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect();
    rows.sort();
    rows
}

fn sessions(db: &GrafeoDB) -> [(&'static str, Session); 2] {
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    [("all rules", db.session()), ("plain", plain)]
}

/// Each form of a scenario, with `@` for the expression, gives what `want` says, under every form,
/// index and rule set.
fn gives(want: Want, forms: &[&str]) {
    for setup in SETUPS {
        let db = graph(setup);
        for (expression, holders) in FORMS {
            let wanted = want.rows(holders);
            for form in forms {
                let query = form.replace('@', expression);
                for (rules, session) in sessions(&db) {
                    assert_eq!(
                        rows_of(&session, &query),
                        wanted,
                        "{setup:?}, {rules}: {query}"
                    );
                }
            }
        }
    }
}

/// A node alone.
#[test]
fn a_node_alone() {
    gives(
        Want::Holders,
        &[
            "MATCH (x@) WHERE x.name <> 'Amsterdam' RETURN x.name",
            "MATCH (x@ {k: 1}) RETURN x.name",
        ],
    );
}

/// First node of a path.
#[test]
fn first_node() {
    gives(
        Want::Holders,
        &[
            "MATCH (x@)-[:R]->(t:T) RETURN x.name",
            "MATCH (x@)-[:R]->(t) RETURN x.name",
            "MATCH (x@)-[:R]->(:T) RETURN x.name",
        ],
    );
}

/// First node, incoming and undirected.
#[test]
fn first_node_directions() {
    gives(
        Want::Holders,
        &[
            "MATCH (x@)<-[:S]-(t:T) RETURN x.name",
            "MATCH (x@)-[:R]-(t:T) RETURN x.name",
            "MATCH (x@)<-[:S]-(t) RETURN x.name",
        ],
    );
}

/// First node, with WHERE and properties.
#[test]
fn first_node_where() {
    gives(
        Want::Holders,
        &[
            "MATCH (x@ {k: 1})-[:R]->(t:T) RETURN x.name",
            "MATCH (x@ WHERE x.k = 1)-[:R]->(t:T) RETURN x.name",
            "MATCH (x@)-[:R]->(t:T) WHERE x.k = 1 RETURN x.name",
        ],
    );
}

/// First node, far end pinned by a key.
#[test]
fn first_node_pinned_far_end() {
    gives(
        Want::Holders,
        &[
            "MATCH (x@)-[:R]->(t:T {tk: 1}) RETURN x.name",
            "MATCH (x@)-[:R]->(t:T) WHERE t.tk = 1 RETURN x.name",
            "MATCH (x@)-[:R]->(t {tk: 1}) RETURN x.name",
        ],
    );
}

/// First node of a longer path.
#[test]
fn first_node_two_hops() {
    gives(
        Want::Holders,
        &["MATCH (x@)-[:R]->(t:T)-[:S]->(y {name: 'Alix'}) RETURN x.name"],
    );
}

/// Named path.
#[test]
fn named_path() {
    gives(
        Want::Holders,
        &[
            "MATCH p = (x@)-[:R]->(t:T) RETURN x.name",
            "MATCH p = (x@)-[:R]->(t:T)-[:S]->(y {name: 'Alix'}) RETURN x.name",
        ],
    );
}

/// Path modes.
#[test]
fn path_modes() {
    gives(
        Want::Holders,
        &[
            "MATCH TRAIL (x@)-[:R]->(t:T) RETURN x.name",
            "MATCH ACYCLIC (x@)-[:R]->(t:T) RETURN x.name",
            "MATCH SIMPLE (x@)-[:R]->(t:T) RETURN x.name",
        ],
    );
}

/// Variable-length hop.
#[test]
fn variable_length() {
    gives(
        Want::Holders,
        &[
            "MATCH (x@)-[:R*1..2]->(t:T) RETURN x.name",
            "MATCH (x@)-[:R]->{1,2}(t:T) RETURN x.name",
            "MATCH (x@)-[:R]->+(t:T) RETURN x.name",
        ],
    );
}

/// Shortest path.
#[test]
fn shortest() {
    gives(
        Want::Holders,
        &[
            "MATCH ANY SHORTEST (x@)-[:R]->+(t:T) RETURN x.name",
            "MATCH p = ALL SHORTEST (x@)-[:R]->+(t:T) RETURN x.name",
            "MATCH SHORTEST 1 PATH (x@)-[:R]->+(t:T) RETURN x.name",
            "MATCH ANY SHORTEST (t:T)-[:S]->+(x@) RETURN x.name",
        ],
    );
}

/// Parenthesized path.
#[test]
fn parenthesized() {
    gives(
        Want::Holders,
        &[
            "MATCH ((x@)-[:R]->(t:T)) RETURN x.name",
            "MATCH ((x@)-[:R]->(t:T) WHERE x.k = 1) RETURN x.name",
        ],
    );
}

/// Juxtaposed pattern, the node before the segment.
#[test]
fn juxtaposed_outer_node() {
    gives(
        Want::Holders,
        &[
            "MATCH (x@)((a)-[:R]->(b)){1,1}(t:T) RETURN x.name",
            "MATCH (x@)(-[:R]->(m)){1,1}(t:T) RETURN x.name",
            "MATCH (x@)((a)-[:R]->(m)-[:S]->(b)){1,1}(y {name: 'Alix'}) RETURN x.name",
        ],
    );
}

/// Juxtaposed pattern, the segment's own nodes.
#[test]
fn juxtaposed_segment() {
    gives(
        Want::Holders,
        &[
            "MATCH (x)((a@)-[:R]->(b)){1,1}(t:T) RETURN x.name",
            "MATCH (s:T)((a)-[:S]->(b@)){1,1}(x) RETURN x.name",
            "MATCH (s:T)((a)-[:S]->(b)){1,1}(x@) RETURN x.name",
            "MATCH (s {name: 'Alix'})((a)-[:R]->(m)-[:S]->(b@)){1,1}(x) RETURN x.name",
        ],
    );
}

/// Comma-separated patterns.
#[test]
fn comma_separated() {
    gives(
        Want::Holders,
        &[
            "MATCH (x@)-[:R]->(t:T), (x)-[:R]->(u) RETURN x.name",
            "MATCH (u:T), (x@)-[:R]->(u) RETURN x.name",
            "MATCH (x@), (x)-[:R]->(u:T) RETURN x.name",
            "MATCH (t:T)-[:S]->(w {name: 'Alix'}), (x@)-[:R]->(t) RETURN x.name",
        ],
    );
}

/// A source the input already binds.
#[test]
fn source_already_bound() {
    gives(
        Want::Holders,
        &[
            "MATCH (x {k: 1}) MATCH (x@)-[:R]->(t:T) RETURN x.name",
            "MATCH (x {k: 1}) WITH x MATCH (x@)-[:R]->(t:T) RETURN x.name",
        ],
    );
}

/// First node after other input.
#[test]
fn after_input() {
    gives(
        Want::Holders,
        &[
            "UNWIND [1] AS i MATCH (x@)-[:R]->(t:T) RETURN x.name",
            "MATCH (u:T) MATCH (x@)-[:R]->(u) RETURN x.name",
        ],
    );
}

/// EXISTS on a node from outside.
#[test]
fn exists_on_a_bound_node() {
    gives(
        Want::Holders,
        &[
            "MATCH (y {k: 1}) WHERE EXISTS { MATCH (y@)-[:R]->(t:T) } RETURN y.name",
            "MATCH (y {k: 1}) WHERE EXISTS { MATCH (t:T)-[:S]->(y@) } RETURN y.name",
        ],
    );
}

/// NOT EXISTS on a node from outside.
#[test]
fn not_exists_on_a_bound_node() {
    gives(
        Want::Complement,
        &["MATCH (y {k: 1}) WHERE NOT EXISTS { MATCH (y@)-[:R]->(t:T) } RETURN y.name"],
    );
}

/// EXISTS with a node of its own.
#[test]
fn exists_on_its_own_node() {
    gives(
        Want::HubIfAny,
        &[
            "MATCH (t:T) WHERE EXISTS { MATCH (x@)-[:R]->(t) } RETURN t.name",
            "MATCH (t:T) WHERE EXISTS { MATCH (t)-[:S]->(x@) } RETURN t.name",
        ],
    );
}

/// NOT EXISTS with a node of its own.
#[test]
fn not_exists_on_its_own_node() {
    gives(
        Want::HubIfNone,
        &["MATCH (t:T) WHERE NOT EXISTS { MATCH (x@)-[:R]->(t) } RETURN t.name"],
    );
}

/// COUNT subquery.
#[test]
fn count_subquery() {
    gives(
        Want::Count,
        &[
            "MATCH (t:T) RETURN COUNT { MATCH (x@)-[:R]->(t) } AS c",
            "MATCH (t:T) RETURN COUNT { MATCH (t)-[:S]->(x@) } AS c",
        ],
    );
}

/// COUNT subquery on a node from outside.
#[test]
fn count_on_a_bound_node() {
    gives(
        Want::Holders,
        &["MATCH (y {k: 1}) WHERE COUNT { MATCH (y@)-[:R]->() } = 1 RETURN y.name"],
    );
}

/// CALL subquery.
#[test]
fn call_subquery() {
    gives(
        Want::Holders,
        &[
            "CALL { MATCH (x@)-[:R]->(t:T) RETURN x.name AS r } RETURN r",
            "MATCH (y {k: 1}) CALL { WITH y MATCH (y@)-[:R]->(t:T) RETURN y.name AS r } RETURN r",
            "CALL { MATCH (t:T)-[:S]->(x@) RETURN x.name AS r } RETURN r",
        ],
    );
}

/// OPTIONAL MATCH, first node.
#[test]
fn optional_first_node() {
    gives(
        Want::Optional,
        &[
            "OPTIONAL MATCH (x@)-[:R]->(t:T) RETURN x.name",
            "MATCH (t:T) OPTIONAL MATCH (x@)-[:R]->(t) RETURN x.name",
        ],
    );
}

/// OPTIONAL MATCH, a source the input binds.
#[test]
fn optional_bound_source() {
    gives(
        Want::Kept,
        &["MATCH (y {k: 1}) OPTIONAL MATCH (y@)-[:R]->(t:T) RETURN y.name, t.name"],
    );
}

/// OPTIONAL MATCH, far end.
#[test]
fn optional_far_end() {
    gives(
        Want::Optional,
        &["MATCH (t:T) OPTIONAL MATCH (t)-[:S]->(x@) RETURN x.name"],
    );
}

/// First node without a variable.
#[test]
fn no_variable() {
    gives(Want::Count, &["MATCH (@)-[:R]->(t:T) RETURN count(*) AS c"]);
}

/// Far end.
#[test]
fn far_end() {
    gives(
        Want::Holders,
        &[
            "MATCH (t:T)-[:S]->(x@) RETURN x.name",
            "MATCH (t:T)<-[:R]-(x@) RETURN x.name",
            "MATCH (t)-[:S]->(x@) RETURN x.name",
            "MATCH (t:T {tk: 1})-[:S]->(x@) RETURN x.name",
        ],
    );
}

/// A later node, in the middle.
#[test]
fn later_node() {
    gives(
        Want::Holders,
        &[
            "MATCH (t:T)-[:S]->(x@)-[:R]->(u:T) RETURN x.name",
            "MATCH (y {name: 'Alix'})-[:R]->(t:T)-[:S]->(x@) RETURN x.name",
        ],
    );
}

// ---------------------------------------------------------------------------
// What a statement after the match changes
// ---------------------------------------------------------------------------

/// Every node of the graph: name, `k`, `hit`, labels, and whether its edge to the hub stands.
fn everything(session: &Session) -> Vec<String> {
    let with_edge = rows_of(session, "MATCH (x)-[:R]->() RETURN x.name");
    let mut rows: Vec<String> = rows_of(session, "MATCH (x) RETURN x.name, x.k, x.hit, labels(x)")
        .into_iter()
        .map(|row| {
            let name = row.split('|').next().unwrap().to_string();
            format!("{row}|{}", with_edge.contains(&name))
        })
        .collect();
    rows.sort();
    rows
}

/// The graph's rows as written, with `change` applied to the nodes the expression holds for.
fn expected(holders: &[&str], change: fn(&mut Node)) -> Vec<String> {
    let mut nodes = vec![
        Node::new("Alix", &["A"]),
        Node::new("Gus", &["B"]),
        Node::new("Jules", &["C"]),
        Node::new("Mia", &[]),
        Node::new("Vincent", &["A", "B"]),
    ];
    for node in &mut nodes {
        if holders.contains(&node.name) {
            change(node);
        }
    }
    let hub = Node {
        name: HUB,
        k: None,
        hit: None,
        labels: vec!["T"],
        edge: false,
        deleted: false,
    };
    let mut rows: Vec<String> = nodes
        .iter()
        .chain([&hub])
        .filter(|node| !node.deleted)
        .map(Node::row)
        .collect();
    rows.sort();
    rows
}

struct Node {
    name: &'static str,
    k: Option<i64>,
    hit: Option<bool>,
    labels: Vec<&'static str>,
    edge: bool,
    deleted: bool,
}

impl Node {
    fn new(name: &'static str, labels: &[&'static str]) -> Self {
        Self {
            name,
            k: Some(1),
            hit: None,
            labels: labels.to_vec(),
            edge: true,
            deleted: false,
        }
    }

    fn row(&self) -> String {
        let mut labels = self.labels.clone();
        labels.sort_unstable();
        format!(
            "{}|{}|{}|[{}]|{}",
            self.name,
            self.k.map_or("null".to_string(), |k| k.to_string()),
            self.hit.map_or("null".to_string(), |hit| hit.to_string()),
            labels.join(","),
            self.edge
        )
    }
}

/// A statement with `@` for the expression changes the nodes the expression holds for, and every other
/// node's data stays, under every index and rule set.
fn changes_only_its_holders(statement: &str, change: fn(&mut Node)) {
    for setup in SETUPS {
        for (expression, holders) in FORMS {
            for rules in [PlannerRules::all(), PlannerRules::plain()] {
                let db = graph(setup);
                let mut session = db.session();
                session.set_planner_rules(rules);
                let query = statement.replace('@', expression);
                session
                    .execute(&query)
                    .unwrap_or_else(|error| panic!("{setup:?}: {query}: {error}"));
                assert_eq!(
                    everything(&session),
                    expected(holders, change),
                    "{setup:?}: {query}"
                );
            }
        }
    }
}

#[test]
fn set_a_property_changes_only_its_holders() {
    changes_only_its_holders("MATCH (x@)-[:R]->(t:T) SET x.hit = true", |node| {
        node.hit = Some(true);
    });
}

#[test]
fn set_a_label_changes_only_its_holders() {
    changes_only_its_holders("MATCH (x@)-[:R]->(t:T) SET x:New", |node| {
        node.labels.push("New");
    });
}

#[test]
fn remove_a_property_changes_only_its_holders() {
    changes_only_its_holders("MATCH (x@)-[:R]->(t:T) REMOVE x.k", |node| {
        node.k = None;
    });
}

#[test]
fn remove_a_label_changes_only_its_holders() {
    changes_only_its_holders("MATCH (x@)-[:R]->(t:T) REMOVE x:A", |node| {
        node.labels.retain(|label| *label != "A");
    });
}

#[test]
fn delete_an_edge_changes_only_its_holders() {
    changes_only_its_holders("MATCH (x@)-[r:R]->(t:T) DELETE r", |node| {
        node.edge = false;
    });
}

#[test]
fn detach_delete_removes_only_its_holders() {
    changes_only_its_holders("MATCH (x@)-[:R]->(t:T) DETACH DELETE x", |node| {
        node.deleted = true;
    });
}

/// A `DELETE` of a node with edges fails and leaves everything as it was. With no node to delete, it
/// succeeds and changes nothing, which is what a dropped expression got wrong both ways.
#[test]
fn delete_a_node_with_edges_fails_only_when_the_expression_holds() {
    for setup in SETUPS {
        for (expression, holders) in FORMS {
            let db = graph(setup);
            let session = db.session();
            let query = format!("MATCH (x{expression})-[:R]->(t:T) DELETE x");
            let result = session.execute(&query);
            assert_eq!(result.is_err(), !holders.is_empty(), "{setup:?}: {query}");
            assert_eq!(
                everything(&session),
                expected(&[], |_| {}),
                "{setup:?}: {query} leaves everything"
            );
        }
    }
}

/// A rollback brings back what a `DETACH DELETE` with an expression removed.
#[test]
fn a_rollback_restores_what_the_statement_changed() {
    let db = graph(Setup::NoIndex);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (x:A&B)-[:R]->(t:T) DETACH DELETE x")
        .unwrap();
    assert_eq!(
        everything(&session),
        expected(&["Vincent"], |node| node.deleted = true)
    );
    session.rollback().unwrap();
    assert_eq!(everything(&session), expected(&[], |_| {}));
}

// ---------------------------------------------------------------------------
// The other languages
// ---------------------------------------------------------------------------

/// SQL/PGQ's `GRAPH_TABLE` takes GQL's patterns: first node, later nodes and a lone node.
#[test]
fn sql_pgq_holds_it_in_every_position() {
    for setup in SETUPS {
        let db = graph(setup);
        for (expression, holders) in FORMS {
            for pattern in [
                format!("(x{expression})-[:R]->(t IS T)"),
                format!("(x{expression})<-[:S]-(t IS T)"),
                format!("(x{expression} WHERE x.k = 1)-[:R]->(t IS T)"),
                format!("(x{expression})-[:R]->(t IS T)-[:S]->(y {{name: 'Alix'}})"),
                format!("(u IS T), (x{expression})-[:R]->(u)"),
                format!("(t IS T)-[:S]->(x{expression})"),
                format!("(t IS T)-[:S]->(x{expression})-[:R]->(u IS T)"),
                format!("(x{expression} {{k: 1}})"),
            ] {
                let query =
                    format!("SELECT r FROM GRAPH_TABLE (MATCH {pattern} COLUMNS (x.name AS r))");
                for (rules, session) in sessions(&db) {
                    let result = session
                        .execute_sql(&query)
                        .unwrap_or_else(|error| panic!("{query}: {error}"));
                    let mut found: Vec<String> =
                        result.rows().iter().map(|row| cell(&row[0])).collect();
                    found.sort();
                    assert_eq!(found, holders.to_vec(), "{setup:?}, {rules}: {query}");
                }
            }
        }
    }
}

/// openCypher's node pattern takes `:A:B` and no expression: a syntax error, not a dropped label.
#[test]
fn cypher_refuses_an_expression_on_a_node() {
    let db = graph(Setup::NoIndex);
    let session = db.session();
    for query in [
        "MATCH (x:A&B)-[:R]->(t:T) RETURN x.name",
        "MATCH (x:A|B)-[:R]->(t:T) RETURN x.name",
        "MATCH (x:!A)-[:R]->(t:T) RETURN x.name",
        "MATCH (x:%)-[:R]->(t:T) RETURN x.name",
        "MATCH (x:(A))-[:R]->(t:T) RETURN x.name",
        "MATCH (x IS A)-[:R]->(t:T) RETURN x.name",
        "MATCH (x:A&B) RETURN x.name",
        "MATCH (t:T)-[:S]->(x:A|B) RETURN x.name",
        "CREATE (n:A&B {k: 1})",
        "CREATE (n IS A {k: 1})",
        "MERGE (n:A&B {k: 1})",
        "MERGE (n IS A {k: 1})",
    ] {
        assert!(session.execute_cypher(query).is_err(), "{query}");
    }
    let count = |query: &str| session.execute_cypher(query).unwrap().row_count();
    assert_eq!(count("MATCH (x:A:B)-[:R]->(t:T) RETURN x.name"), 1);
    assert_eq!(count("MATCH (x:A)-[:R]->(t:T) RETURN x.name"), 2);
    assert_eq!(
        count("MATCH (x)-[:R]->(t:T) WHERE x:A OR x:B RETURN x.name"),
        3
    );
    assert_eq!(
        rows_of(&session, "MATCH (x) RETURN count(x)"),
        vec!["6"],
        "the refused writes wrote nothing"
    );
}
