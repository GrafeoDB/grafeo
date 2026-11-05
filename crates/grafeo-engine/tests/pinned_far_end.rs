//! A chain whose far end an index or `id()` pins: the last hop finds the nodes of that key
//! once, before it reads its input, and goes on from them; the filter stays and decides
//! (#714, `PlannerRule::PinnedFarEnd`).
//!
//! `MATCH (a:Person {id: $x})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: $y}) RETURN ...`
//!
//! Every answer is checked three ways: the rule on, the rule off (and `SoughtEnd` off, and
//! no rule at all), and a count over this test's own list of people and edges.

#![cfg(all(feature = "gql", feature = "cypher"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few thousand nodes"
)]

use std::collections::HashMap;

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Way {
    Out,
    In,
    Either,
}

impl Way {
    const ALL: [Self; 3] = [Self::Out, Self::In, Self::Either];

    fn text(self) -> &'static str {
        match self {
            Self::Out => "-[:KNOWS]->",
            Self::In => "<-[:KNOWS]-",
            Self::Either => "-[:KNOWS]-",
        }
    }
}

#[derive(Clone, Debug)]
struct Person {
    name: String,
    key: Option<i64>,
    label: &'static str,
}

#[derive(Clone, Debug)]
struct Knows {
    number: usize,
    from: String,
    to: String,
}

/// The graph as this test knows it. Keys repeat: Vincent and Jules share 3.
#[derive(Clone, Debug)]
struct People {
    people: Vec<Person>,
    edges: Vec<Knows>,
}

fn people() -> People {
    let person = |name: &str, key, label| Person {
        name: name.to_string(),
        key,
        label,
    };
    let edge = |number, from: &str, to: &str| Knows {
        number,
        from: from.to_string(),
        to: to.to_string(),
    };
    People {
        people: vec![
            person("Alix", Some(1), "Person"),
            person("Gus", Some(2), "Person"),
            person("Vincent", Some(3), "Person"),
            person("Jules", Some(3), "Person"),
            person("Mia", Some(4), "Person"),
            person("Butch", None, "Person"),
            person("Amsterdam", Some(1), "City"),
            person("Django", Some(6), "Person"),
        ],
        edges: vec![
            edge(0, "Alix", "Gus"),
            edge(1, "Gus", "Vincent"),
            edge(2, "Gus", "Jules"),
            edge(3, "Vincent", "Alix"),
            // Two edges between one pair, and one each way.
            edge(4, "Vincent", "Alix"),
            edge(5, "Alix", "Vincent"),
            // A self-loop.
            edge(6, "Gus", "Gus"),
            edge(7, "Jules", "Butch"),
            edge(8, "Butch", "Alix"),
            edge(9, "Alix", "Amsterdam"),
            edge(10, "Amsterdam", "Gus"),
        ],
    }
}

impl People {
    fn person(&self, name: &str) -> &Person {
        self.people.iter().find(|p| p.name == name).unwrap()
    }

    /// The people `name` reaches over one hop, by edge.
    fn hops(&self, name: &str, way: Way) -> Vec<(usize, &str)> {
        self.edges
            .iter()
            .filter_map(|edge| match way {
                Way::Out => (edge.from == name).then_some((edge.number, edge.to.as_str())),
                Way::In => (edge.to == name).then_some((edge.number, edge.from.as_str())),
                Way::Either => {
                    if edge.from == name {
                        Some((edge.number, edge.to.as_str()))
                    } else {
                        (edge.to == name).then_some((edge.number, edge.from.as_str()))
                    }
                }
            })
            .collect()
    }

    /// (a, c, b) of the chain, sorted, by `pin_a` and `pin_b` on the people's keys.
    fn chain(
        &self,
        ways: (Way, Way),
        pin_a: &dyn Fn(&Person) -> bool,
        pin_b: &dyn Fn(&Person) -> bool,
        edges_may_repeat: bool,
    ) -> Vec<Vec<String>> {
        let mut rows = Vec::new();
        for a in self
            .people
            .iter()
            .filter(|p| p.label == "Person" && pin_a(p))
        {
            for (first, c) in self.hops(&a.name, ways.0) {
                if self.person(c).label != "Person" {
                    continue;
                }
                for (second, b) in self.hops(c, ways.1) {
                    let far = self.person(b);
                    if far.label != "Person" || !pin_b(far) {
                        continue;
                    }
                    if !edges_may_repeat && first == second {
                        continue;
                    }
                    rows.push(vec![a.name.clone(), c.to_string(), b.to_string()]);
                }
            }
        }
        rows.sort();
        rows
    }
}

/// A database with `model`'s people and edges and an index on `id`, and the ids of its nodes.
fn database(model: &People, config: Config) -> (GrafeoDB, HashMap<String, NodeId>) {
    let db = GrafeoDB::with_config(config).unwrap();
    db.create_property_index("id").unwrap();
    let mut ids = HashMap::new();
    for person in &model.people {
        let mut properties = vec![("name", Value::from(person.name.as_str()))];
        if let Some(key) = person.key {
            properties.push(("id", Value::Int64(key)));
        }
        let id = db
            .create_node_with_props(&[person.label], properties)
            .unwrap();
        ids.insert(person.name.clone(), id);
    }
    for edge in &model.edges {
        db.create_edge(ids[&edge.from], ids[&edge.to], "KNOWS")
            .unwrap();
    }
    (db, ids)
}

fn in_memory() -> Config {
    Config::in_memory().with_history_retention(HistoryRetention::Forever)
}

/// The rule sets that must agree, by name.
fn rule_sets() -> Vec<(&'static str, PlannerRules)> {
    vec![
        ("all rules", PlannerRules::all()),
        (
            "all but PinnedFarEnd",
            PlannerRules::all().with(PlannerRule::PinnedFarEnd, false),
        ),
        (
            "all but SoughtEnd",
            PlannerRules::all().with(PlannerRule::SoughtEnd, false),
        ),
        (
            "all but ExpandInto",
            PlannerRules::all().with(PlannerRule::ExpandInto, false),
        ),
        (
            "only PinnedFarEnd",
            PlannerRules::plain().with(PlannerRule::PinnedFarEnd, true),
        ),
        ("no rule", PlannerRules::plain()),
    ]
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Language {
    Gql,
    Cypher,
}

fn run(
    session: &Session,
    language: Language,
    query: &str,
    parameters: &HashMap<String, Value>,
) -> Result<Vec<Vec<String>>, String> {
    let result = match language {
        Language::Gql => session.execute_with_params(query, parameters.clone()),
        Language::Cypher => session.execute_cypher_with_params(query, parameters.clone()),
    }
    .map_err(|error| error.to_string())?;
    let mut rows: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    other => format!("{other:?}"),
                })
                .collect()
        })
        .collect();
    rows.sort();
    Ok(rows)
}

fn parameters(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

/// Every chain direction, in both languages, for keys that match several nodes, one, none,
/// or a and b alike: each rule set answers what the list of people says. Twice, because the
/// second call runs the cached plan with the parameters filled in.
fn every_rule_set_agrees(session: &mut Session, model: &People, stage: &str) {
    const PAIRS: [(i64, i64); 8] = [
        (1, 2),
        (2, 3),
        (3, 1),
        (3, 3),
        (1, 1),
        (2, 2),
        (1, 99),
        (99, 2),
    ];
    for (first, second) in Way::ALL.into_iter().flat_map(|w| Way::ALL.map(|v| (w, v))) {
        let query = format!(
            "MATCH (a:Person {{id: $x}}){}(c:Person){}(b:Person {{id: $y}}) RETURN a.name, c.name, b.name",
            first.text(),
            second.text()
        );
        for language in [Language::Gql, Language::Cypher] {
            for (name, rules) in rule_sets() {
                session.set_planner_rules(rules);
                for (x, y) in PAIRS {
                    let expected = model.chain(
                        (first, second),
                        &|p| p.key == Some(x),
                        &|p| p.key == Some(y),
                        language != Language::Cypher,
                    );
                    let values = parameters(&[("x", Value::Int64(x)), ("y", Value::Int64(y))]);
                    let got = run(session, language, &query, &values);
                    assert_eq!(
                        got.as_ref(),
                        Ok(&expected),
                        "{stage}, {language:?}, {name}, x {x} y {y}: {query}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The rows, at every stage of the graph
// ---------------------------------------------------------------------------

#[test]
fn committed_rows_agree() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    every_rule_set_agrees(&mut db.session(), &model, "committed");
}

/// The transaction's own nodes and edges, a key it changes (the seek reads an index) and
/// what it removes, before and after a rollback; another session sees none of it.
#[test]
fn a_transaction_seeks_what_it_sees() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    let mut watcher = db.session();
    session.begin_transaction().unwrap();

    // A new person with a key that exists, and one with a new key.
    session
        .execute("INSERT (:Person {name: 'Marcus', id: 2}), (:Person {name: 'Lucas', id: 7})")
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (m:Person {name: 'Marcus'}), (l:Person {name: 'Lucas'})
             INSERT (a)-[:KNOWS]->(m), (m)-[:KNOWS]->(l), (l)-[:KNOWS]->(a)",
        )
        .unwrap();
    let mut seen = model.clone();
    let add = |seen: &mut People, name: &str, key| {
        seen.people.push(Person {
            name: name.to_string(),
            key: Some(key),
            label: "Person",
        });
    };
    add(&mut seen, "Marcus", 2);
    add(&mut seen, "Lucas", 7);
    let edge = |number, from: &str, to: &str| Knows {
        number,
        from: from.to_string(),
        to: to.to_string(),
    };
    seen.edges.extend([
        edge(11, "Alix", "Marcus"),
        edge(12, "Marcus", "Lucas"),
        edge(13, "Lucas", "Alix"),
    ]);
    every_rule_set_agrees(&mut session, &seen, "own new nodes");
    every_rule_set_agrees(&mut watcher, &model, "another session, own new nodes");

    // A key changed: the old key finds nobody, the new one finds the node.
    session
        .execute("MATCH (g:Person {name: 'Gus'}) SET g.id = 99")
        .unwrap();
    seen.people
        .iter_mut()
        .find(|p| p.name == "Gus")
        .unwrap()
        .key = Some(99);
    every_rule_set_agrees(&mut session, &seen, "own changed key");
    every_rule_set_agrees(&mut watcher, &model, "another session, changed key");

    // A key removed.
    session
        .execute("MATCH (v:Person {name: 'Vincent'}) REMOVE v.id")
        .unwrap();
    seen.people
        .iter_mut()
        .find(|p| p.name == "Vincent")
        .unwrap()
        .key = None;
    every_rule_set_agrees(&mut session, &seen, "own removed key");

    // An edge and a node gone.
    session
        .execute("MATCH (:Person {name: 'Alix'})-[e:KNOWS]->(:Person {name: 'Gus'}) DELETE e")
        .unwrap();
    seen.edges.retain(|e| e.number != 0);
    session
        .execute("MATCH (j:Person {name: 'Jules'}) DETACH DELETE j")
        .unwrap();
    seen.people.retain(|p| p.name != "Jules");
    seen.edges.retain(|e| e.from != "Jules" && e.to != "Jules");
    every_rule_set_agrees(&mut session, &seen, "own deletes");
    every_rule_set_agrees(&mut watcher, &model, "another session, deletes");

    session.rollback().unwrap();
    every_rule_set_agrees(&mut session, &model, "after the rollback");
    every_rule_set_agrees(&mut watcher, &model, "another session, after the rollback");
}

/// A past epoch: keys and edges as they were then, whatever the index holds now.
#[test]
fn a_past_epoch_seeks_what_it_saw() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    let epoch = db.current_epoch();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) SET g.id = 3")
        .unwrap();
    session
        .execute("MATCH (v:Person {name: 'Vincent'}) SET v.id = 2")
        .unwrap();
    session
        .execute("MATCH (j:Person {name: 'Jules'}) DETACH DELETE j")
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}) INSERT (a)-[:KNOWS]->(:Person {name: 'Hans', id: 2})",
        )
        .unwrap();
    session.set_viewing_epoch(epoch);
    every_rule_set_agrees(&mut session, &model, "past epoch");
    session.clear_viewing_epoch();

    let mut now = model.clone();
    now.people.iter_mut().find(|p| p.name == "Gus").unwrap().key = Some(3);
    now.people
        .iter_mut()
        .find(|p| p.name == "Vincent")
        .unwrap()
        .key = Some(2);
    now.people.retain(|p| p.name != "Jules");
    now.edges.retain(|e| e.from != "Jules" && e.to != "Jules");
    now.people.push(Person {
        name: "Hans".to_string(),
        key: Some(2),
        label: "Person",
    });
    now.edges.push(Knows {
        number: 11,
        from: "Alix".to_string(),
        to: "Hans".to_string(),
    });
    every_rule_set_agrees(&mut session, &now, "now");
}

/// Read back from a file: the index and the sorted adjacency are built by the load.
#[test]
fn a_reopened_database_seeks_the_same() {
    let model = people();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("people.grafeo");
    {
        let (db, _) = database(&model, Config::persistent(&path));
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    every_rule_set_agrees(&mut db.session(), &model, "reopened");
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// The other shapes of a pin
// ---------------------------------------------------------------------------

/// `id()` pins as a property does: on either end, in WHERE, as a constant and a parameter.
#[test]
fn an_id_pins_the_far_end() {
    let model = people();
    let (db, ids) = database(&model, in_memory());
    let mut session = db.session();
    let id = |name: &str| i64::try_from(ids[name].as_u64()).unwrap();
    for (name_a, name_b) in [
        ("Alix", "Vincent"),
        ("Alix", "Alix"),
        ("Gus", "Gus"),
        ("Butch", "Gus"),
    ] {
        for (first, second) in Way::ALL.into_iter().flat_map(|w| Way::ALL.map(|v| (w, v))) {
            let expected = model.chain(
                (first, second),
                &|p| p.name == name_a,
                &|p| p.name == name_b,
                true,
            );
            let hops = format!(
                "(a:Person){}(c:Person){}(b:Person)",
                first.text(),
                second.text()
            );
            for query in [
                format!(
                    "MATCH {hops} WHERE id(a) = {} AND id(b) = {} RETURN a.name, c.name, b.name",
                    id(name_a),
                    id(name_b)
                ),
                format!(
                    "MATCH {hops} WHERE id(a) = $x AND id(b) = $y RETURN a.name, c.name, b.name"
                ),
                format!(
                    "MATCH {hops} WHERE a.name = '{name_a}' AND id(b) = $y RETURN a.name, c.name, b.name"
                ),
            ] {
                let values = parameters(&[
                    ("x", Value::Int64(id(name_a))),
                    ("y", Value::Int64(id(name_b))),
                ]);
                for (name, rules) in rule_sets() {
                    session.set_planner_rules(rules);
                    let got = run(&session, Language::Gql, &query, &values);
                    assert_eq!(got.as_ref(), Ok(&expected), "{name}: {query}");
                }
            }
        }
    }
}

/// A null never matches. A value of another type: the plans must agree, whatever the
/// comparison answers (ISO GA04: an integer key and a string are not equal, 1.0 and 1 are).
#[test]
fn a_parameter_that_is_null_or_of_another_type() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    let query = "MATCH (a:Person {id: $x})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: $y}) \
                 RETURN a.name, c.name, b.name";
    for (x, y) in [
        (Value::Int64(1), Value::Null),
        (Value::Null, Value::Int64(1)),
        (Value::Null, Value::Null),
    ] {
        let values = parameters(&[("x", x), ("y", y)]);
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            let got = run(&session, Language::Gql, query, &values);
            assert_eq!(got, Ok(Vec::new()), "{name}: {values:?}");
        }
    }
    for y in [
        Value::from("1"),
        Value::from("Gus"),
        Value::Float64(2.0),
        Value::Float64(2.5),
        Value::Bool(true),
        Value::List(vec![Value::Int64(2)].into()),
    ] {
        let values = parameters(&[("x", Value::Int64(1)), ("y", y)]);
        let mut answers = Vec::new();
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            answers.push((name, run(&session, Language::Gql, query, &values)));
        }
        let (_, no_rule) = answers.last().unwrap().clone();
        for (name, got) in &answers {
            // The contract: both succeed with the same rows, or rule off is the one failing.
            if no_rule.is_ok() {
                assert_eq!(got, &no_rule, "{name}: {values:?}");
            }
        }
    }
}

/// The key of the far end comes from the row: per input row, also several rows that seek
/// the same node, none, or a value read from an earlier variable.
#[test]
fn the_key_comes_from_each_input_row() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    let chain = "-[:KNOWS]-(c:Person)-[:KNOWS]-";
    for (name, rules) in rule_sets() {
        session.set_planner_rules(rules);
        for keys in [vec![2], vec![1, 2, 3], vec![2, 2, 99], vec![]] {
            let mut expected = Vec::new();
            for key in &keys {
                for row in model.chain(
                    (Way::Either, Way::Either),
                    &|p| p.key == Some(1),
                    &|p| p.key == Some(*key),
                    true,
                ) {
                    let mut row = row;
                    row.insert(0, format!("Int64({key})"));
                    expected.push(row);
                }
            }
            expected.sort();
            let list: Vec<String> = keys.iter().map(ToString::to_string).collect();
            let query = format!(
                "UNWIND [{}] AS k MATCH (a:Person {{id: 1}}){chain}(b:Person {{id: k}}) \
                 RETURN k, a.name, c.name, b.name",
                list.join(", ")
            );
            let got = run(&session, Language::Gql, &query, &HashMap::new());
            assert_eq!(got.as_ref(), Ok(&expected), "{name}: {query}");
        }
        // From a value of the row's own near end: a and b alike.
        let query = format!(
            "MATCH (a:Person {{id: 3}}){chain}(b:Person) WHERE b.id = a.id RETURN a.name, c.name, b.name"
        );
        let expected = model.chain(
            (Way::Either, Way::Either),
            &|p| p.key == Some(3),
            &|p| p.key == Some(3),
            true,
        );
        let got = run(&session, Language::Gql, &query, &HashMap::new());
        assert_eq!(got.as_ref(), Ok(&expected), "{name}: {query}");
    }
}

/// A label on the far end still counts: Amsterdam has key 1 and no `Person` label.
#[test]
fn the_seek_checks_the_label() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    for (name, rules) in rule_sets() {
        session.set_planner_rules(rules);
        let got = run(
            &session,
            Language::Gql,
            "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b:City {id: 1}) RETURN b.name",
            &HashMap::new(),
        );
        assert_eq!(got, Ok(vec![vec!["Amsterdam".to_string()]]), "{name}");
        let got = run(
            &session,
            Language::Gql,
            "MATCH (a:Person {name: 'Gus'})-[:KNOWS]->(c)-[:KNOWS]->(b:City {id: 3}) RETURN b.name",
            &HashMap::new(),
        );
        assert_eq!(got, Ok(Vec::new()), "{name}");
        // No label on the far end: Alix and Amsterdam both have key 1.
        let got = run(
            &session,
            Language::Gql,
            "MATCH (v:Person {name: 'Vincent'})-[:KNOWS]->(c)-[:KNOWS]->(b {id: 1}) RETURN DISTINCT b.name",
            &HashMap::new(),
        );
        assert_eq!(got, Ok(vec![vec!["Amsterdam".to_string()]]), "{name}");
    }
}

/// OPTIONAL over the pattern keeps the row with nulls when nothing matches, a seek that
/// finds nothing included.
#[test]
fn optional_over_the_pattern_keeps_its_row() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    let query = "MATCH (a:Person {id: $x}) \
                 OPTIONAL MATCH (a)-[:KNOWS]->(c:Person)-[:KNOWS]->(b:Person {id: $y}) \
                 RETURN a.name, c.name, b.name";
    for (x, y) in [(1, 3), (1, 99), (99, 3), (4, 1), (3, 3)] {
        let mut expected = Vec::new();
        for a in model
            .people
            .iter()
            .filter(|p| p.label == "Person" && p.key == Some(x))
        {
            let rows = model.chain(
                (Way::Out, Way::Out),
                &|p| p.name == a.name,
                &|p| p.key == Some(y),
                true,
            );
            if rows.is_empty() {
                expected.push(vec![a.name.clone(), "Null".to_string(), "Null".to_string()]);
            }
            expected.extend(rows);
        }
        expected.sort();
        let values = parameters(&[("x", Value::Int64(x)), ("y", Value::Int64(y))]);
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            let got = run(&session, Language::Gql, query, &values);
            assert_eq!(got.as_ref(), Ok(&expected), "{name}, x {x} y {y}");
        }
    }
}

/// The count of the issue, distinct middle nodes, and a count of paths.
#[test]
fn counts_over_the_pattern_agree() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    for (x, y) in [(1, 2), (3, 3), (1, 1), (1, 99), (2, 2)] {
        let rows = model.chain(
            (Way::Either, Way::Either),
            &|p| p.key == Some(x),
            &|p| p.key == Some(y),
            true,
        );
        let mut middle: Vec<&String> = rows.iter().map(|row| &row[1]).collect();
        middle.sort();
        middle.dedup();
        let values = parameters(&[("x", Value::Int64(x)), ("y", Value::Int64(y))]);
        let pattern = "MATCH (a:Person {id: $x})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: $y})";
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            let got = run(
                &session,
                Language::Gql,
                &format!("{pattern} RETURN count(DISTINCT c.name)"),
                &values,
            );
            let want = vec![vec![format!("Int64({})", middle.len())]];
            assert_eq!(got.as_ref(), Ok(&want), "{name}, distinct, x {x} y {y}");
            let got = run(
                &session,
                Language::Gql,
                &format!("{pattern} RETURN count(*)"),
                &values,
            );
            let want = vec![vec![format!("Int64({})", rows.len())]];
            assert_eq!(got.as_ref(), Ok(&want), "{name}, count, x {x} y {y}");
        }
    }
}

/// The columns and rows of a `RETURN *` and of set operations over the pattern.
#[test]
fn every_column_and_set_operation_over_the_pattern() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    let pattern = "MATCH (a:Person {id: 1})-[:KNOWS]->(c:Person)-[:KNOWS]->(b:Person {id: 3})";
    let other = "MATCH (a:Person {id: 2})-[:KNOWS]->(c:Person)-[:KNOWS]->(b:Person {id: 1})";
    let result = |session: &Session, query: &str| {
        session
            .execute(query)
            .map(|result| {
                let mut rows: Vec<String> =
                    result.rows().iter().map(|row| format!("{row:?}")).collect();
                rows.sort();
                (result.columns.clone(), rows)
            })
            .map_err(|error| error.to_string())
    };
    for query in [
        format!("{pattern} RETURN *"),
        format!("{pattern} RETURN a.name AS n UNION {other} RETURN a.name AS n"),
        format!("{pattern} RETURN a.name AS n UNION ALL {other} RETURN a.name AS n"),
        format!("{pattern} RETURN c.name AS n EXCEPT {other} RETURN c.name AS n"),
        format!("{pattern} RETURN c.name AS n INTERSECT {pattern} RETURN c.name AS n"),
        // One name, two meanings: the key of one branch stays in its branch.
        format!(
            "{pattern} RETURN b.name AS n UNION ALL MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name AS n"
        ),
        format!(
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name AS n UNION ALL {pattern} RETURN b.name AS n"
        ),
        format!("{pattern} RETURN c.name ORDER BY c.name LIMIT 1"),
        format!("{pattern} WITH b MATCH (x)-[:KNOWS]->(b) RETURN x.name"),
    ] {
        let mut answers = Vec::new();
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            answers.push((name, result(&session, &query)));
        }
        let (_, plain) = answers.last().unwrap().clone();
        assert!(plain.is_ok(), "{query}: {plain:?}");
        for (name, got) in &answers {
            assert_eq!(got, &plain, "{name}: {query}");
        }
    }
}

/// Keys of every type against stored values of every type: an index that misses a value `=`
/// accepts drops a row. With the rule on and off the rows are the same; ISO GA04 decides what
/// they are, and that is not this test's to say.
#[test]
fn keys_of_other_types_find_what_equals_finds() {
    let stored = [
        "2",
        "2.0",
        "'2'",
        "true",
        "-0.0",
        "0",
        "9007199254740993",
        "9007199254740992.0",
        "[2]",
        "{x: 2}",
        "DATE '2024-03-05'",
        "DATETIME '2024-03-05T00:00:00Z'",
    ];
    let keys = [
        Value::Int64(2),
        Value::Float64(2.0),
        Value::from("2"),
        Value::Bool(true),
        Value::Int64(0),
        Value::Float64(0.0),
        Value::Float64(-0.0),
        Value::Float64(f64::NAN),
        Value::Float64(f64::INFINITY),
        Value::Int64(9_007_199_254_740_993),
        Value::Float64(9_007_199_254_740_992.0),
        Value::List(vec![Value::Int64(2)].into()),
        Value::List(vec![Value::Float64(2.0)].into()),
        Value::Null,
    ];
    let db = GrafeoDB::with_config(in_memory()).unwrap();
    db.create_property_index("id").unwrap();
    let mut session = db.session();
    session.execute("INSERT (:Hub {name: 'Alix'})").unwrap();
    for (index, value) in stored.iter().enumerate() {
        session
            .execute(&format!(
                "MATCH (h:Hub) INSERT (h)-[:KNOWS]->(:Person {{name: 'n{index}', id: {value}}})"
            ))
            .unwrap();
    }
    let query = "MATCH (h:Hub)-[:KNOWS]->(b:Person {id: $y}) RETURN b.name";
    let mut literals = Vec::new();
    for value in stored {
        literals.push(format!(
            "MATCH (h:Hub)-[:KNOWS]->(b:Person {{id: {value}}}) RETURN b.name"
        ));
    }
    // A rule may answer where the plain plan raises, not the other way: only an answer of the
    // plain plan is compared, and most keys must give one, or this compares nothing.
    let mut compared = 0;
    for key in keys {
        let values = parameters(&[("y", key.clone())]);
        let mut answers = Vec::new();
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            answers.push((name, run(&session, Language::Gql, query, &values)));
        }
        let (_, plain) = answers.last().unwrap().clone();
        for (name, got) in &answers {
            if plain.is_ok() {
                assert_eq!(got, &plain, "{name}: key {key:?}");
                compared += 1;
            }
        }
    }
    assert!(compared > 0, "the plain plan raised for every key");
    for literal in &literals {
        let mut answers = Vec::new();
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            answers.push((name, run(&session, Language::Gql, literal, &HashMap::new())));
        }
        let (_, plain) = answers.last().unwrap().clone();
        assert!(plain.is_ok(), "the plain plan raised for {literal}");
        for (name, got) in &answers {
            assert_eq!(got, &plain, "{name}: {literal}");
        }
    }
    // An id of another type: the key must not find more or fewer than `=` does.
    let id_query = "MATCH (h:Hub)-[:KNOWS]->(b:Person) WHERE id(b) = $y RETURN b.name";
    let some_id = session
        .execute("MATCH (b {name: 'n0'}) RETURN id(b)")
        .unwrap()
        .rows()[0][0]
        .as_int64()
        .unwrap();
    for key in [
        Value::Int64(some_id),
        Value::Float64(some_id as f64),
        Value::from(some_id.to_string().as_str()),
        Value::Int64(-1),
        Value::Int64(1 << 40),
        Value::Float64(0.5),
        Value::Bool(true),
        Value::Null,
    ] {
        let values = parameters(&[("y", key.clone())]);
        let mut answers = Vec::new();
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            answers.push((name, run(&session, Language::Gql, id_query, &values)));
        }
        let (_, plain) = answers.last().unwrap().clone();
        for (name, got) in &answers {
            if plain.is_ok() {
                assert_eq!(got, &plain, "{name}: id key {key:?}");
            }
        }
    }
}

/// What a whole database holds, to compare after a statement that writes.
fn dump(session: &Session) -> Vec<String> {
    let mut lines = Vec::new();
    for query in [
        "MATCH (n) RETURN n.name, n.id, labels(n)",
        "MATCH (a)-[e]->(b) RETURN a.name, a.id, type(e), b.name, b.id",
    ] {
        let result = session.execute(query).unwrap();
        lines.extend(result.rows().iter().map(|row| format!("{row:?}")));
    }
    lines.sort();
    lines
}

/// A statement that writes over the pattern: the key is found once, and a later row may
/// have changed what it finds. Each rule set runs it on a database of its own.
#[test]
fn a_statement_that_writes_over_the_pattern_leaves_the_same_graph() {
    let model = people();
    for statement in [
        // Changes the key the hop finds, row by row.
        "MATCH (a:Person)-[:KNOWS]->(b:Person {id: 3}) SET b.id = 2 RETURN a.name, b.name",
        "MATCH (a:Person)-[:KNOWS]->(b:Person {id: 2}) SET a.id = 2 RETURN a.name, b.name",
        "MATCH (a:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(b:Person {id: 3}) SET c.id = 3 RETURN a.name, c.name, b.name",
        // Makes a new node of the key, or a new edge to it, for every row.
        "MATCH (a:Person)-[:KNOWS]->(b:Person {id: 3}) INSERT (:Person {name: 'Marcus', id: 3}) RETURN a.name",
        "MATCH (a:Person)-[:KNOWS]->(b:Person {id: 3}) INSERT (a)-[:KNOWS]->(b) RETURN a.name, b.name",
        "MATCH (a:Person)-[:KNOWS]->(b:Person {id: 3}) INSERT (b)-[:KNOWS]->(:Person {name: 'Lucas', id: 3}) RETURN a.name",
        // Removes what it reads.
        "MATCH (a:Person)-[e:KNOWS]->(b:Person {id: 3}) DELETE e RETURN a.name, b.name",
        "MATCH (a:Person)-[:KNOWS]->(b:Person {id: 3}) DETACH DELETE b RETURN a.name",
        "MATCH (a:Person)-[:KNOWS]->(b:Person {id: 3}) DETACH DELETE a RETURN b.name",
        // Over a hop that closes a cycle.
        "MATCH (a:Person)-[:KNOWS]->(b)-[e:KNOWS]->(a) DELETE e RETURN a.name, b.name",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(a) INSERT (a)-[:KNOWS]->(b) RETURN a.name, b.name",
        // Row by row through UNWIND, each reading what the last wrote.
        "UNWIND [1, 2, 3] AS i MATCH (a:Person {id: 1})-[:KNOWS]->(b:Person {id: 3}) SET b.id = 3 + i RETURN i, b.name",
        "UNWIND [1, 2] AS i MATCH (a:Person {id: 1})-[:KNOWS]->(b:Person {id: 3}) INSERT (a)-[:KNOWS]->(:Person {name: 'Harm', id: 3}) RETURN i, b.name",
        // A write below the hop.
        "MATCH (s:Person {id: 1}) INSERT (s)-[:KNOWS]->(:Person {name: 'Maxence', id: 3}) WITH s MATCH (s)-[:KNOWS]->(b:Person {id: 3}) RETURN b.name",
        "MATCH (s:Person {id: 1}) SET s.id = 3 WITH s MATCH (x:Person)-[:KNOWS]->(b:Person {id: 3}) RETURN x.name, b.name",
        "MATCH (s:Person {id: 1})-[:KNOWS]->(t) SET t.id = 3 WITH s MATCH (s)-[:KNOWS]->(b:Person {id: 3}) RETURN b.name",
        "MATCH (s:Person {id: 1}) INSERT (s)-[:KNOWS]->(s) WITH s MATCH (s)-[:KNOWS]->(b)-[:KNOWS]->(s) RETURN b.name",
        // MERGE and a subquery that writes see their own rows: a later row's hop finds
        // what an earlier row made, and a key found once would miss it.
        "UNWIND [1, 2, 3] AS i MATCH (a:Person {name: 'Alix'}) OPTIONAL MATCH (a)-[:KNOWS]->(b:Person {id: 77}) MERGE (a)-[:KNOWS]->(:Person {id: 77}) RETURN i, b.id",
        "UNWIND [1, 2, 3] AS i MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {id: 2}) MERGE (a)-[:KNOWS]->(:Person {id: 2}) RETURN i, b.name",
        "UNWIND [1, 2, 3] AS i MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {id: 2}) CALL { WITH a INSERT (a)-[:KNOWS]->(:Person {name: 'Harm', id: 2}) } RETURN i, b.name",
        "UNWIND [1, 2, 3] AS i MATCH (a:Person {name: 'Alix'}) OPTIONAL MATCH (a)-[:KNOWS]->(b:Person {id: 77}) CALL { WITH a INSERT (a)-[:KNOWS]->(:Person {name: 'Harm', id: 77}) } RETURN i, b.name",
        "UNWIND [1, 2, 3] AS i MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {id: 2}) CALL { WITH b SET b.id = 5 } RETURN i, b.name",
        // A subquery that writes, per row.
        "MATCH (x:Person) CALL { WITH x INSERT (x)-[:KNOWS]->(:Person {name: 'Hans', id: 3}) } WITH x MATCH (x)-[:KNOWS]->(b:Person {id: 3}) RETURN x.name, b.name",
        "MATCH (x:Person {id: 1}) MERGE (x)-[:KNOWS]->(m:Person {id: 3}) RETURN m.name",
    ] {
        let mut answers = Vec::new();
        for (name, rules) in rule_sets() {
            let (db, _) = database(&model, in_memory());
            let mut session = db.session();
            session.set_planner_rules(rules);
            let rows = session
                .execute(statement)
                .map(|result| {
                    let mut rows: Vec<String> =
                        result.rows().iter().map(|row| format!("{row:?}")).collect();
                    rows.sort();
                    rows
                })
                .map_err(|error| error.to_string());
            answers.push((name, rows, dump(&session)));
        }
        let (_, rows, graph) = answers.last().unwrap().clone();
        for (name, got_rows, got_graph) in &answers {
            if rows.is_ok() {
                assert_eq!(got_rows, &rows, "{name}, rows: {statement}");
                assert_eq!(got_graph, &graph, "{name}, graph: {statement}");
            }
        }
    }
}

/// An expand that runs again per row: inside a subquery, with nothing found and found.
#[test]
fn the_hop_runs_again_for_each_row_of_a_subquery() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    for query in [
        "MATCH (x:Person) CALL { WITH x MATCH (x)-[:KNOWS]->(b:Person {id: 3}) RETURN b.name AS n } RETURN x.name, n",
        "MATCH (x:Person) CALL { WITH x MATCH (x)-[:KNOWS]->(b:Person {id: 99}) RETURN b.name AS n } RETURN x.name, n",
        "MATCH (x:Person) CALL { MATCH (y:Person)-[:KNOWS]->(b:Person {id: 3}) RETURN count(*) AS n } RETURN x.name, n",
        "MATCH (x:Person) OPTIONAL MATCH (x)-[:KNOWS]->(b:Person {id: 3}) RETURN x.name, b.name",
        "MATCH (x:Person) OPTIONAL MATCH (x)-[:KNOWS]->(b:Person {id: 99}) RETURN x.name, b.name",
        "MATCH (x:Person) WHERE EXISTS { MATCH (x)-[:KNOWS]->(b:Person {id: 3}) } RETURN x.name",
        "MATCH (x:Person) WHERE NOT EXISTS { MATCH (x)-[:KNOWS]->(c)-[:KNOWS]->(b:Person {id: 3}) } RETURN x.name",
        "MATCH (x:Person) RETURN x.name, COUNT { MATCH (x)-[:KNOWS]->(c)-[:KNOWS]->(b:Person {id: 3}) } AS n",
        "UNWIND [1, 3, 1, 99, 3] AS k MATCH (x:Person {id: k})-[:KNOWS]->(b:Person {id: 3}) RETURN k, x.name, b.name",
        "MATCH (b:Person {id: 3}) MATCH (x)-[:KNOWS]->(b) RETURN x.name",
    ] {
        let mut answers = Vec::new();
        for (name, rules) in rule_sets() {
            session.set_planner_rules(rules);
            answers.push((name, run(&session, Language::Gql, query, &HashMap::new())));
        }
        let (_, plain) = answers.last().unwrap().clone();
        assert!(plain.is_ok(), "{query}: {plain:?}");
        for (name, got) in &answers {
            assert_eq!(got, &plain, "{name}: {query}");
        }
    }
}

/// A plan cached with an index must not outlive it, and the rows stay the same.
#[test]
fn dropping_the_index_keeps_the_rows() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let session = db.session();
    let query = "MATCH (a:Person {id: $x})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: $y}) \
                 RETURN a.name, c.name, b.name";
    let values = parameters(&[("x", Value::Int64(1)), ("y", Value::Int64(3))]);
    let before = run(&session, Language::Gql, query, &values).unwrap();
    assert_ne!(before.len(), 0);
    db.drop_property_index("id").unwrap();
    assert_eq!(
        run(&session, Language::Gql, query, &values).unwrap(),
        before
    );
    assert!(!explain(&session, query).contains("expand to key"));
    db.create_property_index("id").unwrap();
    assert_eq!(
        run(&session, Language::Gql, query, &values).unwrap(),
        before
    );
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

fn explain(session: &Session, query: &str) -> String {
    let result = session.execute(&format!("EXPLAIN {query}")).unwrap();
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => format!("{other:?}"),
    }
}

/// `[expand to key: ...]` on the check of `b`, with the expand to `b` below it.
fn assert_seeks_the_far_end(plan: &str, key: &str, query: &str) {
    let lines: Vec<&str> = plan.lines().collect();
    let hint = lines
        .iter()
        .position(|line| line.contains(&format!("[expand to key: {key}]")));
    let hop = lines
        .iter()
        .position(|line| line.contains("Expand (") && line.contains("(b)"));
    assert!(
        hint.is_some_and(|hint| hop.is_some_and(|hop| hint < hop)),
        "no key on the hop to b: {query}\n{plan}"
    );
}

const PINNED: [&str; 8] = [
    // A named path over single hops binds them all: rows and columns do not change.
    "MATCH p = (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: 3}) RETURN length(p)",
    "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: 3}) RETURN c.name",
    "MATCH (a:Person {id: 1})-[:KNOWS]->(c:Person)-[:KNOWS]->(b:Person {id: 3}) RETURN c.name",
    "MATCH (a:Person {id: 1})<-[:KNOWS]-(c:Person)<-[:KNOWS]-(b:Person {id: 3}) RETURN c.name",
    "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person) WHERE b.id = 3 RETURN c.name",
    "MATCH (a:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: 3}) WHERE a.name = 'Alix' RETURN c.name",
    "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(d:Person)-[:KNOWS]-(b:Person {id: 3}) RETURN c.name",
    "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: 3}) RETURN count(DISTINCT c.name)",
];

#[test]
fn the_plan_seeks_the_far_end_and_closes_on_it() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let usual = db.session();
    let mut off = db.session();
    off.set_planner_rules(PlannerRules::all().with(PlannerRule::PinnedFarEnd, false));
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    for query in PINNED {
        assert_seeks_the_far_end(&explain(&usual, query), "id", query);
        assert!(!explain(&off, query).contains("expand to key"), "{query}");
        assert!(!explain(&plain, query).contains("expand to key"), "{query}");
    }
}

#[test]
fn an_id_pin_shows_the_same_shape() {
    let model = people();
    let (db, ids) = database(&model, in_memory());
    let usual = db.session();
    let (a, b) = (ids["Alix"].as_u64(), ids["Vincent"].as_u64());
    let query = format!(
        "MATCH (a:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person) WHERE id(a) = {a} AND id(b) = {b} RETURN c.name"
    );
    assert_seeks_the_far_end(&explain(&usual, &query), "id()", &query);
}

/// With no index, a property pin looks nothing up: the plain plan, whatever the rule says.
#[test]
fn without_an_index_the_rule_stays_off() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (a:Person {id: 1})-[:KNOWS]->(c:Person {id: 2}), (c)-[:KNOWS]->(:Person {id: 3})",
        )
        .unwrap();
    let mut off = db.session();
    off.set_planner_rules(PlannerRules::all().with(PlannerRule::PinnedFarEnd, false));
    for query in PINNED {
        assert_eq!(explain(&session, query), explain(&off, query), "{query}");
    }
}

/// Shapes the rule must leave alone.
#[test]
fn the_far_end_is_not_pinned() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let usual = db.session();
    let mut off = db.session();
    off.set_planner_rules(PlannerRules::all().with(PlannerRule::PinnedFarEnd, false));
    for query in [
        // Not one key.
        "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person) WHERE b.id > 3 RETURN c.name",
        "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person) WHERE b.id <> 3 RETURN c.name",
        "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person) WHERE b.id = 3 OR b.id = 2 RETURN c.name",
        "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person) WHERE NOT b.id = 3 RETURN c.name",
        // A property with no index.
        "MATCH (a:Person {id: 1})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {name: 'Gus'}) RETURN c.name",
        // A quantified hop has no one edge to look up.
        "MATCH (a:Person {id: 1})-[:KNOWS]->{1,2}(b:Person {id: 3}) RETURN b.name",
        // Writes below the hop.
        "MATCH (s:Person {id: 1}) INSERT (s)-[:KNOWS]->(:Person {id: 3}) WITH s MATCH (s)-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: 3}) RETURN c.name",
    ] {
        assert_eq!(explain(&usual, query), explain(&off, query), "{query}");
    }
}

/// What it is for: a middle node with a thousand others is looked into, not listed, and a
/// key nobody has walks nothing.
#[test]
fn the_seek_reads_little() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id").unwrap();
    let session = db.session();
    session
        .execute(
            "INSERT (a:Person {id: 1})-[:KNOWS]->(h:Person {id: 2}),
                    (a)-[:KNOWS]->(i:Person {id: 3}),
                    (h)-[:KNOWS]->(b:Person {id: 9}),
                    (i)-[:KNOWS]->(b)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (h:Person) WHERE h.id = 2 OR h.id = 3
             UNWIND range(1, 2000) AS n INSERT (h)-[:KNOWS]->(:Person {id: 1000})",
        )
        .unwrap();
    let mut off = db.session();
    off.set_planner_rules(PlannerRules::all().with(PlannerRule::PinnedFarEnd, false));
    let query = "MATCH (a:Person {id: 1})-[:KNOWS]->(c)-[:KNOWS]->(b:Person {id: $y}) RETURN b.id";
    let read = |session: &Session, y: i64| {
        let values = parameters(&[("y", Value::Int64(y))]);
        let result = session.execute_with_params(query, values).unwrap();
        (result.row_count(), result.rows_scanned().unwrap())
    };
    let (rows, listed) = read(&off, 9);
    assert_eq!(rows, 2);
    assert!(listed >= 4000, "the list read {listed}");
    let (rows, looked_up) = read(&session, 9);
    assert_eq!(rows, 2);
    assert!(looked_up < 200, "the lookup read {looked_up}");
    let (rows, nobody) = read(&session, 12345);
    assert_eq!(rows, 0);
    assert!(nobody < 5, "a key nobody has read {nobody}");
}

/// A key many nodes hold is not read node by node: one row, three edges, 5,000 holders. The
/// reads stay near those of the plain plan, and the rows are the same.
#[test]
fn a_key_many_nodes_hold_lists_as_usual() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id").unwrap();
    let start = db.create_node(&["Start"]).unwrap();
    let mut holders = Vec::new();
    for key in 0..5_000_i64 {
        let holder = db
            .create_node_with_props(
                &["Person"],
                [("id", Value::Int64(7)), ("k", Value::Int64(key))],
            )
            .unwrap();
        holders.push(holder);
    }
    for key in [10, 20, 30] {
        db.create_edge(start, holders[key], "KNOWS").unwrap();
    }
    let usual = db.session();
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::all().with(PlannerRule::PinnedFarEnd, false));
    let query = "MATCH (a:Start)-[:KNOWS]->(b:Person {id: 7}) RETURN b.k ORDER BY b.k";
    let (on, off) = (usual.execute(query).unwrap(), plain.execute(query).unwrap());
    assert_eq!(on.rows(), off.rows());
    assert_eq!(on.row_count(), 3);
    let (on_reads, off_reads) = (on.rows_scanned().unwrap(), off.rows_scanned().unwrap());
    assert!(on_reads <= off_reads + 20, "{on_reads} against {off_reads}");
    assert!(on_reads < 1_024, "read {on_reads}, nearly every holder");
}

/// An integer key finds a stored float, and a float key a stored integer: `=` says they are
/// equal (ISO/IEC 39075 19.3), so an index that misses one loses a row.
#[test]
fn a_number_key_finds_the_other_kind_of_number() {
    let db = GrafeoDB::with_config(in_memory()).unwrap();
    db.create_property_index("id").unwrap();
    let session = db.session();
    session
        .execute(
            "INSERT (h:Hub)-[:KNOWS]->(:Person {name: 'Alix', id: 2}),
                    (h)-[:KNOWS]->(:Person {name: 'Gus', id: 2.0}),
                    (h)-[:KNOWS]->(:Person {name: 'Vincent', id: 3}),
                    (h)-[:KNOWS]->(:Person {name: 'Jules', id: 2.5})",
        )
        .unwrap();
    let mut session = session;
    let both = vec![vec!["Alix".to_string()], vec!["Gus".to_string()]];
    let query = "MATCH (:Hub)-[:KNOWS]->(b:Person {id: $y}) RETURN b.name";
    for (name, rules) in rule_sets() {
        session.set_planner_rules(rules);
        for key in [Value::Int64(2), Value::Float64(2.0)] {
            let values = parameters(&[("y", key.clone())]);
            let got = run(&session, Language::Gql, query, &values);
            assert_eq!(got.as_ref(), Ok(&both), "{name}: key {key:?}");
        }
        for (literal, expected) in [("2", &both), ("2.0", &both)] {
            let query =
                format!("MATCH (:Hub)-[:KNOWS]->(b:Person {{id: {literal}}}) RETURN b.name");
            let got = run(&session, Language::Gql, &query, &HashMap::new());
            assert_eq!(got.as_ref(), Ok(expected), "{name}: literal {literal}");
        }
        let values = parameters(&[("y", Value::Float64(2.5))]);
        let got = run(&session, Language::Gql, query, &values);
        assert_eq!(got, Ok(vec![vec!["Jules".to_string()]]), "{name}");
    }
}

/// A subquery runs its hop again for each row: with a key found and with one nobody holds,
/// the rows are the list's, and that key reads nothing more on the later runs.
#[test]
fn a_rerun_keeps_what_the_key_found() {
    let model = people();
    let (db, _) = database(&model, in_memory());
    let mut session = db.session();
    let query = |key: i64| {
        format!(
            "MATCH (x:Person) CALL {{ WITH x MATCH (x)-[:KNOWS]->(b:Person {{id: {key}}}) \
             RETURN b.name AS n }} RETURN x.name, n"
        )
    };
    let expected = |key: i64| -> Vec<Vec<String>> {
        let mut rows = Vec::new();
        for x in model.people.iter().filter(|p| p.label == "Person") {
            for (_, b) in model.hops(&x.name, Way::Out) {
                let far = model.person(b);
                if far.label == "Person" && far.key == Some(key) {
                    rows.push(vec![x.name.clone(), b.to_string()]);
                }
            }
        }
        rows.sort();
        rows
    };
    for (name, rules) in rule_sets() {
        session.set_planner_rules(rules);
        for key in [3, 2, 99] {
            let got = run(&session, Language::Gql, &query(key), &HashMap::new());
            assert_eq!(got.as_ref(), Ok(&expected(key)), "{name}, key {key}");
        }
    }
    session.set_planner_rules(PlannerRules::all());
    let nobody = session.execute(&query(99)).unwrap();
    let people_count = model.people.len() as u64;
    assert!(
        nobody.rows_scanned().unwrap() <= people_count + 2,
        "read {:?}",
        nobody.rows_scanned()
    );
}
