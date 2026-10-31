//! A parenthesized quantified path pattern (#678, item 2; ISO/IEC 39075:2024
//! 16.7): its `WHERE` holds for each repetition, reading that repetition's
//! nodes and edges, and a pattern of more than one edge repeats as a whole,
//! within its bounds.
//!
//! The expected rows come from a brute-force enumeration of the walks
//! (`oracle`), in each direction and path mode, with the same condition per
//! step.

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// A person: name, `x` and `limit`.
const PEOPLE: [(&str, i64, i64); 5] = [
    ("Alix", 1, 2),
    ("Gus", 2, 0),
    ("Vincent", 3, 4),
    ("Mia", 0, 8),
    ("Jules", 5, 1),
];

/// `:R` edges: source, target, `w`. A cycle Alix, Vincent, Mia, a branch to
/// Jules, and two ways from Alix to Vincent.
const EDGES: [(&str, &str, i64); 6] = [
    ("Alix", "Gus", 3),
    ("Gus", "Vincent", 1),
    ("Vincent", "Mia", 5),
    ("Alix", "Vincent", 7),
    ("Mia", "Alix", 9),
    ("Vincent", "Jules", 2),
];

fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (name, x, limit) in PEOPLE {
        db.execute(&format!(
            "INSERT (:P {{name: '{name}', x: {x}, limit: {limit}}})"
        ))
        .unwrap();
    }
    for (from, to, w) in EDGES {
        db.execute(&format!(
            "MATCH (a:P {{name: '{from}'}}), (b:P {{name: '{to}'}}) INSERT (a)-[:R {{w: {w}}}]->(b)"
        ))
        .unwrap();
    }
    db
}

fn person(name: &str) -> (i64, i64) {
    let (_, x, limit) = PEOPLE.iter().find(|(n, _, _)| *n == name).unwrap();
    (*x, *limit)
}

#[derive(Clone, Copy)]
enum Direction {
    Out,
    In,
    Both,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Walk,
    Trail,
    Acyclic,
}

/// One step of a path: the step's first node `a`, its edge (index into
/// `EDGES`) and its last node `b`. On `<-` the step's `a` is the edge's
/// target.
#[derive(Clone, Copy)]
struct Step {
    a: &'static str,
    edge: usize,
    b: &'static str,
}

/// Every path from `start` of `min..=max` steps in `direction` and `mode`
/// whose every step satisfies `keep`: (the last node, the `w` of each edge,
/// the names of the nodes), sorted.
fn oracle(
    start: &'static str,
    direction: Direction,
    (min, max): (usize, usize),
    mode: Mode,
    keep: &dyn Fn(&Step) -> bool,
) -> Vec<(String, Vec<i64>, Vec<String>)> {
    fn steps_from(node: &'static str, direction: Direction) -> Vec<Step> {
        let mut steps = Vec::new();
        for (edge, (from, to, _)) in EDGES.iter().enumerate() {
            if matches!(direction, Direction::Out | Direction::Both) && *from == node {
                steps.push(Step {
                    a: node,
                    edge,
                    b: to,
                });
            }
            if matches!(direction, Direction::In | Direction::Both) && *to == node {
                steps.push(Step {
                    a: node,
                    edge,
                    b: from,
                });
            }
        }
        steps
    }
    let mut found = Vec::new();
    let mut open: Vec<Vec<Step>> = vec![Vec::new()];
    while let Some(path) = open.pop() {
        let end = path.last().map_or(start, |step| step.b);
        if path.len() >= min {
            found.push((
                end.to_string(),
                path.iter().map(|step| EDGES[step.edge].2).collect(),
                std::iter::once(start)
                    .chain(path.iter().map(|step| step.b))
                    .map(ToString::to_string)
                    .collect(),
            ));
        }
        if path.len() == max {
            continue;
        }
        for step in steps_from(end, direction) {
            if !keep(&step) {
                continue;
            }
            let repeats = match mode {
                Mode::Walk => false,
                Mode::Trail => path.iter().any(|earlier| earlier.edge == step.edge),
                Mode::Acyclic => step.b == start || path.iter().any(|earlier| earlier.b == step.b),
            };
            if !repeats {
                let mut longer = path.clone();
                longer.push(step);
                open.push(longer);
            }
        }
    }
    found.sort();
    found
}

/// A value as text: strings bare, numbers, lists in brackets.
fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(text).collect::<Vec<_>>().join(", ")
        ),
        other => format!("{other:?}"),
    }
}

/// The rows of the GQL `query` as text, sorted.
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = db
        .execute(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect())
        .collect();
    rows.sort();
    rows
}

/// The oracle's paths as the rows of `RETURN t.name, [x IN e | x.w]`.
fn as_rows(paths: &[(String, Vec<i64>, Vec<String>)]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = paths
        .iter()
        .map(|(end, weights, _)| {
            vec![
                end.clone(),
                text(&Value::List(
                    weights
                        .iter()
                        .map(|w| Value::Int64(*w))
                        .collect::<Vec<_>>()
                        .into(),
                )),
            ]
        })
        .collect();
    rows.sort();
    rows
}

const ARROWS: [(&str, Direction); 3] = [
    ("(a)-[e:R]->(b)", Direction::Out),
    ("(a)<-[e:R]-(b)", Direction::In),
    ("(a)-[e:R]-(b)", Direction::Both),
];

/// The step conditions: as GQL, and as the oracle checks them.
fn conditions() -> Vec<(&'static str, Box<dyn Fn(&Step) -> bool>)> {
    vec![
        ("e.w > 2", Box::new(|step: &Step| EDGES[step.edge].2 > 2)),
        (
            "a.x < b.x",
            Box::new(|step: &Step| person(step.a).0 < person(step.b).0),
        ),
        (
            "e.w > 2 AND a.x <> b.x",
            Box::new(|step: &Step| EDGES[step.edge].2 > 2 && person(step.a).0 != person(step.b).0),
        ),
    ]
}

/// The control: without a condition the quantified pattern already matches
/// the enumeration, in every direction and mode, so the oracle reads
/// directions, undirected edges and modes as the engine does.
#[test]
fn without_a_condition_the_pattern_matches_the_enumeration() {
    let db = graph();
    for (arrow, direction) in ARROWS {
        for (keyword, mode) in [
            ("", Mode::Walk),
            ("TRAIL", Mode::Trail),
            ("ACYCLIC", Mode::Acyclic),
        ] {
            for (start, ..) in PEOPLE {
                let query = format!(
                    "MATCH {keyword} (s:P {{name: '{start}'}})({arrow}){{1,3}}(t) \
                     RETURN t.name, [x IN e | x.w]"
                );
                let expected = as_rows(&oracle(start, direction, (1, 3), mode, &|_: &Step| true));
                assert_eq!(rows(&db, &query), expected, "{query}");
            }
        }
    }
}

/// The WHERE of the quantified pattern holds for each step, in every
/// direction, from every start: an edge condition, a node condition (a is
/// the step's first node, b its last) and both.
#[test]
fn a_step_condition_holds_for_each_step_in_every_direction() {
    let db = graph();
    for (arrow, direction) in ARROWS {
        for (condition, keep) in conditions() {
            for (start, ..) in PEOPLE {
                let query = format!(
                    "MATCH (s:P {{name: '{start}'}})({arrow} WHERE {condition}){{1,3}}(t) \
                     RETURN t.name, [x IN e | x.w]"
                );
                let expected = as_rows(&oracle(start, direction, (1, 3), Mode::Walk, &*keep));
                assert_eq!(rows(&db, &query), expected, "{query}");
            }
        }
    }
}

/// The same with the path named: its nodes are the steps' nodes in order,
/// its length the number of steps.
#[test]
fn a_named_path_holds_the_steps_the_condition_kept() {
    let db = graph();
    for (arrow, direction) in ARROWS {
        let query = format!(
            "MATCH p = (s:P {{name: 'Alix'}})({arrow} WHERE e.w > 2){{1,3}}(t) \
             RETURN t.name, length(p), [n IN nodes(p) | n.name]"
        );
        let mut expected: Vec<Vec<String>> =
            oracle("Alix", direction, (1, 3), Mode::Walk, &|step: &Step| {
                EDGES[step.edge].2 > 2
            })
            .into_iter()
            .map(|(end, weights, nodes)| {
                vec![
                    end,
                    weights.len().to_string(),
                    format!("[{}]", nodes.join(", ")),
                ]
            })
            .collect();
        expected.sort();
        assert_eq!(rows(&db, &query), expected, "{query}");
    }
}

/// TRAIL and ACYCLIC, inside the quantified pattern and in front of the
/// whole one, keep only the paths of their mode whose every step holds. The
/// cycle Alix, Vincent, Mia tells them apart from WALK.
#[test]
fn trail_and_acyclic_steps_hold_the_condition_and_the_mode() {
    let db = graph();
    for (keyword, mode) in [("TRAIL", Mode::Trail), ("ACYCLIC", Mode::Acyclic)] {
        for (arrow, direction) in ARROWS {
            let expected = as_rows(&oracle("Alix", direction, (1, 4), mode, &|step: &Step| {
                EDGES[step.edge].2 > 2
            }));
            for query in [
                format!(
                    "MATCH (s:P {{name: 'Alix'}})({keyword} {arrow} WHERE e.w > 2){{1,4}}(t) \
                     RETURN t.name, [x IN e | x.w]"
                ),
                format!(
                    "MATCH {keyword} (s:P {{name: 'Alix'}})({arrow} WHERE e.w > 2){{1,4}}(t) \
                     RETURN t.name, [x IN e | x.w]"
                ),
            ] {
                assert_eq!(rows(&db, &query), expected, "{query}");
            }
        }
    }
    // WALK goes round the cycle; TRAIL and ACYCLIC do not, so they differ.
    let walk = oracle("Alix", Direction::Out, (1, 4), Mode::Walk, &|_: &Step| true);
    let trail = oracle("Alix", Direction::Out, (1, 4), Mode::Trail, &|_: &Step| {
        true
    });
    let acyclic = oracle(
        "Alix",
        Direction::Out,
        (1, 4),
        Mode::Acyclic,
        &|_: &Step| true,
    );
    assert!(walk.len() > trail.len() && trail.len() > acyclic.len());
}

/// A condition may read a variable bound outside the quantified pattern:
/// each start's own `limit`.
#[test]
fn a_step_condition_reads_an_outer_variable() {
    let db = graph();
    let query = "MATCH (s:P)((a)-[e:R]->(b) WHERE e.w > s.limit){1,3}(t) \
                 RETURN s.name, t.name, [x IN e | x.w]";
    let mut expected: Vec<Vec<String>> = Vec::new();
    for (start, _, limit) in PEOPLE {
        for row in as_rows(&oracle(
            start,
            Direction::Out,
            (1, 3),
            Mode::Walk,
            &|step: &Step| EDGES[step.edge].2 > limit,
        )) {
            expected.push(std::iter::once(start.to_string()).chain(row).collect());
        }
    }
    expected.sort();
    assert_eq!(rows(&db, query), expected, "{query}");
}

/// A condition that fails on the first step leaves no path of one or more
/// steps; `{0,2}` still has the path of no steps.
#[test]
fn a_condition_that_fails_on_the_first_step_leaves_only_the_empty_path() {
    let db = graph();
    // Gus's only edge has w 1.
    assert_eq!(
        rows(
            &db,
            "MATCH (s:P {name: 'Gus'})((a)-[e:R]->(b) WHERE e.w > 2){1,3}(t) RETURN t.name"
        ),
        Vec::<Vec<String>>::new()
    );
    assert_eq!(
        rows(
            &db,
            "MATCH (s:P {name: 'Gus'})((a)-[e:R]->(b) WHERE e.w > 2){0,2}(t) \
             RETURN t.name, [x IN e | x.w]"
        ),
        vec![vec!["Gus".to_string(), "[]".to_string()]]
    );
}

// ---------------------------------------------------------------------------
// A quantified pattern of more than one edge repeats as a whole
// ---------------------------------------------------------------------------

/// The two-edge segments from `start`, repeated `min..=max` times: (the last
/// node, the `w` of the first edges, the `w` of the second edges), as rows.
fn segments(
    start: &'static str,
    (min, max): (usize, usize),
    keep: &dyn Fn(&Step, &Step) -> bool,
) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    // Every walk of 2 * repetitions edges, checked per repetition.
    fn walks(node: &'static str, left: usize, path: &mut Vec<Step>, out: &mut Vec<Vec<Step>>) {
        if left == 0 {
            out.push(path.clone());
            return;
        }
        for (edge, (from, to, _)) in EDGES.iter().enumerate() {
            if *from == node {
                path.push(Step {
                    a: from,
                    edge,
                    b: to,
                });
                walks(to, left - 1, path, out);
                path.pop();
            }
        }
    }
    for repetitions in min..=max {
        let mut all = Vec::new();
        walks(start, 2 * repetitions, &mut Vec::new(), &mut all);
        for walk in all {
            if walk.chunks(2).all(|pair| keep(&pair[0], &pair[1])) {
                let w = |steps: Vec<&Step>| {
                    text(&Value::List(
                        steps
                            .iter()
                            .map(|step| Value::Int64(EDGES[step.edge].2))
                            .collect::<Vec<_>>()
                            .into(),
                    ))
                };
                rows.push(vec![
                    walk.last().map_or(start, |step| step.b).to_string(),
                    w(walk.iter().step_by(2).collect()),
                    w(walk.iter().skip(1).step_by(2).collect()),
                ]);
            }
        }
    }
    rows.sort();
    rows
}

#[test]
fn a_two_edge_pattern_repeats_within_its_bounds() {
    let db = graph();
    for (bounds, (min, max)) in [("{1,3}", (1, 3)), ("{2,2}", (2, 2)), ("{0,1}", (0, 1))] {
        for (start, ..) in PEOPLE {
            let query = format!(
                "MATCH (s:P {{name: '{start}'}})((a)-[e1:R]->(m)-[e2:R]->(b)){bounds}(t) \
                 RETURN t.name, [x IN e1 | x.w], [x IN e2 | x.w]"
            );
            assert_eq!(
                rows(&db, &query),
                segments(start, (min, max), &|_, _| true),
                "{query}"
            );
        }
    }
}

/// The WHERE of a two-edge pattern holds for each repetition.
#[test]
fn a_two_edge_pattern_checks_its_condition_per_repetition() {
    let db = graph();
    for (start, ..) in PEOPLE {
        let query = format!(
            "MATCH (s:P {{name: '{start}'}})((a)-[e1:R]->(m)-[e2:R]->(b) WHERE e1.w < e2.w \
             AND a.x <> b.x){{1,3}}(t) RETURN t.name, [x IN e1 | x.w], [x IN e2 | x.w]"
        );
        let expected = segments(start, (1, 3), &|first, second| {
            EDGES[first.edge].2 < EDGES[second.edge].2 && person(first.a).0 != person(second.b).0
        });
        assert_eq!(rows(&db, &query), expected, "{query}");
    }
}

/// A two-edge pattern without an upper bound is refused with an error that
/// names the bounded form, not run once.
#[test]
fn an_unbounded_multi_edge_pattern_is_refused() {
    let db = graph();
    for quantifier in ["{1,}", "+", "*"] {
        let query = format!(
            "MATCH (s:P {{name: 'Alix'}})((a)-[:R]->(m)-[:R]->(b)){quantifier}(t) RETURN t.name"
        );
        let error = db
            .execute(&query)
            .expect_err(&format!("{query}: an unbounded multi-edge pattern"));
        assert_eq!(
            error.error_code().as_str(),
            "GRAFEO-Q004",
            "{query}: {error}"
        );
        assert!(
            error.to_string().contains("bound"),
            "{query}: the message names the bounded form: {error}"
        );
    }
}

/// The repetitions a multi-edge pattern unrolls to are capped: past the cap
/// is refused with an error that names it, at the cap the pattern runs.
#[test]
fn a_multi_edge_pattern_past_the_repetition_cap_is_refused() {
    // A chain long enough that every repetition count has its own row.
    let db = GrafeoDB::new_in_memory();
    db.execute("FOR i IN range(0, 400) INSERT (:N {i: i})")
        .unwrap();
    db.execute("MATCH (a:N), (b:N) WHERE b.i = a.i + 1 INSERT (a)-[:NEXT]->(b)")
        .unwrap();
    let query = |max: u64| {
        format!("MATCH (s:N {{i: 0}})((a)-[:NEXT]->(m)-[:NEXT]->(b)){{1,{max}}}(t) RETURN count(*)")
    };
    let error = db
        .execute(&query(1_000_000))
        .expect_err("a million repetitions are past the cap");
    let message = error.to_string();
    // The message names the cap: "at most N repetitions".
    let cap: u64 = message
        .split("at most ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|number| number.parse().ok())
        .unwrap_or_else(|| panic!("the message names the cap: {message}"));
    assert!(cap >= 1, "{message}");
    let result = db
        .execute(&query(cap))
        .unwrap_or_else(|e| panic!("{} repetitions, the cap: {e}", cap));
    // Up to 200 repetitions fit in the chain of 400 edges.
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(i64::try_from(cap.min(200)).unwrap()),
        "{cap} repetitions"
    );
    assert!(db.execute(&query(cap + 1)).is_err(), "one past the cap");
}
