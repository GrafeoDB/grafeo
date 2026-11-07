//! Under SQL/PGQ a comparison across kinds raises 22G04, and a path's `WHERE` applies to the
//! paths the pattern matches: a far node no edge reaches is compared with nothing. A plan that
//! starts at the far end (the far-end turn, through an index or a scan) evaluates the predicate
//! on every far node first, so under SQL/PGQ no far-end turn is taken, and planning never raises
//! on stored values (`EXPLAIN` of a valid statement answers). GQL and Cypher keep their plans.
//!
//! Every row compares the rows with those of `PlannerRules::plain()` and with the rule off, and
//! the rows come before the plan, so a wrong answer is reported as one.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sql_pgq_far_end_turn_guard::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

/// Many `:Start` nodes with a far node each, so a turn has a clear winner; Alix, Gus and Vincent
/// reach the far node `k: 7`; one far node holds a string and no edge reaches it.
const GRAPH: &[&str] = &[
    "UNWIND range(1, 5000) AS i INSERT (a:Start {s: i + 1000}), (f:Far {k: i + 1000}), (a)-[:R]->(f)",
    "INSERT (a:Start {s: 1, name: 'Alix'}), (g:Start {s: 2, name: 'Gus'}), \
     (v:Start {s: 3, name: 'Vincent'}), (m:Far {k: 7}), (a)-[:R]->(m), (g)-[:R]->(m), (v)-[:R]->(m)",
    "INSERT (:Far {k: 'x'})",
];

const INDEX: &str = "CREATE INDEX by_far FOR (n:Far) ON (n.k)";

fn database(index: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in GRAPH.iter().copied().chain(index.then_some(INDEX)) {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    db
}

/// The sessions every row is run under: all rules, the turn's rule off, and every rule off.
fn sessions(db: &GrafeoDB) -> [(&'static str, Session); 3] {
    let mut rule_off = db.session();
    rule_off.set_planner_rules(PlannerRules::all().with(PlannerRule::SoughtEnd, false));
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    [
        ("all rules", db.session()),
        ("rule off", rule_off),
        ("plain", plain),
    ]
}

/// A language the pattern is written in.
#[derive(Clone, Copy, Debug)]
enum Language {
    Sql,
    Gql,
    Cypher,
}

fn run(session: &Session, language: Language, query: &str) -> Result<QueryResult> {
    match language {
        Language::Sql => session.execute_sql(query),
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    }
}

/// The pattern with the far end's filter `far`, in `language`, returning the source's name.
fn query(language: Language, far: &str) -> String {
    match language {
        Language::Sql => format!(
            "SELECT a FROM GRAPH_TABLE (MATCH (a:Start)-[:R]->(m:Far) WHERE {far} \
             COLUMNS (a.name AS a))"
        ),
        Language::Gql | Language::Cypher => {
            format!("MATCH (a:Start)-[:R]->(m:Far) WHERE {far} RETURN a.name AS a")
        }
    }
}

/// The rows as text, sorted: no order is owed without `ORDER BY`.
fn rows(result: &QueryResult) -> Vec<String> {
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    rows.sort();
    rows
}

/// The plan `EXPLAIN` gives, or why there is none: the error it raised, or that it ran the
/// statement instead (its columns are the statement's, not `plan`).
fn plan(session: &Session, language: Language, query: &str) -> std::result::Result<String, String> {
    let result =
        run(session, language, &format!("EXPLAIN {query}")).map_err(|error| error.to_string())?;
    if result.columns != ["plan"] {
        return Err(format!(
            "EXPLAIN ran the statement (columns {:?})",
            result.columns
        ));
    }
    Ok(result.rows()[0][0].as_str().unwrap_or_default().to_string())
}

/// Whether the plan starts at the far end `m` and walks back to the source.
fn starts_at_the_far_end(plan: &str) -> bool {
    plan.contains("Expand (m)")
}

fn status(error: &Error) -> String {
    GqlStatus::from(error).as_str().to_string()
}

const NAMES: [&str; 3] = ["Alix", "Gus", "Vincent"];

/// The probe: the string far node is reached by no edge, so the equality meets only `k: 7`.
/// With and without the index, under every rule set, the three names, and `EXPLAIN` answers.
#[test]
fn an_unreached_far_node_of_another_kind_is_compared_with_nothing() {
    let mut wrong = Vec::new();
    for index in [true, false] {
        let db = database(index);
        let text = query(Language::Sql, "m.k = 7");
        for (name, session) in sessions(&db) {
            match run(&session, Language::Sql, &text) {
                Ok(result) if rows(&result) == NAMES => {}
                Ok(result) => wrong.push(format!("index {index}, {name}: {:?}", rows(&result))),
                Err(error) => wrong.push(format!("index {index}, {name}: {error}")),
            }
            if let Err(error) = plan(&session, Language::Sql, &text) {
                wrong.push(format!("index {index}, {name}, EXPLAIN: {error}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The control: once an edge reaches the string far node, the equality meets it and SQL/PGQ
/// raises 22G04 whatever the plan.
#[test]
fn a_reached_far_node_of_another_kind_raises() {
    for index in [true, false] {
        let db = database(index);
        db.execute("MATCH (a:Start {name: 'Alix'}), (x:Far {k: 'x'}) INSERT (a)-[:R]->(x)")
            .unwrap();
        let text = query(Language::Sql, "m.k = 7");
        for (name, session) in sessions(&db) {
            let error = run(&session, Language::Sql, &text)
                .map(|result| rows(&result))
                .expect_err(&format!(
                    "index {index}, {name}: the reached 'x' is compared"
                ));
            assert_eq!(status(&error), "22G04", "index {index}, {name}: {error}");
        }
    }
}

/// What a statement gives: its rows, or the status it raised.
fn outcome(session: &Session, query: &str) -> std::result::Result<Vec<String>, String> {
    run(session, Language::Sql, query)
        .map(|result| rows(&result))
        .map_err(|error| status(&error))
}

/// Every far-end form, with and without an index: under SQL/PGQ the outcome is the plain
/// plan's, which compares only the far nodes a hop reaches; no plan starts at the far end, and
/// `EXPLAIN` answers. The integer forms meet `k: 7` only and give the three names. A string
/// literal meets the reached integer keys of the crowd, so `m.k = 'x'` raises 22G04 under every
/// plan, and so does `IN` with a string, as SQL's `IN` is `= ANY` (ISO/IEC 9075-2 8.4). A null in
/// the list is unknown, not another kind: it raises nothing.
#[test]
fn every_far_end_form_answers_as_the_plain_plan() {
    let names = || Ok(NAMES.map(String::from).to_vec());
    let raises = || Err("22G04".to_string());
    let forms: [(&str, std::result::Result<Vec<String>, String>); 9] = [
        ("m.k = 7", names()),
        ("m.k IN [7, 8]", names()),
        ("m.k >= 7 AND m.k < 8", names()),
        ("m.k <= 7", names()),
        ("m.k IN [7, null]", names()),
        ("m.k IN [null]", Ok(Vec::new())),
        ("m.k = 'x'", raises()),
        ("m.k IN ['x', 'y']", raises()),
        ("m.k IN ['x', null]", raises()),
    ];
    let mut wrong = Vec::new();
    for index in [true, false] {
        let db = database(index);
        let [all, rule_off, plain_session] = sessions(&db);
        for (far, want) in &forms {
            let text = query(Language::Sql, far);
            let oracle = outcome(&plain_session.1, &text);
            if &oracle != want {
                wrong.push(format!(
                    "index {index}, plain, {far}: {oracle:?}, want {want:?}"
                ));
            }
            for (name, session) in [&all, &rule_off] {
                let got = outcome(session, &text);
                if got != oracle {
                    wrong.push(format!(
                        "index {index}, {name}, {far}: {got:?}, plain gives {oracle:?}"
                    ));
                }
                match plan(session, Language::Sql, &text) {
                    Ok(plan) if starts_at_the_far_end(&plan) => wrong.push(format!(
                        "index {index}, {name}, {far}: the plan starts at the far end\n{plan}"
                    )),
                    Ok(_) => {}
                    Err(error) => {
                        wrong.push(format!("index {index}, {name}, {far}, EXPLAIN: {error}"));
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// GQL and Cypher keep the indexed turn: a string and a number are unequal there, so reading
/// the far end first is safe, and the plan still starts at `m` through the index.
#[test]
fn gql_and_cypher_keep_the_indexed_turn() {
    let db = database(true);
    for language in [Language::Gql, Language::Cypher] {
        for far in ["m.k = 7", "m.k IN [7, 8]"] {
            let text = query(language, far);
            let got = rows(&run(&db.session(), language, &text).unwrap());
            assert_eq!(got, NAMES, "{language:?} {far}");
            let plan = plan(&db.session(), language, &text).unwrap();
            assert!(
                starts_at_the_far_end(&plan) && plan.contains("[index: k]"),
                "{language:?} {far}: the plan starts at the far end\n{plan}"
            );
        }
    }
}

/// Every rule set the class runs under: all rules, every rule off, and each rule turned off
/// from all and on from none, so each key read form (an index lookup, an expand to the key, the
/// typed column read, `IN`, a range) is reached with and without the rule that makes it.
fn every_rule_set() -> Vec<(String, PlannerRules)> {
    let mut sets = vec![
        ("all rules".to_string(), PlannerRules::all()),
        ("plain".to_string(), PlannerRules::plain()),
    ];
    for rule in PlannerRule::ALL {
        sets.push((
            format!("all but {rule:?}"),
            PlannerRules::all().with(rule, false),
        ));
        sets.push((
            format!("only {rule:?}"),
            PlannerRules::plain().with(rule, true),
        ));
    }
    sets
}

/// The class over key reads: a value of another kind that the statement reaches raises 22G04
/// exactly when the plain scan does, and an unreached one never does, for every key read form
/// and every rule set, with and without an index; `EXPLAIN` plans every statement without
/// raising (planning compares with the order that does not raise).
#[test]
fn every_key_read_raises_exactly_when_the_scan_does() {
    let filters = ["m.k = 7", "m.k IN [7, 8]", "m.k >= 7 AND m.k < 8"];
    // The far node alone (every Far node is in scope, the string one too) and the hop.
    let shapes = [
        (
            "the far node alone",
            "SELECT k FROM GRAPH_TABLE (MATCH (m:Far) WHERE {far} COLUMNS (m.k AS k))",
        ),
        (
            "the hop",
            "SELECT a FROM GRAPH_TABLE (MATCH (a:Start)-[:R]->(m:Far) WHERE {far} COLUMNS (a.name AS a))",
        ),
    ];
    let mut wrong = Vec::new();
    for index in [true, false] {
        for reached in [false, true] {
            let db = database(index);
            if reached {
                db.execute("MATCH (a:Start {name: 'Alix'}), (x:Far {k: 'x'}) INSERT (a)-[:R]->(x)")
                    .unwrap();
            }
            for (shape, template) in shapes {
                for far in filters {
                    let text = template.replace("{far}", far);
                    let mut plain = db.session();
                    plain.set_planner_rules(PlannerRules::plain());
                    let oracle = outcome(&plain, &text);
                    // The string node is in scope for the far node alone, and for the hop once
                    // an edge reaches it.
                    let in_scope = reached || shape == "the far node alone";
                    if in_scope != (oracle == Err("22G04".to_string())) {
                        wrong.push(format!(
                            "index {index}, reached {reached}, {shape}, {far}: plain gives {oracle:?}"
                        ));
                    }
                    for (rules, set) in every_rule_set() {
                        let mut session = db.session();
                        session.set_planner_rules(set);
                        let got = outcome(&session, &text);
                        if got != oracle {
                            wrong.push(format!(
                                "index {index}, reached {reached}, {shape}, {far}, {rules}: {got:?}, plain gives {oracle:?}"
                            ));
                        }
                        if let Err(error) = plan(&session, Language::Sql, &text) {
                            wrong.push(format!(
                                "index {index}, reached {reached}, {shape}, {far}, {rules}, EXPLAIN: {error}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// An `IN` list holding a value of another kind raises 22G04 under SQL/PGQ whatever the order of
/// its items: a match earlier in the list does not end the comparison before the later items'
/// kinds are seen. For a short list (compared item by item) and a list of 16 or more items (a
/// set), with and without a null item, with the key indexed or not, under every rule set, for a
/// property and for a literal on the left. Controls: lists of one kind answer, a null item alone
/// is unknown.
#[test]
fn an_in_list_mixing_kinds_raises_in_either_order() {
    let ints = |from: i64| (from..from + 16).map(|i| i.to_string()).collect::<Vec<_>>();
    let set = |items: Vec<String>| format!("[{}]", items.join(", "));
    let with_string_last = set([ints(1), vec!["'a'".to_string()]].concat());
    let with_string_first = set([vec!["'a'".to_string()], ints(1)].concat());
    let with_string_and_null = set([ints(1), vec!["null".to_string(), "'a'".to_string()]].concat());
    let mixed: Vec<String> = vec![
        "[1, 'a']".to_string(),
        "['a', 1]".to_string(),
        "[1, null, 'a']".to_string(),
        "['a', null, 1]".to_string(),
        "[null, 1, 'a']".to_string(),
        with_string_last,
        with_string_first,
        with_string_and_null,
    ];
    let answers: Vec<(String, bool)> = vec![
        ("[1, 2]".to_string(), true),
        ("[2, null]".to_string(), false),
        (set(ints(1)), true),
        (set([ints(2), vec!["null".to_string()]].concat()), false),
    ];
    let mut wrong = Vec::new();
    for index in [false, true] {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:One {k: 1, name: 'Alix'})").unwrap();
        if index {
            db.execute("CREATE INDEX by_one FOR (n:One) ON (n.k)")
                .unwrap();
        }
        for left in ["n.k", "1"] {
            let text = |list: &str| {
                format!(
                    "SELECT * FROM GRAPH_TABLE (MATCH (n:One) WHERE {left} IN {list} \
                     COLUMNS (n.name AS a))"
                )
            };
            for (name, session) in sessions(&db) {
                for list in &mixed {
                    let query = text(list);
                    match outcome(&session, &query) {
                        Err(status) if status == "22G04" => {}
                        other => wrong.push(format!(
                            "index {index}, {name}, {query}: {other:?}, want 22G04"
                        )),
                    }
                }
                for (list, found) in &answers {
                    let query = text(list);
                    let want = if *found {
                        vec!["Alix".to_string()]
                    } else {
                        Vec::new()
                    };
                    match outcome(&session, &query) {
                        Ok(got) if got == want => {}
                        other => wrong.push(format!(
                            "index {index}, {name}, {query}: {other:?}, want {want:?}"
                        )),
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
