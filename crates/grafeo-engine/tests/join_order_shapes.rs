//! The join order of comma patterns (#714, `PlannerRule::JoinOrder`): which plan each shape
//! gets, and that the written tree stands wherever the order is not free to change.
//!
//! Rows are checked here against the plan with the rule off, and against a brute-force
//! enumerator in `planner_rules_results.rs`.

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few thousand nodes"
)]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

fn text(result: &grafeo_engine::database::QueryResult) -> String {
    let mut out = String::new();
    for row in result.rows() {
        if let Some(Value::String(line)) = row.first() {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn explain(session: &Session, query: &str) -> String {
    text(&session.execute(&format!("EXPLAIN {query}")).unwrap())
}

fn explain_cypher(session: &Session, query: &str) -> String {
    text(&session.execute_cypher(&format!("EXPLAIN {query}")).unwrap())
}

fn sorted_rows(result: &grafeo_engine::database::QueryResult) -> Vec<String> {
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

fn written(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::JoinOrder, false));
    session
}

/// How many lines of a plan hold `word`.
fn count(plan: &str, word: &str) -> usize {
    plan.lines().filter(|line| line.contains(word)).count()
}

/// People with interests in tags, posts with tags and creators: the graph of a benchmark that
/// took the written order into a hash join of 30 million rows.
fn social() -> GrafeoDB {
    // Written plans of two patterns are cross products: a slow build must not make them a test of
    // the query timeout, while a bad plan still fails by name, well inside the job's own limit.
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_query_timeout(std::time::Duration::from_secs(600)),
    )
    .unwrap();
    let tags: Vec<NodeId> = (0..20)
        .map(|i| {
            db.create_node_with_props(
                &["Tag"],
                [("name", Value::from(format!("tag{i:02}").as_str()))],
            )
            .unwrap()
        })
        .collect();
    let people: Vec<NodeId> = (0..1000_i64)
        .map(|i| {
            db.create_node_with_props(
                &["Person"],
                [("k", Value::Int64(i)), ("age", Value::Int64(20 + i % 50))],
            )
            .unwrap()
        })
        .collect();
    for (i, person) in people.iter().enumerate() {
        for step in 0..15 {
            db.create_edge(*person, tags[(i + step) % 20], "HAS_INTEREST")
                .unwrap();
        }
    }
    for i in 0..1000_usize {
        let post = db.create_node(&["Post"]).unwrap();
        db.create_edge(post, tags[i % 20], "HAS_TAG").unwrap();
        db.create_edge(post, people[i], "HAS_CREATOR").unwrap();
    }
    db
}

const PATTERN_JOIN: &str = "MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), \
     (m)-[:HAS_CREATOR]->(p) RETURN t.name, count(*)";
const PATTERN_JOIN_SQL: &str = "SELECT name, COUNT(*) AS c FROM GRAPH_TABLE (\
     MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), (m)-[:HAS_CREATOR]->(p) \
     COLUMNS (t.name AS name)) GROUP BY name";

/// Starts at one side, expands, and closes by a lookup: no hash join.
#[test]
fn a_cycle_of_comma_patterns_expands_and_closes_by_a_lookup() {
    let db = social();
    let usual = db.session();
    let plan = explain(&usual, PATTERN_JOIN);
    assert_eq!(count(&plan, "Join ("), 0, "{plan}");
    assert!(count(&plan, "Expand (") >= 3, "{plan}");
    assert!(
        plan.contains("[expand into:") || plan.contains("[IntersectExpand into:"),
        "{plan}"
    );
}

/// The SQL/PGQ twin answers as its written tree does. Its plan cannot be read here (its `EXPLAIN`
/// runs the query), and reads across plan shapes say nothing.
#[test]
fn the_sql_twin_answers_as_the_written_tree() {
    let db = social();
    let usual = db.session();
    let off = written(&db);
    let (on_result, off_result) = (
        usual.execute_sql(PATTERN_JOIN_SQL).unwrap(),
        off.execute_sql(PATTERN_JOIN_SQL).unwrap(),
    );
    assert_eq!(sorted_rows(&on_result), sorted_rows(&off_result));
    assert_eq!(on_result.row_count(), 20);
}

/// And the rows are the rows of the written tree, without its hash joins. Reads compare only
/// plans of one shape: an intersection counts every probe and a hash join neither its build nor
/// its output, so the plan that reads three times as much here runs a thousand times faster on
/// medium SNB (5 ms against 7,505 ms).
#[test]
fn the_cycle_answers_as_the_written_tree_without_its_joins() {
    let db = social();
    let usual = db.session();
    let off = written(&db);
    let (on_result, off_result) = (
        usual.execute(PATTERN_JOIN).unwrap(),
        off.execute(PATTERN_JOIN).unwrap(),
    );
    assert_eq!(sorted_rows(&on_result), sorted_rows(&off_result));
    assert_eq!(on_result.row_count(), 20);
    let plan = explain(&usual, PATTERN_JOIN);
    assert_eq!(count(&plan, "Join (Inner)"), 0, "{plan}");
}

/// Off, the written tree stands: the join of the patterns as they are written.
#[test]
fn turning_the_rule_off_keeps_the_written_tree() {
    let db = social();
    let off = written(&db);
    let plan = explain(&off, PATTERN_JOIN);
    assert_eq!(count(&plan, "Join (Inner)"), 2, "{plan}");
    assert_eq!(plan, explain(&off, PATTERN_JOIN));
    assert_ne!(plan, explain(&db.session(), PATTERN_JOIN));
}

/// Few rows on each side and a large list in the middle: expanding through the middle reads
/// all of it, a hash join of the two sides reads almost nothing.
#[test]
fn two_selective_chains_over_a_large_middle_join_by_hash() {
    let db = GrafeoDB::new_in_memory();
    let middle: Vec<NodeId> = (0..4)
        .map(|_| db.create_node(&["Middle"]).unwrap())
        .collect();
    for hub in &middle {
        for _ in 0..3_000 {
            let other = db.create_node(&["Other"]).unwrap();
            db.create_edge(other, *hub, "TO").unwrap();
        }
    }
    for i in 0..5 {
        let left = db.create_node(&["Left"]).unwrap();
        db.create_edge(left, middle[i % 4], "TO").unwrap();
        let right = db.create_node(&["Right"]).unwrap();
        db.create_edge(right, middle[i % 4], "TO").unwrap();
    }
    let query = "MATCH (l:Left)-[:TO]->(m:Middle), (r:Right)-[:TO]->(m) RETURN id(l), id(r)";
    let usual = db.session();
    let plan = explain(&usual, query);
    assert_eq!(count(&plan, "Join (Inner)"), 1, "{plan}");
    let result = usual.execute(query).unwrap();
    assert_eq!(
        sorted_rows(&result),
        sorted_rows(&written(&db).execute(query).unwrap())
    );
    assert!(
        result.rows_scanned().unwrap() < 200,
        "{:?}",
        result.rows_scanned()
    );
    // With `(m)` unlabelled no count or sample sees the hubs. Rows only, and what it reads is
    // reported: whether it must hash too is the architect's to say.
    let query = "MATCH (l:Left)-[:TO]->(m), (r:Right)-[:TO]->(m) RETURN id(l), id(r)";
    let result = usual.execute(query).unwrap();
    assert_eq!(
        sorted_rows(&result),
        sorted_rows(&written(&db).execute(query).unwrap())
    );
    eprintln!(
        "unlabelled middle reads {:?}:\n{}",
        result.rows_scanned(),
        explain(&usual, query)
    );
}

/// A Cypher comma pattern binds each edge once: reordering keeps the checks that its hops of
/// one type are different edges, above the whole tree.
#[test]
fn cypher_keeps_its_edge_checks_above_the_reordered_tree() {
    let db = social();
    let usual = db.session();
    let off = written(&db);
    let query = "MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (q:Person)-[:HAS_INTEREST]->(t), \
                 (m:Post)-[:HAS_CREATOR]->(p), (m)-[:HAS_TAG]->(t) WHERE p.k < 40 AND q.k < 40 \
                 RETURN count(*)";
    let plan = explain_cypher(&usual, query);
    let check = plan
        .lines()
        .position(|line| line.contains("Filter") && line.contains("Ne"))
        .unwrap_or_else(|| panic!("no edge check:\n{plan}"));
    let first_step = plan
        .lines()
        .position(|line| line.contains("Expand (") || line.contains("Join ("))
        .unwrap();
    assert!(check < first_step, "{plan}");
    assert_eq!(
        sorted_rows(&usual.execute_cypher(query).unwrap()),
        sorted_rows(&off.execute_cypher(query).unwrap()),
    );
}

/// A filter that can raise stays with its relation, once, above the relation's label check, so
/// it overtakes no check and reads what the written plan reads at most.
#[test]
fn a_raising_filter_stays_with_its_relation() {
    let db = social();
    let usual = db.session();
    let query = "MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), \
                 (m)-[:HAS_CREATOR]->(p) WHERE 100 / p.age > 0 RETURN count(*)";
    let plan = explain(&usual, query);
    assert_eq!(count(&plan, "Div"), 1, "the filter runs once:\n{plan}");
    let division = plan
        .lines()
        .position(|line| line.contains("Div"))
        .expect(&plan);
    let label = plan
        .lines()
        .position(|line| line.contains("hasLabel(p, \"Person\")"));
    if let Some(label) = label {
        assert!(
            division < label,
            "it runs before the label it was written under:\n{plan}"
        );
    }
    assert_eq!(
        sorted_rows(&usual.execute(query).unwrap()),
        sorted_rows(&written(&db).execute(query).unwrap()),
    );
    // The check written first still guards it: age 0 on a person in no answer. Rule off may
    // fail on it for now (the contract allows that, and a later FILTER running first is its own
    // bug); rule on must succeed, with the rows the guard leaves.
    let unguarded = usual.execute(query).unwrap();
    db.session()
        .execute("INSERT (:Person {k: 5000, age: 0})")
        .unwrap();
    let guarded = "MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), \
                   (m)-[:HAS_CREATOR]->(p) FILTER p.age <> 0 FILTER 100 / p.age > 0 RETURN count(*)";
    let on = usual
        .execute(guarded)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(sorted_rows(&on), sorted_rows(&unguarded));
    if let Ok(off) = written(&db).execute(guarded) {
        assert_eq!(sorted_rows(&off), sorted_rows(&on));
    }
}

/// Parts that share nothing: the product of their rows, in any order, with each part planned
/// alone and nothing joined across them.
#[test]
fn parts_that_share_nothing_answer_their_product() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (:Tag {k: 1}), (:Tag {k: 2}),
                    (a:Person {k: 1})-[:KNOWS]->(b:Person {k: 2}), (b)-[:KNOWS]->(a),
                    (:Person {k: 3}), (:Post {k: 1}), (:Post {k: 2}), (:Post {k: 3})",
        )
        .unwrap();
    let usual = db.session();
    let off = written(&db);
    for (query, expected) in [
        ("MATCH (t:Tag), (p:Person) RETURN count(*)", 6),
        ("MATCH (p:Person), (t:Tag) RETURN count(*)", 6),
        ("MATCH (t:Tag), (p:Person), (m:Post) RETURN count(*)", 18),
        (
            "MATCH (t:Tag), (a:Person)-[:KNOWS]->(b:Person) RETURN count(*)",
            4,
        ),
        (
            "MATCH (t:Tag), (a:Person)-[:KNOWS]->(b), (b)-[:KNOWS]->(c) RETURN count(*)",
            4,
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->(b), (m:Post), (b)-[:KNOWS]->(a) RETURN count(*)",
            6,
        ),
    ] {
        for (name, session) in [("on", &usual), ("off", &off)] {
            let rows = session.execute(query).unwrap();
            assert_eq!(rows.rows()[0][0], Value::Int64(expected), "{name}: {query}");
        }
    }
}

/// Column order is the query's own where it shows: `RETURN *`, also in a branch of a set
/// operation, keeps the written tree. Named columns are free, in a branch too: rows only.
#[test]
fn every_column_keeps_the_written_tree() {
    let db = social();
    let usual = db.session();
    let off = written(&db);
    let hops = "MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), (m)-[:HAS_CREATOR]->(p)";
    for query in [
        format!("{hops} RETURN *"),
        format!("{hops} RETURN * UNION {hops} RETURN *"),
    ] {
        assert_eq!(explain(&usual, &query), explain(&off, &query), "{query}");
    }
    let small = "MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), (m)-[:HAS_CREATOR]->(p) WHERE p.k < 20";
    for query in [
        format!("{small} RETURN t.name AS n UNION {small} RETURN t.name AS n"),
        format!("{small} RETURN t.name AS n UNION ALL {small} RETURN t.name AS n"),
    ] {
        assert_eq!(
            sorted_rows(&usual.execute(&query).unwrap()),
            sorted_rows(&off.execute(&query).unwrap()),
            "{query}"
        );
    }
}

/// Over 12 relations the written order stands, and planning stays quick.
#[test]
fn past_twelve_relations_the_written_order_stands() {
    let db = social();
    let usual = db.session();
    let off = written(&db);
    let hops = |count: usize| -> String {
        let mut paths = Vec::new();
        for i in 0..count {
            paths.push(format!("(n{i}:Person)-[:HAS_INTEREST]->(t)"));
        }
        format!("MATCH {} RETURN count(*)", paths.join(", "))
    };
    assert_eq!(explain(&usual, &hops(13)), explain(&off, &hops(13)));
    assert_ne!(explain(&usual, &hops(12)), "");
}

/// A condition the search cannot read would be lost from a reordered plan: the written tree
/// stands, and its rows.
#[test]
fn a_condition_the_search_cannot_read_keeps_the_written_tree() {
    let db = social();
    let usual = db.session();
    let off = written(&db);
    for query in [
        "MATCH (p:Person), (q:Person) WHERE p.k + 1 = q.k AND p.k < 5 RETURN p.k, q.k",
        "MATCH (p:Person), (q:Person) WHERE p.k = q.k OR p.k = 1 AND q.k < 2 RETURN count(*)",
    ] {
        assert_eq!(
            sorted_rows(&usual.execute(query).unwrap()),
            sorted_rows(&off.execute(query).unwrap()),
            "{query}"
        );
    }
    let query = "MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(u:Tag) WHERE t.name + 'x' = u.name + 'x' RETURN count(*)";
    assert_eq!(explain(&usual, query), explain(&off, query));
}

/// One plan on every call and for a fresh database of the same data.
#[test]
fn the_plan_is_the_same_on_every_run() {
    let queries = [
        PATTERN_JOIN,
        "MATCH (l:Tag), (p:Person)-[:HAS_INTEREST]->(u), (m:Post)-[:HAS_TAG]->(u) RETURN count(*)",
        "MATCH (p:Person)-[:HAS_INTEREST]->(t), (q:Person)-[:HAS_INTEREST]->(t) RETURN count(*)",
    ];
    let first = social();
    let plans: Vec<String> = queries
        .iter()
        .map(|q| explain(&first.session(), q))
        .collect();
    for _ in 0..3 {
        let db = social();
        for (query, plan) in queries.iter().zip(&plans) {
            assert_eq!(&explain(&db.session(), query), plan, "{query}");
        }
    }
}

/// `a.x = b.y` and `a = b` stay conditions of the join, whatever the order.
#[test]
fn equalities_between_patterns_stay_conditions() {
    let db = social();
    let usual = db.session();
    let off = written(&db);
    for query in [
        "MATCH (p:Person), (q:Person) WHERE p = q AND p.k < 3 RETURN p.k, q.k",
        "MATCH (p:Person), (q:Person) WHERE p.k = q.k AND q.k < 3 RETURN p.k, q.k",
        "MATCH (p:Person)-[:HAS_INTEREST]->(t), (q:Person)-[:HAS_INTEREST]->(u) \
         WHERE p = q AND t = u AND p.k < 3 RETURN p.k, t.name, u.name",
        "MATCH (p:Person)-[:HAS_INTEREST]->(t), (q:Person)-[:HAS_INTEREST]->(u) \
         WHERE p.k = q.k AND t.name = u.name AND p.k < 3 RETURN p.k, t.name",
    ] {
        assert_eq!(
            sorted_rows(&usual.execute(query).unwrap()),
            sorted_rows(&off.execute(query).unwrap()),
            "{query}"
        );
    }
    let rows = usual
        .execute("MATCH (p:Person), (q:Person) WHERE p = q AND p.k < 3 RETURN p.k, q.k")
        .unwrap();
    assert_eq!(rows.row_count(), 3);
}

/// A hop that continues from a node another side binds loses its scan, and with it the scan's
/// label and filters: they must come back as checks, also when it runs against its direction.
#[test]
fn a_hop_keeps_the_label_and_filters_of_its_source() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (a:Person {name: 'Alix', age: 30})-[:K]->(:Item {name: 'b1'}),
                    (a)-[:L]->(:Item {name: 'c1'}),
                    (g:Person {name: 'Gus', age: 40})-[:K]->(:Item {name: 'b2'}),
                    (g)-[:L]->(:Item {name: 'c2'}),
                    (d:City {name: 'Amsterdam', age: 30})-[:K]->(:Item {name: 'b3'}),
                    (d)-[:L]->(:Item {name: 'c3'}),
                    (s:Person {name: 'Vincent'})-[:K]->(t:Tag {name: 't'}),
                    (m:City {name: 'Berlin', age: 1})-[:L]->(t),
                    (n:Person {name: 'Mia', age: 2})-[:L]->(t)",
        )
        .unwrap();
    // Many of the second kind, so that the search has a reason to start there.
    session
        .execute("MATCH (a:Person {name: 'Alix'}) UNWIND range(1, 300) AS i INSERT (a)-[:K]->(:Item {name: 'many'})")
        .unwrap();
    let usual = db.session();
    let off = written(&db);
    let names = |rows: &grafeo_engine::database::QueryResult| {
        let mut out: Vec<String> = rows
            .rows()
            .iter()
            .map(|row| format!("{:?}", row[0]))
            .collect();
        out.sort();
        out.dedup();
        out
    };
    for (query, expected) in [
        // City Amsterdam has both edges and must stay out.
        (
            "MATCH (a:Person)-[:K]->(b), (a)-[:L]->(c) RETURN a.name",
            vec!["String(\"Alix\")", "String(\"Gus\")"],
        ),
        (
            "MATCH (a:Person {age: 40})-[:K]->(b), (a)-[:L]->(c) RETURN a.name",
            vec!["String(\"Gus\")"],
        ),
        (
            "MATCH (a:Person)-[:K]->(b), (a)-[:L]->(c) WHERE a.age = 30 RETURN a.name",
            vec!["String(\"Alix\")"],
        ),
        // Run from t, against the direction: the source's label stays a check.
        (
            "MATCH (x:Person)-[:K]->(t:Tag), (y:City)-[:L]->(t) RETURN y.name",
            vec!["String(\"Berlin\")"],
        ),
        (
            "MATCH (x:Person)-[:K]->(t:Tag), (y {age: 2})-[:L]->(t) RETURN y.name",
            vec!["String(\"Mia\")"],
        ),
    ] {
        let (on, plain) = (usual.execute(query).unwrap(), off.execute(query).unwrap());
        assert_eq!(sorted_rows(&on), sorted_rows(&plain), "{query}");
        assert_eq!(names(&on), expected, "{query}");
    }
}

/// Planning 12 relations, the most the search takes, stays quick. The bound is far above what
/// it should take, to catch a search that clones its way past a few relations.
#[test]
fn planning_twelve_relations_stays_quick() {
    let db = social();
    let usual = db.session();
    let shapes = [
        // A star on one variable.
        (0..12)
            .map(|i| format!("(n{i}:Person)-[:HAS_INTEREST]->(t)"))
            .collect::<Vec<_>>(),
        // A chain.
        (0..12)
            .map(|i| format!("(n{i})-[:HAS_INTEREST]->(n{})", i + 1))
            .collect::<Vec<_>>(),
        // A cycle.
        (0..12)
            .map(|i| format!("(n{i})-[:HAS_INTEREST]->(n{})", (i + 1) % 12))
            .collect::<Vec<_>>(),
    ];
    for paths in shapes {
        let query = format!("MATCH {} RETURN count(*)", paths.join(", "));
        let start = std::time::Instant::now();
        let plan = explain(&usual, &query);
        let took = start.elapsed();
        assert_ne!(plan, "");
        assert!(took.as_secs() < 5, "{took:?} for {query}");
    }
}

/// Reordering may lift a comparison of a property with a value above other filters only if it
/// cannot raise: never, for any two kinds, with any operator, as a literal or a parameter.
#[test]
fn a_property_against_a_value_never_raises() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let literals = [
        "1",
        "1.5",
        "'x'",
        "true",
        "DATE '2024-03-05'",
        "TIME '12:00:00'",
        "DATETIME '2024-03-05T12:00:00Z'",
        "DURATION 'P1D'",
        "[1, 2]",
        "{a: 1}",
        "null",
    ];
    for literal in literals {
        session
            .execute(&format!("INSERT (:T {{p: {literal}}})"))
            .unwrap();
    }
    session.execute("INSERT (:T)").unwrap();
    // A pattern of two parts is what can be reordered: the rows of the plain plan, with its
    // filters where they were written.
    let mut plain = db.session();
    plain.set_planner_rules(grafeo_engine::query::optimizer::PlannerRules::plain());
    for operator in ["=", "<>", "<", "<=", ">", ">="] {
        for literal in literals {
            let query = format!("MATCH (n:T) WHERE n.p {operator} {literal} RETURN count(*)");
            session
                .execute(&query)
                .unwrap_or_else(|error| panic!("{query}: {error}"));
            let two_parts = format!(
                "MATCH (n:T), (m:T) WHERE n.p {operator} {literal} AND m.p = 1 RETURN count(*)"
            );
            let usual = session
                .execute(&two_parts)
                .unwrap_or_else(|error| panic!("{two_parts}: {error}"));
            let written = plain
                .execute(&two_parts)
                .unwrap_or_else(|error| panic!("{two_parts}, plain: {error}"));
            assert_eq!(usual.rows(), written.rows(), "{two_parts}");
        }
        for value in [
            Value::Int64(1),
            Value::Float64(1.5),
            Value::from("x"),
            Value::Bool(true),
            Value::Null,
            Value::List(vec![Value::Int64(1)].into()),
        ] {
            let query = format!("MATCH (n:T) WHERE n.p {operator} $v RETURN count(*)");
            let parameters = std::collections::HashMap::from([("v".to_string(), value.clone())]);
            session
                .execute_with_params(&query, parameters)
                .unwrap_or_else(|error| panic!("{query} with {value:?}: {error}"));
        }
    }
}

/// The graph of the benchmark `pattern_join` runs on, at a tenth of its medium size: people with 1
/// to 5 interests, posts with one creator and 1 to 3 tags, and twice as many comments that share
/// those edge types, so a count of the edges overstates what a post has. On this shape the posts
/// are the cheapest start by rows read (a model of all 12 orders: 1.33 times less than the tags).
fn social_with_comments() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let mut state = 42_u64;
    let mut next = move |bound: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    let tags: Vec<NodeId> = (0..20)
        .map(|i| {
            db.create_node_with_props(
                &["Tag"],
                [("name", Value::from(format!("tag{i:02}").as_str()))],
            )
            .unwrap()
        })
        .collect();
    let people: Vec<NodeId> = (0..1_000_i64)
        .map(|i| {
            db.create_node_with_props(
                &["Person"],
                [("k", Value::Int64(i)), ("age", Value::Int64(20 + i % 50))],
            )
            .unwrap()
        })
        .collect();
    for person in &people {
        for _ in 0..=next(5) {
            db.create_edge(*person, tags[next(20) as usize], "HAS_INTEREST")
                .unwrap();
        }
    }
    for _ in 0..1_000 {
        let post = db.create_node(&["Post"]).unwrap();
        db.create_edge(post, people[next(1_000) as usize], "HAS_CREATOR")
            .unwrap();
        for _ in 0..=next(3) {
            db.create_edge(post, tags[next(20) as usize], "HAS_TAG")
                .unwrap();
        }
    }
    for _ in 0..2_000 {
        let comment = db.create_node(&["Comment"]).unwrap();
        db.create_edge(comment, people[next(1_000) as usize], "HAS_CREATOR")
            .unwrap();
        for _ in 0..next(3) {
            db.create_edge(comment, tags[next(20) as usize], "HAS_TAG")
                .unwrap();
        }
    }
    db
}

/// The to-one edge decides: `pattern_join` starts at the posts (user's sampling decision).
#[test]
fn pattern_join_starts_at_the_to_one_edge() {
    let db = social_with_comments();
    let usual = db.session();
    let plan = explain(&usual, PATTERN_JOIN);
    let deepest = plan
        .lines()
        .rfind(|line| line.contains("NodeScan ("))
        .expect(&plan);
    assert!(deepest.contains("(m:Post)"), "{plan}");
    assert_eq!(
        sorted_rows(&usual.execute(PATTERN_JOIN).unwrap()),
        sorted_rows(&written(&db).execute(PATTERN_JOIN).unwrap()),
    );
}

/// Every HAS_INTEREST target is a Tag: the label check after that hop must not divide by the
/// share of Tags among all nodes, which would make the plan through it look cheap. On a graph
/// where it is the most expensive start, it stays last.
#[test]
fn a_label_check_after_a_typed_hop_does_not_collapse_the_estimate() {
    let db = social_with_comments();
    let usual = db.session();
    let plan = explain(&usual, PATTERN_JOIN);
    let deepest = plan
        .lines()
        .rfind(|line| line.contains("NodeScan ("))
        .expect(&plan);
    assert!(!deepest.contains("(t:Tag)"), "{plan}");
}

/// A query's own variable with the name a closing check would have: it stays bound and its
/// rows stay right, read above the hop, in every plan.
#[test]
fn a_variable_named_like_a_closing_check_keeps_its_rows() {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (a:Person {k: 1})-[:KNOWS]->(b:Person {k: 2}),
                    (b)-[:KNOWS]->(c:Person {k: 3}),
                    (c)-[:KNOWS]->(a),
                    (b)-[:KNOWS]->(a),
                    (a)-[:KNOWS]->(a)",
        )
        .unwrap();
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    let usual = db.session();
    let off = written(&db);
    for name in [
        "_cycle_end_0",
        "_cycle_end_1",
        "_cycle_0",
        "_cycle",
        "_cycle_x",
    ] {
        for query in [
            format!("MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->({name}) RETURN a.k, {name}.k"),
            format!(
                "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->({name}) WHERE {name} = a RETURN a.k, {name}.k"
            ),
            format!(
                "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->({name})-[:KNOWS]->(a) RETURN a.k, {name}.k"
            ),
            format!(
                "MATCH (a)-[:KNOWS]->(b), (b)-[:KNOWS]->({name}), ({name})-[:KNOWS]->(a) RETURN a.k, {name}.k"
            ),
            format!(
                "MATCH (a)-[:KNOWS]->({name}), ({name})-[:KNOWS]->(c), (c)-[:KNOWS]->(a) RETURN a.k, {name}.k, c.k"
            ),
            format!(
                "MATCH ({name})-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->({name}) RETURN {name}.k, b.k"
            ),
            format!(
                "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) MATCH (a)-[:KNOWS]->({name}) RETURN {name}.k"
            ),
            format!(
                "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->({name}) WHERE {name} = a RETURN count({name}) AS n"
            ),
        ] {
            let want = sorted_rows(
                &plain
                    .execute(&query)
                    .unwrap_or_else(|e| panic!("{query}: {e}")),
            );
            for (rules, session) in [("all", &usual), ("all but JoinOrder", &off)] {
                let got = session
                    .execute(&query)
                    .unwrap_or_else(|e| panic!("{rules}: {query}: {e}"));
                assert_eq!(sorted_rows(&got), want, "{rules}: {query}");
            }
        }
    }
}
