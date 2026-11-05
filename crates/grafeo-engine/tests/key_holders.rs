//! Whoever holds a key is found once, whichever way it is looked up (key-lookup design, "Tests every
//! step's lookups need").
//!
//! The nodes that hold `k` are of one label, of two (each returned once, though two indexes hold it),
//! of a label with no index, and of none. Each goes through the labelled form, no label, a union and
//! a hop's far end, under each index, hot and reopened, with the rules on and off. `key_holders.gtest`
//! runs the same table on every binding. The rest of this file holds what the lookups are for: how
//! little they read, and the versions a lookup must still see.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main key_holders::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

const GRAPH: &str = "INSERT (s:Start {name: 'Amsterdam'}), (b:Start {name: 'Berlin'}), (a:Far {name: 'Alix', k: 7}), (g:Near {name: 'Gus', k: 7}), (v:Far:Near {name: 'Vincent', k: 7}), (m {name: 'Mia', k: 7}), (j:Other {name: 'Jules', k: 7}), (c:Far {name: 'Butch', k: 8}), (d:Near {name: 'Django', k: 8}), (h {name: 'Hans', k: 8}), (s)-[:R]->(a), (s)-[:R]->(a), (s)-[:R]->(g), (s)-[:R]->(v), (s)-[:R]->(m), (s)-[:R]->(j), (s)-[:R]->(c), (s)-[:R]->(d), (s)-[:R]->(h), (b)-[:R]->(v), (b)-[:R]->(m), (s)-[:S]->(g)";

/// Name, query, whether Cypher reads it too, and its rows, each its cells joined.
const FORMS: &[(&str, &str, bool, &[&str])] = &[
    (
        "label_far",
        "MATCH (n:Far {k: 7}) RETURN n.name",
        true,
        &["Alix", "Vincent"],
    ),
    (
        "label_near",
        "MATCH (n:Near {k: 7}) RETURN n.name",
        true,
        &["Gus", "Vincent"],
    ),
    (
        "label_other",
        "MATCH (n:Other {k: 7}) RETURN n.name",
        true,
        &["Jules"],
    ),
    (
        "label_far_where",
        "MATCH (n:Far) WHERE n.k = 7 RETURN n.name",
        true,
        &["Alix", "Vincent"],
    ),
    (
        "label_far_in",
        "MATCH (n:Far) WHERE n.k IN [7, 70] RETURN n.name",
        true,
        &["Alix", "Vincent"],
    ),
    (
        "two_labels",
        "MATCH (n:Far:Near {k: 7}) RETURN n.name",
        true,
        &["Vincent"],
    ),
    (
        "no_label",
        "MATCH (n {k: 7}) RETURN n.name",
        true,
        &["Alix", "Gus", "Jules", "Mia", "Vincent"],
    ),
    (
        "no_label_where",
        "MATCH (n) WHERE n.k = 7 RETURN n.name",
        true,
        &["Alix", "Gus", "Jules", "Mia", "Vincent"],
    ),
    (
        "no_label_in",
        "MATCH (n) WHERE n.k IN [7, 70] RETURN n.name",
        true,
        &["Alix", "Gus", "Jules", "Mia", "Vincent"],
    ),
    (
        "union",
        "MATCH (n) WHERE (n:Far OR n:Near) AND n.k = 7 RETURN n.name",
        true,
        &["Alix", "Gus", "Vincent"],
    ),
    (
        "union_expression",
        "MATCH (n:Far|Near {k: 7}) RETURN n.name",
        false,
        &["Alix", "Gus", "Vincent"],
    ),
    (
        "union_with_a_label_of_no_index",
        "MATCH (n) WHERE (n:Far OR n:Other) AND n.k = 7 RETURN n.name",
        true,
        &["Alix", "Jules", "Vincent"],
    ),
    (
        "hop_far_label",
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name, m.name",
        true,
        &[
            "Amsterdam|Alix",
            "Amsterdam|Alix",
            "Amsterdam|Vincent",
            "Berlin|Vincent",
        ],
    ),
    (
        "hop_far_label_where",
        "MATCH (a:Start)-[:R]->(m:Far) WHERE m.k = 7 RETURN a.name, m.name",
        true,
        &[
            "Amsterdam|Alix",
            "Amsterdam|Alix",
            "Amsterdam|Vincent",
            "Berlin|Vincent",
        ],
    ),
    (
        "hop_no_label",
        "MATCH (a:Start)-[:R]->(m {k: 7}) RETURN a.name, m.name",
        true,
        &[
            "Amsterdam|Alix",
            "Amsterdam|Alix",
            "Amsterdam|Gus",
            "Amsterdam|Jules",
            "Amsterdam|Mia",
            "Amsterdam|Vincent",
            "Berlin|Mia",
            "Berlin|Vincent",
        ],
    ),
    (
        "hop_union",
        "MATCH (a:Start)-[:R]->(m) WHERE (m:Far OR m:Near) AND m.k = 7 RETURN a.name, m.name",
        true,
        &[
            "Amsterdam|Alix",
            "Amsterdam|Alix",
            "Amsterdam|Gus",
            "Amsterdam|Vincent",
            "Berlin|Vincent",
        ],
    ),
    (
        "hop_other_label",
        "MATCH (a:Start)-[:R]->(m:Other {k: 7}) RETURN a.name, m.name",
        true,
        &["Amsterdam|Jules"],
    ),
    (
        "hop_written_backwards",
        "MATCH (m:Far {k: 7})<-[:R]-(a:Start) RETURN a.name, m.name",
        true,
        &[
            "Amsterdam|Alix",
            "Amsterdam|Alix",
            "Amsterdam|Vincent",
            "Berlin|Vincent",
        ],
    ),
    (
        "hop_undirected",
        "MATCH (a:Start)-[r]-(m:Near {k: 7}) RETURN a.name, m.name",
        true,
        &[
            "Amsterdam|Gus",
            "Amsterdam|Gus",
            "Amsterdam|Vincent",
            "Berlin|Vincent",
        ],
    ),
    (
        "hop_two_types",
        "MATCH (a:Start)-[:R|S]->(m:Near {k: 7}) RETURN a.name, m.name",
        true,
        &[
            "Amsterdam|Gus",
            "Amsterdam|Gus",
            "Amsterdam|Vincent",
            "Berlin|Vincent",
        ],
    ),
    (
        "hop_one_start",
        "MATCH (a:Start {name: 'Berlin'})-[:R]->(m {k: 7}) RETURN m.name",
        true,
        &["Mia", "Vincent"],
    ),
    (
        "hop_from_rows",
        "UNWIND [7, 7] AS x MATCH (a:Start {name: 'Berlin'})-[:R]->(m:Far {k: x}) RETURN m.name",
        true,
        &["Vincent", "Vincent"],
    ),
];

/// The indexes a lookup must not depend on, beside a key-wide one, which only the API makes.
const INDEXES: &[(&str, &[&str])] = &[
    ("no_index", &[]),
    ("on_far", &["CREATE INDEX on_far FOR (n:Far) ON (n.k)"]),
    ("on_near", &["CREATE INDEX on_near FOR (n:Near) ON (n.k)"]),
    (
        "on_far_near",
        &[
            "CREATE INDEX on_far FOR (n:Far) ON (n.k)",
            "CREATE INDEX on_near FOR (n:Near) ON (n.k)",
        ],
    ),
    (
        "on_every_label",
        &[
            "CREATE INDEX on_far FOR (n:Far) ON (n.k)",
            "CREATE INDEX on_near FOR (n:Near) ON (n.k)",
            "CREATE INDEX on_other FOR (n:Other) ON (n.k)",
        ],
    ),
];

fn cell(value: &Value) -> String {
    match value {
        Value::String(text) => text.as_str().to_string(),
        Value::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}

fn rows_of(session: &Session, cypher: bool, query: &str) -> Vec<String> {
    let result = if cypher {
        session.execute_cypher(query)
    } else {
        session.execute(query)
    }
    .unwrap_or_else(|error| panic!("cypher {cypher}: {query}: {error}"));
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect();
    rows.sort();
    rows
}

/// Every rule set a lookup must answer alike under.
fn rule_sets() -> Vec<(String, PlannerRules)> {
    let mut sets = vec![
        ("all".to_string(), PlannerRules::all()),
        ("plain".to_string(), PlannerRules::plain()),
    ];
    for rule in [
        PlannerRule::IndexLookup,
        PlannerRule::KeyColumnLookup,
        PlannerRule::KeySeek,
        PlannerRule::PinnedFarEnd,
        PlannerRule::SoughtEnd,
    ] {
        sets.push((
            format!("no {rule:?}"),
            PlannerRules::all().with(rule, false),
        ));
    }
    sets
}

fn check_every_form(db: &GrafeoDB, stage: &str) {
    for (name, query, in_cypher, wanted) in FORMS {
        let mut wanted: Vec<String> = wanted.iter().map(ToString::to_string).collect();
        wanted.sort();
        for (rules_name, rules) in rule_sets() {
            let mut session = db.session();
            session.set_planner_rules(rules);
            for cypher in [false, true] {
                if cypher && !in_cypher {
                    continue;
                }
                assert_eq!(
                    rows_of(&session, cypher, query),
                    wanted,
                    "{stage}, {rules_name}, cypher {cypher}: {name}: {query}"
                );
            }
        }
    }
}

fn build(db: &GrafeoDB, statements: &[&str], key_wide: bool) {
    let session = db.session();
    session.execute(GRAPH).unwrap();
    for statement in statements {
        session.execute(statement).unwrap();
    }
    if key_wide {
        db.create_property_index("k").unwrap();
    }
}

#[test]
fn every_form_finds_each_holder_once_hot() {
    for (index, statements) in INDEXES {
        let db = GrafeoDB::new_in_memory();
        build(&db, statements, false);
        check_every_form(&db, index);
    }
    let db = GrafeoDB::new_in_memory();
    build(&db, &[], true);
    check_every_form(&db, "key-wide");
}

#[test]
fn every_form_finds_each_holder_once_reopened() {
    for (index, statements) in INDEXES {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("holders.grafeo");
        {
            let db = GrafeoDB::open(&path).unwrap();
            build(&db, statements, false);
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        check_every_form(&db, index);
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Versions: a lookup sees what its reader sees
// ---------------------------------------------------------------------------

const NO_LABEL: &str = "MATCH (n {k: 7}) RETURN n.name";
const ON_FAR: &str = "MATCH (n:Far {k: 7}) RETURN n.name";
const HOP: &str = "MATCH (a:Start {name: 'Berlin'})-[:R]->(m:Far {k: 7}) RETURN m.name";
const HOP_NO_LABEL: &str = "MATCH (a:Start {name: 'Berlin'})-[:R]->(m {k: 7}) RETURN m.name";

fn names(wanted: &[&str]) -> Vec<String> {
    wanted.iter().map(ToString::to_string).collect()
}

/// A reader of an old snapshot finds what it saw: a label removed and added, a value changed and a
/// node deleted after it began, by each lookup, whatever the indexes.
#[test]
fn an_old_snapshot_finds_what_it_saw() {
    for (index, statements) in INDEXES {
        let db = GrafeoDB::new_in_memory();
        build(&db, statements, false);
        let mut old = db.session();
        old.begin_transaction().unwrap();
        let before = [
            (NO_LABEL, rows_of(&old, false, NO_LABEL)),
            (ON_FAR, rows_of(&old, false, ON_FAR)),
            (HOP, rows_of(&old, false, HOP)),
            (HOP_NO_LABEL, rows_of(&old, false, HOP_NO_LABEL)),
        ];
        let writer = db.session();
        for statement in [
            "MATCH (n {name: 'Vincent'}) REMOVE n:Far",
            "MATCH (n {name: 'Gus'}) SET n:Far",
            "MATCH (n {name: 'Mia'}) SET n.k = 70",
            "MATCH (n {name: 'Jules'}) DETACH DELETE n",
            "INSERT (:Far {name: 'Shosanna', k: 7})",
        ] {
            writer.execute(statement).unwrap();
        }
        for (query, rows) in &before {
            assert_eq!(&rows_of(&old, false, query), rows, "{index}: {query}");
        }
        old.rollback().unwrap();
        assert_eq!(
            rows_of(&db.session(), false, ON_FAR),
            names(&["Alix", "Gus", "Shosanna"]),
            "{index}: now"
        );
        assert_eq!(
            rows_of(&db.session(), false, NO_LABEL),
            names(&["Alix", "Gus", "Shosanna", "Vincent"]),
            "{index}: now, no label"
        );
        assert_eq!(
            rows_of(&db.session(), false, HOP),
            Vec::<String>::new(),
            "{index}"
        );
    }
}

/// A transaction finds its own label and value, and another reader does not; a rollback takes them
/// back.
#[test]
fn a_transaction_finds_its_own_changes() {
    for (index, statements) in INDEXES {
        let db = GrafeoDB::new_in_memory();
        build(&db, statements, false);
        let mut own = db.session();
        own.begin_transaction().unwrap();
        for statement in [
            "MATCH (n {name: 'Hans'}) SET n:Far, n.k = 7",
            "MATCH (n {name: 'Alix'}) REMOVE n:Far",
            "INSERT (:Near:Far {name: 'Shosanna', k: 7})",
        ] {
            own.execute(statement).unwrap();
        }
        assert_eq!(
            rows_of(&own, false, ON_FAR),
            names(&["Hans", "Shosanna", "Vincent"]),
            "{index}: own"
        );
        assert_eq!(
            rows_of(&db.session(), false, ON_FAR),
            names(&["Alix", "Vincent"]),
            "{index}: another reader"
        );
        assert!(
            rows_of(&own, false, NO_LABEL).contains(&"Hans".to_string()),
            "{index}"
        );
        assert!(
            !rows_of(&db.session(), false, NO_LABEL).contains(&"Hans".to_string()),
            "{index}"
        );
        own.rollback().unwrap();
        assert_eq!(
            rows_of(&own, false, ON_FAR),
            names(&["Alix", "Vincent"]),
            "{index}: rollback"
        );
    }
}

// ---------------------------------------------------------------------------
// The no-label lookup: the union of the labels' indexes and a remainder
// ---------------------------------------------------------------------------

fn scanned(session: &Session, query: &str) -> usize {
    usize::try_from(session.execute(query).unwrap().rows_scanned().unwrap()).unwrap()
}

fn plan_of(session: &Session, query: &str) -> String {
    let result = session.execute(&format!("EXPLAIN {query}")).unwrap();
    result.rows()[0][0].as_str().unwrap().to_string()
}

/// Thousands of nodes of each label, all holding `k`.
fn crowd(statements: &[&str]) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "UNWIND range(1, 2000) AS i INSERT (:Far {k: i})",
        "UNWIND range(2001, 4000) AS i INSERT (:Near {k: i})",
        "UNWIND range(4001, 4010) AS i INSERT (:Far:Near {k: i})",
    ] {
        session.execute(statement).unwrap();
    }
    for statement in statements {
        session.execute(statement).unwrap();
    }
    db
}

const BOTH_INDEXES: [&str; 2] = [
    "CREATE INDEX on_far FOR (n:Far) ON (n.k)",
    "CREATE INDEX on_near FOR (n:Near) ON (n.k)",
];

/// Every node holding the key has a label with an index, so nothing remains to compare, and no value
/// is read. Each node once, though `:Far:Near` is in two indexes.
#[test]
fn no_label_reads_no_value_when_every_holder_has_an_index() {
    let db = crowd(&BOTH_INDEXES);
    let session = db.session();
    for (query, name) in [
        ("MATCH (n {k: 4005}) RETURN n.k", "two indexes"),
        ("MATCH (n {k: 7}) RETURN n.k", "one"),
    ] {
        assert_eq!(session.execute(query).unwrap().row_count(), 1, "{name}");
        assert!(
            scanned(&session, query) < 100,
            "{name}: {}",
            plan_of(&session, query)
        );
    }
}

/// With some labels unindexed, the remainder is what is compared: the read stops short of the labels
/// the indexes cover.
#[test]
fn no_label_compares_only_the_rows_the_indexes_do_not_cover() {
    let db = crowd(&BOTH_INDEXES[..1]);
    let session = db.session();
    for (query, rows) in [
        ("MATCH (n {k: 7}) RETURN n.k", 1),
        ("MATCH (n {k: 3000}) RETURN n.k", 1),
        ("MATCH (n {k: 4005}) RETURN n.k", 1),
    ] {
        assert_eq!(session.execute(query).unwrap().row_count(), rows, "{query}");
        assert!(
            scanned(&session, query) <= 2_100,
            "{query} reads the 2,000 of the label with no index, and few more\n{}",
            plan_of(&session, query)
        );
    }
}

/// The note says nothing is read through an index. Over a union, or no label, with an index read, it
/// is not so any more.
#[test]
fn no_note_where_a_union_reads_an_index() {
    let db = crowd(&BOTH_INDEXES[..1]);
    let session = db.session();
    for query in [
        "MATCH (n {k: 7}) RETURN n.k",
        "MATCH (n) WHERE n.k = 7 RETURN n.k",
        "MATCH (n:Far|Near {k: 7}) RETURN n.k",
        "MATCH (n) WHERE (n:Far OR n:Near) AND n.k = 7 RETURN n.k",
    ] {
        let plan = plan_of(&session, query);
        assert!(!plan.contains("[no index"), "{query}\n{plan}");
    }
}

// ---------------------------------------------------------------------------
// A key some nodes hold, at the far end of a long list
// ---------------------------------------------------------------------------

/// A hub with thousands of edges, and a key two far nodes hold.
fn hub(statements: &[&str]) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "INSERT (:Start {s: 1, name: 'Amsterdam'})",
        "UNWIND range(1, 3000) AS i INSERT (:Far {k: i})",
        "INSERT (:Far {k: 77777, name: 'Alix'}), (:Far {k: 77777, name: 'Gus'}), (:Other {k: 77777, name: 'Jules'}), (:Far:Near {k: 88888, name: 'Vincent'})",
        "MATCH (a:Start), (m) WHERE m.k >= 0 INSERT (a)-[:R]->(m)",
    ] {
        session.execute(statement).unwrap();
    }
    for statement in statements {
        session.execute(statement).unwrap();
    }
    db
}

const FAR_INDEX: &str = "CREATE INDEX on_far FOR (n:Far) ON (n.k)";

/// A key two nodes hold costs the hub two reads, not its whole list.
#[test]
fn a_shared_key_reads_each_target_not_the_list() {
    let db = hub(&[FAR_INDEX]);
    let session = db.session();
    let query = "MATCH (a:Start)-[:R]->(m:Far {k: 77777}) RETURN m.name";
    assert_eq!(rows_of(&session, false, query), names(&["Alix", "Gus"]));
    assert_eq!(rows_of(&session, true, query), names(&["Alix", "Gus"]));
    let mut bare = db.session();
    bare.set_planner_rules(PlannerRules::plain());
    assert_eq!(rows_of(&bare, false, query), names(&["Alix", "Gus"]));
    assert!(
        scanned(&session, query) < 100,
        "the hub's list is not walked\n{}",
        plan_of(&session, query)
    );
}

/// Through a key-wide index, the candidate of another label is dropped once, so `:Far` costs as a key
/// two nodes of `:Far` hold, and the third is not read per row.
#[test]
fn a_candidate_without_the_label_is_dropped_once() {
    let db = hub(&[]);
    db.create_property_index("k").unwrap();
    let session = db.session();
    let query = "MATCH (a:Start)-[:R]->(m:Far {k: 77777}) RETURN m.name";
    assert_eq!(rows_of(&session, false, query), names(&["Alix", "Gus"]));
    assert!(
        scanned(&session, query) < 100,
        "{}",
        plan_of(&session, query)
    );
    let every = "MATCH (a:Start)-[:R]->(m {k: 77777}) RETURN m.name";
    assert_eq!(
        rows_of(&session, false, every),
        names(&["Alix", "Gus", "Jules"])
    );
}

/// A candidate's label is read at the view: a reader from before its removal still gets the row.
#[test]
fn a_candidate_whose_label_went_after_the_snapshot_stays_for_that_reader() {
    let db = hub(&[]);
    db.create_property_index("k").unwrap();
    let mut old = db.session();
    old.begin_transaction().unwrap();
    db.session()
        .execute("MATCH (n {name: 'Gus'}) REMOVE n:Far")
        .unwrap();
    let query = "MATCH (a:Start)-[:R]->(m:Far {k: 77777}) RETURN m.name";
    assert_eq!(rows_of(&old, false, query), names(&["Alix", "Gus"]));
    old.rollback().unwrap();
    assert_eq!(rows_of(&db.session(), false, query), names(&["Alix"]));
}

/// A node of two labels is one target; parallel edges each give a row; a hop with no shorter side
/// (undirected, or two types) gives the rows of the walk.
#[test]
fn the_per_target_read_keeps_the_rows_of_the_walk() {
    for statements in [
        &[FAR_INDEX][..],
        &["CREATE INDEX on_near FOR (n:Near) ON (n.k)"][..],
    ] {
        let db = hub(statements);
        let session = db.session();
        session
            .execute(
                "MATCH (a:Start), (m {name: 'Vincent'}) INSERT (a)-[:R]->(m), (a)-[:S]->(m), (m)-[:R]->(a)",
            )
            .unwrap();
        let mut bare = db.session();
        bare.set_planner_rules(PlannerRules::plain());
        for (query, rows) in [
            ("MATCH (a:Start)-[:R]->(m:Near {k: 88888}) RETURN m.name", 2),
            ("MATCH (a:Start)-[:R]->(m:Far {k: 88888}) RETURN m.name", 2),
            (
                "MATCH (a:Start)-[:R]->(m:Far:Near {k: 88888}) RETURN m.name",
                2,
            ),
            (
                "MATCH (a:Start)-[:R|S]->(m:Near {k: 88888}) RETURN m.name",
                3,
            ),
            ("MATCH (a:Start)-[]-(m:Near {k: 88888}) RETURN m.name", 4),
            ("MATCH (a:Start)<-[:R]-(m:Near {k: 88888}) RETURN m.name", 1),
        ] {
            let found = rows_of(&session, true, query);
            assert_eq!(found.len(), rows, "{statements:?}: {query}");
            assert_eq!(
                found,
                rows_of(&bare, true, query),
                "{statements:?}: {query}"
            );
        }
    }
}

/// A short list is walked as it was: the rule applies only when it is cheaper, and the rows hold on
/// both sides of where it applies.
#[test]
fn lists_on_both_sides_of_the_ratio_give_the_rows_of_the_walk() {
    for degree in [1_i64, 2, 3, 5, 8, 10, 11, 12, 20, 64, 100] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Start {s: 1})").unwrap();
        session
            .execute(&format!(
                "UNWIND range(1, {degree}) AS i INSERT (:Far {{k: i % 2}})"
            ))
            .unwrap();
        session
            .execute("MATCH (a:Start), (m:Far) INSERT (a)-[:R]->(m)")
            .unwrap();
        session.execute(FAR_INDEX).unwrap();
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: 1}) RETURN m.k";
        let wanted = usize::try_from(degree.div_euclid(2) + degree.rem_euclid(2)).unwrap();
        assert_eq!(
            session.execute(query).unwrap().row_count(),
            wanted,
            "degree {degree}"
        );
    }
}

// ---------------------------------------------------------------------------
// Turning with no index, the turn's second step
// ---------------------------------------------------------------------------

fn no_index_graph(starts: i64, far: i64) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        format!("UNWIND range(1, {starts}) AS i INSERT (:Start {{s: i}})"),
        format!("UNWIND range(1, {far}) AS i INSERT (:Far {{k: i}})"),
        format!(
            "MATCH (a:Start), (m:Far) WHERE m.k % {starts} = a.s % {starts} INSERT (a)-[:R]->(m)"
        ),
    ] {
        session.execute(&statement).unwrap();
    }
    db
}

/// A far label much smaller than the sources: scanning it costs less than every source's list.
#[test]
#[ignore = "waits for the no-index turn, verified last"]
fn a_small_far_label_and_many_sources_start_at_the_far_end() {
    let db = no_index_graph(5_000, 50);
    let query = "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.s, m.k";
    let session = db.session();
    assert_eq!(
        rows_of(&session, false, query),
        rows_of(
            &{
                let mut bare = db.session();
                bare.set_planner_rules(PlannerRules::plain());
                bare
            },
            false,
            query
        )
    );
    let plan = plan_of(&session, query);
    assert!(plan.contains("Expand (m)"), "{plan}");
}

/// Few starts are the cheap side, whatever the far label holds.
#[test]
fn few_sources_and_a_large_far_label_stay_at_the_source() {
    let db = no_index_graph(5, 5_000);
    let session = db.session();
    let query = "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.s, m.k";
    let plan = plan_of(&session, query);
    assert!(!plan.contains("Expand (m)"), "{plan}");
    assert_eq!(session.execute(query).unwrap().row_count(), 1);
}

/// Rows at every ratio of the two labels, one and two hops, in both languages: the turn may choose
/// either way, and answers alike.
#[test]
fn every_ratio_of_labels_answers_alike() {
    for (starts, far) in [
        (1, 1),
        (1, 500),
        (10, 500),
        (100, 500),
        (500, 500),
        (500, 100),
        (500, 10),
        (500, 1),
    ] {
        let db = no_index_graph(starts, far);
        let session = db.session();
        session
            .execute("MATCH (m:Far) INSERT (m)-[:Q]->(:Past {k: m.k})")
            .unwrap();
        let mut bare = db.session();
        bare.set_planner_rules(PlannerRules::plain());
        for query in [
            "MATCH (a:Start)-[:R]->(m:Far {k: 1}) RETURN a.s, m.k",
            "MATCH (a:Start)-[:R]->(m:Far) WHERE m.k = 1 RETURN a.s, m.k",
            "MATCH (a:Start)-[:R]->(m:Far {k: 123456}) RETURN a.s, m.k",
            "MATCH (a:Start)-[:R]->(b:Far)-[:Q]->(m:Past {k: 1}) RETURN a.s, m.k",
            "MATCH (m:Far {k: 1})<-[:R]-(a:Start) RETURN a.s, m.k",
            "MATCH (a:Start)-[:R]-(m:Far {k: 1}) RETURN a.s, m.k",
        ] {
            for cypher in [false, true] {
                assert_eq!(
                    rows_of(&session, cypher, query),
                    rows_of(&bare, cypher, query),
                    "{starts} starts, {far} far, cypher {cypher}: {query}"
                );
            }
        }
    }
}
