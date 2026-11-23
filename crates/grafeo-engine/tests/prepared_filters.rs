//! A prepared statement whose filters sit over its hops plans once: the cycles' filters, a far
//! end a key pins and a bound the zone maps decide. Each later call opens the recipe with its own
//! values, view and hop targets (per-call physical planning, part 1, the filters over hops).
//!
//! The oracle is today's path: the statement with its values written as literals, which has no
//! prepared plan and so no recipe, on a database with the plan cache off, given the same data,
//! sessions and writes. In a build with `--cfg grafeo_work_counters`, a call after the first
//! whose values are all integers and strings must open the recipe and plan nothing (`PlanBuilt`
//! 0), and its operators must do the work today's do.

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::{Config, GrafeoDB, Session};

/// The triangles through a person.
const TRIANGLE: &str = "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) WHERE b.id < c.id RETURN count(*) AS triangles";

/// The four-cycles through a person.
const SQUARE: &str = "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(d:Person)-[:KNOWS]-(a) WHERE b.id < d.id AND c.id <> $personId RETURN count(*) AS cycles";

/// The people two hops from a person, the person left out.
const HOP_2: &str = "MATCH (s:Person {id: $personId})-[:KNOWS]-(:Person)-[:KNOWS]-(f:Person) WHERE f.id <> $personId RETURN count(DISTINCT f.id) AS persons";

/// The friends two people share: the second person's key pins the far end.
const COMMON: &str = "MATCH (a:Person {id: $person1Id})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: $person2Id}) RETURN count(DISTINCT c.id) AS common";

/// The triangles through a person, each listed: no aggregate above makes the closing hop an
/// intersection when its rule is off, so it reads its target's column (`ExpandInto`).
const TRIANGLE_LISTED: &str = "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) WHERE b.id < c.id RETURN b.id AS b, c.id AS c";

/// A person's friends older than a bound, which the zone maps may prove no one passes.
const OLDER: &str =
    "MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person) WHERE b.age > $min RETURN b.id AS id";

/// Alix (1) in a triangle with Gus (2) and Vincent (3) and a square with Vincent, Jules (4) and
/// Mia (5); Butch ('b', a string key) knows Alix and Jules, and three people share the key 7.
const GRAPH: &str = "INSERT (alix:Person {id: 1, age: 30}), (gus:Person {id: 2, age: 40}), \
    (vincent:Person {id: 3, age: 50}), (jules:Person {id: 4, age: 20}), \
    (mia:Person {id: 5, age: 35}), (butch:Person {id: 'b', age: 45}), \
    (alix)-[:KNOWS]->(gus), (gus)-[:KNOWS]->(vincent), (vincent)-[:KNOWS]->(alix), \
    (vincent)-[:KNOWS]->(jules), (jules)-[:KNOWS]->(mia), (mia)-[:KNOWS]->(alix), \
    (gus)-[:KNOWS]->(jules), (butch)-[:KNOWS]->(alix), (butch)-[:KNOWS]->(jules), \
    (:Person {id: 7, age: 60})-[:KNOWS]->(gus), (:Person {id: 7, age: 61})-[:KNOWS]->(jules), \
    (s7:Person {id: 7, age: 62})-[:KNOWS]->(alix), (s7)-[:KNOWS]->(jules)";

/// The languages each statement runs in.
const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// The same statements on a database with the plan cache and one without.
struct Pair {
    on: GrafeoDB,
    off: GrafeoDB,
}

impl Pair {
    /// [`GRAPH`] twice, with a property index on `id` when `indexed`.
    fn new(indexed: bool) -> Self {
        Self::of(GRAPH, indexed)
    }

    /// The data `graph` inserts, twice, with a property index on `id` when `indexed`.
    fn of(graph: &str, indexed: bool) -> Self {
        let make = |config: Config| {
            let db = GrafeoDB::with_config(config).unwrap();
            db.execute(graph).unwrap();
            if indexed {
                db.create_property_index("id").unwrap();
            }
            db
        };
        Self {
            on: make(Config::in_memory()),
            off: make(Config::in_memory().with_plan_cache_limit(0)),
        }
    }

    fn execute(&self, statement: &str) {
        self.on.execute(statement).unwrap();
        self.off.execute(statement).unwrap();
    }

    /// A session on each, with `rules`.
    fn sessions(&self, rules: PlannerRules) -> Sessions {
        let session = |db: &GrafeoDB| {
            let mut session = db.session();
            session.set_planner_rules(rules);
            session
        };
        Sessions {
            on: session(&self.on),
            off: session(&self.off),
        }
    }
}

/// A session on each database of a [`Pair`].
struct Sessions {
    on: Session,
    off: Session,
}

/// `value` written as a literal of GQL and Cypher.
fn literal(value: &Value) -> String {
    match value {
        Value::Int64(int) => int.to_string(),
        Value::String(text) => format!("'{text}'"),
        Value::Null => "NULL".to_string(),
        other => panic!("no literal for {other:?}"),
    }
}

/// `query` with each of `values` written as a literal in place of its parameter.
fn written_out(query: &str, values: &HashMap<String, Value>) -> String {
    let mut names: Vec<&String> = values.keys().collect();
    // The longest name first, so `$person1Id` is not read as `$person1` and more.
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let mut text = query.to_string();
    for name in names {
        text = text.replace(&format!("${name}"), &literal(&values[name]));
    }
    text
}

/// The rows of `query` with `values` on `session`, sorted, or its error.
fn rows(
    session: &Session,
    language: &str,
    query: &str,
    values: &HashMap<String, Value>,
) -> Result<Vec<String>, String> {
    session
        .execute_language(query, language, Some(values.clone()))
        .map(|result| {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            rows
        })
        .map_err(|error| error.to_string())
}

impl Sessions {
    /// One call of `query` with `values` on both sessions: the same answer. With work counters,
    /// a call with integer and string values only, `after_the_first`, plans nothing, and its
    /// operators do the work today's do, so they are today's (an empty result where the zone
    /// maps exclude every row, an intersection where today's closes the cycle with one).
    fn call(
        &self,
        language: &str,
        query: &str,
        values: &HashMap<String, Value>,
        after_the_first: bool,
    ) -> Result<Vec<String>, String> {
        #[cfg(grafeo_work_counters)]
        let _ = grafeo_common::work::take();
        let answer = rows(&self.on, language, query, values);
        #[cfg(grafeo_work_counters)]
        let opened = grafeo_common::work::take();
        let today = rows(
            &self.off,
            language,
            &written_out(query, values),
            &HashMap::new(),
        );
        #[cfg(grafeo_work_counters)]
        if after_the_first
            && values
                .values()
                .all(|value| matches!(value, Value::Int64(_) | Value::String(_)))
        {
            use grafeo_common::work::Work;
            let planned = grafeo_common::work::take();
            assert!(
                planned.get(Work::PlanBuilt) > 0,
                "today's path of {query} planned nothing: {}",
                planned.describe()
            );
            assert_eq!(
                opened.get(Work::PlanBuilt),
                0,
                "{query} with {values:?}, in {language}, opened no recipe: {}",
                opened.describe()
            );
            for class in [
                Work::VisibilityTest,
                Work::StampedRead,
                Work::RowGroupLookup,
                Work::AdjacencyRead,
                Work::ColumnRead,
                Work::ChunkBuilt,
                Work::RowBuilt,
                Work::ColumnCopied,
                Work::ValueHash,
                Work::ValueCompare,
            ] {
                assert_eq!(
                    opened.get(class),
                    planned.get(class),
                    "{query} with {values:?}, in {language}: {class:?} differs from today's \
                     operators (opened: {}; today: {})",
                    opened.describe(),
                    planned.describe()
                );
            }
        }
        #[cfg(not(grafeo_work_counters))]
        let _ = after_the_first;
        assert_eq!(answer, today, "{query} with {values:?}, in {language}");
        answer
    }

    /// Calls of `query` with each of `calls`, after a first call that caches its plan.
    fn calls(&self, language: &str, query: &str, calls: &[HashMap<String, Value>]) {
        self.call(language, query, &calls[0], false).ok();
        for values in calls {
            self.call(language, query, values, true).ok();
        }
    }
}

/// `name` bound to `value`, and the further pairs.
fn values(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

/// The keys of one person: integers in and out of the graph, a string, a key three people
/// share, NULL and an integer again after the string.
fn person_keys() -> Vec<Value> {
    vec![
        Value::Int64(1),
        Value::Int64(2),
        Value::Int64(3),
        Value::Int64(4),
        Value::Int64(5),
        Value::from("b"),
        Value::Int64(7),
        Value::Int64(99),
        Value::Null,
        Value::Int64(1),
    ]
}

/// The rule sets the cycles run under: every rule, and every rule but the intersection, which
/// leaves a closing hop that reads its target's column.
fn rule_sets() -> [PlannerRules; 2] {
    [
        PlannerRules::all(),
        PlannerRules::all().with(PlannerRule::IntersectExpand, false),
    ]
}

/// Triangle, square and two hops give today's rows for every key, in GQL and Cypher, with and
/// without an index on the key and with and without the intersection.
#[test]
fn the_cycles_give_todays_rows() {
    for indexed in [false, true] {
        let pair = Pair::new(indexed);
        for rules in rule_sets() {
            let sessions = pair.sessions(rules);
            for language in LANGUAGES {
                for query in [TRIANGLE, SQUARE, HOP_2, TRIANGLE_LISTED] {
                    let calls: Vec<_> = person_keys()
                        .into_iter()
                        .map(|key| values(&[("personId", key)]))
                        .collect();
                    sessions.calls(language, query, &calls);
                }
            }
        }
    }
}

/// The friends two people share, the far end pinned by the second key: a key nobody holds, one
/// three people hold, NULL on either side, an integer then a string, in GQL and Cypher.
#[test]
fn a_far_end_pinned_by_a_key_gives_todays_rows() {
    let pairs = [
        (Value::Int64(1), Value::Int64(4)),
        (Value::Int64(4), Value::Int64(1)),
        (Value::Int64(2), Value::Int64(3)),
        (Value::Int64(1), Value::Int64(7)),
        (Value::Int64(7), Value::Int64(1)),
        (Value::Int64(1), Value::Int64(99)),
        (Value::Int64(1), Value::from("b")),
        (Value::from("b"), Value::Int64(1)),
        (Value::Int64(1), Value::Null),
        (Value::Null, Value::Int64(1)),
        (Value::Int64(4), Value::Int64(1)),
    ];
    for indexed in [false, true] {
        let pair = Pair::new(indexed);
        let sessions = pair.sessions(PlannerRules::all());
        for language in LANGUAGES {
            let calls: Vec<_> = pairs
                .iter()
                .map(|(first, second)| {
                    values(&[("person1Id", first.clone()), ("person2Id", second.clone())])
                })
                .collect();
            sessions.calls(language, COMMON, &calls);
        }
    }
}

/// An index on the pinned key created and then dropped between calls: each call gives today's
/// rows, as the far end is or is not looked up.
#[test]
fn an_index_created_and_dropped_between_calls_is_followed() {
    let pair = Pair::new(false);
    let sessions = pair.sessions(PlannerRules::all());
    let common = |label: &str| {
        for language in LANGUAGES {
            let answer = sessions.call(
                language,
                COMMON,
                &values(&[
                    ("person1Id", Value::Int64(1)),
                    ("person2Id", Value::Int64(4)),
                ]),
                true,
            );
            assert!(answer.is_ok(), "{label}: {answer:?}");
        }
    };
    common("no index");
    pair.on.create_property_index("id").unwrap();
    pair.off.create_property_index("id").unwrap();
    common("index created");
    assert!(pair.on.drop_property_index("id").unwrap());
    assert!(pair.off.drop_property_index("id").unwrap());
    common("index dropped");
    pair.on.create_property_index("age").unwrap();
    pair.off.create_property_index("age").unwrap();
    for min in [100, 25] {
        let answer = sessions.call(
            "cypher",
            OLDER,
            &values(&[("personId", Value::Int64(1)), ("min", Value::Int64(min))]),
            true,
        );
        assert!(answer.is_ok(), "index on the bound property: {answer:?}");
    }
}

/// `OLDER` for Alix with `min`.
fn older_than(min: i64) -> HashMap<String, Value> {
    values(&[("personId", Value::Int64(1)), ("min", Value::Int64(min))])
}

/// A bound past every age gives no rows; a write between calls that makes an age pass it gives
/// that row at the next call; an age removed, and a person deleted, between calls give none.
#[test]
fn the_zone_maps_decide_at_each_call() {
    let pair = Pair::new(false);
    let sessions = pair.sessions(PlannerRules::all());
    for language in LANGUAGES {
        sessions
            .call(language, OLDER, &older_than(25), false)
            .unwrap();
        assert_eq!(
            sessions
                .call(language, OLDER, &older_than(100), true)
                .unwrap(),
            Vec::<String>::new(),
            "a bound past every age, in {language}"
        );
    }
    pair.execute("MATCH (p:Person {id: 2}) SET p.age = 150");
    for language in LANGUAGES {
        assert_eq!(
            sessions
                .call(language, OLDER, &older_than(100), true)
                .unwrap()
                .len(),
            1,
            "an age written between calls, in {language}"
        );
    }
    pair.execute("MATCH (p:Person {id: 2}) REMOVE p.age");
    pair.execute("MATCH (p:Person {id: 3}) SET p.age = 160");
    pair.execute("MATCH (p:Person {id: 3}) DETACH DELETE p");
    for language in LANGUAGES {
        assert_eq!(
            sessions
                .call(language, OLDER, &older_than(100), true)
                .unwrap(),
            Vec::<String>::new(),
            "an age removed and a person deleted between calls, in {language}"
        );
    }
}

/// Inside one transaction, its own uncommitted write that makes an age pass the bound gives that
/// row at its next call; after the rollback the row is gone. In GQL and Cypher.
#[test]
fn the_zone_maps_see_the_transactions_own_write() {
    for language in LANGUAGES {
        the_transactions_own_write_in(language);
    }
}

/// [`the_zone_maps_see_the_transactions_own_write`] in `language`.
fn the_transactions_own_write_in(language: &str) {
    let pair = Pair::new(false);
    let mut sessions = pair.sessions(PlannerRules::all());
    sessions
        .call(language, OLDER, &older_than(25), false)
        .unwrap();
    sessions.on.begin_transaction().unwrap();
    sessions.off.begin_transaction().unwrap();
    assert_eq!(
        sessions
            .call(language, OLDER, &older_than(200), true)
            .unwrap(),
        Vec::<String>::new(),
        "a bound past every age"
    );
    for session in [&sessions.on, &sessions.off] {
        session
            .execute("MATCH (p:Person {id: 2}) SET p.age = 250")
            .unwrap();
    }
    assert_eq!(
        sessions
            .call(language, OLDER, &older_than(200), true)
            .unwrap()
            .len(),
        1,
        "the transaction's own uncommitted age"
    );
    sessions.on.rollback().unwrap();
    sessions.off.rollback().unwrap();
    assert_eq!(
        sessions
            .call(language, OLDER, &older_than(200), true)
            .unwrap(),
        Vec::<String>::new(),
        "rolled back"
    );
}

/// A person's age: a RETURN directly over the pinned scan.
const AGE: &str = "MATCH (a:Person {id: $personId}) RETURN a.age AS age";

/// Keys against the zone maps of a column that holds integers alone (1 to 5, 7 and 10): past
/// the greatest and below the least, today's planner proves the pinned scan empty from them
/// and plans no operators past it, and the recipe's seek takes the same empty tree with the
/// call's key; a key inside the range that nobody holds, and Alix's, seek. With the zone-map
/// rule off every key seeks. With and without an index on the key, in GQL and Cypher, under a
/// triangle and under a RETURN directly over the scan.
#[test]
fn a_key_the_zone_maps_exclude_gives_todays_tree() {
    let graph = GRAPH.replace("{id: 'b'", "{id: 10");
    let without_zone_maps = PlannerRules::all().with(PlannerRule::ZoneMaps, false);
    for indexed in [false, true] {
        let pair = Pair::of(&graph, indexed);
        for rules in [PlannerRules::all(), without_zone_maps] {
            let sessions = pair.sessions(rules);
            for language in LANGUAGES {
                for query in [TRIANGLE, AGE] {
                    let key = |id: i64| values(&[("personId", Value::Int64(id))]);
                    sessions.call(language, query, &key(1), false).unwrap();
                    for id in [99, -5, 6, 1] {
                        sessions
                            .call(language, query, &key(id), true)
                            .unwrap_or_else(|error| {
                                panic!(
                                    "{query} with {id}, indexed {indexed}, in {language}: {error}"
                                )
                            });
                    }
                }
            }
        }
    }
}

/// A call that goes back to today's path (a key of a kind the seek does not read) shows the
/// statement's warnings once, as a call that opens the recipe does: the recipe raised the plan's
/// before it declined, and today's path raises them again. The deprecated list `+` is raised when
/// a row evaluates it, so each key matches a row.
#[test]
fn a_call_back_on_todays_path_shows_its_warnings_once() {
    const QUERY: &str = "MATCH (a:Person) WHERE a.id = $personId RETURN [a.age] + [1] AS ages";
    let pair = Pair::new(true);
    let session = pair.on.session();
    let warnings = |key: Value| {
        let result = session
            .execute_language(QUERY, "gql", Some(values(&[("personId", key)])))
            .unwrap();
        result.warnings().to_vec()
    };
    let first = warnings(Value::Int64(1));
    assert_eq!(first.len(), 1, "the deprecated list `+`: {first:?}");
    assert_eq!(
        warnings(Value::Int64(2)),
        first,
        "a call that opens the recipe"
    );
    #[cfg(grafeo_work_counters)]
    let _ = grafeo_common::work::take();
    // A float key the seek does not read, equal to Alix's: her row evaluates the `+`.
    assert_eq!(
        warnings(Value::Float64(1.0)),
        first,
        "a call back on today's path"
    );
    #[cfg(grafeo_work_counters)]
    assert!(
        grafeo_common::work::take().get(grafeo_common::work::Work::PlanBuilt) > 0,
        "the float key went back to today's path"
    );
}

/// A filter whose route reads the store while planning (a value join of two scans) gives
/// today's rows, call after call.
#[test]
fn an_uncovered_filter_gives_todays_rows() {
    let pair = Pair::new(true);
    let sessions = pair.sessions(PlannerRules::all());
    let query =
        "MATCH (a:Person {id: $personId}), (b:Person) WHERE b.age = a.age + $gap RETURN b.id AS id";
    for language in LANGUAGES {
        for (person, gap) in [(1, 10), (2, 10), (1, 0), (99, 10)] {
            sessions
                .call(
                    language,
                    query,
                    &values(&[
                        ("personId", Value::Int64(person)),
                        ("gap", Value::Int64(gap)),
                    ]),
                    false,
                )
                .unwrap();
        }
    }
}

/// Each kind of hop target a filter notes is reached, as today's plan of the statement shows
/// (its PROFILE names the hint), and the call that opens the recipe gives today's rows (with work
/// counters, today's work): an intersection, a column a closing hop checks, and a key that pins
/// the far end.
#[test]
fn each_hop_target_kind_is_reached() {
    let pair = Pair::new(true);
    let alix = values(&[("personId", Value::Int64(1))]);
    let alix_and_jules = values(&[
        ("person1Id", Value::Int64(1)),
        ("person2Id", Value::Int64(4)),
    ]);
    for (rules, language, query, call, hint) in [
        (
            PlannerRules::all(),
            "cypher",
            TRIANGLE,
            &alix,
            "[IntersectExpand into: a]",
        ),
        (
            PlannerRules::all().with(PlannerRule::IntersectExpand, false),
            "cypher",
            TRIANGLE_LISTED,
            &alix,
            "[expand into: a]",
        ),
        (
            PlannerRules::all(),
            "gql",
            COMMON,
            &alix_and_jules,
            "[expand to key: id]",
        ),
        (
            PlannerRules::all(),
            "cypher",
            COMMON,
            &alix_and_jules,
            "[expand to key: id]",
        ),
    ] {
        let mut session = pair.off.session();
        session.set_planner_rules(rules);
        let profile = session
            .execute_language(
                &format!("PROFILE {}", written_out(query, call)),
                language,
                None,
            )
            .unwrap();
        let Value::String(text) = &profile.rows()[0][0] else {
            panic!("PROFILE {query} gave no text");
        };
        assert!(text.contains(hint), "{query} in {language}: {text}");
        let sessions = pair.sessions(rules);
        sessions.call(language, query, call, false).unwrap();
        sessions.call(language, query, call, true).unwrap();
    }
}

/// SQL/PGQ's triangle through a person.
#[cfg(feature = "sql-pgq")]
const SQL_TRIANGLE: &str = "SELECT count(*) AS triangles FROM GRAPH_TABLE (MATCH (a:Person)-[k1:KNOWS]-(b:Person)-[k2:KNOWS]-(c:Person)-[k3:KNOWS]-(a:Person) WHERE a.id = $personId AND b.id < c.id COLUMNS (c.id AS cid))";

/// SQL/PGQ's common friends of two people.
#[cfg(feature = "sql-pgq")]
const SQL_COMMON: &str = "SELECT count(DISTINCT cid) AS common FROM GRAPH_TABLE (MATCH (a:Person)-[k1:KNOWS]-(c:Person)-[k2:KNOWS]-(b:Person) WHERE a.id = $person1Id AND b.id = $person2Id COLUMNS (c.id AS cid))";

/// SQL/PGQ's friends older than a bound, whose zone check reads the kinds a column holds.
#[cfg(feature = "sql-pgq")]
const SQL_OLDER: &str = "SELECT bid FROM GRAPH_TABLE (MATCH (a:Person)-[k:KNOWS]-(b:Person) WHERE a.id = $personId AND b.age > $min COLUMNS (b.id AS bid))";

/// The SQL/PGQ triangle, common friends and older friends give today's rows for every key, with
/// and without an index on the key, and the bound past every age gives none. SQL/PGQ's `=`
/// raises across kinds, so with Butch's string key stored an integer key raises, as today; on
/// a graph whose keys are all integers the calls give rows.
#[cfg(feature = "sql-pgq")]
#[test]
fn the_sql_shapes_give_todays_rows() {
    let alix = values(&[("personId", Value::Int64(1))]);
    for indexed in [false, true] {
        let pair = Pair::new(indexed);
        let sessions = pair.sessions(PlannerRules::all());
        // A stored key of another kind sends each call back to today's path (the seek's key
        // read declines where `=` could raise), so these calls may plan.
        for _ in 0..2 {
            let raised = sessions.call("sql", SQL_TRIANGLE, &alix, false);
            assert!(
                raised.is_err(),
                "a stored string key, indexed {indexed}: {raised:?}"
            );
        }
        let pair = Pair::of(&GRAPH.replace("{id: 'b'", "{id: 6"), indexed);
        let sessions = pair.sessions(PlannerRules::all());
        sessions.call("sql", SQL_TRIANGLE, &alix, false).unwrap();
        assert_eq!(
            sessions.call("sql", SQL_TRIANGLE, &alix, true).unwrap(),
            vec!["[Int64(1)]".to_string()],
            "Alix's triangles, indexed {indexed}"
        );
        // A string key against the integer keys goes back to today's path too, for its rows.
        sessions
            .call(
                "sql",
                SQL_TRIANGLE,
                &values(&[("personId", Value::from("b"))]),
                false,
            )
            .ok();
        let people: Vec<_> = person_keys()
            .into_iter()
            .filter(|key| !matches!(key, Value::String(_)))
            .map(|key| values(&[("personId", key)]))
            .collect();
        sessions.calls("sql", SQL_TRIANGLE, &people);
        let pairs: Vec<_> = [(1, 4), (4, 1), (1, 7), (1, 99), (2, 3)]
            .into_iter()
            .map(|(first, second)| {
                values(&[
                    ("person1Id", Value::Int64(first)),
                    ("person2Id", Value::Int64(second)),
                ])
            })
            .collect();
        sessions.calls("sql", SQL_COMMON, &pairs);
        assert!(
            sessions
                .call("sql", SQL_COMMON, &pairs[0], true)
                .is_ok_and(|rows| rows.len() == 1),
            "common friends, indexed {indexed}"
        );
        sessions
            .call("sql", SQL_OLDER, &older_than(25), false)
            .unwrap();
        assert_eq!(
            sessions
                .call("sql", SQL_OLDER, &older_than(100), true)
                .unwrap(),
            Vec::<String>::new(),
            "a bound past every age, indexed {indexed}"
        );
        pair.execute("MATCH (p:Person {id: 2}) SET p.age = 150");
        assert_eq!(
            sessions
                .call("sql", SQL_OLDER, &older_than(100), true)
                .unwrap()
                .len(),
            1,
            "an age written between calls, indexed {indexed}"
        );
    }
}
