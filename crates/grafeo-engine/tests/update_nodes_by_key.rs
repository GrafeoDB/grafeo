//! `GrafeoDB::update_nodes(labels, key, rows)`: each row's properties merge
//! into the existing nodes with all of `labels` whose `key` equals the row's
//! key; no node is created. With a property index on the key and without:
//!
//! - a float key equal to an integer one matches it (2.0 matches 2);
//! - a node with only some of the labels is not matched;
//! - a row without the key, and a key no node has, are skipped and listed;
//! - a key two nodes share updates both and counts as one row;
//! - the properties merge: the others stay;
//! - one write: a row that breaks a UNIQUE constraint writes nothing;
//! - in a session's transaction, a rollback undoes it;
//! - a key compares as `=` does: the text "5", "05" or "+5" matches the
//!   integer 5, the text "5.0" or "5e0" does not, as `MATCH (n {id: $k})`
//!   finds them;
//! - a null value removes the property, as `SET n += r` does;
//! - it gives the nodes `UNWIND $rows AS r MATCH ... SET n += r` gives, with
//!   a few rows (which look their nodes up through an index on the key) and
//!   with many (which read the label once), and the same change events;
//! - in a transaction, with an index, it finds the transaction's own nodes;
//! - a persistent database replays it from the WAL after a crash.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main update_nodes_by_key::
//! ```

#![cfg(feature = "gql")]

use std::collections::{BTreeMap, HashMap};

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// Methods 1 to 5 (Graph:Method, `name` m1 to m5, `tier` 0), method 6 is
/// only a Graph (no Method label), and two Graph:Method nodes share id 7.
fn database(index: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if index {
        db.create_property_index("id").unwrap();
    }
    db.execute(
        "UNWIND range(1, 5) AS i INSERT (:Graph:Method {id: i, name: 'm' + toString(i), tier: 0})",
    )
    .unwrap();
    db.execute("INSERT (:Graph {id: 6, name: 'm6', tier: 0})")
        .unwrap();
    db.execute("INSERT (:Graph:Method {id: 7, name: 'a7', tier: 0}), (:Graph:Method {id: 7, name: 'b7', tier: 0})")
        .unwrap();
    db
}

fn row(pairs: &[(&str, Value)]) -> HashMap<PropertyKey, Value> {
    pairs
        .iter()
        .map(|(k, v)| (PropertyKey::new(*k), v.clone()))
        .collect()
}

/// `(name, tier)` of every node, by name.
fn state(db: &GrafeoDB) -> Vec<(String, i64)> {
    db.execute("MATCH (n:Graph) RETURN n.name, n.tier ORDER BY n.name")
        .unwrap()
        .rows()
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::String(name), Value::Int64(tier)) => (name.to_string(), *tier),
            other => panic!("{other:?}"),
        })
        .collect()
}

fn tiers(db: &GrafeoDB) -> HashMap<String, i64> {
    state(db).into_iter().collect()
}

#[test]
fn the_key_matches_by_value_and_the_labels_must_all_be_there() {
    for index in [false, true] {
        let db = database(index);
        let summary = db
            .update_nodes(
                &["Graph", "Method"],
                "id",
                vec![
                    row(&[("id", Value::Float64(2.0)), ("tier", Value::Int64(1))]),
                    row(&[("id", Value::Int64(6)), ("tier", Value::Int64(1))]),
                    row(&[("tier", Value::Int64(1))]),
                    row(&[("id", Value::Int64(99)), ("tier", Value::Int64(1))]),
                    row(&[("id", Value::Int64(7)), ("tier", Value::Int64(1))]),
                ],
            )
            .unwrap();
        assert_eq!(summary.created, 0, "index {index}");
        assert_eq!(summary.updated, 2, "2.0 and the shared 7; index {index}");
        assert_eq!(summary.skipped, 3, "index {index}");
        assert_eq!(
            summary.skipped_rows,
            [1, 2, 3],
            "the Graph-only 6, no key, 99; index {index}"
        );
        let t = tiers(&db);
        assert_eq!(t["m2"], 1, "2.0 matches 2; index {index}");
        assert_eq!(t["m6"], 0, "a Graph without Method; index {index}");
        assert_eq!((t["a7"], t["b7"]), (1, 1), "both nodes of 7; index {index}");
        assert_eq!((t["m1"], t["m3"]), (0, 0), "index {index}");
        assert_eq!(
            db.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0],
            Value::Int64(8),
            "no node created; index {index}"
        );
    }
}

/// A row's properties merge into the node: `name` stays when only `tier`
/// is set, and a new property is added.
#[test]
fn the_properties_merge() {
    let db = database(false);
    db.update_nodes(
        &["Graph", "Method"],
        "id",
        vec![row(&[
            ("id", Value::Int64(3)),
            ("tier", Value::Int64(9)),
            ("owner", Value::from("Gus")),
        ])],
    )
    .unwrap();
    assert_eq!(
        db.execute("MATCH (n:Method {id: 3}) RETURN n.name, n.tier, n.owner")
            .unwrap()
            .rows(),
        [[Value::from("m3"), Value::Int64(9), Value::from("Gus")]]
    );
}

/// A row that breaks a UNIQUE constraint fails the call, and no row of it
/// is written.
#[test]
fn a_constraint_violation_writes_nothing() {
    let db = database(false);
    db.execute_cypher("CREATE CONSTRAINT FOR (m:Method) REQUIRE m.name IS UNIQUE")
        .unwrap_or_else(|error| panic!("{error}"));
    let before = state(&db);
    let result = db.update_nodes(
        &["Graph", "Method"],
        "id",
        vec![
            row(&[("id", Value::Int64(1)), ("tier", Value::Int64(5))]),
            row(&[("id", Value::Int64(2)), ("name", Value::from("m3"))]),
        ],
    );
    assert!(result.is_err(), "name m3 is taken");
    assert_eq!(state(&db), before);
}

/// In a session's transaction the update is seen inside and undone by the
/// rollback.
#[test]
fn a_rollback_undoes_it() {
    let db = database(false);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let summary = session
        .update_nodes(
            &["Graph", "Method"],
            "id",
            vec![row(&[("id", Value::Int64(4)), ("tier", Value::Int64(3))])],
        )
        .unwrap();
    assert_eq!(summary.updated, 1);
    assert_eq!(
        session
            .execute("MATCH (n:Method {id: 4}) RETURN n.tier")
            .unwrap()
            .rows()[0][0],
        Value::Int64(3)
    );
    session.rollback().unwrap();
    assert_eq!(tiers(&db)["m4"], 0);
}

/// Whether `update_nodes` updates the node with `id: stored` for a row with
/// key `key`, and whether `MATCH (n:Key {id: $k})` matches it, with a
/// property index on `id` or without. Five other nodes make a single row
/// look its node up through the index (see `rows_beat_a_seek`).
fn updates_and_matches(stored: &Value, key: &Value, index: bool) -> (bool, bool) {
    let db = GrafeoDB::new_in_memory();
    if index {
        db.create_property_index("id").unwrap();
    }
    db.execute_with_params(
        "INSERT (:Key {id: $id, tier: 0})",
        HashMap::from([("id".to_string(), stored.clone())]),
    )
    .unwrap();
    db.execute("UNWIND range(100, 104) AS i INSERT (:Key {id: i, tier: 0})")
        .unwrap();
    let matches = db
        .execute_with_params(
            "MATCH (n:Key {id: $k}) RETURN count(n)",
            HashMap::from([("k".to_string(), key.clone())]),
        )
        .unwrap()
        .scalar::<i64>()
        .unwrap();
    let summary = db
        .update_nodes(
            &["Key"],
            "id",
            vec![row(&[("id", key.clone()), ("tier", Value::Int64(1))])],
        )
        .unwrap();
    let updated = db
        .execute("MATCH (n:Key) WHERE n.tier = 1 RETURN count(n)")
        .unwrap()
        .scalar::<i64>()
        .unwrap();
    assert_eq!(summary.updated, usize::try_from(updated).unwrap());
    (updated == 1, matches == 1)
}

/// A key compares with the stored key as `=` does: a string equals no
/// number, whatever it spells ("5", "05", "+5", "5.0" and "5e0" against 5,
/// and a stored "5" against the integer and the float 5, GA04); an integer
/// equals a float of its value; two strings are equal only when identical.
/// `update_nodes` updates exactly the nodes `MATCH (n {id: $k})` matches,
/// with an index on the key and without.
#[test]
fn a_key_matches_as_equals_in_a_query_does() {
    let cases = [
        (Value::Int64(5), Value::from("5"), false),
        (Value::Int64(5), Value::from("05"), false),
        (Value::Int64(5), Value::from("+5"), false),
        (Value::Int64(5), Value::from("5.0"), false),
        (Value::Int64(5), Value::from("5e0"), false),
        (Value::Int64(5), Value::Float64(5.0), true),
        (Value::Int64(5), Value::Int64(5), true),
        (Value::Int64(5), Value::from("6"), false),
        (Value::from("5"), Value::Int64(5), false),
        (Value::from("5"), Value::Float64(5.0), false),
        (Value::from("5"), Value::from("5"), true),
        (Value::from("5"), Value::from("05"), false),
        (Value::from("5"), Value::Int64(6), false),
    ];
    for index in [false, true] {
        for (stored, key, equal) in &cases {
            let (updated, matched) = updates_and_matches(stored, key, index);
            assert_eq!(
                (updated, matched),
                (*equal, *equal),
                "stored {stored:?}, key {key:?}, index {index}: (updated, matched)"
            );
        }
    }
}

/// The property names of the Method with id `id`, sorted.
fn keys_of(db: &GrafeoDB, id: i64) -> Vec<String> {
    let mut keys: Vec<String> = db
        .get_node(db.find_nodes_by_property("id", &Value::Int64(id)).unwrap()[0])
        .unwrap()
        .unwrap()
        .properties
        .iter()
        .map(|(key, _)| key.as_str().to_string())
        .collect();
    keys.sort();
    keys
}

/// A null value removes the property, as `SET n += {tier: null}` does.
#[test]
fn a_null_value_removes_the_property() {
    for index in [false, true] {
        let db = database(index);
        let summary = db
            .update_nodes(
                &["Method"],
                "id",
                vec![row(&[("id", Value::Int64(3)), ("tier", Value::Null)])],
            )
            .unwrap();
        assert_eq!(summary.updated, 1, "index {index}");
        assert_eq!(
            keys_of(&db, 3),
            ["id", "name"],
            "tier is gone; index {index}"
        );
        assert_eq!(keys_of(&db, 4), ["id", "name", "tier"], "index {index}");
    }
}

/// 300 Methods (`id` 1 to 300, `name` m1 to m300, `tier` 0), a Graph-only
/// node of id 6000, and two Methods that share id 7000.
fn large(config: grafeo_engine::Config, index: bool) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    if index {
        db.create_property_index("id").unwrap();
    }
    db.execute(
        "UNWIND range(1, 300) AS i \
         INSERT (:Graph:Method {id: i, name: 'm' + toString(i), tier: 0})",
    )
    .unwrap();
    db.execute(
        "INSERT (:Graph {id: 6000, name: 'g6000', tier: 0}), \
         (:Graph:Method {id: 7000, name: 'a7000', tier: 0}), \
         (:Graph:Method {id: 7000, name: 'b7000', tier: 0})",
    )
    .unwrap();
    db
}

/// `(name, tier, owner)` of every Graph node, by name.
fn full_state(db: &GrafeoDB) -> Vec<Vec<Value>> {
    db.execute("MATCH (n:Graph) RETURN n.name, n.tier, n.owner ORDER BY n.name")
        .unwrap()
        .rows()
        .to_vec()
}

/// Six rows (a few: with an index each looks its nodes up through it): a
/// float key, the Graph-only node, the shared key, a key no node has, a
/// null that removes `tier`, and a new property.
fn few_rows() -> Vec<HashMap<PropertyKey, Value>> {
    vec![
        row(&[("id", Value::Float64(2.0)), ("tier", Value::Int64(1))]),
        row(&[("id", Value::Int64(6000)), ("tier", Value::Int64(1))]),
        row(&[("id", Value::Int64(7000)), ("tier", Value::Int64(2))]),
        row(&[("id", Value::Int64(99_999)), ("tier", Value::Int64(1))]),
        row(&[("id", Value::Int64(5)), ("tier", Value::Null)]),
        row(&[("id", Value::Int64(9)), ("owner", Value::from("Gus"))]),
    ]
}

/// A row per Method and one more (many: they read the label once).
fn many_rows() -> Vec<HashMap<PropertyKey, Value>> {
    (1..=301)
        .map(|i| row(&[("id", Value::Int64(i)), ("tier", Value::Int64(i % 7))]))
        .collect()
}

const STATEMENT: &str = "UNWIND $rows AS r MATCH (n:Graph:Method {id: r.id}) SET n += r";

fn statement_params(rows: Vec<HashMap<PropertyKey, Value>>) -> HashMap<String, Value> {
    let rows: Vec<Value> = rows
        .into_iter()
        .map(|row| Value::Map(row.into_iter().collect::<BTreeMap<_, _>>().into()))
        .collect();
    HashMap::from([("rows".to_string(), Value::List(rows.into()))])
}

/// With and without an index, a few rows and many give the nodes the
/// statement gives.
#[test]
fn it_gives_what_the_statement_gives() {
    for index in [false, true] {
        for (shape, rows) in [("few", few_rows()), ("many", many_rows())] {
            let by_call = large(grafeo_engine::Config::in_memory(), index);
            let by_statement = large(grafeo_engine::Config::in_memory(), index);
            let summary = by_call
                .update_nodes(&["Graph", "Method"], "id", rows.clone())
                .unwrap();
            by_statement
                .execute_with_params(STATEMENT, statement_params(rows))
                .unwrap();
            let state = full_state(&by_call);
            assert_eq!(
                state,
                full_state(&by_statement),
                "{shape} rows, index {index}"
            );
            if shape == "many" {
                assert_eq!(
                    (summary.updated, summary.skipped_rows),
                    (300, vec![300]),
                    "index {index}"
                );
                continue;
            }
            assert_eq!(summary.updated, 4, "index {index}");
            assert_eq!(summary.skipped_rows, [1, 3], "index {index}");
            let of = |name: &str| {
                state
                    .iter()
                    .find(|r| r[0] == Value::from(name))
                    .unwrap()
                    .clone()
            };
            assert_eq!(of("m2")[1], Value::Int64(1), "2.0 matches 2; index {index}");
            assert_eq!(of("m5")[1], Value::Null, "removed; index {index}");
            assert_eq!(of("m9")[1..], [Value::Int64(0), Value::from("Gus")]);
            assert_eq!(
                (of("a7000")[1].clone(), of("b7000")[1].clone()),
                (Value::Int64(2), Value::Int64(2))
            );
            assert_eq!(
                of("g6000")[1],
                Value::Int64(0),
                "not a Method; index {index}"
            );
        }
    }
}

/// The call reports to change data capture what the statement reports.
#[cfg(feature = "cdc")]
#[test]
fn it_reports_the_change_events_the_statement_reports() {
    type Properties = Option<BTreeMap<String, Value>>;
    fn events(
        db: &GrafeoDB,
        from: grafeo_common::types::EpochId,
    ) -> Vec<(String, String, Properties, Properties)> {
        let sorted = |map: &Option<HashMap<String, Value>>| {
            map.as_ref()
                .map(|map| map.clone().into_iter().collect::<BTreeMap<_, _>>())
        };
        let mut events: Vec<_> = db
            .changes_between(from, db.current_epoch())
            .unwrap()
            .iter()
            .map(|event| {
                (
                    format!("{:?}", event.entity_id),
                    format!("{:?}", event.kind),
                    sorted(&event.before),
                    sorted(&event.after),
                )
            })
            .collect();
        events.sort_by_key(|event| format!("{event:?}"));
        events
    }
    for index in [false, true] {
        for (shape, rows) in [("few", few_rows()), ("many", many_rows())] {
            let config = || grafeo_engine::Config::in_memory().with_cdc();
            let by_call = large(config(), index);
            let by_statement = large(config(), index);
            let from = grafeo_common::types::EpochId::new(by_call.current_epoch().as_u64() + 1);
            by_call
                .update_nodes(&["Graph", "Method"], "id", rows.clone())
                .unwrap();
            by_statement
                .execute_with_params(STATEMENT, statement_params(rows))
                .unwrap();
            let expected = events(&by_statement, from);
            assert!(
                expected.len() >= 4,
                "{shape} rows, index {index}: {expected:?}"
            );
            assert_eq!(
                events(&by_call, from),
                expected,
                "{shape} rows, index {index}"
            );
        }
    }
}

/// In a transaction, with an index on the key, a row finds a node the
/// transaction created, and the commit keeps the update.
#[test]
fn in_a_transaction_it_finds_the_transactions_own_nodes() {
    let db = large(grafeo_engine::Config::in_memory(), true);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Graph:Method {id: 5000, name: 'm5000', tier: 0})")
        .unwrap();
    let summary = session
        .update_nodes(
            &["Graph", "Method"],
            "id",
            vec![
                row(&[("id", Value::Int64(5000)), ("tier", Value::Int64(4))]),
                row(&[("id", Value::Int64(8)), ("tier", Value::Int64(4))]),
            ],
        )
        .unwrap();
    assert_eq!((summary.updated, summary.skipped), (2, 0));
    session.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (n:Method) WHERE n.id IN [8, 5000] RETURN n.id, n.tier ORDER BY n.id")
            .unwrap()
            .rows(),
        [
            [Value::Int64(8), Value::Int64(4)],
            [Value::Int64(5000), Value::Int64(4)]
        ]
    );
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod crash {
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};

    use super::{few_rows, full_state, large};

    const PATH_VAR: &str = "GRAFEO_UPDATE_NODES_CRASH_PATH";

    fn config(path: &Path) -> Config {
        Config::persistent(path).with_storage_format(StorageFormat::Auto)
    }

    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// Child-process entry for [`a_crash_after_it_replays_it`]; a no-op
    /// when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = large(config(Path::new(&path)), true);
        db.update_nodes(&["Graph", "Method"], "id", few_rows())
            .unwrap();
        // Exit without closing: no checkpoint, the WAL stays.
        std::process::exit(0);
    }

    /// A child process updates and exits without closing the database, as a
    /// crash does; the reopened database holds what the same steps give in
    /// memory.
    #[test]
    fn a_crash_after_it_replays_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("updates.grafeo");
        let status =
            child_process::run(grafeo_common::child_test!("crash_child").env(PATH_VAR, &path))
                .unwrap();
        assert!(status.success(), "the child process failed");
        assert!(sidecar_wal(&path).exists(), "a crash leaves the WAL");

        let expected = large(Config::in_memory(), true);
        expected
            .update_nodes(&["Graph", "Method"], "id", few_rows())
            .unwrap();
        let reopened = GrafeoDB::with_config(config(&path)).unwrap();
        assert_eq!(full_state(&reopened), full_state(&expected));
    }
}
