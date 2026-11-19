//! Leaving soft-deleted nodes out of search (#742): the `filters` every search takes. `$ne`
//! and `$nin` keep a node without the property (equality and the other operators need it);
//! a map of several operators keeps it only when each is `$ne` or `$nin`. A null filter value
//! and a malformed filter (an unknown operator, a non-list `$in`/`$nin`, a non-string
//! `$contains`) are invalid-argument errors naming the key, for every search kind, before the
//! search runs. `CALL grafeo.search.text`, `.vector` and `.mmr` take the same filters map and
//! keep `k`. The filter runs before the top-k cut, so `k` counts matching nodes, and text
//! scores stay those of the whole index.
//!
//! The oracle is the unfiltered search kept to the nodes the decided rules allow (text and
//! vector: the same list, in order, with the same scores; MMR and hybrid, whose ranks depend
//! on what else is in the list: the same set).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main search_soft_delete::
//! ```

#![cfg(all(feature = "hybrid-search", feature = "gql"))]

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, ErrorCode};
use grafeo_engine::GrafeoDB;

const TEXT: &str = "canals";
/// Closest to Gus: the vector index ranks Gus, Vincent, Alix, Jules, Mia.
const VECTOR: [f32; 2] = [1.0, 0.0];

/// Who has `deleted` and how: `None` is a node without it.
const DOCS: [(&str, &str, Option<bool>, [f32; 2]); 5] = [
    ("Alix", "canals of Amsterdam", Some(false), [0.7, 0.3]),
    ("Gus", "canals and bridges", Some(true), [1.0, 0.0]),
    ("Vincent", "canals at night", None, [0.9, 0.1]),
    (
        "Jules",
        "Amsterdam canals in winter",
        Some(false),
        [0.5, 0.5],
    ),
    ("Mia", "a quiet street", Some(false), [0.0, 1.0]),
];

fn documents() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (owner, text, deleted, emb) in DOCS {
        let mut properties = HashMap::new();
        properties.insert(PropertyKey::new("owner"), Value::from(owner));
        properties.insert(PropertyKey::new("text"), Value::from(text));
        properties.insert(PropertyKey::new("emb"), Value::Vector(emb.to_vec().into()));
        if let Some(deleted) = deleted {
            properties.insert(PropertyKey::new("deleted"), Value::Bool(deleted));
        }
        db.create_node_with_props(&["Doc"], properties).unwrap();
    }
    db.create_text_index("Doc", "text").unwrap();
    db.create_vector_index("Doc", "emb", Some(2), Some("cosine"), None, None, None)
        .unwrap();
    db
}

fn owner(db: &GrafeoDB, node: NodeId) -> String {
    match db
        .get_node(node)
        .unwrap()
        .expect("a hit exists")
        .properties
        .get(&PropertyKey::new("owner"))
    {
        Some(Value::String(owner)) => owner.to_string(),
        other => panic!("{node:?}: {other:?}"),
    }
}

fn ops(pairs: &[(&str, Value)]) -> Value {
    Value::Map(Arc::new(
        pairs
            .iter()
            .map(|(op, value)| (PropertyKey::new(*op), value.clone()))
            .collect(),
    ))
}

fn list(values: &[Value]) -> Value {
    Value::List(values.to_vec().into())
}

fn filter(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

/// The decided filters and the owners each allows, of all five.
fn cases() -> Vec<(
    &'static str,
    HashMap<String, Value>,
    &'static [&'static str],
)> {
    let t = Value::Bool(true);
    let f = Value::Bool(false);
    vec![
        (
            "equal to false: a node without the flag fails",
            filter(&[("deleted", f.clone())]),
            &["Alix", "Jules", "Mia"],
        ),
        (
            "$ne true: a node without the flag is kept",
            filter(&[("deleted", ops(&[("$ne", t.clone())]))]),
            &["Alix", "Vincent", "Jules", "Mia"],
        ),
        (
            "$nin [true]: kept too",
            filter(&[("deleted", ops(&[("$nin", list(std::slice::from_ref(&t)))]))]),
            &["Alix", "Vincent", "Jules", "Mia"],
        ),
        (
            "$in [false]: needs the flag",
            filter(&[("deleted", ops(&[("$in", list(std::slice::from_ref(&f)))]))]),
            &["Alix", "Jules", "Mia"],
        ),
        (
            "$ne and $nin together: kept",
            filter(&[(
                "deleted",
                ops(&[("$ne", t.clone()), ("$nin", list(std::slice::from_ref(&t)))]),
            )]),
            &["Alix", "Vincent", "Jules", "Mia"],
        ),
        (
            "$ne with $in: the $in needs the flag",
            filter(&[(
                "deleted",
                ops(&[("$ne", t.clone()), ("$in", list(std::slice::from_ref(&f)))]),
            )]),
            &["Alix", "Jules", "Mia"],
        ),
        (
            "every hit filtered",
            filter(&[("owner", Value::from("Gus")), ("deleted", f.clone())]),
            &[],
        ),
        (
            "a property no node has",
            filter(&[("archived", t.clone())]),
            &[],
        ),
        (
            "$ne on a property no node has",
            filter(&[("archived", ops(&[("$ne", t.clone())]))]),
            &["Alix", "Gus", "Vincent", "Jules", "Mia"],
        ),
        (
            "an empty map",
            HashMap::new(),
            &["Alix", "Gus", "Vincent", "Jules", "Mia"],
        ),
    ]
}

type Hits = Vec<(String, u64)>;

/// Each search kind with `filters`: its hits as owners and score bits, best first.
fn searches() -> Vec<(
    &'static str,
    fn(&GrafeoDB, usize, Option<&HashMap<String, Value>>) -> Result<Hits, Error>,
)> {
    fn named(db: &GrafeoDB, hits: Vec<(NodeId, f64)>) -> Hits {
        hits.into_iter()
            .map(|(node, score)| (owner(db, node), score.to_bits()))
            .collect()
    }
    vec![
        ("text", |db, k, filters| {
            db.text_search("Doc", "text", TEXT, k, filters)
                .map(|hits| named(db, hits))
        }),
        ("vector", |db, k, filters| {
            db.vector_search("Doc", "emb", &VECTOR, k, None, filters)
                .map(|hits| {
                    named(
                        db,
                        hits.into_iter().map(|(n, d)| (n, f64::from(d))).collect(),
                    )
                })
        }),
        ("batch vector", |db, k, filters| {
            db.batch_vector_search("Doc", "emb", &[VECTOR.to_vec()], k, None, filters)
                .map(|mut batches| {
                    named(
                        db,
                        batches
                            .remove(0)
                            .into_iter()
                            .map(|(n, d)| (n, f64::from(d)))
                            .collect(),
                    )
                })
        }),
        ("mmr", |db, k, filters| {
            db.mmr_search("Doc", "emb", &VECTOR, k, Some(10), None, None, filters)
                .map(|hits| {
                    named(
                        db,
                        hits.into_iter().map(|(n, d)| (n, f64::from(d))).collect(),
                    )
                })
        }),
        ("hybrid", |db, k, filters| {
            db.hybrid_search(
                "Doc",
                "text",
                "emb",
                TEXT,
                Some(&VECTOR[..]),
                k,
                None,
                filters,
            )
            .map(|hits| named(db, hits))
        }),
    ]
}

/// Whether the search's ranks are its own (text, vector) or depend on the list (MMR, hybrid).
fn ranks_hold(kind: &str) -> bool {
    matches!(kind, "text" | "vector" | "batch vector")
}

fn check_every_case(db: &GrafeoDB, note: &str) {
    for (kind, search) in searches() {
        let unfiltered = search(db, 10, None).unwrap();
        assert!(!unfiltered.is_empty(), "{kind}");
        assert_eq!(
            search(db, 10, Some(&HashMap::new())).unwrap(),
            unfiltered,
            "{kind}: an empty map is no filter"
        );
        for (case, filters, allowed) in cases() {
            let found = search(db, 10, Some(&filters))
                .unwrap_or_else(|e| panic!("{note} {kind}, {case}: {e}"));
            let kept: Hits = unfiltered
                .iter()
                .filter(|(owner, _)| allowed.contains(&owner.as_str()))
                .cloned()
                .collect();
            if ranks_hold(kind) {
                assert_eq!(found, kept, "{note} {kind}, {case}");
            } else {
                let names = |hits: &Hits| -> BTreeSet<String> {
                    hits.iter().map(|(owner, _)| owner.clone()).collect()
                };
                assert_eq!(names(&found), names(&kept), "{note} {kind}, {case}");
            }
            // k counts the matching nodes: k = 1 is the best one the filter allows.
            if ranks_hold(kind) {
                let best = search(db, 1, Some(&filters)).unwrap();
                assert_eq!(
                    best,
                    kept.into_iter().take(1).collect::<Hits>(),
                    "{note} {kind}, {case}, k = 1"
                );
            }
        }
    }
}

/// Rules 1 to 7 and 12, in every search kind.
#[test]
fn every_search_keeps_the_nodes_the_filter_allows() {
    check_every_case(&documents(), "no index on deleted:");
}

/// Rule 9: a property index on `deleted` changes no result.
#[test]
fn a_property_index_on_the_flag_changes_nothing() {
    let db = documents();
    db.create_property_index("deleted").unwrap();
    check_every_case(&db, "with an index on deleted:");
}

/// Rule 8: a flag set to null counts as no flag.
#[test]
fn a_flag_set_to_null_counts_as_none() {
    let db = documents();
    db.execute("MATCH (d:Doc {owner: 'Alix'}) SET d.deleted = NULL")
        .unwrap();
    let names = |filters: HashMap<String, Value>| -> BTreeSet<String> {
        db.text_search("Doc", "text", TEXT, 10, Some(&filters))
            .unwrap()
            .into_iter()
            .map(|(node, _)| owner(&db, node))
            .collect()
    };
    assert_eq!(
        names(filter(&[("deleted", ops(&[("$ne", Value::Bool(true))]))])),
        BTreeSet::from(["Alix".into(), "Vincent".into(), "Jules".into()])
    );
    assert_eq!(
        names(filter(&[("deleted", Value::Bool(false))])),
        BTreeSet::from(["Jules".into()])
    );
}

/// `error` is an invalid-argument error naming `key` (and `also`, when given).
fn assert_invalid(case: &str, result: Result<Hits, Error>, key: &str, also: &str) {
    let error = result.expect_err(case);
    let message = error.to_string();
    assert_eq!(
        error.error_code(),
        ErrorCode::InvalidInput,
        "{case}: {message}"
    );
    assert!(message.contains(key), "{case}: names {key}: {message}");
    assert!(message.contains(also), "{case}: names {also}: {message}");
}

/// Rule 8 and the ruling: a null filter value points to `$ne`; a malformed filter names its
/// operator. Every search kind refuses them alike.
#[test]
fn null_values_and_malformed_filters_are_refused_by_every_search() {
    let db = documents();
    let null = Value::Null;
    let nulls = [
        ("equal to null", filter(&[("deleted", null.clone())])),
        (
            "$ne null",
            filter(&[("deleted", ops(&[("$ne", null.clone())]))]),
        ),
        (
            "null in $in",
            filter(&[(
                "deleted",
                ops(&[("$in", list(std::slice::from_ref(&null)))]),
            )]),
        ),
        (
            "null in $nin",
            filter(&[(
                "deleted",
                ops(&[("$nin", list(&[Value::Bool(false), null.clone()]))]),
            )]),
        ),
    ];
    let malformed = [
        (
            "an unknown operator",
            filter(&[("deleted", ops(&[("$near", Value::Int64(1))]))]),
            "$near",
        ),
        (
            "$in with a non-list",
            filter(&[("deleted", ops(&[("$in", Value::Bool(true))]))]),
            "$in",
        ),
        (
            "$nin with a non-list",
            filter(&[("deleted", ops(&[("$nin", Value::from("x"))]))]),
            "$nin",
        ),
        (
            "$contains with a non-string",
            filter(&[("owner", ops(&[("$contains", Value::Int64(5))]))]),
            "$contains",
        ),
    ];
    for (kind, search) in searches() {
        for (case, filters) in &nulls {
            assert_invalid(
                &format!("{kind}, {case}"),
                search(&db, 10, Some(filters)),
                "deleted",
                "$ne",
            );
        }
        for (case, filters, operator) in &malformed {
            let key = filters.keys().next().unwrap();
            assert_invalid(
                &format!("{kind}, {case}"),
                search(&db, 10, Some(filters)),
                key,
                operator,
            );
        }
    }
}

/// The rows `CALL` gives, as owners and score bits.
fn called(db: &GrafeoDB, query: &str, filters: Option<Value>) -> Result<Hits, Error> {
    let params = filters.map_or_else(HashMap::new, |filters| {
        HashMap::from([("filters".to_string(), filters)])
    });
    let result = db.execute_with_params(query, params)?;
    Ok(result
        .rows()
        .iter()
        .map(|row| {
            let node = NodeId::new(u64::try_from(row[0].as_int64().unwrap()).unwrap());
            let score = match &row[1] {
                Value::Float64(score) => *score,
                other => panic!("a score: {other:?}"),
            };
            (owner(db, node), score.to_bits())
        })
        .collect())
}

/// Rule 10: the three procedures take the filters map (here as a parameter), keep k and give
/// the API's hits with the same scores; without it they search as before; a non-map is a
/// parameter error.
#[test]
fn the_search_procedures_take_the_filters() {
    let db = documents();
    let not_deleted = Value::Map(Arc::new(
        [(
            PropertyKey::new("deleted"),
            ops(&[("$ne", Value::Bool(true))]),
        )]
        .into_iter()
        .collect(),
    ));
    let as_map = filter(&[("deleted", ops(&[("$ne", Value::Bool(true))]))]);
    for (kind, call, unfiltered_call) in [
        (
            "text",
            "CALL grafeo.search.text('Doc', 'text', 'canals', 1, $filters) YIELD node_id, score \
             RETURN node_id, score",
            "CALL grafeo.search.text('Doc', 'text', 'canals', 1) YIELD node_id, score \
             RETURN node_id, score",
        ),
        (
            "vector",
            "CALL grafeo.search.vector('Doc', 'emb', [1.0, 0.0], 1, $filters) \
             YIELD node_id, distance RETURN node_id, distance",
            "CALL grafeo.search.vector('Doc', 'emb', [1.0, 0.0], 1) \
             YIELD node_id, distance RETURN node_id, distance",
        ),
        (
            "mmr",
            "CALL grafeo.search.mmr('Doc', 'emb', [1.0, 0.0], 1, 10, 0.5, $filters) \
             YIELD node_id, distance RETURN node_id, distance",
            "CALL grafeo.search.mmr('Doc', 'emb', [1.0, 0.0], 1, 10, 0.5) \
             YIELD node_id, distance RETURN node_id, distance",
        ),
    ] {
        let found =
            called(&db, call, Some(not_deleted.clone())).unwrap_or_else(|e| panic!("{kind}: {e}"));
        assert_eq!(found.len(), 1, "{kind}: k is kept");
        assert_ne!(found[0].0, "Gus", "{kind}: Gus is deleted");
        let api = match kind {
            "text" => db
                .text_search("Doc", "text", TEXT, 1, Some(&as_map))
                .unwrap()
                .into_iter()
                .map(|(n, s)| (owner(&db, n), s.to_bits()))
                .collect::<Hits>(),
            _ => db
                .vector_search("Doc", "emb", &VECTOR, 1, None, Some(&as_map))
                .unwrap()
                .into_iter()
                .map(|(n, d)| (owner(&db, n), f64::from(d).to_bits()))
                .collect::<Hits>(),
        };
        if kind != "mmr" {
            assert_eq!(found, api, "{kind}: the API's hit and score");
        }
        let unfiltered = called(&db, unfiltered_call, None).unwrap();
        assert_eq!(unfiltered.len(), 1, "{kind}");
        let error = called(&db, call, Some(Value::from("deleted"))).expect_err(kind);
        assert!(error.to_string().contains("filters"), "{kind}: {error}");
    }
}

/// Rule 11: a call in a transaction sees its uncommitted writes; another session and the API
/// read the published state; after the rollback the node is back.
#[test]
fn a_call_in_a_transaction_filters_on_its_own_writes() {
    let db = documents();
    let call = "CALL grafeo.search.text('Doc', 'text', 'canals', 10, $filters) \
                YIELD node_id, score RETURN node_id, score";
    let live = Value::Map(Arc::new(
        [(PropertyKey::new("deleted"), Value::Bool(false))]
            .into_iter()
            .collect(),
    ));
    let params = || HashMap::from([("filters".to_string(), live.clone())]);
    let owners_of = |rows: &[Vec<Value>]| -> BTreeSet<String> {
        rows.iter()
            .map(|row| {
                owner(
                    &db,
                    NodeId::new(u64::try_from(row[0].as_int64().unwrap()).unwrap()),
                )
            })
            .collect()
    };
    let both: BTreeSet<String> = ["Alix".into(), "Jules".into()].into();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (d:Doc {owner: 'Jules'}) SET d.deleted = true")
        .unwrap();
    let inside = writer.execute_with_params(call, params()).unwrap();
    assert_eq!(owners_of(inside.rows()), BTreeSet::from(["Alix".into()]));
    let other = db.session().execute_with_params(call, params()).unwrap();
    assert_eq!(owners_of(other.rows()), both, "another session");
    let api: BTreeSet<String> = db
        .text_search(
            "Doc",
            "text",
            TEXT,
            10,
            Some(&filter(&[("deleted", Value::Bool(false))])),
        )
        .unwrap()
        .into_iter()
        .map(|(node, _)| owner(&db, node))
        .collect();
    assert_eq!(api, both, "the API reads the published epoch");
    writer.rollback().unwrap();
    let after = writer.execute_with_params(call, params()).unwrap();
    assert_eq!(owners_of(after.rows()), both, "after the rollback");
}
