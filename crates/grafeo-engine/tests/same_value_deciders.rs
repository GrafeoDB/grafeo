//! Class test 2: every place that decides whether two values are the same agrees with one of two
//! references (class-tests design, 2026-11-05).
//!
//! - **Equality form** (ISO/IEC 39075 19.3 General Rule 2): `a = b` as the evaluator answers it, three
//!   valued. A filter, `<>`, `IN` (short and long), a lookup with and without an index (also after a
//!   reopen), a join, `MERGE` and `update_nodes` find `a` for the key `b` exactly when `a = b` is true.
//! - **Grouping form** (22.12): `a` and `b` are the same when both are null, both NaN, two collections
//!   whose elements are pairwise the same, or `a = b` is true. `DISTINCT`, GROUP BY, `UNION`,
//!   `INTERSECT`, `EXCEPT`, `count(DISTINCT)`, `collect(DISTINCT)`, UNIQUE and NODE KEY keep one of each
//!   such set.
//!
//! The values are representatives of each `Value` variant the catalogue (`tests/catalogue/value.txt`)
//! names; every name needs a row here. A variant a property cannot hold is checked in the forms that
//! read parameters only, and the list of those is pinned too.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main same_value_deciders::
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use grafeo_common::types::rdf::Term;
use grafeo_common::types::{
    Date, Decimal, Duration, PropertyKey, Time, Timestamp, Value, ZonedDatetime,
};
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::PlannerRules;
use grafeo_engine::session::Session;

const CATALOGUE: &str = include_str!("../../../tests/catalogue/value.txt");

fn list(items: Vec<Value>) -> Value {
    Value::List(items.into())
}

fn map(pairs: &[(&str, Value)]) -> Value {
    Value::Map(Arc::new(
        pairs
            .iter()
            .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
            .collect::<BTreeMap<_, _>>(),
    ))
}

fn decimal(text: &str) -> Value {
    Value::Decimal(Decimal::parse(text).unwrap())
}

fn counts(pairs: &[(&str, u64)]) -> Arc<HashMap<String, u64>> {
    Arc::new(
        pairs
            .iter()
            .map(|(replica, count)| ((*replica).to_string(), *count))
            .collect(),
    )
}

/// The representatives of the catalogue's variant `name`: the spellings that test a decider. Where the
/// two forms part (null, NaN, a null inside a collection), a variant has two values that only the
/// grouping form calls the same.
fn representatives(name: &str) -> Vec<Value> {
    match name {
        "Null" => vec![Value::Null, Value::Null],
        "Bool" => vec![Value::Bool(false), Value::Bool(true)],
        "Int64" => vec![
            Value::Int64(0),
            Value::Int64(1),
            Value::Int64(-1),
            Value::Int64(5),
            Value::Int64(1000),
            Value::Int64(9_007_199_254_740_993),
        ],
        "Float64" => vec![
            Value::Float64(0.0),
            Value::Float64(-0.0),
            Value::Float64(1.0),
            Value::Float64(1.5),
            Value::Float64(5.0),
            Value::Float64(9_007_199_254_740_992.0),
            Value::Float64(f64::NAN),
            Value::Float64(-f64::NAN),
            Value::Float64(f64::INFINITY),
        ],
        "Decimal" => vec![
            decimal("1.5"),
            decimal("1.50"),
            decimal("5"),
            decimal("1000.0"),
        ],
        "String" => vec![
            Value::from(""),
            Value::from("1"),
            Value::from("5"),
            Value::from("05"),
            Value::from("5.0"),
            Value::from("NaN"),
            Value::from("true"),
            Value::from("Alix"),
        ],
        "Bytes" => vec![
            Value::Bytes(vec![1_u8, 2].into()),
            Value::Bytes(vec![1_u8].into()),
        ],
        "Timestamp" => vec![Value::Timestamp(Timestamp::from_micros(
            1_704_103_200_000_000,
        ))],
        "Date" => vec![
            Value::Date(Date::parse("2024-01-01").unwrap()),
            Value::Date(Date::parse("2024-06-15").unwrap()),
        ],
        "Time" => vec![
            Value::Time(Time::parse("10:00:00").unwrap()),
            Value::Time(Time::parse("10:00:00+01:00").unwrap()),
            Value::Time(Time::parse("09:00:00Z").unwrap()),
        ],
        "Duration" => vec![
            Value::Duration(Duration::parse("P1D").unwrap()),
            Value::Duration(Duration::parse("PT24H").unwrap()),
        ],
        "ZonedDatetime" => vec![
            Value::ZonedDatetime(ZonedDatetime::parse("2024-01-01T10:00:00+01:00").unwrap()),
            Value::ZonedDatetime(ZonedDatetime::parse("2024-01-01T09:00:00Z").unwrap()),
        ],
        "List" => vec![
            list(vec![]),
            list(vec![Value::Int64(1)]),
            list(vec![Value::Float64(1.0)]),
            list(vec![Value::from("1")]),
            list(vec![Value::Null]),
            list(vec![Value::Null]),
            list(vec![Value::Int64(1), Value::Null]),
            list(vec![Value::Float64(1.0), Value::Null]),
            list(vec![list(vec![Value::Null])]),
            list(vec![Value::Float64(f64::NAN)]),
        ],
        "Map" => vec![
            map(&[]),
            map(&[("a", Value::Int64(1))]),
            map(&[("a", Value::Float64(1.0))]),
            map(&[("a", Value::from("1"))]),
            map(&[("a", Value::Null)]),
            map(&[("a", Value::Null)]),
        ],
        "Vector" => vec![
            Value::Vector(vec![1.0_f32, 2.0].into()),
            Value::Vector(vec![0.0_f32].into()),
            Value::Vector(vec![-0.0_f32].into()),
        ],
        "Path" => vec![Value::Path {
            nodes: vec![map(&[("_id", Value::Int64(0))])].into(),
            edges: vec![].into(),
        }],
        "GCounter" => vec![
            Value::GCounter(counts(&[("r1", 2)])),
            Value::GCounter(counts(&[("r1", 1), ("r2", 1)])),
        ],
        "OnCounter" => vec![Value::OnCounter {
            pos: counts(&[("r1", 3)]),
            neg: counts(&[("r1", 1)]),
        }],
        "RdfTerm" => vec![
            Value::RdfTerm(Arc::new(Term::iri("urn:a"))),
            Value::RdfTerm(Arc::new(Term::iri("urn:b"))),
        ],
        other => panic!("the catalogue names {other}: give it a row in `representatives`"),
    }
}

/// The variants a property cannot hold: they are read through parameters only.
const NOT_STORED: &[&str] = &[];

fn catalogue_names() -> Vec<&'static str> {
    CATALOGUE
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// Every representative, with its variant's name.
fn values() -> Vec<(&'static str, Value)> {
    catalogue_names()
        .into_iter()
        .flat_map(|name| representatives(name).into_iter().map(move |v| (name, v)))
        .collect()
}

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

fn rows(session: &Session, query: &str, pairs: &[(&str, Value)]) -> Vec<Vec<Value>> {
    session
        .execute_with_params(query, params(pairs))
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

fn int(value: &Value) -> i64 {
    match value {
        Value::Int64(number) => *number,
        other => panic!("not an integer: {other:?}"),
    }
}

/// An index or a count as the integer a query reads.
fn wide(n: usize) -> i64 {
    i64::try_from(n).unwrap()
}

/// An integer a query returns as an index.
fn index(value: &Value) -> usize {
    usize::try_from(int(value)).unwrap()
}

/// The indexes of the stored nodes a query returns as its first column.
fn found(session: &Session, query: &str, pairs: &[(&str, Value)]) -> BTreeSet<usize> {
    rows(session, query, pairs)
        .iter()
        .map(|row| usize::try_from(int(&row[0])).unwrap())
        .collect()
}

/// `a = b` for every pair, from one query: the equality form's reference.
struct Reference {
    values: Vec<Value>,
    equal: Vec<Vec<Option<bool>>>,
    leaves: std::cell::RefCell<HashMap<(String, String), Option<bool>>>,
    session: Session,
}

impl Reference {
    fn new(db: &GrafeoDB, values: &[Value]) -> Self {
        let session = db.session();
        let n = values.len();
        let mut equal = vec![vec![None; n]; n];
        let all = list(values.to_vec());
        for row in rows(
            &session,
            "UNWIND range(0, $n - 1) AS i UNWIND range(0, $n - 1) AS j \
             RETURN i, j, $vals[i] = $vals[j] AS e",
            &[("n", Value::Int64(wide(n))), ("vals", all)],
        ) {
            let (i, j) = (index(&row[0]), index(&row[1]));
            equal[i][j] = match &row[2] {
                Value::Bool(flag) => Some(*flag),
                Value::Null => None,
                other => panic!("{other:?}"),
            };
        }
        Self {
            values: values.to_vec(),
            equal,
            leaves: std::cell::RefCell::new(HashMap::new()),
            session,
        }
    }

    /// `a = b` of two values, which need not be representatives (collection elements).
    fn equal_values(&self, a: &Value, b: &Value) -> Option<bool> {
        let key = (format!("{a:?}"), format!("{b:?}"));
        if let Some(known) = self.leaves.borrow().get(&key) {
            return *known;
        }
        let answer = match &rows(
            &self.session,
            "RETURN $a = $b AS e",
            &[("a", a.clone()), ("b", b.clone())],
        )[0][0]
        {
            Value::Bool(flag) => Some(*flag),
            _ => None,
        };
        self.leaves.borrow_mut().insert(key, answer);
        answer
    }

    /// The grouping form (22.12).
    fn same_values(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Null, _) | (_, Value::Null) => false,
            (Value::Float64(x), Value::Float64(y)) if x.is_nan() && y.is_nan() => true,
            (Value::List(x), Value::List(y)) => {
                x.len() == y.len() && x.iter().zip(y.iter()).all(|(x, y)| self.same_values(x, y))
            }
            (Value::Map(x), Value::Map(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|((kx, x), (ky, y))| kx == ky && self.same_values(x, y))
            }
            _ => self.equal_values(a, b) == Some(true),
        }
    }

    fn same(&self, i: usize, j: usize) -> bool {
        self.same_values(&self.values[i], &self.values[j])
    }
}

/// The values at `indexes`, for a message.
fn shown<'a>(values: &'a [Value], indexes: &BTreeSet<usize>) -> Vec<&'a Value> {
    indexes.iter().map(|&i| &values[i]).collect()
}

#[test]
fn every_variant_of_the_catalogue_has_representatives() {
    for name in catalogue_names() {
        assert!(!representatives(name).is_empty(), "{name}");
    }
}

/// Stores each representative as `(:V {i, v})`; returns the indexes stored.
fn store(session: &Session, values: &[(&str, Value)], label: &str) -> BTreeSet<usize> {
    let mut stored = BTreeSet::new();
    for (i, (_, value)) in values.iter().enumerate() {
        let insert = format!("INSERT (:{label} {{i: $i, v: $v}})");
        if session
            .execute_with_params(
                &insert,
                params(&[("i", Value::Int64(wide(i))), ("v", value.clone())]),
            )
            .is_ok()
        {
            stored.insert(i);
        }
    }
    stored
}

/// The lookups of the equality form, each key `b` against every stored value.
fn equality_lookups(
    session: &Session,
    reference: &Reference,
    stored: &BTreeSet<usize>,
    stage: &str,
) -> Vec<String> {
    let values = &reference.values;
    let mut wrong = Vec::new();
    let pads: Vec<Value> = (0..20)
        .map(|k| Value::from(format!("\u{1}pad{k}").as_str()))
        .collect();
    for (b, key) in values.iter().enumerate() {
        let equal: BTreeSet<usize> = stored
            .iter()
            .copied()
            .filter(|&a| reference.equal[a][b] == Some(true))
            .collect();
        let unequal: BTreeSet<usize> = stored
            .iter()
            .copied()
            .filter(|&a| reference.equal[a][b] == Some(false))
            .collect();
        let mut long = vec![key.clone()];
        long.extend(pads.iter().cloned());
        for (form, query, want) in [
            ("filter", "MATCH (n:V) WHERE n.v = $b RETURN n.i", &equal),
            ("pattern", "MATCH (n:V {v: $b}) RETURN n.i", &equal),
            ("no label", "MATCH (n) WHERE n.v = $b RETURN n.i", &equal),
            ("<>", "MATCH (n:V) WHERE n.v <> $b RETURN n.i", &unequal),
            (
                "IN short",
                "MATCH (n:V) WHERE n.v IN [$b] RETURN n.i",
                &equal,
            ),
            (
                "IN long",
                "MATCH (n:V) WHERE n.v IN $long RETURN n.i",
                &equal,
            ),
        ] {
            let got = found(
                session,
                query,
                &[("b", key.clone()), ("long", list(long.clone()))],
            );
            if &got != want {
                wrong.push(format!(
                    "{stage}, {form}, key {key:?}: found {:?}, want {:?}",
                    shown(values, &got),
                    shown(values, want)
                ));
            }
        }
    }
    wrong
}

#[test]
fn equality_deciders_find_what_equals_finds() {
    let named = values();
    let values: Vec<Value> = named.iter().map(|(_, v)| v.clone()).collect();
    let db = GrafeoDB::new_in_memory();
    let reference = Reference::new(&db, &values);
    let session = db.session();
    let stored = store(&session, &named, "V");
    let refused: Vec<&str> = named
        .iter()
        .enumerate()
        .filter(|(i, _)| !stored.contains(i))
        .map(|(_, (name, _))| *name)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    assert_eq!(refused, NOT_STORED, "the variants a property cannot hold");
    let mut wrong = Vec::new();
    // The plan of each rule set: the scan's filter, the key's column, then the indexes.
    let plain = {
        let mut session = db.session();
        session.set_planner_rules(PlannerRules::plain());
        session
    };
    wrong.extend(equality_lookups(&plain, &reference, &stored, "plain"));
    wrong.extend(equality_lookups(
        &session,
        &reference,
        &stored,
        "key column",
    ));
    session
        .execute("CREATE INDEX v_of_v FOR (n:V) ON (n.v)")
        .unwrap();
    wrong.extend(equality_lookups(
        &session,
        &reference,
        &stored,
        "label index",
    ));
    db.create_property_index("v").unwrap();
    wrong.extend(equality_lookups(
        &session,
        &reference,
        &stored,
        "property index",
    ));

    // A join on the values: the pairs `=` holds of.
    let joined = db.session();
    store(&joined, &named, "A");
    store(&joined, &named, "B");
    let got: BTreeSet<(usize, usize)> = rows(
        &joined,
        "MATCH (a:A), (b:B) WHERE a.v = b.v RETURN a.i, b.i",
        &[],
    )
    .iter()
    .map(|row| (index(&row[0]), index(&row[1])))
    .collect();
    let want: BTreeSet<(usize, usize)> = stored
        .iter()
        .flat_map(|&a| stored.iter().map(move |&b| (a, b)))
        .filter(|&(a, b)| reference.equal[a][b] == Some(true))
        .collect();
    for pair in got.symmetric_difference(&want) {
        wrong.push(format!(
            "join: {:?} = {:?}: found {}, want {}",
            values[pair.0],
            values[pair.1],
            got.contains(pair),
            want.contains(pair)
        ));
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The same lookups after a reopen, from the file's columns and its index.
#[test]
fn equality_deciders_find_what_equals_finds_reopened() {
    let named = values();
    let values: Vec<Value> = named.iter().map(|(_, v)| v.clone()).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("same_value.grafeo");
    let stored = {
        let db = GrafeoDB::open(&path).unwrap();
        let stored = store(&db.session(), &named, "V");
        db.close().unwrap();
        stored
    };
    let db = GrafeoDB::open(&path).unwrap();
    let reference = Reference::new(&db, &values);
    let session = db.session();
    let mut wrong = equality_lookups(&session, &reference, &stored, "reopened, key column");
    db.create_property_index("v").unwrap();
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    wrong.extend(equality_lookups(
        &session,
        &reference,
        &stored,
        "reopened, index",
    ));
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    db.close().unwrap();
}

/// `MERGE` and `update_nodes` match a key as `=` does: one key at a time, against every stored value.
#[test]
fn merge_and_upsert_match_what_equals_finds() {
    let named = values();
    let values: Vec<Value> = named.iter().map(|(_, v)| v.clone()).collect();
    let db = GrafeoDB::new_in_memory();
    let reference = Reference::new(&db, &values);
    let mut wrong = Vec::new();
    for (b, key) in values.iter().enumerate() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let stored = store(&session, &named, "V");
        let want: BTreeSet<usize> = stored
            .iter()
            .copied()
            .filter(|&a| reference.equal[a][b] == Some(true))
            .collect();
        // A null key matches nothing and MERGE refuses it; an upsert row without its key is skipped.
        if !stored.contains(&b) || *key == Value::Null {
            continue;
        }
        // A MERGE that matches nothing creates a node, without `i`.
        let merged: BTreeSet<i64> = rows(
            &session,
            "MERGE (n:V {v: $b}) RETURN coalesce(n.i, -1) AS i",
            &[("b", key.clone())],
        )
        .iter()
        .map(|row| int(&row[0]))
        .collect();
        let merge_want: BTreeSet<i64> = if want.is_empty() {
            BTreeSet::from([-1])
        } else {
            want.iter().map(|&i| wide(i)).collect()
        };
        if merged != merge_want {
            wrong.push(format!("MERGE {key:?}: {merged:?}, want {merge_want:?}"));
        }
        let fresh = GrafeoDB::new_in_memory();
        let fresh_session = fresh.session();
        store(&fresh_session, &named, "V");
        let row = HashMap::from([
            (PropertyKey::new("v"), key.clone()),
            (PropertyKey::new("tier"), Value::Int64(1)),
        ]);
        let summary = fresh.update_nodes(&["V"], "v", vec![row]).unwrap();
        let updated = found(
            &fresh_session,
            "MATCH (n:V) WHERE n.tier = 1 RETURN n.i",
            &[],
        );
        // The summary counts rows: the one row updated its nodes, or matched none and was skipped.
        let rows_reported = (summary.updated, summary.skipped);
        let rows_wanted = if want.is_empty() { (0, 1) } else { (1, 0) };
        if updated != want || rows_reported != rows_wanted {
            wrong.push(format!(
                "update_nodes {key:?}: {updated:?} (rows updated, skipped: {rows_reported:?}), \
                 want {want:?} ({rows_wanted:?})"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The grouping deciders keep one of each set of values the grouping form calls the same.
#[test]
fn grouping_deciders_keep_one_of_each_same_set() {
    let named = values();
    let values: Vec<Value> = named.iter().map(|(_, v)| v.clone()).collect();
    let db = GrafeoDB::new_in_memory();
    let reference = Reference::new(&db, &values);
    let session = db.session();
    let n = values.len();
    let mut wrong = Vec::new();

    // GROUP BY: the classes themselves.
    let mut want_classes: Vec<BTreeSet<usize>> = Vec::new();
    for i in 0..n {
        match want_classes
            .iter_mut()
            .find(|class| reference.same(*class.iter().next().unwrap(), i))
        {
            Some(class) => {
                class.insert(i);
            }
            None => want_classes.push(BTreeSet::from([i])),
        }
    }
    let got_classes: BTreeSet<BTreeSet<usize>> = rows(
        &session,
        // Through WITH: a grouping key built from a parameter fails today (found 2026-11-06).
        "UNWIND range(0, $n - 1) AS i WITH i, $vals[i] AS x RETURN x, collect(i) AS members",
        &[("n", Value::Int64(wide(n))), ("vals", list(values.clone()))],
    )
    .iter()
    .map(|row| match &row[1] {
        Value::List(members) => members.iter().map(index).collect(),
        other => panic!("{other:?}"),
    })
    .collect();
    let want_set: BTreeSet<BTreeSet<usize>> = want_classes.iter().cloned().collect();
    for class in got_classes.symmetric_difference(&want_set) {
        let side = if got_classes.contains(class) {
            "found"
        } else {
            "wanted"
        };
        wrong.push(format!(
            "GROUP BY, {side} {:?}",
            class.iter().map(|&i| &values[i]).collect::<Vec<_>>()
        ));
    }

    // Each value against the others: whether it joins a set the others hold.
    for b in 0..n {
        let others: Vec<Value> = (0..n)
            .filter(|&j| j != b)
            .map(|j| values[j].clone())
            .collect();
        let joins = (0..n).any(|j| j != b && reference.same(j, b));
        let classes_of = |indexes: &[usize]| -> usize {
            let mut kept: Vec<usize> = Vec::new();
            for &i in indexes {
                if !kept.iter().any(|&k| reference.same(k, i)) {
                    kept.push(i);
                }
            }
            kept.len()
        };
        let all_idx: Vec<usize> = (0..n).collect();
        let with_b = classes_of(&all_idx);
        let non_null = |indexes: &[usize]| -> Vec<usize> {
            indexes
                .iter()
                .copied()
                .filter(|&i| values[i] != Value::Null)
                .collect()
        };
        let key = [("b", values[b].clone()), ("others", list(others.clone()))];
        let count = |query: &str| -> i64 {
            let result = rows(&session, query, &key);
            match &result[0][0] {
                Value::Int64(c) => *c,
                Value::List(items) => wide(items.len()),
                other => panic!("{query}: {other:?}"),
            }
        };
        for (form, got, want) in [
            (
                "DISTINCT",
                wide(
                    rows(
                        &session,
                        "UNWIND $others + [$b] AS x RETURN DISTINCT x",
                        &key,
                    )
                    .len(),
                ),
                wide(with_b),
            ),
            (
                "UNION",
                wide(
                    rows(
                        &session,
                        "UNWIND $others AS x RETURN x UNION RETURN $b AS x",
                        &key,
                    )
                    .len(),
                ),
                wide(with_b),
            ),
            (
                "INTERSECT",
                wide(
                    rows(
                        &session,
                        "RETURN $b AS x INTERSECT UNWIND $others AS x RETURN x",
                        &key,
                    )
                    .len(),
                ),
                i64::from(joins),
            ),
            (
                "EXCEPT",
                wide(
                    rows(
                        &session,
                        "RETURN $b AS x EXCEPT UNWIND $others AS x RETURN x",
                        &key,
                    )
                    .len(),
                ),
                i64::from(!joins),
            ),
            (
                "count(DISTINCT)",
                count("UNWIND $others + [$b] AS x RETURN count(DISTINCT x) AS c"),
                wide(classes_of(&non_null(&all_idx))),
            ),
            (
                "collect(DISTINCT)",
                count("UNWIND $others + [$b] AS x RETURN collect(DISTINCT x) AS c"),
                wide(classes_of(&non_null(&all_idx))),
            ),
        ] {
            if got != want {
                wrong.push(format!("{form}, {:?}: {got}, want {want}", values[b]));
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

/// UNIQUE and NODE KEY refuse a value the grouping form calls the same as one stored before; NODE KEY
/// refuses a null too.
#[test]
fn unique_and_node_key_refuse_the_same_value() {
    let named = values();
    let values: Vec<Value> = named.iter().map(|(_, v)| v.clone()).collect();
    let db = GrafeoDB::new_in_memory();
    let reference = Reference::new(&db, &values);
    let mut wrong = Vec::new();
    for (constraint, nulls_refused) in [
        ("CREATE CONSTRAINT FOR (n:U) ON (n.v) UNIQUE", false),
        ("CREATE CONSTRAINT u_key FOR (n:U) ON (n.v) NODE KEY", true),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let free = store(&session, &named, "Free");
        session.execute(constraint).unwrap();
        let mut kept: Vec<usize> = Vec::new();
        for (i, value) in values.iter().enumerate() {
            if !free.contains(&i) {
                continue;
            }
            let null = *value == Value::Null;
            let want_refused =
                (null && nulls_refused) || (!null && kept.iter().any(|&k| reference.same(k, i)));
            let refused = session
                .execute_with_params(
                    "INSERT (:U {i: $i, v: $v})",
                    params(&[("i", Value::Int64(wide(i))), ("v", value.clone())]),
                )
                .is_err();
            if refused != want_refused {
                wrong.push(format!(
                    "{constraint}: {value:?} refused {refused}, want {want_refused}"
                ));
            }
            if !refused && !null {
                kept.push(i);
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
