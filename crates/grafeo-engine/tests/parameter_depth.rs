//! A parameter value nested deeper than a property value may be (128 levels
//! of lists and maps, `MAX_PROPERTY_VALUE_DEPTH`) is refused with a clear
//! error, on every way a parameter reaches a query. Before, cloning,
//! comparing and dropping a parameter recursed without a limit, so a value
//! nested deep enough overflowed the stack and ended the process.
//!
//! The deep cases run in a child process: a stack overflow there ends the
//! child, which the parent sees, instead of ending the test run.

#![cfg(feature = "gql")]

use std::collections::{BTreeMap, HashMap};
use std::process::Command;
use std::sync::Arc;

use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
use grafeo_common::testing::child_process;
use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

const CHILD_VAR: &str = "GRAFEO_PARAMETER_DEPTH_CHILD";
/// Exit code of a child whose call failed with the depth error.
const REFUSED: i32 = 19;
/// Exit code of a child whose call succeeded where it should have failed.
const ACCEPTED: i32 = 23;

/// A list nested `depth` levels deep around the integer 1, built without
/// recursion.
fn deep_list(depth: usize) -> Value {
    let mut value = Value::Int64(1);
    for _ in 0..depth {
        value = Value::List(vec![value].into());
    }
    value
}

/// A map nested `depth` levels deep (`{a: {a: ... 1}}`).
fn deep_map(depth: usize) -> Value {
    let mut value = Value::Int64(1);
    for _ in 0..depth {
        value = Value::Map(Arc::new(BTreeMap::from([(PropertyKey::new("a"), value)])));
    }
    value
}

/// The value of case `shape:depth` (`list:129`, `map:100000`, `mixed:...`).
fn value_of(shape: &str, depth: usize) -> Value {
    match shape {
        "list" => deep_list(depth),
        "map" => deep_map(depth),
        // A shallow map whose one entry is a deep list.
        "mixed" => Value::Map(Arc::new(BTreeMap::from([(
            PropertyKey::new("inner"),
            deep_list(depth - 1),
        )]))),
        other => panic!("unknown shape {other}"),
    }
}

/// The ways a parameter reaches a query, as the child names them.
const ENTRIES: [&str; 5] = [
    "db-gql",
    "db-cypher",
    "session-gql",
    "session-transaction",
    "insert-property",
];

/// The child: runs `entry` with the parameter `shape:depth` and exits with
/// [`REFUSED`] when it fails with the depth error, printing the message. A
/// no-op when run directly.
#[test]
fn parameter_depth_child() {
    let Ok(which) = std::env::var(CHILD_VAR) else {
        return;
    };
    let mut parts = which.split(':');
    let (entry, shape, depth) = (
        parts.next().unwrap().to_string(),
        parts.next().unwrap().to_string(),
        parts.next().unwrap().parse::<usize>().unwrap(),
    );
    let db = GrafeoDB::new_in_memory();
    let params = HashMap::from([("p".to_string(), value_of(&shape, depth))]);
    let result = match entry.as_str() {
        "db-gql" => db.execute_with_params("RETURN $p AS p", params).map(|_| ()),
        #[cfg(feature = "cypher")]
        "db-cypher" => db
            .execute_cypher_with_params("RETURN $p AS p", params)
            .map(|_| ()),
        #[cfg(not(feature = "cypher"))]
        "db-cypher" => db.execute_with_params("RETURN $p AS p", params).map(|_| ()),
        "session-gql" => db
            .session()
            .execute_with_params("RETURN $p = $p AS same", params)
            .map(|_| ()),
        "session-transaction" => {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            let result = session
                .execute_with_params("RETURN $p AS x", params)
                .map(|_| ());
            let _ = session.rollback();
            result
        }
        "insert-property" => db
            .execute_with_params("INSERT (:N {p: $p})", params)
            .map(|_| ()),
        other => panic!("unknown entry {other}"),
    };
    match result {
        Ok(()) => {
            println!("ACCEPTED");
            std::process::exit(ACCEPTED);
        }
        Err(error) => {
            println!("REFUSED {error}");
            // The database still answers.
            assert_eq!(
                db.execute("RETURN 1 AS one").unwrap().rows()[0][0],
                Value::Int64(1)
            );
            std::process::exit(REFUSED);
        }
    }
}

/// Runs the child for `entry:shape:depth`; returns its exit code and output.
fn child(entry: &str, shape: &str, depth: usize) -> (Option<i32>, String) {
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "parameter_depth_child", "--nocapture"])
            .env(CHILD_VAR, format!("{entry}:{shape}:{depth}")),
    )
    .unwrap();
    (
        output.status.code(),
        format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

/// Past the limit, at one level too many and far past it, every entry
/// refuses the parameter with an error that names the limit, and the process
/// lives.
#[test]
fn a_parameter_nested_past_the_limit_is_refused_everywhere() {
    // Every case runs; the failures are listed together.
    let mut wrong = Vec::new();
    for depth in [MAX_PROPERTY_VALUE_DEPTH + 1, 100_000] {
        for shape in ["list", "map", "mixed"] {
            for entry in ENTRIES {
                let (code, output) = child(entry, shape, depth);
                let named = output.contains(&MAX_PROPERTY_VALUE_DEPTH.to_string());
                if code != Some(REFUSED) || !named {
                    let how = match code {
                        Some(ACCEPTED) => "accepted it".to_string(),
                        Some(REFUSED) => "refused it without naming the limit".to_string(),
                        other => format!("ended the process ({other:?})"),
                    };
                    wrong.push(format!("{entry}, {shape} {depth} deep: {how}"));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "each must fail with the depth error and the process live:\n{}",
        wrong.join("\n")
    );
}

/// The control: a parameter at the limit is a value like any other, in a
/// query and as a property.
#[test]
fn a_parameter_at_the_limit_works() {
    let db = GrafeoDB::new_in_memory();
    for value in [
        deep_list(MAX_PROPERTY_VALUE_DEPTH),
        deep_map(MAX_PROPERTY_VALUE_DEPTH),
    ] {
        let params = HashMap::from([("p".to_string(), value.clone())]);
        let rows = db
            .execute_with_params("RETURN $p AS p, $p = $p AS same", params.clone())
            .unwrap();
        assert_eq!(rows.rows()[0][0], value);
        assert_eq!(rows.rows()[0][1], Value::Bool(true));
        db.execute_with_params("INSERT (:N {p: $p})", params)
            .unwrap();
    }
    assert_eq!(
        db.execute("MATCH (n:N) RETURN count(n)").unwrap().rows()[0][0],
        Value::Int64(2)
    );
}
