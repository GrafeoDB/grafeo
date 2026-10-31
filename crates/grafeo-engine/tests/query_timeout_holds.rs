//! The query timeout stops every query shape, close to its limit: a count,
//! a sort or a filter over a long `UNWIND` or a cross product, and a
//! Cartesian `MATCH`. The deadline was checked only between the batches a
//! query returns, so a query that aggregates, sorts or filters billions of
//! rows before returning one ran on far past it (a 1 s limit held for none
//! of these after 25 s), and a host that runs generated queries, such as an
//! LLM agent or a server, hung. Each shape runs in a child process that the
//! test kills past its own deadline, so a query that ignores the timeout
//! fails the test instead of hanging it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test query_timeout_holds
//! ```

#![cfg(feature = "gql")]

use std::process::Command;
use std::time::{Duration, Instant};

use grafeo_common::testing::child_process;
use grafeo_engine::{Config, GrafeoDB};

/// The variable naming the shape the child runs.
const CHILD_VAR: &str = "GRAFEO_QUERY_TIMEOUT_CHILD";

/// The child's exit codes.
const TIMED_OUT: i32 = 0;
const FINISHED: i32 = 3;
const OTHER_ERROR: i32 = 4;
const TIMED_OUT_LATE: i32 = 5;

/// The limit the child's database sets, and how late its error may come:
/// one batch of work past the limit, on a slow CI runner.
const LIMIT: Duration = Duration::from_secs(1);
const SLACK: Duration = Duration::from_secs(4);

/// Each shape takes far longer than the limit on any machine.
const SHAPES: &[(&str, &str)] = &[
    (
        "a count over a long UNWIND",
        "UNWIND range(1, 3000000000) AS x RETURN count(x)",
    ),
    (
        "a count over a cross product",
        "UNWIND range(1, 100000) AS x UNWIND range(1, 100000) AS y RETURN count(*)",
    ),
    (
        "a sort over a cross product",
        "UNWIND range(1, 100000) AS x UNWIND range(1, 100000) AS y \
         RETURN x, y ORDER BY x DESC LIMIT 1",
    ),
    (
        "a filter no row passes over a cross product",
        "UNWIND range(1, 100000) AS x UNWIND range(1, 100000) AS y \
         FILTER x = y + 200000 RETURN x",
    ),
    (
        "rows returned from a long UNWIND",
        "UNWIND range(1, 300000000) AS x RETURN x",
    ),
    (
        "a Cartesian MATCH of thirty patterns",
        "MATCH (a:P), (b:P), (c:P), (d:P), (e:P), (f:P), (g:P), (h:P), (i:P), (j:P), \
         (k:P), (l:P), (m:P), (n:P), (o:P), (p:P), (q:P), (r:P), (s:P), (t:P), \
         (u:P), (v:P), (w:P), (x:P), (y:P), (z:P), (a2:P), (b2:P), (c2:P), (d2:P) \
         RETURN count(*)",
    ),
];

/// A database with the limit and two nodes for the MATCH.
fn db() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_query_timeout(LIMIT)).unwrap();
    db.execute("INSERT (:P {n: 1}), (:P {n: 2})").unwrap();
    db
}

/// The child: runs shape `index` and exits with [`TIMED_OUT`] when it fails
/// with the timeout error within the slack. A no-op when run directly.
#[test]
fn query_timeout_child() {
    let Ok(index) = std::env::var(CHILD_VAR) else {
        return;
    };
    let (_, query) = SHAPES[index.parse::<usize>().unwrap()];
    let db = db();
    let started = Instant::now();
    let outcome = db.execute(query);
    let elapsed = started.elapsed();
    println!("ELAPSED {elapsed:?}");
    match outcome {
        Ok(result) => {
            println!("FINISHED with {} rows", result.rows().len());
            std::process::exit(FINISHED);
        }
        Err(error) => {
            let message = error.to_string();
            println!("ERROR {message}");
            if !message.contains("timeout") {
                std::process::exit(OTHER_ERROR);
            }
            // The database still answers.
            assert!(db.execute("MATCH (p:P) RETURN count(p)").is_ok());
            std::process::exit(if elapsed <= LIMIT + SLACK {
                TIMED_OUT
            } else {
                TIMED_OUT_LATE
            });
        }
    }
}

/// Every shape fails with the timeout error, close to the limit.
#[test]
fn the_query_timeout_stops_every_shape() {
    let mut wrong = Vec::new();
    for (index, (name, _)) in SHAPES.iter().enumerate() {
        let started = Instant::now();
        let outcome = child_process::output_with_deadline(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "query_timeout_child", "--nocapture"])
                .env(CHILD_VAR, index.to_string()),
            LIMIT + SLACK + Duration::from_secs(15),
        );
        let output = match outcome {
            Ok(output) => output,
            Err(error) => {
                wrong.push(format!(
                    "{name}: still running after {:?} ({error})",
                    started.elapsed()
                ));
                continue;
            }
        };
        let text = String::from_utf8_lossy(&output.stdout);
        let said = |prefix: &str| {
            text.lines()
                .find(|line| line.starts_with(prefix))
                .unwrap_or("")
                .to_string()
        };
        match output.status.code() {
            Some(TIMED_OUT) => {}
            Some(TIMED_OUT_LATE) => {
                wrong.push(format!("{name}: timed out late, {}", said("ELAPSED")));
            }
            Some(FINISHED) => wrong.push(format!("{name}: finished, {}", said("FINISHED"))),
            Some(OTHER_ERROR) => wrong.push(format!("{name}: {}", said("ERROR"))),
            other => wrong.push(format!(
                "{name}: the process ended ({other:?}): {}",
                String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .rev()
                    .take(2)
                    .collect::<Vec<_>>()
                    .join(" / ")
            )),
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// Control: a query well within the limit is not stopped, and a database
/// without a timeout runs a long query to its end.
#[test]
fn queries_within_the_limit_finish() {
    let db = db();
    let rows = db
        .execute("UNWIND range(1, 100000) AS x RETURN count(x) AS c")
        .unwrap();
    assert_eq!(
        rows.rows()[0][0],
        grafeo_common::types::Value::Int64(100_000)
    );
    let unlimited = GrafeoDB::with_config(Config::in_memory().without_query_timeout()).unwrap();
    let rows = unlimited
        .execute("UNWIND range(1, 1000) AS x UNWIND range(1, 1000) AS y RETURN count(*) AS c")
        .unwrap();
    assert_eq!(
        rows.rows()[0][0],
        grafeo_common::types::Value::Int64(1_000_000)
    );
}
