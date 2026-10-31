//! Grouping and DISTINCT over timestamps and zoned datetimes that spill
//! under memory pressure follow each language's `=`, as they do in memory:
//! GQL reads a timestamp at the session time zone (UTC without one), so a
//! timestamp and a zoned datetime of one instant are one group; Cypher keeps
//! them apart. The groups are hashed and compared on reload, so the key of
//! a spilled group must be the language's identity, not the stored bytes.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test temporal_grouping_spill
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "spill"))]

use std::collections::HashMap;

use grafeo_common::memory::buffer::{MemoryGrant, MemoryRegion, PressureLevel};
use grafeo_common::types::{Timestamp, Value, ZonedDatetime};
use grafeo_engine::{Config, GrafeoDB};

/// The instants: one a minute from 2024-03-15T00:00:00Z.
const INSTANTS: i64 = 1024;
const BASE: i64 = 1_710_460_800;
/// Rounds of the rows, so the grouping runs long enough under pressure.
const ROUNDS: i64 = 11;

/// Every instant twice per round: as a timestamp (its UTC wall time) and as
/// a zoned datetime at +01:00.
fn rows() -> Value {
    let rows: Vec<Value> = (0..ROUNDS)
        .flat_map(|_| {
            (0..INSTANTS).flat_map(|i| {
                let at = Timestamp::from_secs(BASE + 60 * i);
                [
                    Value::Timestamp(at),
                    Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(at, 3600)),
                ]
            })
        })
        .collect();
    Value::List(rows.into())
}

/// A database whose buffer manager is under high pressure, so grouping
/// spills, and the allocation that holds it there.
fn under_pressure(spill_root: &std::path::Path) -> (GrafeoDB, MemoryGrant) {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_memory_limit(64 * 1024 * 1024)
            .with_spill_path(spill_root)
            .without_query_timeout(),
    )
    .unwrap();
    let manager = db.buffer_manager();
    let pressure = manager
        .try_allocate(
            manager.config().budget * 9 / 10,
            MemoryRegion::ExecutionBuffers,
        )
        .unwrap();
    assert_eq!(manager.pressure_level(), PressureLevel::High);
    (db, pressure)
}

/// The number of groups and the sizes of the groups, sorted.
fn groups(result: &grafeo_engine::database::QueryResult) -> (usize, Vec<i64>) {
    let mut sizes: Vec<i64> = result
        .rows()
        .iter()
        .map(|row| row[1].as_int64().unwrap())
        .collect();
    sizes.sort_unstable();
    sizes.dedup();
    (result.rows().len(), sizes)
}

const GROUP_BY: &str = "UNWIND $rows AS t RETURN t, count(*) AS c";
const DISTINCT: &str = "UNWIND $rows AS t RETURN count(DISTINCT t) AS d";

/// GQL at UTC: one group per instant, each with both kinds, every round.
#[test]
fn spilled_gql_groups_put_an_instant_in_one_group() {
    let spill_root = tempfile::tempdir().unwrap();
    let (db, _pressure) = under_pressure(spill_root.path());
    let params = HashMap::from([("rows".to_string(), rows())]);
    let result = db.execute_with_params(GROUP_BY, params.clone()).unwrap();
    assert_eq!(
        groups(&result),
        (usize::try_from(INSTANTS).unwrap(), vec![2 * ROUNDS])
    );
    let distinct = db.execute_with_params(DISTINCT, params).unwrap();
    assert_eq!(distinct.rows()[0][0], Value::Int64(INSTANTS));
}

/// Cypher: a timestamp is a local datetime, never equal to a zoned one, so
/// each instant is two groups.
#[test]
fn spilled_cypher_groups_keep_local_and_zoned_apart() {
    let spill_root = tempfile::tempdir().unwrap();
    let (db, _pressure) = under_pressure(spill_root.path());
    let params = HashMap::from([("rows".to_string(), rows())]);
    let result = db
        .execute_cypher_with_params(GROUP_BY, params.clone())
        .unwrap();
    assert_eq!(
        groups(&result),
        (2 * usize::try_from(INSTANTS).unwrap(), vec![ROUNDS])
    );
    let distinct = db.execute_cypher_with_params(DISTINCT, params).unwrap();
    assert_eq!(distinct.rows()[0][0], Value::Int64(2 * INSTANTS));
}

/// GQL at +02:00: a timestamp reads two hours earlier, so it groups with
/// the zoned value of 120 instants before it. The first 120 timestamps and
/// the last 120 zoned values have no partner: 1024 + 120 groups.
#[test]
fn spilled_gql_groups_read_timestamps_at_the_session_zone() {
    let spill_root = tempfile::tempdir().unwrap();
    let (db, _pressure) = under_pressure(spill_root.path());
    let session = db.session();
    session.execute("SESSION SET TIME ZONE '+02:00'").unwrap();
    let params = HashMap::from([("rows".to_string(), rows())]);
    let result = session
        .execute_with_params(GROUP_BY, params.clone())
        .unwrap();
    let (count, sizes) = groups(&result);
    assert_eq!(
        (count, sizes),
        (
            usize::try_from(INSTANTS + 120).unwrap(),
            vec![ROUNDS, 2 * ROUNDS]
        )
    );
    let distinct = session.execute_with_params(DISTINCT, params).unwrap();
    assert_eq!(distinct.rows()[0][0], Value::Int64(INSTANTS + 120));
}
