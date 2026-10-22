//! Integration tests for temporal types: Date, Time, Duration.
//!
//! Tests temporal constructors, arithmetic, comparisons, and typed literals
//! across GQL, Cypher, and SPARQL query languages.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test temporal_types
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Creates 2 Person nodes (Alix birthday 1990-06-15, Gus birthday 2000-01-01).
fn create_test_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Alix".into())),
                (
                    "birthday",
                    Value::Date(grafeo_common::types::Date::from_ymd(1990, 6, 15).unwrap()),
                ),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Gus".into())),
                (
                    "birthday",
                    Value::Date(grafeo_common::types::Date::from_ymd(2000, 1, 1).unwrap()),
                ),
            ],
        )
        .unwrap();
    let _ = session.commit();
    db
}

// ============================================================================
// Cypher temporal constructors (standalone RETURN)
// ============================================================================

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_function() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute_cypher("RETURN date('2024-01-15') AS d").unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 1);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_function() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute_cypher("RETURN duration('P1Y') AS d").unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.months(), 12);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_time_function() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute_cypher("RETURN time('23:59:59') AS t").unwrap();
    let row = &result.rows()[0];
    let t = row[0].as_time().expect("expected Time value");
    assert_eq!(t.hour(), 23);
    assert_eq!(t.minute(), 59);
    assert_eq!(t.second(), 59);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_datetime_function() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN datetime('2024-03-15T14:30:00Z') AS dt")
        .unwrap();
    let row = &result.rows()[0];
    let ts = row[0].as_timestamp().expect("expected Timestamp value");
    let date = ts.to_date();
    assert_eq!(date.year(), 2024);
    assert_eq!(date.month(), 3);
    assert_eq!(date.day(), 15);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_with_components() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN duration('P1Y2M3D') AS d")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.months(), 14); // 1Y2M = 14 months
    assert_eq!(d.days(), 3);
}

// ============================================================================
// GQL typed literal syntax (needs MATCH context)
// ============================================================================

#[test]
#[cfg(feature = "gql")]
fn gql_date_typed_literal() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DATE '2024-03-15' AS d LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 3);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "gql")]
fn gql_time_typed_literal() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN TIME '14:30:00' AS t LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let t = row[0].as_time().expect("expected Time value");
    assert_eq!(t.hour(), 14);
    assert_eq!(t.minute(), 30);
    assert_eq!(t.second(), 0);
}

#[test]
#[cfg(feature = "gql")]
fn gql_duration_typed_literal() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DURATION 'P1Y2M3D' AS d LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.months(), 14);
    assert_eq!(d.days(), 3);
}

#[test]
#[cfg(feature = "gql")]
fn gql_datetime_typed_literal() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DATETIME '2024-03-15T14:30:00Z' AS dt LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let ts = row[0].as_timestamp().expect("expected Timestamp value");
    let date = ts.to_date();
    assert_eq!(date.year(), 2024);
    assert_eq!(date.month(), 3);
    assert_eq!(date.day(), 15);
}

// ============================================================================
// Component extraction
// ============================================================================

#[test]
#[cfg(feature = "cypher")]
fn cypher_year_month_day_extraction() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher(
            "RETURN year(date('2024-03-15')) AS y, month(date('2024-03-15')) AS m, day(date('2024-03-15')) AS d",
        )
        .unwrap();
    let row = &result.rows()[0];
    assert_eq!(row[0].as_int64(), Some(2024));
    assert_eq!(row[1].as_int64(), Some(3));
    assert_eq!(row[2].as_int64(), Some(15));
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_hour_minute_second_extraction() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher(
            "RETURN hour(time('14:30:45')) AS h, minute(time('14:30:45')) AS m, second(time('14:30:45')) AS s",
        )
        .unwrap();
    let row = &result.rows()[0];
    assert_eq!(row[0].as_int64(), Some(14));
    assert_eq!(row[1].as_int64(), Some(30));
    assert_eq!(row[2].as_int64(), Some(45));
}

// ============================================================================
// Temporal arithmetic
// ============================================================================

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_plus_duration() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN date('2024-01-15') + duration('P1M') AS next_month")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 2);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_minus_date() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN date('2024-03-15') - date('2024-01-01') AS diff")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    // 74 days between Jan 1 and Mar 15
    assert_eq!(d.days(), 74);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_arithmetic() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN duration('P1Y') + duration('P6M') AS combined")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.months(), 18); // 12 + 6
}

// ============================================================================
// GQL temporal arithmetic with typed literals (MATCH context)
// ============================================================================

#[test]
#[cfg(feature = "gql")]
fn gql_date_plus_duration() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DATE '2024-01-15' + DURATION 'P1M' AS next_month LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 2);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "gql")]
fn gql_date_minus_date() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) RETURN DATE '2024-03-15' - DATE '2024-01-01' AS diff LIMIT 1")
        .unwrap();
    let row = &result.rows()[0];
    let d = row[0].as_duration().expect("expected Duration value");
    assert_eq!(d.days(), 74);
}

// ============================================================================
// Temporal comparison in WHERE
// ============================================================================

#[test]
#[cfg(feature = "gql")]
fn gql_date_comparison_where() {
    let db = create_test_db();
    let result = db
        .execute("MATCH (p:Person) WHERE p.birthday > DATE '1995-01-01' RETURN p.name AS name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("Gus"));
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_comparison_where() {
    let db = create_test_db();
    let result = db
        .execute_cypher(
            "MATCH (p:Person) WHERE p.birthday > date('1995-01-01') RETURN p.name AS name",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("Gus"));
}

// ============================================================================
// JSON parameter roundtrip
// ============================================================================

#[test]
#[cfg(feature = "gql")]
fn gql_date_json_param_roundtrip() {
    let db = create_test_db();
    let mut params = std::collections::HashMap::new();
    params.insert(
        "d".to_string(),
        Value::Date(grafeo_common::types::Date::from_ymd(2024, 6, 15).unwrap()),
    );
    let result = db
        .execute_with_params("MATCH (p:Person) RETURN $d AS val LIMIT 1", params)
        .unwrap();
    let d = result.rows()[0][0]
        .as_date()
        .expect("expected Date value from param");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 6);
    assert_eq!(d.day(), 15);
}

// ============================================================================
// SPARQL xsd: typed literals
// ============================================================================

#[test]
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn sparql_xsd_date_literal() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_sparql(
            r#"SELECT ?d WHERE { BIND("2024-03-15"^^<http://www.w3.org/2001/XMLSchema#date> AS ?d) }"#,
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "expected at least one row");
    let d = result.rows()[0][0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 3);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn sparql_xsd_duration_literal() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_sparql(
            r#"SELECT ?d WHERE { BIND("P1Y6M"^^<http://www.w3.org/2001/XMLSchema#duration> AS ?d) }"#,
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "expected at least one row");
    let d = result.rows()[0][0]
        .as_duration()
        .expect("expected Duration value");
    assert_eq!(d.months(), 18);
}

// ============================================================================
// Temporal map constructors: date({year:...}), time({hour:...}), etc.
// ============================================================================

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_map_constructor() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN date({year: 2024, month: 3, day: 15}) AS d")
        .unwrap();
    let d = result.rows()[0][0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 3);
    assert_eq!(d.day(), 15);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_date_map_defaults() {
    let db = GrafeoDB::new_in_memory();
    // month and day should default to 1 when omitted
    let result = db.execute_cypher("RETURN date({year: 2024}) AS d").unwrap();
    let d = result.rows()[0][0].as_date().expect("expected Date value");
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 1);
    assert_eq!(d.day(), 1);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_time_map_constructor() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN time({hour: 14, minute: 30, second: 45}) AS t")
        .unwrap();
    let t = result.rows()[0][0].as_time().expect("expected Time value");
    assert_eq!(t.hour(), 14);
    assert_eq!(t.minute(), 30);
    assert_eq!(t.second(), 45);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_datetime_map_constructor() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher(
            "RETURN datetime({year: 2024, month: 3, day: 15, hour: 14, minute: 30}) AS dt",
        )
        .unwrap();
    let ts = result.rows()[0][0]
        .as_timestamp()
        .expect("expected Timestamp value");
    let d = ts.to_date();
    assert_eq!(d.year(), 2024);
    assert_eq!(d.month(), 3);
    assert_eq!(d.day(), 15);
    let t = ts.to_time();
    assert_eq!(t.hour(), 14);
    assert_eq!(t.minute(), 30);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_map_constructor() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN duration({years: 1, months: 2, days: 3, hours: 4}) AS dur")
        .unwrap();
    let d = result.rows()[0][0]
        .as_duration()
        .expect("expected Duration value");
    assert_eq!(d.months(), 14); // 1 year + 2 months
    assert_eq!(d.days(), 3);
    // 4 hours in nanoseconds = 4 * 3_600_000_000_000
    assert_eq!(d.nanos(), 4 * 3_600_000_000_000);
}

#[test]
#[cfg(feature = "cypher")]
fn cypher_duration_map_with_weeks() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute_cypher("RETURN duration({weeks: 2, days: 3}) AS dur")
        .unwrap();
    let d = result.rows()[0][0]
        .as_duration()
        .expect("expected Duration value");
    assert_eq!(d.days(), 17); // 2 weeks + 3 days
}
