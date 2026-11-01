//! Named GQL OPTIONAL calls retain an incoming row when the procedure has no output.
//! The procedures feature is required, as it is for ordinary named CALL.

#![cfg(feature = "gql")]

#[cfg(feature = "algos")]
mod supported {
    use grafeo_common::types::Value::{Int64, Null};
    use grafeo_common::utils::error::Error;
    use grafeo_engine::GrafeoDB;
    use grafeo_engine::transaction::IsolationLevel;

    use grafeo_common::types::Value;
    use grafeo_engine::database::OperatorProfile;

    fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
        db.execute(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"))
            .rows()
            .to_vec()
    }

    fn registered() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "CREATE PROCEDURE optional_values() RETURNS (value INTEGER, tag INTEGER) AS { \
             UNWIND [7, 7, 9] AS v RETURN v AS value, 100 AS tag ORDER BY value }",
        )
        .unwrap();
        db.execute(
            "CREATE PROCEDURE optional_empty() RETURNS (value INTEGER, tag INTEGER) AS { \
             MATCH (:Absent) RETURN 7 AS value, 100 AS tag }",
        )
        .unwrap();
        db.execute(
            "CREATE PROCEDURE optional_people() RETURNS (value INTEGER) AS { \
             MATCH (p:Person) RETURN p.id AS value ORDER BY value }",
        )
        .unwrap();
        db
    }

    #[test]
    fn an_empty_named_optional_call_keeps_the_standalone_and_incoming_rows() {
        let db = GrafeoDB::new_in_memory();
        let ordinary = db.execute("CALL db.labels()").unwrap();
        assert_eq!(ordinary.columns, ["label"]);
        assert_eq!(ordinary.rows(), &[] as &[Vec<Value>]);
        let optional = db
            .execute("OPTIONAL CALL db.labels() YIELD label RETURN label")
            .unwrap();
        assert_eq!(optional.columns, ["label"]);
        assert_eq!(optional.rows(), &[vec![Null]]);
        assert_eq!(
            rows(
                &db,
                "UNWIND [3, 19, 3] AS token OPTIONAL CALL db.labels() \
                 YIELD label RETURN token, label ORDER BY token"
            ),
            [[Int64(3), Null], [Int64(3), Null], [Int64(19), Null]],
        );
        assert_eq!(
            rows(
                &db,
                "UNWIND [3, 19, 3] AS token CALL db.labels() \
                    YIELD label RETURN token, label"
            ),
            [] as [Vec<Value>; 0]
        );
        assert_eq!(
            rows(
                &db,
                "UNWIND [] AS token OPTIONAL CALL db.labels() \
                 YIELD label RETURN token, label"
            ),
            [] as [Vec<Value>; 0]
        );
        assert_eq!(
            rows(
                &db,
                "OPTIONAL CALL db.labels() YIELD label AS chosen RETURN chosen"
            ),
            [[Null]],
        );
    }

    #[test]
    fn a_named_optional_call_preserves_duplicate_inputs_and_outputs() {
        let db = registered();
        let want = [
            [Int64(1), Int64(7), Int64(100)],
            [Int64(1), Int64(7), Int64(100)],
            [Int64(1), Int64(7), Int64(100)],
            [Int64(1), Int64(7), Int64(100)],
            [Int64(1), Int64(9), Int64(100)],
            [Int64(1), Int64(9), Int64(100)],
            [Int64(2), Int64(7), Int64(100)],
            [Int64(2), Int64(7), Int64(100)],
            [Int64(2), Int64(9), Int64(100)],
        ];
        for call in ["CALL", "OPTIONAL CALL"] {
            let query = format!(
                "UNWIND [1, 1, 2] AS token {call} optional_values() YIELD value, tag \
                 RETURN token, value, tag ORDER BY token, value"
            );
            assert_eq!(rows(&db, &query), want, "{query}");
        }
    }

    #[test]
    fn optional_yield_aliases_filter_before_null_extension() {
        let db = registered();
        assert_eq!(
            rows(
                &db,
                "UNWIND [3, 19] AS token OPTIONAL CALL optional_values() \
                 YIELD value AS chosen, tag AS kind WHERE chosen = 8 \
                 RETURN token, chosen, kind ORDER BY token"
            ),
            [[Int64(3), Null, Null], [Int64(19), Null, Null]],
        );
        assert_eq!(
            rows(
                &db,
                "UNWIND [3, 19] AS token CALL optional_values() \
                    YIELD value AS chosen, tag AS kind WHERE chosen = 8 \
                    RETURN token, chosen, kind"
            ),
            [] as [Vec<Value>; 0]
        );
        assert_eq!(
            rows(
                &db,
                "OPTIONAL CALL optional_values() YIELD value AS chosen, tag AS kind \
                 WHERE chosen = 8 RETURN chosen, kind"
            ),
            [[Null, Null]],
        );
    }

    #[test]
    fn optional_default_columns_and_aliased_outputs_keep_earlier_values() {
        let db = registered();
        let empty = db
            .execute("OPTIONAL CALL optional_empty() RETURN *")
            .unwrap();
        assert_eq!(empty.columns, ["value", "tag"]);
        assert_eq!(empty.rows(), &[vec![Null, Null]]);
        let nonempty = db
            .execute("OPTIONAL CALL optional_values() RETURN *")
            .unwrap();
        assert_eq!(nonempty.columns, ["value", "tag"]);
        assert_eq!(
            nonempty.rows(),
            &[
                vec![Int64(7), Int64(100)],
                vec![Int64(7), Int64(100)],
                vec![Int64(9), Int64(100)]
            ]
        );
        for call in ["CALL", "OPTIONAL CALL"] {
            let query = format!("UNWIND [41] AS token {call} optional_values() RETURN *");
            let result = db.execute(&query).unwrap();
            assert_eq!(result.columns, ["token", "value", "tag"], "{query}");
            assert_eq!(
                result.rows(),
                &[
                    vec![Int64(41), Int64(7), Int64(100)],
                    vec![Int64(41), Int64(7), Int64(100)],
                    vec![Int64(41), Int64(9), Int64(100)]
                ],
                "{query}"
            );
        }
        // The original procedure column name collides with an earlier name,
        // but the explicit YIELD alias is fresh and must preserve both values.
        assert_eq!(
            rows(
                &db,
                "UNWIND [41] AS value OPTIONAL CALL optional_values() \
                 YIELD value AS chosen, tag AS kind RETURN value, chosen, kind ORDER BY chosen"
            ),
            [
                [Int64(41), Int64(7), Int64(100)],
                [Int64(41), Int64(7), Int64(100)],
                [Int64(41), Int64(9), Int64(100)]
            ],
        );
        // An explicit underscore alias is an output, not an internal column.
        for call in ["CALL", "OPTIONAL CALL"] {
            let query = format!(
                "{call} optional_values() YIELD value AS _value RETURN _value ORDER BY _value"
            );
            assert_eq!(
                rows(&db, &query),
                [[Int64(7)], [Int64(7)], [Int64(9)]],
                "{query}"
            );
        }
    }

    #[test]
    fn a_matched_optional_call_keeps_ordinary_name_collision_behavior() {
        let db = registered();
        let ordinary = db.execute(
            "UNWIND [41] AS value CALL optional_values() YIELD value \
                                   RETURN value ORDER BY value",
        );
        let optional = db.execute(
            "UNWIND [41] AS value OPTIONAL CALL optional_values() YIELD value \
                                   RETURN value ORDER BY value",
        );
        // This is a compatibility control, not a new shadowing/freshness rule.
        match (ordinary, optional) {
            (Ok(ordinary), Ok(optional)) => {
                assert_eq!(ordinary.rows().len(), 3);
                assert_eq!(optional.columns, ordinary.columns);
                assert_eq!(optional.rows(), ordinary.rows());
            }
            (Err(ordinary), Err(optional)) => {
                assert_eq!(
                    std::mem::discriminant(&optional),
                    std::mem::discriminant(&ordinary)
                );
                if let (Error::Query(ordinary), Error::Query(optional)) = (ordinary, optional) {
                    assert_eq!(optional.kind, ordinary.kind);
                }
            }
            other => panic!("optional call changed the existing collision policy: {other:?}"),
        }
    }

    #[test]
    fn a_named_optional_call_keeps_an_explicit_subquery_import() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(
            rows(
                &db,
                "UNWIND [3, 19] AS token CALL (token) { \
                 OPTIONAL CALL db.labels() YIELD label \
                 RETURN token + 1 AS kept, label AS maybe } \
                 RETURN token, kept, maybe ORDER BY token"
            ),
            [[Int64(3), Int64(4), Null], [Int64(19), Int64(20), Null]],
        );
    }

    #[test]
    fn a_named_optional_call_after_writes_reads_the_whole_input() {
        for call in ["CALL", "OPTIONAL CALL"] {
            let db = registered();
            let query = format!(
                "FOR x IN [1, 2] INSERT (:Person {{id: x}}) WITH x \
                 {call} optional_people() YIELD value RETURN x, value ORDER BY x, value"
            );
            assert_eq!(
                rows(&db, &query),
                [
                    [Int64(1), Int64(1)],
                    [Int64(1), Int64(2)],
                    [Int64(2), Int64(1)],
                    [Int64(2), Int64(2)],
                ],
                "{query}"
            );
            assert_eq!(
                rows(&db, "MATCH (p:Person) RETURN p.id ORDER BY p.id"),
                [[Int64(1)], [Int64(2)]]
            );
        }
    }

    #[test]
    fn named_optional_calls_keep_snapshot_and_own_write_visibility() {
        let db = registered();
        db.execute("INSERT (:Person {id: 1})").unwrap();
        let mut reader = db.session();
        reader
            .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        db.execute("INSERT (:Person {id: 2})").unwrap();
        for call in ["CALL", "OPTIONAL CALL"] {
            let query = format!("{call} optional_people() YIELD value RETURN value ORDER BY value");
            let result = reader.execute(&query).unwrap();
            assert_eq!(result.rows(), &[vec![Int64(1)]], "{query}");
        }
        reader.execute("MATCH (p:Person {id: 1}) DELETE p").unwrap();
        let ordinary = reader
            .execute("CALL optional_people() YIELD value RETURN value")
            .unwrap();
        assert_eq!(ordinary.rows(), &[] as &[Vec<Value>]);
        let optional = reader
            .execute("OPTIONAL CALL optional_people() YIELD value RETURN value")
            .unwrap();
        assert_eq!(optional.rows(), &[vec![Null]]);
        reader.rollback().unwrap();
        assert_eq!(
            rows(&db, "MATCH (p:Person) RETURN p.id ORDER BY p.id"),
            [[Int64(1)], [Int64(2)]]
        );
    }

    #[test]
    fn profile_counts_the_real_procedure_and_null_extended_rows() {
        fn matching<'a>(node: &'a OperatorProfile, name: &str, out: &mut Vec<&'a OperatorProfile>) {
            if node.operator == name {
                out.push(node);
            }
            for child in &node.children {
                matching(child, name, out);
            }
        }

        let db = GrafeoDB::new_in_memory();
        for (query, expected_rows, expected_apply, minimum_procedure_calls) in [
            ("CALL db.labels() YIELD label RETURN label", 0, None, 1),
            (
                "OPTIONAL CALL db.labels() YIELD label RETURN label",
                1,
                Some(1),
                1,
            ),
            (
                "UNWIND [3, 19, 3] AS token CALL db.labels() YIELD label RETURN token, label",
                0,
                Some(0),
                3,
            ),
            (
                "UNWIND [3, 19, 3] AS token OPTIONAL CALL db.labels() YIELD label RETURN token, label",
                3,
                Some(3),
                3,
            ),
        ] {
            let result = db.execute(&format!("PROFILE {query}")).unwrap();
            let profile = result.profile().expect("PROFILE carries its operator tree");
            assert_eq!(profile.root.rows, expected_rows, "{query}: {profile:?}");
            assert!(profile.root.calls > 0, "{query}: {profile:?}");
            let mut procedures = Vec::new();
            matching(&profile.root, "ProcedureCall", &mut procedures);
            procedures.retain(|node| node.detail == "db.labels" && node.children.is_empty());
            assert_eq!(procedures.len(), 1, "{query}: {profile:?}");
            assert_eq!(
                procedures[0].rows, 0,
                "the procedure itself is empty: {query}"
            );
            assert!(
                procedures[0].calls >= minimum_procedure_calls,
                "the real procedure runs for every incoming row: {query}: {profile:?}"
            );
            let mut applies = Vec::new();
            matching(&profile.root, "Apply", &mut applies);
            applies.retain(|node| node.children.len() == 2);
            if let Some(rows) = expected_apply {
                assert_eq!(applies.len(), 1, "{query}: {profile:?}");
                assert_eq!(applies[0].rows, rows, "{query}: {profile:?}");
                assert!(applies[0].calls > 0, "{query}: {profile:?}");
            } else {
                assert!(applies.is_empty(), "{query}: {profile:?}");
            }
        }
    }
}

#[cfg(not(feature = "algos"))]
#[test]
fn named_calls_keep_the_existing_missing_procedures_error() {
    use grafeo_common::utils::error::{Error, QueryErrorKind};
    use grafeo_engine::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    for query in [
        "CALL db.labels() YIELD label RETURN label",
        "OPTIONAL CALL db.labels() YIELD label RETURN label",
    ] {
        let error = db
            .execute(query)
            .expect_err("procedures require the algos feature");
        assert!(
            matches!(&error, Error::Query(query_error) if query_error.kind == QueryErrorKind::Unsupported),
            "{query}: {error}"
        );
        assert!(error.to_string().contains("algos"), "{query}: {error}");
    }
}
