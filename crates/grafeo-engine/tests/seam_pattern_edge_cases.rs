//! Seam tests for pattern matching edge cases (ISO/IEC 39075 Section 16).
//!
//! Tests boundary conditions for quantifiers, path modes, empty results
//! with aggregation, and label expression edge cases.
//!
//! ```bash
//! cargo test -p grafeo-engine --test main seam_pattern_edge_cases::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

/// Creates a small chain: Alix -> Gus -> Vincent
fn chain_graph() -> GrafeoDB {
    let db = db();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) INSERT (g)-[:KNOWS]->(:Person {name: 'Vincent'})")
        .unwrap();
    db
}

// ============================================================================
// 1. Empty result set interactions with aggregation
// ============================================================================

mod empty_result_aggregation {
    use super::*;

    #[test]
    fn count_on_empty_returns_zero() {
        let db = db();
        let session = db.session();

        let result = session
            .execute("MATCH (n:NonExistent) RETURN COUNT(*) AS cnt")
            .unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Aggregate should always return one row"
        );
        assert_eq!(result.rows()[0][0], Value::Int64(0));
    }

    #[test]
    fn count_star_on_empty_returns_zero() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session
            .execute("MATCH (n:Animal) RETURN COUNT(*) AS cnt")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(0));
    }

    #[test]
    fn sum_on_empty_returns_null() {
        let db = db();
        let session = db.session();

        let result = session
            .execute("MATCH (n:NonExistent) RETURN SUM(n.val) AS s")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "SUM on empty should be NULL"
        );
    }

    #[test]
    fn avg_on_empty_returns_null() {
        let db = db();
        let session = db.session();

        let result = session
            .execute("MATCH (n:NonExistent) RETURN AVG(n.val) AS a")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "AVG on empty should be NULL"
        );
    }

    #[test]
    fn min_on_empty_returns_null() {
        let db = db();
        let session = db.session();

        let result = session
            .execute("MATCH (n:NonExistent) RETURN MIN(n.val) AS m")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "MIN on empty should be NULL"
        );
    }

    #[test]
    fn max_on_empty_returns_null() {
        let db = db();
        let session = db.session();

        let result = session
            .execute("MATCH (n:NonExistent) RETURN MAX(n.val) AS m")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "MAX on empty should be NULL"
        );
    }

    #[test]
    fn collect_on_empty_returns_empty_list() {
        let db = db();
        let session = db.session();

        let result = session
            .execute("MATCH (n:NonExistent) RETURN COLLECT(n.val) AS c")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::List(vec![].into()),
            "COLLECT on empty should be empty list"
        );
    }
}

// ============================================================================
// 2. Quantified path patterns
// ============================================================================

mod quantified_paths {
    use super::*;

    #[test]
    fn variable_length_one_hop() {
        let db = chain_graph();
        let session = db.session();

        let result = session
            .execute("MATCH (a:Person {name: 'Alix'})-[:KNOWS*1..1]->(b) RETURN b.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "1..1 should match exactly one hop");
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    }

    #[test]
    fn variable_length_two_hops() {
        let db = chain_graph();
        let session = db.session();

        let result = session
            .execute("MATCH (a:Person {name: 'Alix'})-[:KNOWS*2..2]->(b) RETURN b.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "2..2 should reach Vincent");
        assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
    }

    #[test]
    fn variable_length_range() {
        let db = chain_graph();
        let session = db.session();

        let result = session
            .execute(
                "MATCH (a:Person {name: 'Alix'})-[:KNOWS*1..2]->(b) RETURN b.name ORDER BY b.name",
            )
            .unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "1..2 should reach both Gus and Vincent"
        );
    }

    #[test]
    fn variable_length_no_match() {
        let db = chain_graph();
        let session = db.session();

        let result = session
            .execute("MATCH (a:Person {name: 'Vincent'})-[:KNOWS*1..5]->(b) RETURN b.name")
            .unwrap();
        assert_eq!(result.row_count(), 0, "Vincent has no outgoing KNOWS edges");
    }

    #[test]
    fn variable_length_star() {
        let db = chain_graph();
        let session = db.session();

        // * is shorthand for 1..unlimited
        let result = session
            .execute("MATCH (a:Person {name: 'Alix'})-[:KNOWS*]->(b) RETURN b.name ORDER BY b.name")
            .unwrap();
        assert_eq!(result.row_count(), 2, "* should reach all reachable nodes");
    }
}

// ============================================================================
// 3. OPTIONAL MATCH edge cases
// ============================================================================

mod optional_match {
    use super::*;

    #[test]
    fn optional_match_returns_null_when_no_match() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session
            .execute(
                "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n.name, m.name AS friend",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(
            result.rows()[0][1],
            Value::Null,
            "No match should return NULL"
        );
    }

    #[test]
    fn optional_match_returns_data_when_match_exists() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        let result = session
            .execute(
                "MATCH (n:Person {name: 'Alix'}) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN m.name",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    }

    #[test]
    fn optional_match_mixed_results() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        let result = session
            .execute(
                "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n.name, m.name ORDER BY n.name",
            )
            .unwrap();

        // Alix -> Gus (match), Gus -> null (no outgoing), Vincent -> null (no outgoing)
        assert_eq!(result.row_count(), 3);
    }
}

// ============================================================================
// 4. Label expression edge cases
// ============================================================================

mod label_expressions {
    use super::*;

    #[test]
    fn multi_label_node_matches_each_label() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person:Engineer {name: 'Alix'})")
            .unwrap();

        let r1 = session.execute("MATCH (n:Person) RETURN n").unwrap();
        let r2 = session.execute("MATCH (n:Engineer) RETURN n").unwrap();
        assert_eq!(r1.row_count(), 1);
        assert_eq!(r2.row_count(), 1);
    }

    #[test]
    fn label_disjunction() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute("INSERT (:Animal {species: 'Cat'})")
            .unwrap();
        session.execute("INSERT (:Vehicle {type: 'Car'})").unwrap();

        let result = session
            .execute("MATCH (n IS Person | Animal) RETURN n")
            .unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "Disjunction should match Person and Animal"
        );
    }

    #[test]
    fn label_negation() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute("INSERT (:Animal {species: 'Cat'})")
            .unwrap();

        let result = session.execute("MATCH (n IS !Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "Negation should exclude Person");
    }

    #[test]
    fn label_wildcard() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute("INSERT (:Animal {species: 'Cat'})")
            .unwrap();

        let result = session.execute("MATCH (n IS %) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "Wildcard should match all labeled nodes"
        );
    }

    #[test]
    fn nonexistent_label_returns_empty() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session.execute("MATCH (n:Unicorn) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "Nonexistent label should match nothing"
        );
    }
}

// ============================================================================
// 5. Edge direction edge cases
// ============================================================================

mod edge_directions {
    use super::*;

    #[test]
    fn outgoing_edge_match() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        let result = session
            .execute("MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b) RETURN b.name")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    }

    #[test]
    fn incoming_edge_match() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        let result = session
            .execute("MATCH (b:Person {name: 'Gus'})<-[:KNOWS]-(a) RETURN a.name")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn undirected_edge_match() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        // Undirected should match both directions
        let result = session
            .execute("MATCH (a:Person {name: 'Gus'})-[:KNOWS]-(b) RETURN b.name")
            .unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Undirected should match the edge from either side"
        );
    }

    #[test]
    fn wrong_direction_returns_empty() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        // Alix has no incoming KNOWS edges
        let result = session
            .execute("MATCH (a:Person {name: 'Alix'})<-[:KNOWS]-(b) RETURN b.name")
            .unwrap();
        assert_eq!(result.row_count(), 0, "Wrong direction should return empty");
    }
}
