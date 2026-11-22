//! A LIMIT cuts a range scan only when nothing between them drops rows: in Gremlin, a
//! `.limit()` after a range `has`, a traversal and another `has` returned fewer rows than it
//! should (0.6.3 returned none of the 99 matches), as the range scan below was cut first.
//!
//! The oracle is the same traversal without the limit: a limit of `n` keeps `min(n, all)` rows.

#![cfg(feature = "gremlin")]

use grafeo_engine::GrafeoDB;

/// People `k` 0 to 999, each knowing the next, with and without an index on `k`.
fn chain(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("k").unwrap();
    }
    db.execute("UNWIND range(0, 999) AS i INSERT (:Person {k: i})")
        .unwrap();
    db.execute("MATCH (a:Person), (b:Person) WHERE b.k = a.k + 1 INSERT (a)-[:KNOWS]->(b)")
        .unwrap();
    db
}

/// Each traversal with `.limit(n)` keeps `min(n, all)` rows, where `all` is its row count
/// without the limit. The cases: a range `has`, a traversal and a `has` on the far end; the
/// same with a second `has` stacked on the range; and a range `has` with a second `has` and no
/// traversal.
#[test]
fn a_limit_keeps_the_rows_a_range_scan_below_it_leads_to() {
    for indexed in [false, true] {
        let db = chain(indexed);
        for traversal in [
            "g.V().hasLabel('Person').has('k', gt(0)).out('KNOWS').has('k', gt(900))",
            "g.V().hasLabel('Person').has('k', gt(0)).has('k', lt(990)).out('KNOWS').has('k', gt(900))",
            "g.V().hasLabel('Person').has('k', gt(0)).has('k', gt(900))",
            "g.V().has('k', gt(0)).out('KNOWS').has('k', gt(900)).values('k')",
        ] {
            let all = db.execute_gremlin(traversal).unwrap().row_count();
            assert!(all > 0, "{traversal}: the case has rows");
            for limit in [1, 3, 50, 2000] {
                let limited = db
                    .execute_gremlin(&format!("{traversal}.limit({limit})"))
                    .unwrap()
                    .row_count();
                assert_eq!(
                    limited,
                    all.min(limit),
                    "{traversal}.limit({limit}), indexed {indexed}"
                );
            }
        }
    }
}
