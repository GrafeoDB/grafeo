//! Cypher's `^` is left associative and binds tighter than `*` and `+`, as
//! openCypher's grammar has it (`UnaryAddOrSubtractExpression ( '^'
//! UnaryAddOrSubtractExpression )*`; TCK 2024.3 Precedence2 [2] [3]): `2 ^ 3 ^
//! 2` is `(2 ^ 3) ^ 2`, 64. It was read right to left, 512.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cypher_power_associativity::
//! ```

#![cfg(feature = "cypher")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn power_is_left_associative_and_binds_tightest() {
    let db = GrafeoDB::new_in_memory();
    for (query, expected) in [
        ("RETURN 2 ^ 3 ^ 2 AS r", 64.0),
        ("RETURN 4 ^ (3 * 2) ^ 3 AS r", 68_719_476_736.0),
        ("RETURN 2 ^ 2 ^ 2 ^ 2 AS r", 256.0),
        ("RETURN 4 ^ 3 * 2 ^ 3 AS r", 512.0),
        ("RETURN 2 * 3 ^ 2 AS r", 18.0),
        ("RETURN 2 + 3 ^ 2 AS r", 11.0),
        ("RETURN -2 ^ 2 AS r", 4.0),
        ("RETURN 2 ^ (3 ^ 2) AS r", 512.0),
    ] {
        let rows = db.execute_cypher(query).unwrap().rows().to_vec();
        assert_eq!(rows, [vec![Value::Float64(expected)]], "{query}");
    }
}
