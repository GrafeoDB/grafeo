//! Property-based fuzz test for the GQL parser.
//!
//! Generates random GQL-like queries and asserts the parser never panics.
//! Works with `cargo test` (no nightly, no libfuzzer needed).
// Fuzz RNG values are intentionally truncated/wrapped for variety
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
//!
//! ```bash
//! cargo test -p grafeo-adapters --test main gql_fuzz_test:: -- --nocapture
//! ```

#[cfg(feature = "gql")]
mod tests {
    use std::fmt::Write;

    use grafeo_adapters::query::gql::parse;

    /// Labels, properties, and variables used for generation.
    const LABELS: &[&str] = &["Person", "City", "Company", "Node", "Item", "Tag", "Sensor"];
    const PROPS: &[&str] = &["name", "age", "city", "val", "score", "status"];
    const VARS: &[&str] = &["n", "m", "a", "b", "c", "x", "y", "p"];
    const STRINGS: &[&str] = &["Alix", "Gus", "Vincent", "Amsterdam", "Berlin"];

    /// Simple seeded pseudo-random number generator (xorshift32).
    struct Rng(u32);

    impl Rng {
        fn new(seed: u32) -> Self {
            Self(if seed == 0 { 1 } else { seed })
        }

        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }

        fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
            items[(self.next() as usize) % items.len()]
        }

        fn coin(&mut self) -> bool {
            self.next().is_multiple_of(2)
        }

        fn range(&mut self, max: u32) -> u32 {
            self.next() % max
        }
    }

    /// Generate a random GQL-like query.
    fn generate_query(rng: &mut Rng) -> String {
        let clause = match rng.range(6) {
            0 => "MATCH",
            1 => "OPTIONAL MATCH",
            2 => "INSERT",
            3 => "MERGE",
            4 => "MATCH",
            _ => "MATCH",
        };

        let mut q = String::from(clause);
        q.push(' ');

        // Node pattern
        q.push('(');
        let var = rng.pick(VARS);
        q.push_str(var);
        if rng.coin() {
            q.push(':');
            q.push_str(rng.pick(LABELS));
        }
        if rng.coin() {
            q.push_str(" {");
            q.push_str(rng.pick(PROPS));
            q.push_str(": ");
            match rng.range(4) {
                0 => write!(q, "{}", rng.next() as i32 % 100).unwrap(),
                1 => {
                    q.push('\'');
                    q.push_str(rng.pick(STRINGS));
                    q.push('\'');
                }
                2 => q.push_str("true"),
                _ => q.push_str("null"),
            }
            q.push('}');
        }
        q.push(')');

        // Optional edge pattern
        if rng.coin() && clause == "MATCH" {
            match rng.range(3) {
                0 => q.push_str("-["),
                1 => q.push_str("<-["),
                _ => q.push_str("-["),
            }
            if rng.coin() {
                q.push_str(rng.pick(VARS));
            }
            if rng.coin() {
                q.push(':');
                q.push_str(rng.pick(LABELS));
            }
            q.push_str("]->");
            q.push('(');
            q.push_str(rng.pick(VARS));
            if rng.coin() {
                q.push(':');
                q.push_str(rng.pick(LABELS));
            }
            q.push(')');
        }

        // WHERE clause
        if rng.coin() && (clause == "MATCH" || clause == "OPTIONAL MATCH") {
            q.push_str(" WHERE ");
            q.push_str(var);
            q.push('.');
            q.push_str(rng.pick(PROPS));
            match rng.range(6) {
                0 => q.push_str(" = "),
                1 => q.push_str(" <> "),
                2 => q.push_str(" > "),
                3 => q.push_str(" < "),
                4 => q.push_str(" >= "),
                _ => q.push_str(" IS NULL"),
            }
            if rng.range(6) < 5 {
                write!(q, "{}", rng.next() as i32 % 100).unwrap();
            }
        }

        // RETURN clause (for MATCH/OPTIONAL MATCH)
        if clause == "MATCH" || clause == "OPTIONAL MATCH" {
            q.push_str(" RETURN ");
            if rng.coin() {
                q.push('*');
            } else {
                q.push_str(var);
                q.push('.');
                q.push_str(rng.pick(PROPS));
                if rng.coin() {
                    q.push_str(", count(");
                    q.push_str(var);
                    q.push_str(") AS cnt");
                }
            }
        }

        // Modifiers
        if rng.coin() {
            q.push_str(" LIMIT ");
            write!(q, "{}", rng.range(20) + 1).unwrap();
        }

        q
    }

    #[test]
    fn gql_parser_never_panics_on_generated_queries() {
        let mut rng = Rng::new(42);
        let mut parse_ok = 0;
        let mut parse_err = 0;

        for _ in 0..10_000 {
            let query = generate_query(&mut rng);
            match parse(&query) {
                Ok(_) => parse_ok += 1,
                Err(_) => parse_err += 1,
            }
        }

        // At least some should parse successfully (proves the generator works)
        assert!(parse_ok > 0, "No queries parsed successfully out of 10,000");
        eprintln!("Fuzz results: {parse_ok} parsed OK, {parse_err} returned errors, 0 panics");
    }

    #[test]
    fn gql_parser_never_panics_on_random_bytes() {
        let mut rng = Rng::new(12345);

        for _ in 0..5_000 {
            let len = (rng.range(200) + 1) as usize;
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    // Bias toward ASCII printable range for more interesting inputs
                    let b = rng.next() as u8;
                    if rng.coin() {
                        (b % 95) + 32 // ASCII printable
                    } else {
                        b
                    }
                })
                .collect();

            if let Ok(query) = std::str::from_utf8(&bytes) {
                let _ = parse(query);
            }
        }
        // If we get here, no panics occurred
    }

    #[test]
    fn gql_parser_handles_edge_case_inputs() {
        let long_label = format!("MATCH (n:{}) RETURN n", "A".repeat(1000));
        let edge_cases: Vec<&str> = vec![
            "",
            " ",
            "\n",
            "\t\t\t",
            "MATCH",
            "MATCH ()",
            "MATCH () RETURN",
            "MATCH (n) RETURN *",
            "MATCH (n:) RETURN n",
            "MATCH (:Person) RETURN *",
            "MATCH (n:Person {}) RETURN n",
            "MATCH (n:Person {name:}) RETURN n",
            "MATCH (n)-[]->(m) RETURN n, m",
            "MATCH (n)-[:]->(m) RETURN n",
            "MATCH (n)<-[]-(m) RETURN n",
            "RETURN 1 + 2",
            "RETURN 'hello'",
            "RETURN null",
            "RETURN true AND false",
            "RETURN [1, 2, 3]",
            "RETURN {a: 1}",
            "INSERT ()",
            "INSERT (:X)",
            "INSERT (:X {a: 1})-[:R]->(:Y)",
            "MERGE (:X {a: 1})",
            "UNWIND [1, 2, 3] AS x RETURN x",
            "UNWIND [] AS x RETURN x",
            // Deeply nested
            "MATCH ((((n)))) RETURN n",
            // Very long label
            &long_label,
            // Many properties
            "MATCH (n {a:1, b:2, c:3, d:4, e:5, f:6, g:7, h:8}) RETURN n",
            // Unicode
            "MATCH (n {name: '\u{1F600}'}) RETURN n",
            "MATCH (n {name: '\u{4E16}\u{754C}'}) RETURN n",
            // Operators
            "RETURN 1 + 2 * 3 - 4 / 5 % 6",
            // SQL keywords
            "SELECT * FROM graph",
            "DELETE FROM nodes",
            // Unterminated string
            "MATCH (n {name: 'unclosed) RETURN n",
            // Mismatched parens
            "MATCH (n RETURN n",
            "MATCH n) RETURN n",
            // Empty CASE
            "RETURN CASE END",
            "RETURN CASE WHEN true THEN 1 END",
        ];

        for input in &edge_cases {
            let _ = parse(input); // must not panic
        }
    }
}
