//! GQL names in backticks and string literals with a doubled quote.
//!
//! A backticked name in SET and REMOVE is the name inside the backticks, as
//! in INSERT, MATCH and RETURN: `SET n.`two words` = 2` stored the key with
//! its backticks, and `REMOVE n.`odd key`` missed the real key. A doubled
//! backtick inside one is one backtick. In a character string literal a
//! doubled quote is the quote itself (ISO/IEC 39075:2024 21.2,
//! <character string literal>): `'O''Brien'` and `"say ""hi"""` were syntax
//! errors.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test gql_quoted_names_and_strings
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|e| panic!("`{query}`: {e}"))
        .rows()
        .to_vec()
}

fn strings(values: &[&str]) -> Value {
    Value::List(
        values
            .iter()
            .map(|v| Value::from(*v))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// Alix, with a property whose key needs backticks.
fn alix() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', `odd key`: 1})")
        .unwrap();
    db
}

#[test]
fn set_stores_the_key_inside_the_backticks() {
    let db = alix();
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person) SET n.`two words` = 2 RETURN n.`two words` AS v"
        ),
        [[Value::Int64(2)]]
    );
    assert_eq!(
        rows(&db, "MATCH (n:Person) RETURN keys(n) AS k"),
        [[strings(&["name", "odd key", "two words"])]]
    );
}

#[test]
fn set_adds_the_labels_inside_the_backticks() {
    let db = alix();
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person) SET n:`Odd Label`:`A` RETURN labels(n) AS l"
        ),
        [[strings(&["A", "Odd Label", "Person"])]]
    );
    assert_eq!(
        rows(&db, "MATCH (n:`Odd Label`) RETURN n.name AS name"),
        [[Value::from("Alix")]],
        "the label matches as written elsewhere"
    );
}

#[test]
fn remove_removes_the_key_and_label_inside_the_backticks() {
    let db = alix();
    db.execute("MATCH (n:Person) SET n:`Odd Label`").unwrap();
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person) REMOVE n.`odd key`, n:`Odd Label` RETURN keys(n) AS k, labels(n) AS l"
        ),
        [[strings(&["name"]), strings(&["Person"])]]
    );
}

/// A backticked variable is the variable inside the backticks.
#[test]
fn a_backticked_variable_in_set_and_remove() {
    let db = alix();
    assert_eq!(
        rows(
            &db,
            "MATCH (`the person`:Person) SET `the person`.age = 30 REMOVE `the person`.`odd key` \
             RETURN `the person`.age AS age, keys(`the person`) AS k"
        ),
        [[Value::Int64(30), strings(&["age", "name"])]]
    );
}

/// A doubled backtick inside a backticked name is one backtick.
#[test]
fn a_doubled_backtick_is_one_backtick() {
    let db = alix();
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person) SET n.`a``b` = 1 RETURN keys(n) AS k, n.`a``b` AS v"
        ),
        [[strings(&["a`b", "name", "odd key"]), Value::Int64(1)]]
    );
    db.execute("INSERT (:Person {name: 'Gus', `c``d`: 2})")
        .unwrap();
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person {name: 'Gus'}) RETURN keys(n) AS k, n.`c``d` AS v"
        ),
        [[strings(&["c`d", "name"]), Value::Int64(2)]],
        "in INSERT and RETURN too"
    );
}

#[test]
fn a_doubled_quote_in_a_string_is_the_quote() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        rows(
            &db,
            "RETURN 'O''Brien' AS a, \"say \"\"hi\"\"\" AS b, '' AS empty, '''' AS one_quote, 'a''' AS trailing"
        ),
        [[
            Value::from("O'Brien"),
            Value::from("say \"hi\""),
            Value::from(""),
            Value::from("'"),
            Value::from("a'"),
        ]]
    );
    db.execute("INSERT (:Person {name: 'O''Brien'})").unwrap();
    assert_eq!(
        rows(
            &db,
            "MATCH (p:Person) WHERE p.name = 'O''Brien' RETURN p.name AS name"
        ),
        [[Value::from("O'Brien")]],
        "stored and matched as one quote"
    );
}

/// The other quote kind and the backslash escapes keep their meaning next
/// to a doubled quote.
#[test]
fn a_doubled_quote_next_to_other_escapes() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        rows(
            &db,
            concat!(
                "RETURN 'a\"\"b' AS other_kind, \"a''b\" AS other_kind_too, ",
                "'x\\'y' AS backslash, 'p\\\\''q' AS backslash_then_doubled"
            )
        ),
        [[
            Value::from("a\"\"b"),
            Value::from("a''b"),
            Value::from("x'y"),
            Value::from("p\\'q"),
        ]]
    );
}
