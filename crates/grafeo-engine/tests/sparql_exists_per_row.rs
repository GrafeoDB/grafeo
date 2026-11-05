//! `EXISTS` reads the row by the SPARQL 1.2 draft's definition (17.4.1.4, 18.3.2.7): each group in its pattern starts
//! from the row, as a solution. A sub-SELECT's body is a group, so a variable it does not project is reached; both
//! sides of a `MINUS` start from the row, so any right row compatible with a left row removes it. A `BIND` or alias to
//! a variable the row binds must agree with it. Nothing is refused: forms that a run without the row would answer
//! wrongly run per row. `exists_per_row.gtest` runs the same tables on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_exists_per_row::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

fn cell(value: &Value) -> String {
    match value {
        Value::String(text) => text.as_str().to_string(),
        Value::Null => "null".to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        // A term by its plain value: an IRI without its brackets, a literal without its quotes and type.
        Value::RdfTerm(term) => {
            let text = term.to_ntriples();
            match text.strip_prefix('<') {
                Some(iri) => iri.trim_end_matches('>').to_string(),
                None => text.split('"').nth(1).unwrap_or("").to_string(),
            }
        }
        other => format!("{other:?}"),
    }
}

fn rows_in_order(session: &Session, query: &str) -> Result<Vec<String>, String> {
    let result = session
        .execute_sparql(query)
        .map_err(|error| error.to_string())?;
    Ok(result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect())
}

fn rows_of(session: &Session, query: &str) -> Result<Vec<String>, String> {
    let mut rows = rows_in_order(session, query)?;
    rows.sort();
    Ok(rows)
}

fn holding(data: &str) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session().execute_sparql(data).unwrap();
    db
}

/// Every case runs, so one failure names all that fail with it.
fn each(data: &str, cases: &[(&str, &str, &[&str])]) {
    let db = holding(data);
    let session = db.session();
    let mut wrong = Vec::new();
    for (name, query, want) in cases {
        let mut want: Vec<String> = want.iter().map(ToString::to_string).collect();
        want.sort();
        let found = rows_of(&session, query);
        if found.as_ref() != Ok(&want) {
            wrong.push(format!(
                "{name}: {query}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} wrong:\n{}",
        wrong.len(),
        cases.len(),
        wrong.join("\n")
    );
}

const DATA: &str = "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }";

const CORRELATED_DATA: &str = "INSERT DATA { <urn:a> <urn:p> 1 . <urn:a> <urn:q> 2 . <urn:a> <urn:q> 3 . <urn:a> <urn:r> 0 . <urn:b> <urn:p> 3 . <urn:b> <urn:q> 2 . <urn:b> <urn:t> 0 . <urn:e> <urn:q> 9 . GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:q> 2 } GRAPH <urn:g2> { <urn:d> <urn:p> 5 . <urn:d> <urn:q> 2 . <urn:e> <urn:p> 6 } }";

/// The forms of the design, each with `EXISTS` and `NOT EXISTS`.
const CASES: &[(&str, &str, &[&str])] = &[
    (
        "case_1",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { ?s :p ?o } }",
        &["urn:a"],
    ),
    (
        "case_1_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { ?s :p ?o } }",
        &["urn:b"],
    ),
    (
        "case_2",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { FILTER(?s = :a) } }",
        &["urn:a"],
    ),
    (
        "case_2_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { FILTER(?s = :a) } }",
        &["urn:b"],
    ),
    (
        "case_3",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o FILTER(?s = :a) } }",
        &["urn:a"],
    ),
    (
        "case_3_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { :d :q ?o FILTER(?s = :a) } }",
        &["urn:b"],
    ),
    (
        "case_4",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o { FILTER(?s = :a) } } }",
        &["urn:a"],
    ),
    (
        "case_4_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { :d :q ?o { FILTER(?s = :a) } } }",
        &["urn:b"],
    ),
    (
        "case_5",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_5_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } } }",
        &[],
    ),
    (
        "case_6",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_6_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } } }",
        &[],
    ),
    (
        "case_7",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { ?s :t :k MINUS { ?s :t :k } } }",
        &[],
    ),
    (
        "case_7_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { ?s :t :k MINUS { ?s :t :k } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_8",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o MINUS { :d :q :e } } }",
        &[],
    ),
    (
        "case_8_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { :d :q ?o MINUS { :d :q :e } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_9",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o MINUS { :nobody :q ?z } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_9_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { :d :q ?o MINUS { :nobody :q ?z } } }",
        &[],
    ),
    (
        "case_10",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { BIND(:a AS ?s) } }",
        &["urn:a"],
    ),
    (
        "case_10_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { BIND(:a AS ?s) } }",
        &["urn:b"],
    ),
    (
        "case_11",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { BIND(?s AS ?y) FILTER(?y = :a) } }",
        &["urn:a"],
    ),
    (
        "case_11_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { BIND(?s AS ?y) FILTER(?y = :a) } }",
        &["urn:b"],
    ),
    (
        "case_12",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?s { ?s :p ?o } } }",
        &["urn:a"],
    ),
    (
        "case_12_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT ?s { ?s :p ?o } } }",
        &["urn:b"],
    ),
    (
        "case_13",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?o { ?s :p ?o } } }",
        &["urn:a"],
    ),
    (
        "case_13_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT ?o { ?s :p ?o } } }",
        &["urn:b"],
    ),
    (
        "case_14",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } } }",
        &["urn:a"],
    ),
    (
        "case_14_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } } }",
        &["urn:b"],
    ),
    (
        "case_15",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_15_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 } }",
        &[],
    ),
    (
        "case_16",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_16_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } } }",
        &[],
    ),
    (
        "case_17",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s } }",
        &["urn:a"],
    ),
    (
        "case_17_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s } }",
        &["urn:b"],
    ),
    (
        "case_18",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) } }",
        &[],
    ),
    (
        "case_18_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_19",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT (?s AS ?y) { :d :q ?o } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_19_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT (?s AS ?y) { :d :q ?o } } }",
        &[],
    ),
    (
        "case_20",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_20_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } } }",
        &[],
    ),
    (
        "case_21",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { GRAPH <urn:g> { ?s :r* ?y } } }",
        &[],
    ),
    (
        "case_21_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { GRAPH <urn:g> { ?s :r* ?y } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_22",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } } }",
        &["urn:a"],
    ),
    (
        "case_22_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } } }",
        &["urn:b"],
    ),
    (
        "case_23",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o FILTER(BOUND(?s)) } }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_23_not",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { :d :q ?o FILTER(BOUND(?s)) } }",
        &[],
    ),
];
/// Each form at each place an expression can hold it.
const POSITIONS: &[(&str, &str, &[&str])] = &[
    (
        "case_1_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { ?s :p ?o } && true) }",
        &["urn:a"],
    ),
    (
        "case_1_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { ?s :p ?o }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_1_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { ?s :p ?o } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_1_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { ?s :p ?o } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_1_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { ?s :p ?o }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_2_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { FILTER(?s = :a) } && true) }",
        &["urn:a"],
    ),
    (
        "case_2_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { FILTER(?s = :a) }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_2_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { FILTER(?s = :a) } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_2_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { FILTER(?s = :a) } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_2_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { FILTER(?s = :a) }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_3_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { :d :q ?o FILTER(?s = :a) } && true) }",
        &["urn:a"],
    ),
    (
        "case_3_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { :d :q ?o FILTER(?s = :a) }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_3_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { :d :q ?o FILTER(?s = :a) } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_3_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { :d :q ?o FILTER(?s = :a) } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_3_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { :d :q ?o FILTER(?s = :a) }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_4_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { :d :q ?o { FILTER(?s = :a) } } && true) }",
        &["urn:a"],
    ),
    (
        "case_4_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { :d :q ?o { FILTER(?s = :a) } }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_4_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { :d :q ?o { FILTER(?s = :a) } } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_4_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { :d :q ?o { FILTER(?s = :a) } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_4_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { :d :q ?o { FILTER(?s = :a) } }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_5_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } } && true) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_5_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } }, true, false)) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_5_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_5_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_5_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } }) } }",
        &["urn:a|urn:e", "urn:b|urn:e"],
    ),
    (
        "case_6_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } } && true) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_6_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } }, true, false)) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_6_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_6_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_6_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } }) } }",
        &["urn:a|urn:e", "urn:b|urn:e"],
    ),
    (
        "case_7_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { ?s :t :k MINUS { ?s :t :k } } && true) }",
        &[],
    ),
    (
        "case_7_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { ?s :t :k MINUS { ?s :t :k } }, true, false)) }",
        &[],
    ),
    (
        "case_7_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { ?s :t :k MINUS { ?s :t :k } } AS ?e) }",
        &["urn:a|false", "urn:b|false"],
    ),
    (
        "case_7_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { ?s :t :k MINUS { ?s :t :k } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|false", "urn:b|false"],
    ),
    (
        "case_7_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { ?s :t :k MINUS { ?s :t :k } }) } }",
        &["urn:a|null", "urn:b|null"],
    ),
    (
        "case_8_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { :d :q ?o MINUS { :d :q :e } } && true) }",
        &[],
    ),
    (
        "case_8_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { :d :q ?o MINUS { :d :q :e } }, true, false)) }",
        &[],
    ),
    (
        "case_8_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { :d :q ?o MINUS { :d :q :e } } AS ?e) }",
        &["urn:a|false", "urn:b|false"],
    ),
    (
        "case_8_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { :d :q ?o MINUS { :d :q :e } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|false", "urn:b|false"],
    ),
    (
        "case_8_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { :d :q ?o MINUS { :d :q :e } }) } }",
        &["urn:a|null", "urn:b|null"],
    ),
    (
        "case_9_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { :d :q ?o MINUS { :nobody :q ?z } } && true) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_9_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { :d :q ?o MINUS { :nobody :q ?z } }, true, false)) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_9_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { :d :q ?o MINUS { :nobody :q ?z } } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_9_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { :d :q ?o MINUS { :nobody :q ?z } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_9_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { :d :q ?o MINUS { :nobody :q ?z } }) } }",
        &["urn:a|urn:e", "urn:b|urn:e"],
    ),
    (
        "case_10_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { BIND(:a AS ?s) } && true) }",
        &["urn:a"],
    ),
    (
        "case_10_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { BIND(:a AS ?s) }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_10_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { BIND(:a AS ?s) } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_10_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { BIND(:a AS ?s) } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_10_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { BIND(:a AS ?s) }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_11_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { BIND(?s AS ?y) FILTER(?y = :a) } && true) }",
        &["urn:a"],
    ),
    (
        "case_11_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { BIND(?s AS ?y) FILTER(?y = :a) }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_11_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { BIND(?s AS ?y) FILTER(?y = :a) } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_11_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { BIND(?s AS ?y) FILTER(?y = :a) } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_11_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { BIND(?s AS ?y) FILTER(?y = :a) }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_12_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { SELECT ?s { ?s :p ?o } } && true) }",
        &["urn:a"],
    ),
    (
        "case_12_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { SELECT ?s { ?s :p ?o } }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_12_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { SELECT ?s { ?s :p ?o } } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_12_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { SELECT ?s { ?s :p ?o } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_12_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { SELECT ?s { ?s :p ?o } }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_13_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { SELECT ?o { ?s :p ?o } } && true) }",
        &["urn:a"],
    ),
    (
        "case_13_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { SELECT ?o { ?s :p ?o } }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_13_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { SELECT ?o { ?s :p ?o } } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_13_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { SELECT ?o { ?s :p ?o } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_13_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { SELECT ?o { ?s :p ?o } }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_14_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } } && true) }",
        &["urn:a"],
    ),
    (
        "case_14_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_14_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_14_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_14_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_15_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 } && true) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_15_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 }, true, false)) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_15_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_15_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_15_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 }) } }",
        &["urn:a|urn:e", "urn:b|urn:e"],
    ),
    (
        "case_16_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } } && true) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_16_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } }, true, false)) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_16_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_16_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_16_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } }) } }",
        &["urn:a|urn:e", "urn:b|urn:e"],
    ),
    (
        "case_17_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s } && true) }",
        &["urn:a"],
    ),
    (
        "case_17_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_17_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_17_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_17_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_18_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) } && true) }",
        &[],
    ),
    (
        "case_18_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) }, true, false)) }",
        &[],
    ),
    (
        "case_18_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) } AS ?e) }",
        &["urn:a|false", "urn:b|false"],
    ),
    (
        "case_18_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|false", "urn:b|false"],
    ),
    (
        "case_18_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) }) } }",
        &["urn:a|null", "urn:b|null"],
    ),
    (
        "case_19_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { SELECT (?s AS ?y) { :d :q ?o } } && true) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_19_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { SELECT (?s AS ?y) { :d :q ?o } }, true, false)) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_19_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { SELECT (?s AS ?y) { :d :q ?o } } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_19_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { SELECT (?s AS ?y) { :d :q ?o } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_19_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { SELECT (?s AS ?y) { :d :q ?o } }) } }",
        &["urn:a|urn:e", "urn:b|urn:e"],
    ),
    (
        "case_20_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } } && true) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_20_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } }, true, false)) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_20_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_20_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_20_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } }) } }",
        &["urn:a|urn:e", "urn:b|urn:e"],
    ),
    (
        "case_21_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { GRAPH <urn:g> { ?s :r* ?y } } && true) }",
        &[],
    ),
    (
        "case_21_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { GRAPH <urn:g> { ?s :r* ?y } }, true, false)) }",
        &[],
    ),
    (
        "case_21_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { GRAPH <urn:g> { ?s :r* ?y } } AS ?e) }",
        &["urn:a|false", "urn:b|false"],
    ),
    (
        "case_21_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { GRAPH <urn:g> { ?s :r* ?y } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|false", "urn:b|false"],
    ),
    (
        "case_21_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { GRAPH <urn:g> { ?s :r* ?y } }) } }",
        &["urn:a|null", "urn:b|null"],
    ),
    (
        "case_22_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } } && true) }",
        &["urn:a"],
    ),
    (
        "case_22_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } }, true, false)) }",
        &["urn:a"],
    ),
    (
        "case_22_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_22_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "case_22_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } }) } }",
        &["urn:a|urn:e", "urn:b|null"],
    ),
    (
        "case_23_conjunct",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { :d :q ?o FILTER(BOUND(?s)) } && true) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_23_if",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER(IF(EXISTS { :d :q ?o FILTER(BOUND(?s)) }, true, false)) }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_23_bind",
        "PREFIX : <urn:> SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { :d :q ?o FILTER(BOUND(?s)) } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_23_select",
        "PREFIX : <urn:> SELECT ?s (EXISTS { :d :q ?o FILTER(BOUND(?s)) } AS ?e) WHERE { ?s :t :k }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "case_23_optional_filter",
        "PREFIX : <urn:> SELECT ?s ?z WHERE { ?s :t :k OPTIONAL { :d :q ?z FILTER(EXISTS { :d :q ?o FILTER(BOUND(?s)) }) } }",
        &["urn:a|urn:e", "urn:b|urn:e"],
    ),
];
/// Each form in `ORDER BY`: the rows it holds for come first.
const ORDERED: &[(&str, &str, &[&str])] = &[
    (
        "case_1_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { ?s :p ?o }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_2_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { FILTER(?s = :a) }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_3_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { :d :q ?o FILTER(?s = :a) }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_4_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { :d :q ?o { FILTER(?s = :a) } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_5_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_6_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_7_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { ?s :t :k MINUS { ?s :t :k } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_8_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { :d :q ?o MINUS { :d :q :e } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_9_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { :d :q ?o MINUS { :nobody :q ?z } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_10_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { BIND(:a AS ?s) }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_11_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { BIND(?s AS ?y) FILTER(?y = :a) }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_12_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { SELECT ?s { ?s :p ?o } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_13_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { SELECT ?o { ?s :p ?o } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_14_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_15_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_16_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_17_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_18_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_19_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { SELECT (?s AS ?y) { :d :q ?o } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_20_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_21_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { GRAPH <urn:g> { ?s :r* ?y } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_22_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } }) ?s",
        &["urn:a", "urn:b"],
    ),
    (
        "case_23_order_by",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { :d :q ?o FILTER(BOUND(?s)) }) ?s",
        &["urn:a", "urn:b"],
    ),
];
/// Each form in an update's `WHERE`: update, the query that shows it, rows.
const WRITES: &[(&str, &str, &str, &[&str])] = &[
    (
        "case_1_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { ?s :p ?o } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_2_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { FILTER(?s = :a) } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_3_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { :d :q ?o FILTER(?s = :a) } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_4_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { :d :q ?o { FILTER(?s = :a) } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_5_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { :d :q ?o OPTIONAL { ?s :p ?y } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_6_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { :d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_7_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { ?s :t :k MINUS { ?s :t :k } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &[],
    ),
    (
        "case_8_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { :d :q ?o MINUS { :d :q :e } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &[],
    ),
    (
        "case_9_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { :d :q ?o MINUS { :nobody :q ?z } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_10_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { BIND(:a AS ?s) } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_11_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { BIND(?s AS ?y) FILTER(?y = :a) } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_12_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { SELECT ?s { ?s :p ?o } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_13_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { SELECT ?o { ?s :p ?o } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_14_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { SELECT ?o { ?x :p ?o FILTER(?x = ?s) } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_15_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { SELECT ?s { ?s :t :k } LIMIT 1 } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_16_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_17_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_18_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a) } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &[],
    ),
    (
        "case_19_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { SELECT (?s AS ?y) { :d :q ?o } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_20_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { { ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a", "urn:b"],
    ),
    (
        "case_21_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { GRAPH <urn:g> { ?s :r* ?y } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &[],
    ),
    (
        "case_22_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { :d :q ?o FILTER EXISTS { ?s :p ?y } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a"],
    ),
    (
        "case_23_update",
        "PREFIX : <urn:> INSERT { ?s :hit 1 } WHERE { ?s :t :k FILTER EXISTS { :d :q ?o FILTER(BOUND(?s)) } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :hit 1 }",
        &["urn:a", "urn:b"],
    ),
];

/// Forms with their own data: name, data, query, rows.
const WITH_THEIR_DATA: &[(&str, &str, &str, &[&str])] = &[
    (
        "a_rows_unbound_variable_stays_open",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k OPTIONAL { ?s :w ?u } FILTER EXISTS { :a :p ?u } }",
        &["urn:a", "urn:b"],
    ),
    (
        "a_rows_unbound_variable_stays_open_not",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k OPTIONAL { ?s :w ?u } FILTER NOT EXISTS { :a :p ?u } }",
        &[],
    ),
    (
        "a_blank_node_in_the_row_is_a_term",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . _:n :t :k . :a :p :c }",
        "PREFIX : <urn:> SELECT (COUNT(*) AS ?c) WHERE { ?s :t :k FILTER EXISTS { :a :p ?z OPTIONAL { ?s :p ?y } FILTER(BOUND(?y)) } }",
        &["1"],
    ),
    (
        "a_blank_node_in_the_row_keeps_its_rows",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . _:n :t :k . :a :p :c }",
        "PREFIX : <urn:> SELECT (COUNT(*) AS ?c) WHERE { ?s :t :k FILTER EXISTS { :a :p ?o OPTIONAL { ?s :p ?y } } }",
        &["2"],
    ),
    (
        "the_row_binds_the_graph_variable",
        "PREFIX : <urn:> INSERT DATA { GRAPH :g1 { :a :t :k . :a :p :c } GRAPH :g2 { :b :t :k . :d :p :c } }",
        "PREFIX : <urn:> SELECT ?g ?s WHERE { GRAPH ?g { ?s :t :k } FILTER EXISTS { GRAPH ?g { ?x :p ?o FILTER(?x = ?s) } } }",
        &["urn:g1|urn:a"],
    ),
    (
        "the_row_binds_the_graph_variable_not",
        "PREFIX : <urn:> INSERT DATA { GRAPH :g1 { :a :t :k . :a :p :c } GRAPH :g2 { :b :t :k . :d :p :c } }",
        "PREFIX : <urn:> SELECT ?g ?s WHERE { GRAPH ?g { ?s :t :k } FILTER NOT EXISTS { GRAPH ?g { ?x :p ?o FILTER(?x = ?s) } } }",
        &["urn:g2|urn:b"],
    ),
    (
        "from_named_chooses_the_graph_of_a_per_row_form",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . GRAPH :g1 { :x :p :c } GRAPH :g2 { :a :p :c } }",
        "PREFIX : <urn:> SELECT ?s FROM :a0 FROM NAMED :g1 WHERE { BIND(:a AS ?s) FILTER EXISTS { GRAPH ?h { ?y :p ?o FILTER(?y = ?s) } } }",
        &[],
    ),
    (
        "bind_agrees_with_the_row",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { BIND(:a AS ?s) } }",
        &["urn:a"],
    ),
    (
        "bind_of_an_error_leaves_the_rows_value",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { BIND(1 / 0 AS ?s) } }",
        &["urn:a", "urn:b"],
    ),
    (
        "an_alias_must_agree_with_the_row",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT (:a AS ?s) { :d :q ?o } } }",
        &["urn:a"],
    ),
    (
        "a_nested_exists_reads_the_outer_row",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o FILTER EXISTS { :d :q ?y FILTER(?s = :a) } } }",
        &["urn:a"],
    ),
    (
        "a_nested_not_exists_reads_the_outer_row",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o FILTER NOT EXISTS { :d :q ?y FILTER(?s = :a) } } }",
        &["urn:b"],
    ),
    (
        "a_zero_length_path_of_a_bound_end_is_not_in_the_graph",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { BIND(:nowhere AS ?s) FILTER EXISTS { ?s :r* ?y } }",
        &[],
    ),
    (
        "a_zero_length_path_of_a_constant_keeps_its_row",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :nowhere :r* ?y } }",
        &["urn:a", "urn:b"],
    ),
    (
        "offset_of_a_sub_select_reads_per_row",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?s { ?s :t :k } OFFSET 1 } }",
        &[],
    ),
    (
        "below_an_aggregation_a_key_reads_the_row",
        "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH :g { :c :r :e } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT (COUNT(*) AS ?n) { ?s :p ?o } HAVING(COUNT(*) > 0) } }",
        &["urn:a"],
    ),
];

/// What a run without the row could not answer, over other data.
const READING_THE_ROW: &[(&str, &str, &[&str])] = &[
    (
        "a_filter_in_a_nested_group",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x { FILTER(?x > ?o) } }) }",
        &[],
    ),
    (
        "a_filter_in_a_nested_group_beside_a_triple",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x { ?s <urn:q> ?y FILTER(?y > ?o) } }) }",
        &["urn:a"],
    ),
    (
        "a_filter_in_optional",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x OPTIONAL { ?s <urn:r> ?y FILTER(?y > ?o) } }) }",
        &["urn:a", "urn:b"],
    ),
    (
        "a_filter_in_union",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { { ?s <urn:q> ?x FILTER(?x > ?o) } UNION { ?s <urn:r> ?x } }) }",
        &["urn:a"],
    ),
    (
        "a_filter_in_minus",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x MINUS { ?s <urn:r> ?y FILTER(?y > ?o) } }) }",
        &["urn:a", "urn:b"],
    ),
    (
        "a_bind",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x BIND(?x + ?o AS ?y) }) }",
        &["urn:a", "urn:b"],
    ),
    (
        "a_nested_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(EXISTS { ?s <urn:r> ?y FILTER(?y > ?o) }) }) }",
        &[],
    ),
    (
        "a_nested_not_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(NOT EXISTS { ?s <urn:r> ?y FILTER(?y > ?o) }) }) }",
        &["urn:a", "urn:b"],
    ),
    (
        "a_sub_select",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { { SELECT ?s WHERE { ?s <urn:q> ?x FILTER(?x > ?o) } } }) }",
        &["urn:a"],
    ),
    (
        "a_sub_select_having",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { { SELECT ?s WHERE { ?s <urn:q> ?x } GROUP BY ?s HAVING(COUNT(?x) > ?o) } }) }",
        &[],
    ),
    (
        "may_leave_unbound_and_the_row_binds",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?t <urn:q> ?x OPTIONAL { ?t <urn:r> ?o } }) }",
        &["urn:a", "urn:b"],
    ),
    (
        "may_leave_unbound_read_by_a_filter",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?t <urn:q> ?x OPTIONAL { ?t <urn:r> ?o } FILTER(?o = 1) }) }",
        &["urn:a"],
    ),
    (
        "in_a_bind",
        "SELECT ?s ?e WHERE { ?s <urn:p> ?o BIND(EXISTS { ?s <urn:q> ?x BIND(?x + ?o AS ?y) } AS ?e) }",
        &["urn:a|true", "urn:b|true"],
    ),
    (
        "in_a_projection",
        "SELECT ?s (EXISTS { ?s <urn:q> ?x BIND(?x + ?o AS ?y) } AS ?e) WHERE { ?s <urn:p> ?o }",
        &["urn:a|true", "urn:b|true"],
    ),
];
#[test]
fn exists_reads_the_row_in_each_form() {
    each(DATA, CASES);
}

#[test]
fn each_form_holds_at_each_position() {
    each(DATA, POSITIONS);
}

#[test]
fn each_form_orders_rows() {
    let db = holding(DATA);
    let session = db.session();
    let mut wrong = Vec::new();
    for (name, query, want) in ORDERED {
        let found = rows_in_order(&session, query);
        if found.as_ref().map(Vec::as_slice).map_err(|_| ())
            != Ok(&want.iter().map(ToString::to_string).collect::<Vec<_>>()[..])
        {
            wrong.push(format!(
                "{name}: {query}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn each_form_reads_the_row_in_an_update() {
    let mut wrong = Vec::new();
    for (name, update, query, want) in WRITES {
        let db = holding(DATA);
        let session = db.session();
        if let Err(error) = session.execute_sparql(update) {
            wrong.push(format!("{name}: {update}: {error}"));
            continue;
        }
        let want: Vec<String> = want.iter().map(ToString::to_string).collect();
        let found = rows_of(&session, query);
        if found.as_ref() != Ok(&want) {
            wrong.push(format!(
                "{name}: {update}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn forms_with_their_own_data() {
    let mut wrong = Vec::new();
    for (name, data, query, want) in WITH_THEIR_DATA {
        let db = holding(data);
        let session = db.session();
        let want: Vec<String> = want.iter().map(ToString::to_string).collect();
        let found = rows_of(&session, query);
        if found.as_ref() != Ok(&want) {
            wrong.push(format!(
                "{name}: {query}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Nothing is refused: each form a run once cannot answer answers per row.
#[test]
fn forms_a_run_without_the_row_cannot_answer() {
    each(CORRELATED_DATA, READING_THE_ROW);
}

/// A transaction's own writes are in what `EXISTS` reads, on the per-row route too.
#[test]
fn a_transactions_writes_are_seen_per_row() {
    let db = holding(DATA);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql("PREFIX : <urn:> INSERT DATA { :b :p :c }")
        .unwrap();
    for (query, want) in [
        (
            "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { ?s :p ?o } }",
            &["urn:a", "urn:b"][..],
        ),
        (
            "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { :d :q ?o { FILTER(BOUND(?s)) } OPTIONAL { ?s :p ?y } ?s :p ?z } }",
            &["urn:a", "urn:b"][..],
        ),
        (
            "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?o { ?s :p ?o } } }",
            &["urn:a", "urn:b"][..],
        ),
        (
            "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { SELECT ?o { ?s :p ?o } } }",
            &[][..],
        ),
    ] {
        assert_eq!(rows_of(&session, query).unwrap(), want, "{query}");
    }
    session.rollback().unwrap();
    assert_eq!(
        rows_of(
            &session,
            "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?o { ?s :p ?o } } }"
        )
        .unwrap(),
        ["urn:a"]
    );
}

/// Each statement reads the row's values, repeated, with values the first run never saw.
#[test]
fn a_repeat_reads_the_new_rows() {
    let db = holding(DATA);
    let session = db.session();
    let query =
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k FILTER EXISTS { SELECT ?o { ?s :p ?o } } }";
    assert_eq!(rows_of(&session, query).unwrap(), ["urn:a"]);
    session
        .execute_sparql("PREFIX : <urn:> INSERT DATA { :n :t :k . :n :p :c }")
        .unwrap();
    assert_eq!(rows_of(&session, query).unwrap(), ["urn:a", "urn:n"]);
}
