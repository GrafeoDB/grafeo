//! After NEXT, a procedure call reads the working table the statement before it gave (ISO/IEC
//! 39075 9.2 GR 3): the RETURN that follows returns only what it names, and an aggregate runs
//! over the whole working table. The class test takes every procedure of the registry and
//! compares the NEXT form with the same statement without NEXT, which already answers; the
//! gtest twin is `14.1b_next_then_call.gtest`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main next_then_call::
//! ```

#![cfg(all(feature = "gql", feature = "algos"))]

use grafeo_engine::GrafeoDB;
use grafeo_engine::procedures::BuiltinProcedures;

/// The columns and the rows of `query`, the rows sorted, since a plain RETURN has no order.
fn answer(db: &GrafeoDB, query: &str) -> Result<(Vec<String>, Vec<String>), String> {
    db.execute(query)
        .map(|result| {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            (result.columns.clone(), rows)
        })
        .map_err(|error| error.to_string())
}

/// For every registered procedure that yields a column and runs with no arguments, the NEXT form
/// gives the linear form's answer: one column `c`, the count over the whole working table, and
/// for a plain RETURN only the yielded column. A procedure added later is covered with no change.
#[test]
fn every_procedure_after_next_answers_as_without_next() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:P {name: 'Alix'})-[:K]->(:P {name: 'Gus'})-[:K]->(:P {name: 'Vincent'}), \
         (:Q {name: 'Mia'})",
    )
    .unwrap();
    let mut wrong = Vec::new();
    let mut compared = 0;
    for procedure in BuiltinProcedures::new().list() {
        let Some(column) = procedure.output_columns.first() else {
            continue;
        };
        let call = format!("CALL {}() YIELD {column}", procedure.name);
        for tail in [
            "RETURN count(*) AS c".to_string(),
            format!("RETURN count({column}) AS c"),
            format!("RETURN {column}"),
        ] {
            let linear = format!("UNWIND [1, 2] AS x {call} {tail}");
            let Ok(expected) = answer(&db, &linear) else {
                // Needs arguments: the parser half of the class is in call_followed_by_statements.
                continue;
            };
            compared += 1;
            let next = format!("UNWIND [1, 2] AS x RETURN x NEXT {call} {tail}");
            match answer(&db, &next) {
                Ok(got) if got == expected => {}
                Ok(got) => wrong.push(format!("{next}: {got:?}, want {expected:?}")),
                Err(error) => wrong.push(format!("{next}: {error}, want {expected:?}")),
            }
        }
    }
    assert!(compared >= 30, "only {compared} statements compared");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
