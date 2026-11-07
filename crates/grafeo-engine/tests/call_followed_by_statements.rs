//! A leading named procedure call in GQL is a call query statement, one of the simple statements
//! of a linear query statement (ISO/IEC 39075 14.3), so any simple statement may follow its
//! YIELD, not only RETURN. The class test takes every procedure of the registry and every kind of
//! following statement; Deriva's query is the case that found it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main call_followed_by_statements::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;

/// One statement of each kind that may follow a call, none naming the yielded column except
/// through `{col}`.
#[cfg(feature = "algos")]
const FOLLOWERS: [&str; 9] = [
    "MATCH (n) RETURN count(*) AS c",
    "OPTIONAL MATCH (n) RETURN count(*) AS c",
    "FILTER true RETURN count(*) AS c",
    "LET z = 1 RETURN count(z) AS c",
    "FOR z IN [1, 2] RETURN count(z) AS c",
    "ORDER BY {col} LIMIT 1 RETURN {col}",
    "CALL grafeo.relationshipTypes() YIELD relationshipType AS t RETURN count(t) AS c",
    "MATCH (n) SET n.seen = true RETURN count(*) AS c",
    "INSERT (:Seen) RETURN count(*) AS c",
];

/// Every registered procedure that yields a column, followed by every kind of statement, parses:
/// the plan is given, or the statement fails for another reason (an argument, under the
/// argument rules), never with a syntax error. A procedure added later is covered with no change.
/// The algorithms give the registry most of its procedures.
#[cfg(feature = "algos")]
#[test]
fn every_statement_kind_may_follow_every_procedure() {
    use grafeo_engine::procedures::BuiltinProcedures;

    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {name: 'Alix'})-[:K]->(:P {name: 'Gus'})")
        .unwrap();
    let mut wrong = Vec::new();
    let mut checked = 0;
    for procedure in BuiltinProcedures::new().list() {
        let Some(column) = procedure.output_columns.first() else {
            continue;
        };
        for follower in FOLLOWERS {
            let follower = follower.replace("{col}", column);
            let query = format!(
                "EXPLAIN CALL {}() YIELD {column} {follower}",
                procedure.name
            );
            checked += 1;
            if let Err(error) = db.execute(&query) {
                let status = GqlStatus::from(&error).as_str().to_string();
                if status == "42001" {
                    wrong.push(format!("{query}: {error}"));
                }
            }
        }
    }
    assert!(checked > 100, "only {checked} statements checked");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Deriva's query, as written: a load procedure, its YIELD, then a MATCH over what it loaded,
/// in GQL; and inside a transaction that is rolled back, which leaves nothing.
#[test]
fn derivas_load_then_match_runs_in_gql() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("people.csv");
    std::fs::write(&file, "name\nHans\nShosanna\n").unwrap();
    let path = file.to_string_lossy().replace('\\', "/");
    let query = format!(
        "CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['Leak']}}) YIELD rows \
         MATCH (n:Leak) RETURN rows, n.name AS name ORDER BY name"
    );

    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute(&query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| format!("{} {}", row[0], row[1]))
        .collect();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert!(
        rows[0].contains("Hans") && rows[1].contains("Shosanna") && rows[0].starts_with('2'),
        "{rows:?}"
    );

    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    assert_eq!(session.execute(&query).unwrap().rows().len(), 2);
    session.rollback().unwrap();
    assert_eq!(
        db.execute("MATCH (n:Leak) RETURN n").unwrap().rows().len(),
        0,
        "a rolled-back load left nodes"
    );
}

fn labelled() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})-[:WORKS_AT]->(:Company {name: 'Grafeo OSS'})")
        .unwrap();
    db
}

/// `LIMIT $k` after a leading CALL takes each call's value, also when the same text runs again
/// from the plan cache with another value, in either order.
#[test]
fn a_limit_parameter_after_a_call_takes_each_value() {
    let query = "CALL db.labels() YIELD label RETURN label ORDER BY label LIMIT $k";
    for order in [[1, 2], [2, 1]] {
        let db = labelled();
        for k in order {
            let params = [("k".to_string(), grafeo_common::types::Value::Int64(k))].into();
            let rows = db
                .execute_with_params(query, params)
                .unwrap_or_else(|error| panic!("{query} with k = {k}: {error}"))
                .rows()
                .len();
            assert_eq!(rows, usize::try_from(k).unwrap(), "k = {k} after {order:?}");
        }
    }
}

/// A write after a leading CALL runs once per yielded row and ends the statement without a
/// result; in a transaction that is rolled back it leaves nothing; in a READ ONLY transaction
/// it is refused as read-only and writes nothing.
#[test]
fn a_write_after_a_leading_call() {
    let write = "CALL db.labels() YIELD label INSERT (:Tag {name: label})";
    let tags = |db: &GrafeoDB| {
        db.execute("MATCH (t:Tag) RETURN t.name ORDER BY t.name")
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[0].to_string())
            .collect::<Vec<_>>()
    };

    let db = labelled();
    let result = db
        .execute(write)
        .unwrap_or_else(|error| panic!("{write}: {error}"));
    assert!(result.columns.is_empty(), "no result: {:?}", result.columns);
    assert_eq!(tags(&db).len(), 2, "one Tag per label: {:?}", tags(&db));

    let db = labelled();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute(write).unwrap();
    session.rollback().unwrap();
    assert_eq!(tags(&db), Vec::<String>::new(), "a rollback left Tags");

    let db = labelled();
    let session = db.session();
    session.execute("START TRANSACTION READ ONLY").unwrap();
    let error = session.execute(write).unwrap_err();
    assert_eq!(GqlStatus::from(&error).as_str(), "25G03", "{error}");
    session.execute("ROLLBACK").unwrap();
    assert_eq!(tags(&db), Vec::<String>::new(), "a refused write left Tags");
}
