//! The path search budget as its own setting (#492): `min(memory_limit / 4, 256 MiB)` by
//! default, `Config::with_path_search_budget(bytes)` to set it, never above the memory limit;
//! LPG and RDF read the one value, and a search over it is refused with `GRAFEO-S001` naming
//! the budget and its setting (RDF used to say `GRAFEO-Q006`).
//!
//! The LPG searches over the reported code graph are in `path_search_budget.rs`, which counts
//! what they hold.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main path_search_budget_setting::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::utils::error::{ErrorCode, LimitBound, LimitRefusal};
use grafeo_engine::config::ConfigError;
use grafeo_engine::{Config, GrafeoDB};

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const GIB: usize = 1024 * MIB;

/// The budget follows the decided formula, row by row.
#[test]
fn the_budget_is_a_quarter_of_the_memory_limit_up_to_256_mib_or_the_setting() {
    for (case, config, budget) in [
        ("no memory limit", Config::in_memory(), 256 * MIB),
        (
            "a 64 MiB memory limit",
            Config::in_memory().with_memory_limit(64 * MIB),
            16 * MIB,
        ),
        (
            "a 4 GiB memory limit",
            Config::in_memory().with_memory_limit(4 * GIB),
            256 * MIB,
        ),
        (
            "an explicit 8 MiB",
            Config::in_memory().with_path_search_budget(8 * MIB),
            8 * MIB,
        ),
        (
            "an explicit 512 MiB, no memory limit",
            Config::in_memory().with_path_search_budget(512 * MIB),
            512 * MIB,
        ),
        (
            "an explicit 2 GiB under a 1 GiB memory limit",
            Config::in_memory()
                .with_memory_limit(GIB)
                .with_path_search_budget(2 * GIB),
            GIB,
        ),
        (
            "an explicit 1 MiB under a 1 GiB memory limit",
            Config::in_memory()
                .with_memory_limit(GIB)
                .with_path_search_budget(MIB),
            MIB,
        ),
    ] {
        assert_eq!(config.path_search_budget(), budget, "{case}");
    }
}

#[test]
fn a_zero_path_search_budget_is_refused() {
    let config = Config::in_memory().with_path_search_budget(0);
    assert!(matches!(
        config.validate(),
        Err(ConfigError::ZeroPathSearchBudget)
    ));
    assert!(GrafeoDB::with_config(config).is_err());
}

/// An RDF property path over an explicit budget is refused with the LPG's refusal: S001 and
/// the budget's bound, the figure as set and the setting in the message.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn an_rdf_property_path_over_its_budget_is_refused_naming_it() {
    use std::fmt::Write;

    let budget = 64 * KIB;
    let db = GrafeoDB::with_config(Config::in_memory().with_path_search_budget(budget)).unwrap();
    let mut data = String::from("PREFIX ex: <http://example.org/> INSERT DATA {");
    for i in 0..20_000 {
        write!(data, " ex:n{i} ex:next ex:n{} .", i + 1).unwrap();
    }
    data.push_str(" }");
    db.execute_sparql(&data).unwrap();

    let query = "PREFIX ex: <http://example.org/> \
                 SELECT (COUNT(?x) AS ?c) WHERE { ex:n0 ex:next+ ?x }";
    let error = db
        .execute_sparql(query)
        .expect_err("20,000 terms do not fit in 64 KiB");
    assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}");
    match error.limit_refusal() {
        Some(LimitRefusal::Memory {
            bound: LimitBound::PathSearchBudget,
            limit,
            ..
        }) => assert_eq!(*limit, budget as u64, "{error}"),
        other => panic!("the path search budget's refusal, got {other:?}: {error}"),
    }
    let message = error.to_string();
    assert!(
        message.contains(LimitBound::PathSearchBudget.setting()),
        "{message}"
    );

    // Under a budget that holds the search, the same path answers.
    let roomy =
        GrafeoDB::with_config(Config::in_memory().with_path_search_budget(64 * MIB)).unwrap();
    roomy.execute_sparql(&data).unwrap();
    let rows = roomy.execute_sparql(query).unwrap();
    let count = format!("{:?}", rows.rows()[0][0]);
    assert!(
        count.contains("\"20000\""),
        "every node of the chain: {count}"
    );
}

/// A path search refused inside a transaction leaves the transaction as a failed read leaves it:
/// the writes before and after it commit, and the refusal is not retryable.
#[test]
fn a_refused_path_search_leaves_its_transaction_usable() {
    let db = GrafeoDB::with_config(Config::in_memory().with_path_search_budget(KIB)).unwrap();
    db.execute("INSERT (:City {name: 'Amsterdam'})-[:ROAD]->(:City {name: 'Utrecht'})-[:ROAD]->(:City {name: 'Rotterdam'})")
        .unwrap();
    db.execute(
        "MATCH (a:City {name: 'Rotterdam'}), (b:City {name: 'Amsterdam'}) INSERT (a)-[:ROAD]->(b)",
    )
    .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:City {name: 'Leiden'})").unwrap();
    let error = session
        .execute("MATCH (a:City)-[:ROAD*]->(b) RETURN count(b)")
        .expect_err("an unbounded walk around a cycle passes 1 KiB");
    assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}");
    assert!(!error.error_code().is_retryable(), "{error}");
    session.execute("INSERT (:City {name: 'Delft'})").unwrap();
    session.commit().unwrap();

    let rows = db
        .execute("MATCH (c:City) WHERE c.name IN ['Leiden', 'Delft'] RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(
        rows.rows().len(),
        2,
        "both writes of the transaction committed"
    );
}
