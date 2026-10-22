//! Regression test for issue #308 — `text_match` returns nothing when the
//! database is file-backed (`grafeo-file`). The WAL / CDC store wrappers
//! had empty `GraphStoreSearch` impls, so `has_text_index` / `score_text`
//! fell through to the trait's no-op defaults: pushdown was skipped and
//! per-row evaluation returned null, filtering every row out.

#![cfg(all(feature = "text-index", feature = "gql", feature = "lpg"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn run_reproduction(db: &GrafeoDB) {
    let session = db.session();

    session
        .execute("INSERT (:Article {title: 'A1', body: 'rust database internals'})")
        .expect("insert A1");
    session
        .execute("INSERT (:Article {title: 'A2', body: 'attention mechanisms in graph nets'})")
        .expect("insert A2");
    session
        .execute("CREATE INDEX article_body FOR (n:Article) ON (n.body) USING TEXT")
        .expect("create text index via GQL");

    let result = session
        .execute("MATCH (n:Article) WHERE text_match(n.body, 'rust database') RETURN n.title")
        .expect("query should succeed");

    let titles: Vec<String> = result
        .rows()
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::String(s)) => Some(s.to_string()),
            _ => None,
        })
        .collect();

    assert_eq!(titles, vec!["A1".to_string()]);
}

#[test]
fn issue_308_in_memory() {
    let db = GrafeoDB::new_in_memory();
    run_reproduction(&db);
}

#[cfg(feature = "grafeo-file")]
#[test]
fn issue_308_grafeo_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("issue_308.grafeo");
    let db = GrafeoDB::open(&path).expect("open grafeo file");
    run_reproduction(&db);
}
