//! Live indexes appear in SHOW INDEXES and the public admin inventory.
//! Existing DDL metadata is retained; unnamed API indexes use empty names.
//! Exact label and property names survive checkpoint and reopen.
#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, database::QueryResult};

fn shown(result: QueryResult) -> Vec<[String; 4]> {
    assert_eq!(result.columns, ["name", "type", "label", "property"]);
    let mut rows: Vec<[String; 4]> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(value) => value.to_string(),
                    other => panic!("SHOW INDEXES must retain String cells: {other:?}"),
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("SHOW INDEXES retains exactly four columns")
        })
        .collect();
    rows.sort();
    rows
}

fn wanted(rows: &[[&str; 4]]) -> Vec<[String; 4]> {
    let mut rows: Vec<_> = rows.iter().map(|row| row.map(str::to_string)).collect();
    rows.sort();
    rows
}

fn assert_show(db: &GrafeoDB, rows: &[[&str; 4]]) {
    assert_eq!(shown(db.execute("SHOW INDEXES").unwrap()), wanted(rows));
    #[cfg(feature = "cypher")]
    assert_eq!(
        shown(db.execute_cypher("SHOW INDEXES").unwrap()),
        wanted(rows)
    );
}

fn assert_list_indexes(db: &GrafeoDB, want: &[(&str, &str, &str)]) {
    let mut actual: Vec<_> = db
        .list_indexes()
        .into_iter()
        .map(|index| {
            // Listing a real index does not invent uniqueness, statistics,
            // dimensions, or metrics that the existing report lacks.
            assert!(!index.unique);
            assert_eq!(index.cardinality, None);
            assert_eq!(index.size_bytes, None);
            (index.name, index.index_type, index.target)
        })
        .collect();
    actual.sort();
    let mut want: Vec<_> = want
        .iter()
        .map(|(name, kind, target)| (name.to_string(), kind.to_string(), target.to_string()))
        .collect();
    want.sort();
    assert_eq!(actual, want);
}

#[test]
fn api_property_index_is_shown_without_an_invented_name_or_label() {
    let db = GrafeoDB::new_in_memory();
    assert_show(&db, &[]);
    db.create_property_index("email").unwrap();
    assert!(db.has_property_index("email"));
    assert_show(&db, &[["", "Hash", "", "email"]]);
    // Creating the same physical index twice adds no reporting row.
    db.create_property_index("email").unwrap();
    assert_show(&db, &[["", "Hash", "", "email"]]);
    assert!(db.drop_property_index("email").unwrap());
    assert!(!db.has_property_index("email"));
    assert_show(&db, &[]);
    assert!(!db.drop_property_index("email").unwrap());
    db.create_property_index("email").unwrap();
    assert_show(&db, &[["", "Hash", "", "email"]]);
}

#[test]
fn named_ddl_metadata_is_preserved_without_duplicate_physical_rows() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("email").unwrap();
    assert_show(&db, &[["", "Hash", "", "email"]]);
    db.execute("CREATE INDEX doc_email FOR (d:Doc) ON (d.email)")
        .unwrap();
    db.execute("CREATE INDEX doc_rank FOR (d:Doc) ON (d.rank) USING BTREE")
        .unwrap();
    assert!(db.has_property_index("email"));
    assert!(db.has_property_index("rank"));
    db.create_property_index("email").unwrap();
    db.create_property_index("rank").unwrap();
    assert_show(
        &db,
        &[
            ["doc_email", "Hash", "Doc", "email"],
            ["doc_rank", "BTree", "Doc", "rank"],
        ],
    );
    // Two written DDL names remain two metadata rows even though the
    // current global property index is keyed only by the property.
    db.execute("CREATE INDEX other_email FOR (o:Other) ON (o.email)")
        .unwrap();
    assert_show(
        &db,
        &[
            ["doc_email", "Hash", "Doc", "email"],
            ["doc_rank", "BTree", "Doc", "rank"],
            ["other_email", "Hash", "Other", "email"],
        ],
    );
}

#[test]
fn api_property_rows_follow_the_selected_graph_and_drop_in_that_graph() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("root_key").unwrap();
    db.create_graph("archive").unwrap();
    let archive = db.graph("archive").unwrap();
    let session = archive.session().unwrap();
    session.create_property_index("archive_key").unwrap();
    assert!(session.has_property_index("archive_key"));
    assert!(!session.has_property_index("root_key"));
    assert_show(&db, &[["", "Hash", "", "root_key"]]);
    assert_eq!(
        shown(archive.execute("SHOW INDEXES").unwrap()),
        wanted(&[["", "Hash", "", "archive_key"]])
    );
    db.set_current_graph(Some("archive")).unwrap();
    assert_show(&db, &[["", "Hash", "", "archive_key"]]);
    assert_list_indexes(&db, &[("", "Hash", ":archive_key")]);
    assert!(db.drop_property_index("archive_key").unwrap());
    assert_show(&db, &[]);
    assert_list_indexes(&db, &[]);
    assert_eq!(shown(session.execute("SHOW INDEXES").unwrap()), wanted(&[]));
    db.set_current_graph(None).unwrap();
    assert!(db.has_property_index("root_key"));
    assert_show(&db, &[["", "Hash", "", "root_key"]]);
    assert_list_indexes(&db, &[("", "Hash", ":root_key")]);
}

#[test]
fn public_list_indexes_reports_the_same_api_and_named_targets() {
    let db = GrafeoDB::new_in_memory();
    assert_list_indexes(&db, &[]);
    db.create_property_index("email").unwrap();
    db.execute("CREATE INDEX doc_title FOR (d:Doc) ON (d.title)")
        .unwrap();
    assert!(db.has_property_index("email"));
    assert!(db.has_property_index("title"));
    assert_list_indexes(
        &db,
        &[("", "Hash", ":email"), ("doc_title", "Hash", "Doc:title")],
    );
    assert_show(
        &db,
        &[
            ["", "Hash", "", "email"],
            ["doc_title", "Hash", "Doc", "title"],
        ],
    );
    assert!(db.drop_property_index("email").unwrap());
    assert_list_indexes(&db, &[("doc_title", "Hash", "Doc:title")]);
}

#[cfg(feature = "vector-index")]
mod vector {
    use super::*;

    #[test]
    fn api_vector_index_is_shown_with_its_real_target_and_tracks_drop() {
        let db = GrafeoDB::new_in_memory();
        let id = db
            .create_node_with_props(
                &["Doc"],
                [("emb", Value::Vector(vec![1.0_f32, 0.0, 0.0].into()))],
            )
            .unwrap();
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .unwrap();
        let hits = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id);
        assert_show(&db, &[["", "Vector", "Doc", "emb"]]);
        assert_list_indexes(&db, &[("", "Vector", "Doc:emb")]);
        assert!(db.drop_vector_index("Doc", "emb").unwrap());
        assert_show(&db, &[]);
        assert_list_indexes(&db, &[]);
    }
}

#[cfg(feature = "text-index")]
mod text {
    use super::*;

    #[test]
    fn api_text_index_is_shown_and_a_ddl_name_suppresses_its_unnamed_duplicate() {
        let db = GrafeoDB::new_in_memory();
        let id = db
            .create_node_with_props(&["Doc"], [("body", Value::from("hello grafeo"))])
            .unwrap();
        db.create_text_index("Doc", "body").unwrap();
        assert!(db.text_index_options("Doc", "body").is_some());
        let hits = db.text_search("Doc", "body", "hello", 1, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id);
        assert_show(&db, &[["", "FullText", "Doc", "body"]]);
        assert_list_indexes(&db, &[("", "FullText", "Doc:body")]);
        assert!(db.drop_text_index("Doc", "body").unwrap());
        assert_show(&db, &[]);
        assert_list_indexes(&db, &[]);
        db.execute("CREATE INDEX doc_body FOR (d:Doc) ON (d.body) USING TEXT")
            .unwrap();
        db.create_text_index("Doc", "body").unwrap();
        assert_show(&db, &[["doc_body", "FullText", "Doc", "body"]]);
        assert_list_indexes(&db, &[("doc_body", "FullText", "Doc:body")]);
    }
}

#[cfg(all(feature = "vector-index", feature = "text-index"))]
mod combined {
    use super::*;

    #[test]
    fn a_property_and_text_index_on_the_same_property_are_distinct_real_indexes() {
        let db = GrafeoDB::new_in_memory();
        db.create_node_with_props(
            &["Doc"],
            [
                ("body", Value::from("hello grafeo")),
                ("emb", Value::Vector(vec![1.0_f32, 0.0, 0.0].into())),
            ],
        )
        .unwrap();
        db.create_property_index("body").unwrap();
        db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
            .unwrap();
        db.create_text_index("Doc", "body").unwrap();
        assert!(db.has_property_index("body"));
        assert!(db.text_index_options("Doc", "body").is_some());
        assert_show(
            &db,
            &[
                ["", "Hash", "", "body"],
                ["", "FullText", "Doc", "body"],
                ["", "Vector", "Doc", "emb"],
            ],
        );
        assert_list_indexes(
            &db,
            &[
                ("", "Hash", ":body"),
                ("", "FullText", "Doc:body"),
                ("", "Vector", "Doc:emb"),
            ],
        );
    }
}

// Each pair is exercised in its own database. This checks exact metadata
// without changing the existing joined-key collision behavior.
#[cfg(feature = "vector-index")]
mod colon_vector {
    use super::*;

    #[test]
    fn api_vector_colon_labels_and_properties_are_reported_exactly() {
        for (label, property) in [
            ("Doc:Vector", "emb"),
            ("Doc", "emb:raw"),
            ("Doc:Vector", "emb:raw"),
        ] {
            let db = GrafeoDB::new_in_memory();
            let id = db
                .create_node_with_props(
                    &[label],
                    [(property, Value::Vector(vec![1.0_f32, 0.0, 0.0].into()))],
                )
                .unwrap();
            db.create_vector_index(label, property, Some(3), None, None, None, None)
                .unwrap();
            let hits = db
                .vector_search(label, property, &[1.0, 0.0, 0.0], 1, None, None)
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].0, id);
            assert_show(&db, &[["", "Vector", label, property]]);
            // The existing public target is a joined string. Its format
            // remains unchanged even when the exact SHOW cells need colons.
            assert_list_indexes(&db, &[("", "Vector", &format!("{label}:{property}"))]);
            assert!(db.drop_vector_index(label, property).unwrap());
            assert_show(&db, &[]);
            assert_list_indexes(&db, &[]);
        }
    }
}

#[cfg(feature = "text-index")]
mod colon_text {
    use super::*;

    #[test]
    fn api_text_colon_labels_and_properties_are_reported_exactly() {
        for (label, property) in [
            ("Doc:Text", "body"),
            ("Doc", "body:raw"),
            ("Doc:Text", "body:raw"),
        ] {
            let db = GrafeoDB::new_in_memory();
            let id = db
                .create_node_with_props(&[label], [(property, Value::from("hello grafeo"))])
                .unwrap();
            db.create_text_index(label, property).unwrap();
            assert!(db.text_index_options(label, property).is_some());
            let hits = db.text_search(label, property, "hello", 1, None).unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].0, id);
            assert_show(&db, &[["", "FullText", label, property]]);
            assert_list_indexes(&db, &[("", "FullText", &format!("{label}:{property}"))]);
            assert!(db.drop_text_index(label, property).unwrap());
            assert_show(&db, &[]);
            assert_list_indexes(&db, &[]);
        }
    }
}

#[cfg(feature = "grafeo-file")]
mod persistent {
    use super::*;
    use grafeo_engine::Config;

    #[test]
    fn api_index_inventory_and_colon_targets_survive_checkpoint_and_wal_reopen() {
        // Keep the no-WAL checkpoint control, and exercise the WAL-enabled
        // public configuration when it is built. No crash claim is made.
        #[cfg(feature = "wal")]
        let wal_modes: &[bool] = &[false, true];
        #[cfg(not(feature = "wal"))]
        let wal_modes: &[bool] = &[false];
        for &with_wal in wal_modes {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("indexes.grafeo");
            let open = || {
                let config = Config::persistent(&path);
                GrafeoDB::with_config(if with_wal {
                    config
                } else {
                    config.without_wal()
                })
                .unwrap()
            };
            let db = open();
            db.create_property_index("email").unwrap();
            #[cfg(feature = "vector-index")]
            let vector_id = {
                let id = db
                    .create_node_with_props(
                        &["Doc:Vector"],
                        [("emb:raw", Value::Vector(vec![1.0_f32, 0.0, 0.0].into()))],
                    )
                    .unwrap();
                db.create_vector_index("Doc:Vector", "emb:raw", Some(3), None, None, None, None)
                    .unwrap();
                id
            };
            #[cfg(feature = "text-index")]
            let text_id = {
                let id = db
                    .create_node_with_props(
                        &["Doc:Text"],
                        [("body:raw", Value::from("hello grafeo"))],
                    )
                    .unwrap();
                db.create_text_index("Doc:Text", "body:raw").unwrap();
                id
            };
            let want = &[
                ["", "Hash", "", "email"],
                #[cfg(feature = "vector-index")]
                ["", "Vector", "Doc:Vector", "emb:raw"],
                #[cfg(feature = "text-index")]
                ["", "FullText", "Doc:Text", "body:raw"],
            ];
            let want_list = &[
                ("", "Hash", ":email"),
                #[cfg(feature = "vector-index")]
                ("", "Vector", "Doc:Vector:emb:raw"),
                #[cfg(feature = "text-index")]
                ("", "FullText", "Doc:Text:body:raw"),
            ];
            let check = |db: &GrafeoDB| {
                assert!(db.has_property_index("email"));
                assert_show(db, want);
                assert_list_indexes(db, want_list);
                #[cfg(feature = "vector-index")]
                {
                    let hits = db
                        .vector_search("Doc:Vector", "emb:raw", &[1.0, 0.0, 0.0], 1, None, None)
                        .unwrap();
                    assert_eq!(hits.len(), 1);
                    assert_eq!(hits[0].0, vector_id);
                }
                #[cfg(feature = "text-index")]
                {
                    assert!(db.text_index_options("Doc:Text", "body:raw").is_some());
                    let hits = db
                        .text_search("Doc:Text", "body:raw", "hello", 1, None)
                        .unwrap();
                    assert_eq!(hits.len(), 1);
                    assert_eq!(hits[0].0, text_id);
                }
            };
            check(&db);
            db.close().unwrap();
            let db = open();
            check(&db);
            assert!(db.drop_property_index("email").unwrap());
            #[cfg(feature = "vector-index")]
            assert!(db.drop_vector_index("Doc:Vector", "emb:raw").unwrap());
            #[cfg(feature = "text-index")]
            assert!(db.drop_text_index("Doc:Text", "body:raw").unwrap());
            assert_show(&db, &[]);
            assert_list_indexes(&db, &[]);
            db.close().unwrap();
            let db = open();
            assert!(!db.has_property_index("email"));
            #[cfg(feature = "vector-index")]
            assert!(!db.drop_vector_index("Doc:Vector", "emb:raw").unwrap());
            #[cfg(feature = "text-index")]
            assert_eq!(db.text_index_options("Doc:Text", "body:raw"), None);
            assert_show(&db, &[]);
            assert_list_indexes(&db, &[]);
            db.close().unwrap();
        }
    }
}
