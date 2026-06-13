//! Tests for text index mutation hooks.
//!
//! These verify that text indexes are automatically kept in sync
//! when nodes are created, updated, or deleted.

#![cfg(feature = "text-index")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Helper: create a DB with a text index on :Doc(content).
fn setup_db_with_text_index() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.create_text_index("Doc", "content").unwrap();
    db
}

#[test]
fn test_text_index_auto_insert_via_create_node() {
    let db = setup_db_with_text_index();

    // Create a node AFTER the text index exists
    let id = db
        .create_node_with_props(
            &["Doc"],
            [(
                "content",
                Value::String("The quick brown fox jumps over the lazy dog".into()),
            )],
        )
        .unwrap();

    // Should be searchable immediately
    let results = db
        .text_search("Doc", "content", "quick brown fox", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
}

#[test]
fn test_text_index_auto_update_via_set_property() {
    let db = setup_db_with_text_index();

    let id = db
        .create_node_with_props(
            &["Doc"],
            [(
                "content",
                Value::String("original text about databases".into()),
            )],
        )
        .unwrap();

    // Verify initial text is searchable
    let results = db.text_search("Doc", "content", "databases", 10).unwrap();
    assert_eq!(results.len(), 1);

    // Update the property
    db.set_node_property(
        id,
        "content",
        Value::String("updated text about graph theory".into()),
    )
    .unwrap();

    // Old text should no longer match
    let results = db.text_search("Doc", "content", "databases", 10).unwrap();
    assert!(
        results.is_empty(),
        "Old text should not be found after update"
    );

    // New text should be searchable
    let results = db
        .text_search("Doc", "content", "graph theory", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
}

#[test]
fn test_text_index_auto_delete() {
    let db = setup_db_with_text_index();

    let id = db
        .create_node_with_props(
            &["Doc"],
            [(
                "content",
                Value::String("searchable document content".into()),
            )],
        )
        .unwrap();

    // Verify it's searchable
    let results = db.text_search("Doc", "content", "searchable", 10).unwrap();
    assert_eq!(results.len(), 1);

    // Delete the node
    let deleted = db.delete_node(id).unwrap();
    assert!(deleted);

    // Should no longer appear in search
    let results = db.text_search("Doc", "content", "searchable", 10).unwrap();
    assert!(
        results.is_empty(),
        "Deleted node should not appear in text search"
    );
}

#[test]
fn test_text_index_add_label() {
    let db = GrafeoDB::new_in_memory();
    db.create_text_index("Article", "content").unwrap();

    // Create a node WITHOUT the Article label
    let id = db
        .create_node_with_props(
            &["Doc"],
            [(
                "content",
                Value::String("important article about Rust programming".into()),
            )],
        )
        .unwrap();

    // Not searchable under Article index
    let results = db.text_search("Article", "content", "Rust", 10).unwrap();
    assert!(results.is_empty());

    // Add the Article label
    let added = db.add_node_label(id, "Article");
    assert!(added);

    // Now it should be searchable under the Article text index
    let results = db
        .text_search("Article", "content", "Rust programming", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
}

#[test]
fn test_text_index_non_string_property_ignored() {
    let db = setup_db_with_text_index();

    // Set a non-String property: should not crash or pollute the index
    let id = db
        .create_node_with_props(&["Doc"], [("content", Value::Int64(42))])
        .unwrap();

    let results = db.text_search("Doc", "content", "42", 10).unwrap();
    assert!(
        results.is_empty(),
        "Non-string values should not be indexed"
    );

    // Now set it to a string
    db.set_node_property(
        id,
        "content",
        Value::String("now it is a string value".into()),
    )
    .unwrap();
    let results = db
        .text_search("Doc", "content", "string value", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_text_index_multiple_nodes() {
    let db = setup_db_with_text_index();

    let _id1 = db
        .create_node_with_props(
            &["Doc"],
            [(
                "content",
                Value::String("machine learning with neural networks".into()),
            )],
        )
        .unwrap();
    let _id2 = db
        .create_node_with_props(
            &["Doc"],
            [(
                "content",
                Value::String("graph databases and knowledge graphs".into()),
            )],
        )
        .unwrap();
    let id3 = db
        .create_node_with_props(
            &["Doc"],
            [(
                "content",
                Value::String("machine learning on graphs".into()),
            )],
        )
        .unwrap();

    // "machine learning" should match 2 docs
    let results = db
        .text_search("Doc", "content", "machine learning", 10)
        .unwrap();
    assert_eq!(results.len(), 2);

    // Delete one
    db.delete_node(id3).unwrap();
    let results = db
        .text_search("Doc", "content", "machine learning", 10)
        .unwrap();
    assert_eq!(results.len(), 1);
}
