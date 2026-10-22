//! `hybrid_search` and `text_search` take property filters as
//! `vector_search` does (#397): the text and the vector search both keep
//! only the matching nodes, before the results of a hybrid search are fused,
//! so up to `k` matching nodes come back, ranked as the fusion of the two
//! filtered lists; text scores stay those of the whole index.
//!
//! ```bash
//! cargo test -p grafeo-engine --features hybrid-search,lpg,gql --test search_filters
//! ```

#![cfg(all(feature = "hybrid-search", feature = "lpg"))]

use std::collections::{BTreeSet, HashMap};

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::index::text::{FusionMethod, fuse_results};
use grafeo_engine::GrafeoDB;

/// The text query of every search here.
const QUERY_TEXT: &str = "canals";

/// The vector query of every search here: Gus's embedding.
const QUERY_VECTOR: [f32; 2] = [1.0, 0.0];

/// Six documents with an owner, a city, a rank, a text and a 2-value
/// embedding, with a text index on `:Doc(text)` and a cosine vector index on
/// `:Doc(emb)`.
///
/// For "canals" the text index ranks Alix, Vincent, Jules; for
/// [`QUERY_VECTOR`] the vector index ranks Gus, Mia, Vincent, Jules, Butch,
/// Alix. Alix (a text match only) and Gus (a vector match only) live in
/// Amsterdam, the other four in Berlin.
fn documents() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (owner, city, rank, text, emb) in [
        (
            "Alix",
            "Amsterdam",
            3,
            "Alix rides along canals, canals and canals",
            [0.0, 1.0],
        ),
        (
            "Gus",
            "Amsterdam",
            19,
            "Gus buys museum tickets",
            [1.0, 0.0],
        ),
        ("Vincent", "Berlin", 88, "Vincent paints canals", [0.8, 0.6]),
        (
            "Mia",
            "Berlin",
            3,
            "Mia dances in Berlin clubs",
            [0.95, 0.31],
        ),
        (
            "Jules",
            "Berlin",
            19,
            "Jules swims past the old canals of Berlin at dawn",
            [0.6, 0.8],
        ),
        (
            "Butch",
            "Berlin",
            88,
            "Butch trains at the boxing gym",
            [0.3, 0.954],
        ),
    ] {
        let mut properties = HashMap::new();
        properties.insert(PropertyKey::new("owner"), Value::from(owner));
        properties.insert(PropertyKey::new("city"), Value::from(city));
        properties.insert(PropertyKey::new("rank"), Value::Int64(rank));
        properties.insert(PropertyKey::new("text"), Value::from(text));
        properties.insert(PropertyKey::new("emb"), Value::Vector(emb.to_vec().into()));
        db.create_node_with_props(&["Doc"], properties).unwrap();
    }
    db.create_text_index("Doc", "text").unwrap();
    db.create_vector_index("Doc", "emb", Some(2), Some("cosine"), None, None, None)
        .unwrap();
    db
}

/// The string property `key` of `node`.
fn text_of(db: &GrafeoDB, node: NodeId, key: &str) -> String {
    let found = db.get_node(node).expect("a found node exists");
    match found.properties.get(&PropertyKey::new(key)) {
        Some(Value::String(text)) => text.to_string(),
        other => panic!("node {node:?} has {other:?} for {key}"),
    }
}

/// The owners of `results`, in their order.
fn owners(db: &GrafeoDB, results: &[(NodeId, f64)]) -> Vec<String> {
    results
        .iter()
        .map(|(node, _)| text_of(db, *node, "owner"))
        .collect()
}

/// The filter `city = <city>`.
fn in_city(city: &str) -> HashMap<String, Value> {
    HashMap::from([("city".to_string(), Value::from(city))])
}

/// `hybrid_search` on the documents with `filters`, the text and vector
/// queries above (the vector query only when `with_vector`), and RRF.
fn hybrid(
    db: &GrafeoDB,
    with_vector: bool,
    k: usize,
    filters: Option<&HashMap<String, Value>>,
) -> Vec<(NodeId, f64)> {
    db.hybrid_search(
        "Doc",
        "text",
        "emb",
        QUERY_TEXT,
        with_vector.then_some(&QUERY_VECTOR[..]),
        k,
        None,
        filters,
    )
    .unwrap()
}

#[test]
fn filters_narrow_the_text_and_the_vector_search() {
    let db = documents();
    let unfiltered = owners(&db, &hybrid(&db, true, 10, None));
    for (owner, side) in [("Alix", "text"), ("Gus", "vector")] {
        assert!(
            unfiltered.iter().any(|found| found == owner),
            "unfiltered, the {side} search brings {owner}: {unfiltered:?}"
        );
    }

    let berlin = in_city("Berlin");
    let filtered = hybrid(&db, true, 10, Some(&berlin));
    assert_eq!(
        owners(&db, &filtered).into_iter().collect::<BTreeSet<_>>(),
        ["Butch", "Jules", "Mia", "Vincent"]
            .map(String::from)
            .into_iter()
            .collect(),
        "every node of Berlin, Alix (text) and Gus (vector) left out: {filtered:?}"
    );

    let text_only = hybrid(&db, false, 10, Some(&berlin));
    assert_eq!(
        owners(&db, &text_only),
        ["Vincent", "Jules"],
        "without a query vector, the text matches of Berlin in text order"
    );

    let no_text_match = db
        .hybrid_search(
            "Doc",
            "text",
            "emb",
            "trams",
            Some(&QUERY_VECTOR),
            10,
            None,
            Some(&berlin),
        )
        .unwrap();
    assert_eq!(
        owners(&db, &no_text_match),
        ["Mia", "Vincent", "Jules", "Butch"],
        "without a text match, the vector matches of Berlin nearest first"
    );
}

#[test]
fn k_counts_the_matching_nodes() {
    let db = documents();
    let unfiltered_top_two = owners(&db, &hybrid(&db, true, 2, None));
    assert_eq!(
        unfiltered_top_two,
        ["Vincent", "Jules"],
        "filtering the unfiltered top two afterwards would leave no node of Amsterdam"
    );

    let amsterdam = in_city("Amsterdam");
    assert_eq!(
        owners(&db, &hybrid(&db, true, 2, Some(&amsterdam))),
        ["Alix", "Gus"],
        "Alix matches both searches, Gus the vector one"
    );
    let berlin = in_city("Berlin");
    assert_eq!(
        owners(&db, &hybrid(&db, true, 2, Some(&berlin))),
        ["Vincent", "Jules"]
    );

    // The text search ranks Jules third for canals, after Alix and Vincent:
    // the one text match of rank 19 still comes back for k = 1.
    let rank_19 = HashMap::from([("rank".to_string(), Value::Int64(19))]);
    assert_eq!(
        owners(&db, &hybrid(&db, false, 1, Some(&rank_19))),
        ["Jules"]
    );
}

/// The fusion of the unfiltered text and vector results of the documents,
/// each list cut down to the nodes `keep` accepts, as `hybrid_search` fuses
/// them (a list without results is left out).
fn fusion_of_the_filtered_lists(
    db: &GrafeoDB,
    method: &FusionMethod,
    k: usize,
    keep: impl Fn(NodeId) -> bool,
) -> Vec<(NodeId, f64)> {
    let text: Vec<(NodeId, f64)> = db
        .text_search("Doc", "text", QUERY_TEXT, 100, None)
        .unwrap()
        .into_iter()
        .filter(|(node, _)| keep(*node))
        .collect();
    let vector: Vec<(NodeId, f64)> = db
        .vector_search("Doc", "emb", &QUERY_VECTOR, 100, None, None)
        .unwrap()
        .into_iter()
        .filter(|(node, _)| keep(*node))
        .map(|(node, distance)| (node, -f64::from(distance)))
        .collect();
    let sources: Vec<Vec<(NodeId, f64)>> = [text, vector]
        .into_iter()
        .filter(|list| !list.is_empty())
        .collect();
    fuse_results(&sources, method, k)
}

#[test]
fn filtered_results_are_the_fusion_of_the_filtered_lists() {
    let db = documents();
    let berlin = in_city("Berlin");
    for method in [
        FusionMethod::Rrf { k: 60 },
        FusionMethod::Weighted {
            weights: vec![0.5, 0.5],
        },
    ] {
        let expected = fusion_of_the_filtered_lists(&db, &method, 10, |node| {
            text_of(&db, node, "city") == "Berlin"
        });
        assert!(
            expected.windows(2).all(|pair| pair[0].1 > pair[1].1),
            "the documents give distinct fused scores, so one order: {expected:?}"
        );
        let found = db
            .hybrid_search(
                "Doc",
                "text",
                "emb",
                QUERY_TEXT,
                Some(&QUERY_VECTOR),
                10,
                Some(method.clone()),
                Some(&berlin),
            )
            .unwrap();
        assert_eq!(
            owners(&db, &found),
            owners(&db, &expected),
            "{method:?}: the order of the fused filtered lists"
        );
        for ((node, score), (expected_node, expected_score)) in found.iter().zip(&expected) {
            assert_eq!(node, expected_node);
            assert!(
                (score - expected_score).abs() < 1e-12,
                "{method:?}: {} scores {score}, the fused filtered lists {expected_score}",
                text_of(&db, *node, "owner")
            );
        }
    }
}

#[test]
fn operator_filters_work_as_in_vector_search() {
    let db = documents();
    let filters = HashMap::from([
        ("city".to_string(), Value::from("Berlin")),
        (
            "rank".to_string(),
            Value::Map(std::sync::Arc::new(
                [(PropertyKey::new("$gt"), Value::Int64(19))]
                    .into_iter()
                    .collect(),
            )),
        ),
    ]);
    let vector_side = db
        .vector_search("Doc", "emb", &QUERY_VECTOR, 10, None, Some(&filters))
        .unwrap();
    let vector_owners: Vec<String> = vector_side
        .iter()
        .map(|(node, _)| text_of(&db, *node, "owner"))
        .collect();
    assert_eq!(vector_owners, ["Vincent", "Butch"], "rank 88 in Berlin");
    assert_eq!(
        owners(&db, &hybrid(&db, true, 10, Some(&filters))),
        ["Vincent", "Butch"],
        "Vincent matches both searches, Butch the vector one"
    );
}

#[test]
fn a_filter_no_node_matches_finds_nothing() {
    let db = documents();
    let paris = in_city("Paris");
    assert_eq!(
        hybrid(&db, true, 10, Some(&paris)),
        [],
        "with a query vector"
    );
    assert_eq!(hybrid(&db, false, 10, Some(&paris)), [], "text only");
}

#[test]
fn no_filters_and_empty_filters_search_as_before() {
    let db = documents();
    let unfiltered = hybrid(&db, true, 10, None);
    assert_eq!(
        owners(&db, &unfiltered),
        ["Vincent", "Alix", "Jules", "Gus", "Mia", "Butch"],
        "the RRF of both lists over every document"
    );
    assert_eq!(
        hybrid(&db, true, 10, Some(&HashMap::new())),
        unfiltered,
        "an empty filter map filters nothing"
    );
    assert_eq!(
        unfiltered,
        fusion_of_the_filtered_lists(&db, &FusionMethod::default(), 10, |_| true)
    );
}

/// What `text_search` finds for "canals" with `filters`, as owners and the
/// bits of their scores, best first.
fn text_hits(
    db: &GrafeoDB,
    k: usize,
    filters: Option<&HashMap<String, Value>>,
) -> Vec<(String, u64)> {
    db.text_search("Doc", "text", QUERY_TEXT, k, filters)
        .unwrap()
        .into_iter()
        .map(|(node, score)| (text_of(db, node, "owner"), score.to_bits()))
        .collect()
}

#[test]
fn text_search_filters_keep_the_scores_of_the_whole_index() {
    let db = documents();
    let unfiltered = text_hits(&db, 10, None);
    assert_eq!(
        unfiltered
            .iter()
            .map(|(owner, _)| owner.as_str())
            .collect::<Vec<_>>(),
        ["Alix", "Vincent", "Jules"]
    );
    let berlin = in_city("Berlin");
    assert_eq!(
        text_hits(&db, 10, Some(&berlin)),
        unfiltered[1..],
        "Vincent and Jules, scored as without the filter"
    );
    assert_eq!(
        text_hits(&db, 10, Some(&HashMap::new())),
        unfiltered,
        "an empty filter map filters nothing"
    );
    assert_eq!(text_hits(&db, 10, Some(&in_city("Paris"))), []);
}

#[test]
fn text_search_k_counts_the_matching_nodes() {
    let db = documents();
    let rank_19 = HashMap::from([("rank".to_string(), Value::Int64(19))]);
    let found = text_hits(&db, 1, Some(&rank_19));
    assert_eq!(
        found
            .iter()
            .map(|(owner, _)| owner.as_str())
            .collect::<Vec<_>>(),
        ["Jules"],
        "the best match of rank 19, though Alix and Vincent outscore Jules"
    );
    let ranked_above_19 = HashMap::from([(
        "rank".to_string(),
        Value::Map(std::sync::Arc::new(
            [(PropertyKey::new("$gt"), Value::Int64(19))]
                .into_iter()
                .collect(),
        )),
    )]);
    assert_eq!(
        text_hits(&db, 10, Some(&ranked_above_19))
            .iter()
            .map(|(owner, _)| owner.as_str())
            .collect::<Vec<_>>(),
        ["Vincent"],
        "operator filters as in vector_search"
    );
}
