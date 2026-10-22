//! A read the store cannot do is an error, never an absent answer: the
//! direct reads, the counts, the admin reports, the session's reads and
//! queries all return the store's error.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test read_errors
//! ```

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, ReadView, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::{Edge, Node};
use grafeo_core::graph::{Direction, GraphStore};
use grafeo_core::statistics::Statistics;
use grafeo_engine::{Config, GrafeoDB};

/// A store whose every read fails, as a remote store that lost its
/// connection does.
struct Unreachable;

const MESSAGE: &str = "the store is unreachable";

fn unreachable<T>() -> Result<T> {
    Err(Error::Internal(MESSAGE.to_string()))
}

impl GraphStore for Unreachable {
    fn node(&self, _: NodeId, _: ReadView) -> Result<Option<Node>> {
        unreachable()
    }
    fn edge(&self, _: EdgeId, _: ReadView) -> Result<Option<Edge>> {
        unreachable()
    }
    fn edges(
        &self,
        _: NodeId,
        _: Direction,
        _: &[ArcStr],
        _: ReadView,
    ) -> Result<Vec<(NodeId, EdgeId)>> {
        unreachable()
    }
    fn has_backward_adjacency(&self) -> bool {
        true
    }
    fn node_ids(&self, _: ReadView) -> Result<Vec<NodeId>> {
        unreachable()
    }
    fn all_labels(&self) -> Result<Vec<String>> {
        unreachable()
    }
    fn all_edge_types(&self) -> Result<Vec<String>> {
        unreachable()
    }
    fn all_property_keys(&self) -> Result<Vec<String>> {
        unreachable()
    }
    fn statistics(&self) -> Arc<Statistics> {
        Arc::new(Statistics::default())
    }
    fn committed_epoch(&self) -> EpochId {
        EpochId::INITIAL
    }
}

fn failed<T: std::fmt::Debug>(what: &str, read: Result<T>) {
    match read {
        Err(error) => assert!(error.to_string().contains(MESSAGE), "{what}: {error}"),
        Ok(found) => panic!("{what} answered {found:?} although the store failed"),
    }
}

fn database() -> GrafeoDB {
    GrafeoDB::with_read_store(Arc::new(Unreachable), Config::in_memory()).unwrap()
}

#[test]
fn the_direct_reads_return_the_stores_error() {
    let db = database();
    let (alix, knows) = (NodeId::new(0), EdgeId::new(0));
    failed("get_node", db.get_node(alix));
    failed("get_edge", db.get_edge(knows));
    failed("get_node_labels", db.get_node_labels(alix));
    failed(
        "get_node_edges",
        db.get_node_edges(alix, Direction::Both, &["KNOWS"]),
    );
    failed(
        "find_nodes_by_property",
        db.find_nodes_by_property("name", &Value::from("Alix")),
    );
}

#[test]
fn the_counts_and_reports_return_the_stores_error() {
    let db = database();
    failed("node_count", db.node_count());
    failed("edge_count", db.edge_count());
    failed("label_count", db.label_count());
    failed("edge_type_count", db.edge_type_count());
    failed("property_key_count", db.property_key_count());
    failed("info", db.info());
    failed("detailed_stats", db.detailed_stats());
    failed("schema", db.schema());
    failed("validate", db.validate());
}

#[test]
fn the_sessions_reads_return_the_stores_error() {
    let db = database();
    let session = db.session();
    let alix = NodeId::new(0);
    failed("Session::get_node", session.get_node(alix));
    failed("Session::get_edge", session.get_edge(EdgeId::new(0)));
    failed(
        "Session::get_node_property",
        session.get_node_property(alix, "name"),
    );
    failed("Session::node_exists", session.node_exists(alix));
    failed("Session::get_degree", session.get_degree(alix));
    failed("Session::get_nodes_batch", session.get_nodes_batch(&[alix]));
    failed(
        "Session::get_neighbors_outgoing",
        session.get_neighbors_outgoing(alix),
    );
    failed(
        "Session::find_nodes_by_property",
        session.find_nodes_by_property("name", &Value::from("Alix")),
    );
}

#[test]
fn a_query_returns_the_stores_error() {
    let db = database();
    let session = db.session();
    for query in [
        "MATCH (n) RETURN n",
        "MATCH (a)-[:KNOWS]->(b) RETURN a, b",
        "RETURN info()",
        "RETURN schema()",
    ] {
        failed(
            query,
            session.execute(query).map(|result| result.rows().len()),
        );
    }
}

/// A store without history says so: the history reads are the unsupported
/// error, not an empty history.
#[test]
fn a_store_without_history_does_not_answer_empty() {
    let db = database();
    let unsupported = |what: &str, read: Result<usize>| match read {
        Err(error) => assert!(
            error.to_string().contains("does not support"),
            "{what}: {error}"
        ),
        Ok(found) => panic!("{what} answered {found} versions"),
    };
    unsupported(
        "get_node_history",
        db.get_node_history(NodeId::new(0)).map(|h| h.len()),
    );
    unsupported(
        "get_edge_history",
        db.get_edge_history(EdgeId::new(0)).map(|h| h.len()),
    );
    unsupported(
        "get_node_property_history",
        db.get_node_property_history(NodeId::new(0), "name")
            .map(|h| h.len()),
    );
    unsupported(
        "get_all_node_property_history",
        db.get_all_node_property_history(NodeId::new(0))
            .map(|h| h.len()),
    );
    unsupported(
        "get_node_property_at_epoch",
        db.get_node_property_at_epoch(NodeId::new(0), "name", EpochId::INITIAL)
            .map(|value| usize::from(value.is_some())),
    );
}
