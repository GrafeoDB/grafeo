//! `get_node_edges`: a node's edges as `(neighbor, edge)` pairs without a
//! query (#402), seeing what a query sees: committed edges for the database
//! and a graph handle, the open transaction's own for a session.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test node_edges
//! ```

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_engine::{Direction, GrafeoDB};

fn sorted(mut edges: Vec<(NodeId, EdgeId)>) -> Vec<(NodeId, EdgeId)> {
    edges.sort();
    edges
}

/// Alix knows Gus and Vincent, Gus works with Alix, and Alix has a note to
/// herself (a self-loop).
struct Graph {
    db: GrafeoDB,
    alix: NodeId,
    gus: NodeId,
    vincent: NodeId,
    knows_gus: EdgeId,
    knows_vincent: EdgeId,
    works_with: EdgeId,
    note: EdgeId,
}

fn graph() -> Graph {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    let vincent = db.create_node(&["Person"]).unwrap();
    let knows_gus = db.create_edge(alix, gus, "KNOWS").unwrap();
    let knows_vincent = db.create_edge(alix, vincent, "KNOWS").unwrap();
    let works_with = db.create_edge(gus, alix, "WORKS_WITH").unwrap();
    let note = db.create_edge(alix, alix, "NOTE").unwrap();
    Graph {
        db,
        alix,
        gus,
        vincent,
        knows_gus,
        knows_vincent,
        works_with,
        note,
    }
}

#[test]
fn edges_by_direction() {
    let g = graph();
    assert_eq!(
        sorted(g.db.get_node_edges(g.alix, Direction::Outgoing, &[])),
        sorted(vec![
            (g.gus, g.knows_gus),
            (g.vincent, g.knows_vincent),
            (g.alix, g.note)
        ])
    );
    assert_eq!(
        sorted(g.db.get_node_edges(g.alix, Direction::Incoming, &[])),
        sorted(vec![(g.gus, g.works_with), (g.alix, g.note)])
    );
    // Both directions list the self-loop once.
    assert_eq!(
        sorted(g.db.get_node_edges(g.alix, Direction::Both, &[])),
        sorted(vec![
            (g.gus, g.knows_gus),
            (g.vincent, g.knows_vincent),
            (g.gus, g.works_with),
            (g.alix, g.note)
        ])
    );
    assert_eq!(
        g.db.get_node_edges(g.vincent, Direction::Incoming, &[]),
        vec![(g.alix, g.knows_vincent)]
    );
    assert_eq!(
        g.db.get_node_edges(g.vincent, Direction::Outgoing, &[]),
        Vec::new()
    );
}

#[test]
fn edges_of_some_types() {
    let g = graph();
    assert_eq!(
        sorted(g.db.get_node_edges(g.alix, Direction::Both, &["KNOWS"])),
        sorted(vec![(g.gus, g.knows_gus), (g.vincent, g.knows_vincent)])
    );
    assert_eq!(
        sorted(g.db.get_node_edges(g.alix, Direction::Both, &["WORKS_WITH", "NOTE"])),
        sorted(vec![(g.gus, g.works_with), (g.alix, g.note)])
    );
    assert_eq!(
        g.db.get_node_edges(g.alix, Direction::Both, &["MISSING"]),
        Vec::new()
    );
}

#[test]
fn deleted_edges_and_nodes_are_gone() {
    let g = graph();
    assert!(g.db.delete_edge(g.knows_gus).unwrap());
    assert_eq!(
        sorted(g.db.get_node_edges(g.alix, Direction::Outgoing, &["KNOWS"])),
        vec![(g.vincent, g.knows_vincent)]
    );
    assert_eq!(
        g.db.get_node_edges(g.gus, Direction::Incoming, &[]),
        Vec::new()
    );

    assert!(g.db.detach_delete_node(g.vincent).unwrap());
    assert_eq!(
        g.db.get_node_edges(g.vincent, Direction::Both, &[]),
        Vec::new()
    );
    assert_eq!(
        g.db.get_node_edges(g.alix, Direction::Outgoing, &["KNOWS"]),
        Vec::new()
    );
    assert_eq!(
        g.db.get_node_edges(NodeId(88_888), Direction::Both, &[]),
        Vec::new()
    );
}

#[test]
fn a_session_sees_its_own_uncommitted_edges() {
    let g = graph();
    let mut session = g.db.session();
    session.begin_transaction().unwrap();
    let added = session.create_edge(g.vincent, g.gus, "KNOWS").unwrap();
    assert!(session.delete_edge(g.knows_gus).unwrap());

    assert_eq!(
        session.get_node_edges(g.gus, Direction::Incoming, &["KNOWS"]),
        vec![(g.vincent, added)]
    );
    // The uncommitted edge and the uncommitted delete are the session's own:
    // the database still sees `knows_gus`, and not the new edge (#412).
    assert_eq!(
        g.db.get_node_edges(g.gus, Direction::Incoming, &["KNOWS"]),
        vec![(g.alix, g.knows_gus)]
    );

    session.rollback().unwrap();
    assert_eq!(
        session.get_node_edges(g.gus, Direction::Incoming, &["KNOWS"]),
        vec![(g.alix, g.knows_gus)]
    );
}

#[test]
fn a_graph_handle_lists_the_edges_of_its_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH trips").unwrap();
    let trips = db.graph("trips").unwrap();
    let prague = trips.create_node(&["City"]).unwrap();
    let berlin = trips.create_node(&["City"]).unwrap();
    let route = trips.create_edge(prague, berlin, "ROUTE").unwrap();
    // The default graph's node and its edge are not the named graph's.
    let default_node = db.create_node(&["City"]).unwrap();
    let default_edge = db.create_edge(default_node, default_node, "LOOP").unwrap();

    assert_eq!(
        trips
            .get_node_edges(prague, Direction::Outgoing, &[])
            .unwrap(),
        vec![(berlin, route)]
    );
    assert_eq!(
        db.get_node_edges(default_node, Direction::Both, &[]),
        vec![(default_node, default_edge)]
    );
    assert!(
        !trips
            .get_node_edges(default_node, Direction::Both, &[])
            .unwrap()
            .contains(&(default_node, default_edge))
    );

    db.execute("DROP GRAPH trips").unwrap();
    assert!(
        trips
            .get_node_edges(prague, Direction::Outgoing, &[])
            .is_err()
    );
}
