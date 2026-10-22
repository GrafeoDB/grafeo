//! A row-group root without backward edges (10b) through the engine's
//! writer, checkpoints and opens: its incoming reads find the edges in the
//! edge table, a detach delete deletes the edges into a node, the
//! checkpoint writes no incoming adjacency, and the file reopens without
//! backward edges, as the configuration it stores says.

use grafeo_common::types::ReadView;
use grafeo_core::execution::operators::OperatorError;
use grafeo_core::graph::{Direction, GraphStore};

use super::{Database, images, travels};
use crate::config::Config;

/// The configuration of a database without backward edges.
fn without_backward_edges() -> Config {
    Config::in_memory().without_backward_edges()
}

/// The incoming reads of the root find Gus's edge to Alix in the edge
/// table; a delete of Alix without `DETACH` is refused for it, a detach
/// delete deletes it, and a checkpoint and load into a root without
/// backward edges holds the same, with no incoming adjacency.
#[test]
fn a_detach_delete_finds_the_incoming_edges_without_backward_edges() {
    let database = Database::with_config(&without_backward_edges());
    let travellers = travels(&database);
    let (alix, gus) = (travellers.alix, travellers.gus);
    let trips = database.store(Some("trips"));
    assert!(!database.root.has_backward_adjacency());
    assert!(!trips.has_backward_adjacency(), "trips follows the root");
    let knows = database
        .root
        .edges(gus, Direction::Outgoing, &[], ReadView::latest())
        .unwrap()[0]
        .1;
    assert_eq!(
        database
            .root
            .edges(alix, Direction::Incoming, &[], ReadView::latest())
            .unwrap(),
        vec![(gus, knows)]
    );
    assert_eq!(
        database
            .root
            .degree(
                alix,
                grafeo_core::graph::Direction::Incoming,
                ReadView::latest()
            )
            .unwrap(),
        1
    );
    assert_eq!(
        trips
            .edges(
                travellers.paris,
                Direction::Incoming,
                &[],
                ReadView::latest()
            )
            .unwrap()
            .len(),
        1,
        "Vincent's visit to Paris"
    );

    let open = database.begin(None);
    let refused = open.writer.delete_node(alix, false).unwrap_err();
    assert!(
        matches!(&refused, OperatorError::ConstraintViolation(_, message) if message.contains("1 connected edge")),
        "{refused:?}"
    );
    // The same transaction, which holds the claim on Alix, detaches her.
    assert!(open.writer.delete_node(alix, true).unwrap());
    let epoch = database.commit(open);
    let now = ReadView::committed(epoch);
    assert_eq!(
        database.root.edges(gus, Direction::Both, &[], now).unwrap(),
        Vec::new()
    );
    assert_eq!(
        database
            .root
            .degree(
                gus,
                grafeo_core::graph::Direction::Outgoing,
                ReadView::latest()
            )
            .unwrap(),
        0
    );
    assert!(!database.root.is_node_visible(alix, now).unwrap());
    assert!(!database.root.is_edge_visible(knows, now).unwrap());

    let expected = images(&database);
    let (image, _) = database.image();
    let loaded = Database::load_with(&image, None, &without_backward_edges()).unwrap();
    assert_eq!(images(&loaded), expected);
    assert!(!loaded.root.keeps_backward_edges());
    assert!(!loaded.store(Some("trips")).keeps_backward_edges());
    assert_eq!(loaded.root.memory().incoming_bytes, 0, "root and trips");
    assert_eq!(
        loaded
            .store(Some("trips"))
            .edges(
                travellers.paris,
                Direction::Incoming,
                &[],
                ReadView::latest()
            )
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        loaded
            .root
            .edges(gus, Direction::Both, &[], ReadView::latest())
            .unwrap(),
        Vec::new(),
        "collected at the load"
    );
}

#[cfg(feature = "grafeo-file")]
mod file {
    use grafeo_common::storage::{ChunkNamespace, SectionType};
    use grafeo_common::types::ReadView;
    use grafeo_core::graph::{Direction, GraphStore};

    use super::super::file::FileDatabase;
    use super::super::{assert_holds, images, travels};
    use crate::config::Config;

    /// The adjacency chunks of the LPG section in the database's file:
    /// outgoing, incoming.
    fn adjacency_chunks(database: &FileDatabase) -> (usize, usize) {
        database
            .fm
            .read_image(|image| {
                let source = image.section_source(SectionType::LpgStore).unwrap();
                let count = |namespace: ChunkNamespace| {
                    source
                        .chunks()
                        .iter()
                        .filter(|chunk| chunk.namespace == namespace)
                        .count()
                };
                Ok((
                    count(ChunkNamespace::OutgoingAdjacency),
                    count(ChunkNamespace::IncomingAdjacency),
                ))
            })
            .unwrap()
    }

    /// A database created without backward edges checkpoints no incoming
    /// adjacency, and an open that does not set them reopens it without
    /// them, as its stored configuration says: the root and trips, what it
    /// held, its incoming reads from the edge table, no incoming bytes;
    /// and so after a second close. An open that sets another value for
    /// them is refused (they are fixed at creation), as for a database
    /// created with them.
    #[test]
    fn a_database_without_backward_edges_reopens_without_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("berlin.grafeo");
        let created =
            FileDatabase::create_with(&path, &Config::persistent(&path).without_backward_edges());
        let travellers = travels(&created.database);
        created.checkpoint().unwrap();
        let (outgoing, incoming) = adjacency_chunks(&created);
        assert!(outgoing > 0);
        assert_eq!(incoming, 0, "no incoming adjacency is written");
        let closed = images(&created.database);
        created.close();

        for open in ["the reopen", "the second reopen"] {
            let opened = FileDatabase::open(&path);
            assert_holds(&opened.database, &closed, open);
            let (root, trips) = (&opened.database.root, opened.database.store(Some("trips")));
            assert!(!root.keeps_backward_edges(), "{open}");
            assert!(!trips.keeps_backward_edges(), "{open}");
            assert_eq!(root.memory().incoming_bytes, 0, "{open}");
            assert_eq!(
                root.edges(
                    travellers.alix,
                    Direction::Incoming,
                    &[],
                    ReadView::latest()
                )
                .unwrap()
                .into_iter()
                .map(|(other, _)| other)
                .collect::<Vec<_>>(),
                vec![travellers.gus],
                "{open}: Gus knows Alix"
            );
            assert_eq!(
                trips
                    .degree(
                        travellers.paris,
                        grafeo_core::graph::Direction::Incoming,
                        ReadView::latest()
                    )
                    .unwrap(),
                1,
                "{open}"
            );
            opened.close();
        }

        let error =
            FileDatabase::open_with(&path, Config::persistent(&path).without_backward_edges())
                .map(drop)
                .err();
        assert!(error.is_none(), "the value it was created with: {error:?}");
        let other = tempfile::tempdir().unwrap();
        let kept = other.path().join("paris.grafeo");
        FileDatabase::create(&kept).close();
        let error =
            FileDatabase::open_with(&kept, Config::persistent(&kept).without_backward_edges())
                .err()
                .expect("an open without the backward edges the file was created with is refused");
        assert!(error.to_string().contains("fixed"), "{error}");
    }
}
