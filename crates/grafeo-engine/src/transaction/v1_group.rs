//! The WAL v1 records of a change set: what a transaction's writes logged
//! before they were recorded in a change set (through `WalGraphStore`),
//! built from the set at commit instead. Until the commit writes a WAL v2
//! group from the set, the log stays as it was, record for record.
//!
//! A create is one record and a property record per value it created with,
//! as the store's create and the writes of its values logged them; every
//! other entry is one record. Each record carries the storage key of its
//! graph (`None` for the default graph), for the `SwitchGraph` records the
//! group gets where the graph changes ([`build_group`]). A triple is an RDF
//! record, which names its graph itself.
//!
//! A standalone change (a graph command, a schema statement, an index call)
//! is a group of its own ([`standalone_group`]): a named graph created or
//! dropped is the record it was before, and every other op is a
//! [`WalRecord::Standalone`] holding its WAL v2 log record, which carries
//! the whole catalog record (defaults, parent types, endpoints, every vector
//! index parameter, the graph of an index) that the 0.5 schema records left
//! out.

#[cfg(feature = "lpg")]
use grafeo_common::change::StandaloneOp;
use grafeo_common::change::{ChangeSet, DataModel, DataOp};
#[cfg(feature = "lpg")]
use grafeo_common::storage::LogRecordRef;
#[cfg(feature = "lpg")]
use grafeo_common::types::TransactionId;
#[cfg(feature = "lpg")]
use grafeo_common::utils::error::Result;
use grafeo_storage::wal::WalRecord;

#[cfg(feature = "lpg")]
use super::StandaloneChange;

/// The group of records `change` logs: one record per op, in order, then
/// the commit marker that makes replay apply them all or none.
///
/// # Errors
///
/// The error of a catalog record that does not encode (one past a record's
/// limits): nothing is logged or applied then.
#[cfg(feature = "lpg")]
pub(crate) fn standalone_group(change: &StandaloneChange) -> Result<Vec<WalRecord>> {
    let mut records = Vec::new();
    for op in change.ops() {
        records.push(match op {
            StandaloneOp::CreateGraph { name } => {
                WalRecord::CreateNamedGraph { name: name.clone() }
            }
            StandaloneOp::DropGraph { name } => WalRecord::DropNamedGraph { name: name.clone() },
            StandaloneOp::PutCatalog(_)
            | StandaloneOp::DropCatalog(_)
            | StandaloneOp::RdfGraph(_) => {
                let mut record = Vec::new();
                LogRecordRef::Standalone(op).encode_framed(&mut record)?;
                WalRecord::Standalone { record }
            }
        });
    }
    records.push(WalRecord::TransactionCommit {
        transaction_id: TransactionId::SYSTEM,
    });
    Ok(records)
}

/// A record waiting for its group, with the storage key of the labeled
/// property graph it applies to (`None` for the default graph).
pub(crate) type PendingRecord = (Option<String>, WalRecord);

/// The group of `pending`, closed by `markers`: the records, with
/// `SwitchGraph` wherever the graph changes and a switch back to the default
/// graph before the markers, so replay of every group starts and ends in the
/// default graph. A commit writes its group with one call, so no other
/// session's records land inside it (#411).
pub(crate) fn build_group(pending: Vec<PendingRecord>, markers: &[WalRecord]) -> Vec<WalRecord> {
    let mut group = Vec::with_capacity(pending.len() + markers.len() + 2);
    let mut context: Option<String> = None;
    for (graph, record) in pending {
        if graph != context {
            group.push(WalRecord::SwitchGraph {
                name: graph.clone(),
            });
            context = graph;
        }
        group.push(record);
    }
    if context.is_some() {
        group.push(WalRecord::SwitchGraph { name: None });
    }
    group.extend(markers.iter().cloned());
    group
}

/// The v1 records of `set`'s writes, in the order they were applied (see
/// [`ChangeSet::ops`]: each data entry, and the rows a bulk write kept with
/// its range), each with its graph's storage key. A triple's record names
/// its RDF graph itself and needs no switch, so it carries the default
/// graph's key.
pub(crate) fn v1_records(set: &ChangeSet) -> Vec<PendingRecord> {
    // Sized once: a batch is one entry for many rows.
    let count = set.ops().map(|(_, op, _)| v1_record_count(op)).sum();
    let mut records = Vec::with_capacity(count);
    for (graph, op, _) in set.ops() {
        let key = set
            .graph(graph)
            .and_then(|graph| graph.key.as_ref().map(ToString::to_string));
        // A triple's record names its graph itself: it needs no switch.
        let context = if op.model() == DataModel::Rdf {
            None
        } else {
            key.clone()
        };
        push_v1_records(op, key.as_deref(), |record| {
            records.push((context.clone(), record));
        });
    }
    records
}

/// How many v1 records [`push_v1_records`] pushes for `op`.
fn v1_record_count(op: &DataOp) -> usize {
    match op {
        DataOp::CreateNode { properties, .. } | DataOp::CreateEdge { properties, .. } => {
            1 + properties.len()
        }
        DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {
            usize::from(cfg!(feature = "triple-store"))
        }
        DataOp::DeleteNode { .. }
        | DataOp::DeleteEdge { .. }
        | DataOp::SetNodeProperty { .. }
        | DataOp::RemoveNodeProperty { .. }
        | DataOp::SetEdgeProperty { .. }
        | DataOp::RemoveEdgeProperty { .. }
        | DataOp::AddNodeLabel { .. }
        | DataOp::RemoveNodeLabel { .. } => 1,
    }
}

/// Pushes the v1 records of `op`, a write in the graph with the storage key
/// `graph`, to `push`: a create is one record and a property record per
/// value it created with, as the store's create and the writes of its
/// values logged them; every other op is one record, a triple's naming
/// `graph` as its RDF graph.
pub(crate) fn push_v1_records(op: &DataOp, graph: Option<&str>, mut push: impl FnMut(WalRecord)) {
    #[cfg(not(feature = "triple-store"))]
    let _ = graph;
    match op {
        DataOp::CreateNode {
            id,
            labels,
            properties,
        } => {
            push(WalRecord::CreateNode {
                id: *id,
                labels: labels.iter().map(ToString::to_string).collect(),
            });
            for (property, value) in properties {
                push(WalRecord::SetNodeProperty {
                    id: *id,
                    key: property.as_str().to_string(),
                    value: value.clone(),
                });
            }
        }
        DataOp::DeleteNode { id } => push(WalRecord::DeleteNode { id: *id }),
        DataOp::CreateEdge {
            id,
            src,
            dst,
            edge_type,
            properties,
        } => {
            push(WalRecord::CreateEdge {
                id: *id,
                src: *src,
                dst: *dst,
                edge_type: edge_type.to_string(),
            });
            for (property, value) in properties {
                push(WalRecord::SetEdgeProperty {
                    id: *id,
                    key: property.as_str().to_string(),
                    value: value.clone(),
                });
            }
        }
        DataOp::DeleteEdge { id } => push(WalRecord::DeleteEdge { id: *id }),
        DataOp::SetNodeProperty { id, key, value } => push(WalRecord::SetNodeProperty {
            id: *id,
            key: key.as_str().to_string(),
            value: value.clone(),
        }),
        DataOp::RemoveNodeProperty { id, key } => push(WalRecord::RemoveNodeProperty {
            id: *id,
            key: key.as_str().to_string(),
        }),
        DataOp::SetEdgeProperty { id, key, value } => push(WalRecord::SetEdgeProperty {
            id: *id,
            key: key.as_str().to_string(),
            value: value.clone(),
        }),
        DataOp::RemoveEdgeProperty { id, key } => push(WalRecord::RemoveEdgeProperty {
            id: *id,
            key: key.as_str().to_string(),
        }),
        DataOp::AddNodeLabel { id, label } => push(WalRecord::AddNodeLabel {
            id: *id,
            label: label.to_string(),
        }),
        DataOp::RemoveNodeLabel { id, label } => push(WalRecord::RemoveNodeLabel {
            id: *id,
            label: label.to_string(),
        }),
        #[cfg(feature = "triple-store")]
        DataOp::InsertTriple { triple } => {
            let (subject, predicate, object) = super::ntriples_terms(triple);
            push(WalRecord::InsertRdfTriple {
                subject,
                predicate,
                object,
                graph: graph.map(ToString::to_string),
            });
        }
        #[cfg(feature = "triple-store")]
        DataOp::DeleteTriple { triple } => {
            let (subject, predicate, object) = super::ntriples_terms(triple);
            push(WalRecord::DeleteRdfTriple {
                subject,
                predicate,
                object,
                graph: graph.map(ToString::to_string),
            });
        }
        // A build without the triple store records no triple.
        #[cfg(not(feature = "triple-store"))]
        DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::change::{
        Before, ChangeSet, DataModel, DataOp, GraphRef, NodeImage, PendingVersion,
    };
    use grafeo_common::types::{ArcStr, EdgeId, NodeId, PropertyKey, TransactionId, Value};
    use grafeo_storage::wal::WalRecord;

    use super::{build_group, v1_records};

    /// Every kind of entry maps to the records its write logged before, a
    /// create to one record per value besides its own, each with its graph.
    #[test]
    fn every_entry_maps_to_the_records_its_write_logged() {
        let mut set = ChangeSet::new();
        let default = set
            .slot(GraphRef {
                model: DataModel::Lpg,
                key: None,
            })
            .unwrap();
        let trips = set
            .slot(GraphRef {
                model: DataModel::Lpg,
                key: Some(ArcStr::from("trips")),
            })
            .unwrap();
        let alix = NodeId::new(3);
        let gus = NodeId::new(19);
        let knows = EdgeId::new(88);
        let name = PropertyKey::new("name");
        let created = |op: DataOp| (op, Before::Absent);
        let entries = [
            (
                default,
                created(DataOp::CreateNode {
                    id: alix,
                    labels: [ArcStr::from("Person")].into_iter().collect(),
                    properties: vec![(name.clone(), Value::from("Alix"))],
                }),
            ),
            (
                trips,
                created(DataOp::CreateEdge {
                    id: knows,
                    src: alix,
                    dst: gus,
                    edge_type: ArcStr::from("KNOWS"),
                    properties: vec![(PropertyKey::new("since"), Value::Int64(3))],
                }),
            ),
            (
                default,
                (
                    DataOp::RemoveNodeLabel {
                        id: alix,
                        label: ArcStr::from("Person"),
                    },
                    Before::Labels([ArcStr::from("Person")].into_iter().collect()),
                ),
            ),
            (
                default,
                (
                    DataOp::DeleteNode { id: gus },
                    Before::Node(Box::new(NodeImage {
                        labels: Default::default(),
                        properties: Vec::new(),
                    })),
                ),
            ),
        ];
        for (graph, (op, before)) in entries {
            set.push(graph, op, before, PendingVersion::Created)
                .unwrap();
        }

        let trips = Some("trips".to_string());
        // A record has no equality: compare how they print.
        let printed = |records: &[(Option<String>, WalRecord)]| -> Vec<String> {
            records.iter().map(|record| format!("{record:?}")).collect()
        };
        assert_eq!(
            printed(&v1_records(&set)),
            printed(&[
                (
                    None,
                    WalRecord::CreateNode {
                        id: alix,
                        labels: vec!["Person".to_string()],
                    }
                ),
                (
                    None,
                    WalRecord::SetNodeProperty {
                        id: alix,
                        key: "name".to_string(),
                        value: Value::from("Alix"),
                    }
                ),
                (
                    trips.clone(),
                    WalRecord::CreateEdge {
                        id: knows,
                        src: alix,
                        dst: gus,
                        edge_type: "KNOWS".to_string(),
                    }
                ),
                (
                    trips,
                    WalRecord::SetEdgeProperty {
                        id: knows,
                        key: "since".to_string(),
                        value: Value::Int64(3),
                    }
                ),
                (
                    None,
                    WalRecord::RemoveNodeLabel {
                        id: alix,
                        label: "Person".to_string(),
                    }
                ),
                (None, WalRecord::DeleteNode { id: gus }),
            ])
        );
    }

    /// A graph command logs the record it logged before; every other op
    /// logs its whole WAL v2 log record, which decodes to the op itself (a
    /// default value and an index's graph included); the group ends with
    /// the commit marker, so replay applies all of it or none.
    #[cfg(feature = "lpg")]
    #[test]
    fn a_standalone_change_is_one_record_per_op_then_a_commit() {
        use grafeo_common::change::StandaloneOp;
        use grafeo_common::storage::catalog_record::{
            CatalogKey, CatalogRecord, IndexKeyRecord, IndexKindRecord, IndexRecord,
            NodeTypeRecord, PropertyRecord, PropertyTypeRecord,
        };
        use grafeo_common::storage::{LogRecord, read_log_records};

        use super::super::StandaloneChange;
        use super::standalone_group;

        let city = StandaloneOp::PutCatalog(CatalogRecord::NodeType(NodeTypeRecord {
            name: "City".to_string(),
            properties: vec![PropertyRecord {
                name: "country".to_string(),
                data_type: PropertyTypeRecord::String,
                nullable: true,
                default_value: Some(Value::from("NL")),
            }],
            constraints: Vec::new(),
            parent_types: Vec::new(),
            key_labels: Vec::new(),
        }));
        let index = StandaloneOp::PutCatalog(CatalogRecord::Index(IndexRecord {
            graph: Some("trips".to_string()),
            index: IndexKindRecord::Property {
                key: "name".to_string(),
            },
        }));
        let dropped = StandaloneOp::DropCatalog(CatalogKey::Index {
            graph: None,
            index: IndexKeyRecord::Text {
                label: "Doc".to_string(),
                property: "body".to_string(),
            },
        });
        let mut change = StandaloneChange::new();
        change.push(StandaloneOp::CreateGraph {
            name: "trips".to_string(),
        });
        for op in [&city, &index, &dropped] {
            change.push(op.clone());
        }
        change.push(StandaloneOp::DropGraph {
            name: "archive".to_string(),
        });

        let group = standalone_group(&change).unwrap();
        assert_eq!(group.len(), 6, "{group:?}");
        assert!(matches!(&group[0], WalRecord::CreateNamedGraph { name } if name == "trips"));
        assert!(matches!(&group[4], WalRecord::DropNamedGraph { name } if name == "archive"));
        assert!(matches!(
            group[5],
            WalRecord::TransactionCommit { transaction_id } if transaction_id == TransactionId::SYSTEM
        ));
        for (record, op) in group[1..4].iter().zip([city, index, dropped]) {
            let WalRecord::Standalone { record } = record else {
                panic!("{record:?} holds no log record");
            };
            let mut decoded = Vec::new();
            read_log_records(record, &mut |record| {
                decoded.push(record);
                Ok(())
            })
            .unwrap();
            assert_eq!(decoded, [LogRecord::Standalone(op)]);
        }
    }

    fn create(id: u64) -> WalRecord {
        WalRecord::CreateNode {
            id: NodeId::new(id),
            labels: vec!["N".to_string()],
        }
    }

    fn commit() -> WalRecord {
        WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(7),
        }
    }

    /// Short form of a group for assertions.
    fn shape(group: &[WalRecord]) -> Vec<String> {
        group
            .iter()
            .map(|record| match record {
                WalRecord::CreateNode { id, .. } => format!("node {}", id.as_u64()),
                WalRecord::SwitchGraph { name } => format!("switch {name:?}"),
                WalRecord::TransactionCommit { .. } => "commit".to_string(),
                WalRecord::EpochAdvance { epoch } => format!("epoch {}", epoch.as_u64()),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn default_graph_group_has_no_switches() {
        let group = build_group(vec![(None, create(1)), (None, create(2))], &[commit()]);
        assert_eq!(shape(&group), ["node 1", "node 2", "commit"]);
    }

    #[test]
    fn group_switches_graphs_and_returns_to_default() {
        let pending = vec![
            (Some("g".to_string()), create(1)),
            (Some("g".to_string()), create(2)),
            (None, create(3)),
            (Some("h".to_string()), create(4)),
        ];
        let markers = [
            commit(),
            WalRecord::EpochAdvance {
                epoch: grafeo_common::types::EpochId::new(5),
            },
        ];
        assert_eq!(
            shape(&build_group(pending, &markers)),
            [
                "switch Some(\"g\")",
                "node 1",
                "node 2",
                "switch None",
                "node 3",
                "switch Some(\"h\")",
                "node 4",
                "switch None",
                "commit",
                "epoch 5",
            ]
        );
    }

    #[test]
    fn empty_group_is_only_markers() {
        assert_eq!(shape(&build_group(Vec::new(), &[commit()])), ["commit"]);
        assert!(build_group(Vec::new(), &[]).is_empty());
    }

    /// A triple is the RDF record of its op, which names its graph and
    /// holds its terms as N-Triples strings, and needs no switch.
    #[cfg(feature = "triple-store")]
    #[test]
    fn a_triple_is_an_rdf_record_naming_its_graph() {
        use grafeo_common::storage::log_record::TripleRecord;
        use grafeo_core::graph::rdf::{Term, Triple};

        let mut set = ChangeSet::new();
        let paris = set
            .slot(GraphRef {
                model: DataModel::Rdf,
                key: Some(ArcStr::from("http://example.org/paris")),
            })
            .unwrap();
        let default = set
            .slot(GraphRef {
                model: DataModel::Rdf,
                key: None,
            })
            .unwrap();
        let triple = |object: Term| {
            Box::new(TripleRecord::from(&Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://example.org/name"),
                object,
            )))
        };
        for (graph, op) in [
            (
                paris,
                DataOp::InsertTriple {
                    triple: triple(Term::lang_literal("Alix", "nl")),
                },
            ),
            (
                default,
                DataOp::DeleteTriple {
                    triple: triple(Term::literal("Gus \"de bus\"")),
                },
            ),
        ] {
            set.push(graph, op, Before::Absent, PendingVersion::Created)
                .unwrap();
        }
        let records = v1_records(&set);
        assert_eq!(records.len(), 2, "{records:?}");
        assert!(records.iter().all(|(context, _)| context.is_none()));
        let WalRecord::InsertRdfTriple {
            subject,
            predicate,
            object,
            graph,
        } = &records[0].1
        else {
            panic!("{:?} is no insert", records[0].1);
        };
        assert_eq!(subject, "<http://example.org/alix>");
        assert_eq!(predicate, "<http://example.org/name>");
        assert_eq!(object, "\"Alix\"@nl");
        assert_eq!(graph.as_deref(), Some("http://example.org/paris"));
        let WalRecord::DeleteRdfTriple { object, graph, .. } = &records[1].1 else {
            panic!("{:?} is no delete", records[1].1);
        };
        assert_eq!(object, "\"Gus \\\"de bus\\\"\"");
        assert_eq!(*graph, None);
    }
}
