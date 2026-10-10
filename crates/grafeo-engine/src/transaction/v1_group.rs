//! The WAL v1 records of a change set: what a transaction's writes logged
//! before they were recorded in a change set (through `WalGraphStore`),
//! built from the set at commit instead. Until the commit writes a WAL v2
//! group from the set, the log stays as it was, record for record.
//!
//! A create is one record and a property record per value it created with,
//! as the store's create and the writes of its values logged them; every
//! other entry is one record. Each record carries the storage key of its
//! graph (`None` for the default graph), for the `SwitchGraph` records the
//! group gets where the graph changes.

use grafeo_common::change::{Change, ChangeSet, DataOp};
use grafeo_storage::wal::WalRecord;

/// The v1 records of `set`'s entries, in recorded order, each with its
/// graph's storage key. Bulk ranges and triples have none: a bulk write
/// and the RDF store log their own.
pub(crate) fn v1_records(set: &ChangeSet) -> Vec<(Option<String>, WalRecord)> {
    let mut records = Vec::with_capacity(set.len());
    for change in set.entries() {
        let Change::Data { graph, op, .. } = change else {
            continue;
        };
        let key = set
            .graph(*graph)
            .and_then(|graph| graph.key.as_ref().map(ToString::to_string));
        let mut push = |record: WalRecord| records.push((key.clone(), record));
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
            // The RDF store logs its own triples until W3.6 records them.
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {}
        }
    }
    records
}

#[cfg(test)]
mod tests {
    use grafeo_common::change::{
        Before, ChangeSet, DataModel, DataOp, GraphRef, NodeImage, PendingVersion,
    };
    use grafeo_common::types::{ArcStr, EdgeId, NodeId, PropertyKey, Value};
    use grafeo_storage::wal::WalRecord;

    use super::v1_records;

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
}
