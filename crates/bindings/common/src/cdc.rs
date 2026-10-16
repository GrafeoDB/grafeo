//! Change data capture events as JSON, the one shape every binding returns.

use grafeo_engine::cdc::{ChangeEvent, ChangeKind};

use crate::json::value_to_json;

/// A change event as a JSON object: `entity_id`, `entity_type` (`"node"` or
/// `"edge"`), `graph`, `kind` (`"create"`, `"update"` or `"delete"`),
/// `epoch`, `timestamp`, `before` and `after` (property objects or null),
/// `labels`, `before_labels`, `edge_type`, `src_id`, `dst_id` and
/// `deleted_with` (the node a detach delete removed the edge with).
#[must_use]
pub fn event_to_json(event: &ChangeEvent) -> serde_json::Value {
    let entity_type = if event.entity_id.is_node() {
        "node"
    } else {
        "edge"
    };
    let kind = match event.kind {
        ChangeKind::Create => "create",
        ChangeKind::Update => "update",
        ChangeKind::Delete => "delete",
        _ => "unknown",
    };
    let properties =
        |props: &Option<std::collections::HashMap<String, grafeo_common::types::Value>>| {
            props.as_ref().map_or(serde_json::Value::Null, |props| {
                serde_json::Value::Object(
                    props
                        .iter()
                        .map(|(key, value)| (key.clone(), value_to_json(value)))
                        .collect(),
                )
            })
        };
    serde_json::json!({
        "entity_id": event.entity_id.as_u64(),
        "entity_type": entity_type,
        "graph": event.graph,
        "kind": kind,
        "epoch": event.epoch.0,
        "timestamp": event.timestamp,
        "before": properties(&event.before),
        "after": properties(&event.after),
        "labels": event.labels,
        "before_labels": event.before_labels,
        "edge_type": event.edge_type,
        "src_id": event.src_id,
        "dst_id": event.dst_id,
        "deleted_with": event.deleted_with,
    })
}

/// [`event_to_json`] of every event, as a JSON array.
#[must_use]
pub fn events_to_json(events: &[ChangeEvent]) -> serde_json::Value {
    serde_json::Value::Array(events.iter().map(event_to_json).collect())
}
