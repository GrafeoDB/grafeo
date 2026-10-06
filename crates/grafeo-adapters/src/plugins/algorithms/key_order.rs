//! Nodes in the order of a property value (#566 `key=`).
//!
//! PageRank, Louvain and label propagation follow the order of the node list
//! they get in everything order can change: the visit order, the order of
//! every floating-point sum, which candidate wins a tie, and how communities
//! are numbered. Given the nodes in the order of a property every node holds
//! once (an id from outside the database), their results no longer depend on
//! the order the nodes and edges were inserted in, which decides the internal
//! node ids.

use std::fmt;

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::execution::operators::value_utils::compare_values_total;
use grafeo_core::graph::GraphStore;

/// Why the nodes cannot be put in the order of a key.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyOrderError {
    /// A node has no value for the key (or a null).
    Missing {
        /// The node.
        node: NodeId,
        /// The key.
        key: String,
    },
    /// Two nodes have values for the key that compare equal (`1` and `1.0`
    /// too), so neither can go first.
    Repeated {
        /// The node with the smaller id.
        first: NodeId,
        /// The other node.
        second: NodeId,
        /// The key.
        key: String,
        /// The value both have.
        value: Value,
    },
}

impl fmt::Display for KeyOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { node, key } => {
                write!(f, "node {} has no value for the key '{key}'", node.as_u64())
            }
            Self::Repeated {
                first,
                second,
                key,
                value,
            } => write!(
                f,
                "nodes {} and {} have the same value for the key '{key}': {value}",
                first.as_u64(),
                second.as_u64()
            ),
        }
    }
}

impl std::error::Error for KeyOrderError {}

/// The nodes of `store` in the order of their `key` value, each with that
/// value. Values compare in the total order `ORDER BY` uses, so values of
/// different types are allowed (they order by type).
///
/// # Errors
///
/// Returns [`KeyOrderError::Missing`] for the first node (by id) without a
/// value, and [`KeyOrderError::Repeated`] for two nodes whose values compare
/// equal.
///
/// # Complexity
///
/// O(V log V) value comparisons, after one batch read of the key.
pub fn order_by_key(
    store: &dyn GraphStore,
    key: &str,
) -> Result<Vec<(NodeId, Value)>, KeyOrderError> {
    let mut nodes = store.node_ids();
    nodes.sort_unstable();
    let values = store.get_node_property_batch(&nodes, &PropertyKey::new(key));
    let mut keyed = Vec::with_capacity(nodes.len());
    for (node, value) in nodes.into_iter().zip(values) {
        match value {
            Some(value) if !matches!(value, Value::Null) => keyed.push((node, value)),
            _ => {
                return Err(KeyOrderError::Missing {
                    node,
                    key: key.to_string(),
                });
            }
        }
    }
    // Stable: of two equal values, the smaller id comes first (and is named
    // first in the error).
    keyed.sort_by(|a, b| compare_values_total(&a.1, &b.1));
    if let Some(pair) = keyed
        .windows(2)
        .find(|pair| compare_values_total(&pair[0].1, &pair[1].1).is_eq())
    {
        return Err(KeyOrderError::Repeated {
            first: pair[0].0,
            second: pair[1].0,
            key: key.to_string(),
            value: pair[0].1.clone(),
        });
    }
    Ok(keyed)
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use grafeo_core::graph::lpg::LpgStore;

    fn store_with(keys: &[Option<Value>]) -> (LpgStore, Vec<NodeId>) {
        let store = LpgStore::new().unwrap();
        let ids = keys
            .iter()
            .map(|key| match key {
                Some(value) => store.create_node_with_props(&["Item"], [("key", value.clone())]),
                None => store.create_node(&["Item"]),
            })
            .collect();
        (store, ids)
    }

    #[test]
    fn nodes_come_in_the_order_of_their_key() {
        let (store, ids) = store_with(&[
            Some(Value::from("Gus")),
            Some(Value::from("Alix")),
            Some(Value::from("Vincent")),
        ]);
        let order: Vec<NodeId> = order_by_key(&store, "key")
            .unwrap()
            .into_iter()
            .map(|(node, _)| node)
            .collect();
        assert_eq!(order, vec![ids[1], ids[0], ids[2]]);
    }

    /// Values of different types order by type, as `ORDER BY` orders them.
    #[test]
    fn mixed_types_order_as_order_by_does() {
        let (store, ids) = store_with(&[Some(Value::from("Alix")), Some(Value::Int64(3))]);
        let order: Vec<NodeId> = order_by_key(&store, "key")
            .unwrap()
            .into_iter()
            .map(|(node, _)| node)
            .collect();
        assert_eq!(order, vec![ids[0], ids[1]], "strings before numbers");
    }

    #[test]
    fn a_node_without_the_key_is_an_error() {
        for missing in [None, Some(Value::Null)] {
            let (store, ids) = store_with(&[Some(Value::from("Alix")), missing]);
            assert_eq!(
                order_by_key(&store, "key"),
                Err(KeyOrderError::Missing {
                    node: ids[1],
                    key: "key".to_string()
                })
            );
        }
    }

    /// Two values that compare equal are a repeated key, also across number
    /// types.
    #[test]
    fn a_repeated_key_is_an_error() {
        let (store, ids) = store_with(&[
            Some(Value::Int64(3)),
            Some(Value::Int64(19)),
            Some(Value::Float64(3.0)),
        ]);
        let error = order_by_key(&store, "key").unwrap_err();
        assert_eq!(
            error,
            KeyOrderError::Repeated {
                first: ids[0],
                second: ids[2],
                key: "key".to_string(),
                value: Value::Int64(3),
            }
        );
        assert_eq!(
            error.to_string(),
            format!(
                "nodes {} and {} have the same value for the key 'key': 3",
                ids[0].as_u64(),
                ids[2].as_u64()
            )
        );
    }
}
