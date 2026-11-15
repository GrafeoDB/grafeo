//! The key a MERGE claims before it creates.
//!
//! Two transactions that MERGE one pattern each read a snapshot without the
//! other's element, so both would create it. A MERGE that decides to create
//! claims the pattern's key first, in the transaction (through
//! [`ConstraintValidator::claim_pattern`](super::ConstraintValidator::claim_pattern)),
//! and of two transactions that claim one key the second fails with a
//! retryable write conflict: its retry sees the element and matches it.
//!
//! The key is canonical: two MERGEs claim one key exactly when each would
//! match what the other creates, as a second run of one MERGE matches the
//! first's. In path order it holds, for each element:
//!
//! - a node the MERGE creates: its labels, sorted and once each, and its
//!   match properties, sorted by name, each by the identity a UNIQUE
//!   constraint compares (`1` and `1.0` are one value, a string and a number
//!   two, zoned values by their instant); a null entry is left out, as it
//!   matches an absent property and the created node has none;
//! - a node the row holds: its id;
//! - a relationship: its type, its match properties as for a node, and its
//!   direction.
//!
//! A path and its reverse are one key (the smaller fingerprint of the two
//! orientations), and a pattern with an undirected relationship claims two:
//! the pattern as written, and as created, left to right. A pattern with a
//! value that is not equal to itself (NaN, a list that holds a null) matches
//! nothing, so even a serial second run creates again: it claims nothing.

use std::fmt;
use std::hash::{Hash, Hasher};

use grafeo_common::types::{NodeId, TemporalComparison, Value};
use grafeo_common::utils::hash::hash_one;
use smallvec::SmallVec;

use super::HashableValue;

/// The direction of a relationship in a MERGE pattern, read left to right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClaimDirection {
    /// `-[]->`.
    Out,
    /// `<-[]-`.
    In,
    /// `-[]-`, matched either way and created left to right.
    Either,
}

impl ClaimDirection {
    /// The direction read right to left.
    fn reversed(self) -> Self {
        match self {
            Self::Out => Self::In,
            Self::In => Self::Out,
            Self::Either => Self::Either,
        }
    }
}

/// One element of a pattern's key, borrowed from the pattern as written: the
/// key is read in its canonical form (labels sorted and once each, properties
/// by name without the null entries) when it is hashed or shown, so building
/// one for every row that creates copies nothing.
#[derive(Debug, Clone, Copy)]
enum Element<'a> {
    /// A node the MERGE creates.
    Node {
        labels: &'a [String],
        properties: &'a [(String, Value)],
    },
    /// A node the row holds.
    Bound(NodeId),
    /// A relationship.
    Relationship {
        edge_type: &'a str,
        properties: &'a [(String, Value)],
        direction: ClaimDirection,
    },
}

/// The canonical key of a MERGE pattern (see the module docs), built with
/// [`PatternKey::builder`] or, for a node alone, [`PatternKey::node`]. It
/// borrows the pattern's labels, types and values.
#[derive(Debug, Clone)]
pub struct PatternKey<'a> {
    elements: SmallVec<[Element<'a>; 3]>,
}

/// Builds a [`PatternKey`] element by element, in path order.
#[derive(Debug, Default)]
pub struct PatternKeyBuilder<'a> {
    elements: SmallVec<[Element<'a>; 3]>,
    /// False once a value is not equal to itself: the pattern matches
    /// nothing and claims nothing.
    claimable: bool,
}

/// Whether `value` is equal to itself under `=`, as a MERGE matches: the
/// usual kinds always are, and only a float, a decimal or a value that holds
/// others (a list with a null, say) is checked.
fn equal_to_itself(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::Int64(_) | Value::String(_) | Value::Bytes(_) => true,
        other => {
            super::value_utils::equals(other, other, TemporalComparison::TypesApart) == Some(true)
        }
    }
}

/// Whether every match value of `properties` but the nulls is equal to
/// itself: a pattern with one that is not matches nothing.
fn claimable(properties: &[(String, Value)]) -> bool {
    properties
        .iter()
        .all(|(_, value)| value.is_null() || equal_to_itself(value))
}

/// `labels` sorted and once each, on the stack for the usual few.
fn canonical_labels(labels: &[String]) -> SmallVec<[&str; 4]> {
    let mut sorted: SmallVec<[&str; 4]> = labels.iter().map(String::as_str).collect();
    if sorted.len() > 1 {
        sorted.sort_unstable();
        sorted.dedup();
    }
    sorted
}

/// The match entries of `properties` a key reads: without the nulls (a null
/// entry matches an absent property, and the created element has none), by
/// name, on the stack for the usual few.
fn canonical_properties(properties: &[(String, Value)]) -> SmallVec<[(&str, &Value); 4]> {
    let mut kept: SmallVec<[(&str, &Value); 4]> = properties
        .iter()
        .filter(|(_, value)| !value.is_null())
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    if kept.len() > 1 {
        kept.sort_unstable_by_key(|(name, _)| *name);
    }
    kept
}

/// Hashes `labels` in their canonical form; one label, the usual case, needs
/// no sort.
fn hash_labels<H: Hasher>(labels: &[String], state: &mut H) {
    if let [label] = labels {
        state.write_usize(1);
        label.as_str().hash(state);
        return;
    }
    let sorted = canonical_labels(labels);
    state.write_usize(sorted.len());
    for label in sorted {
        label.hash(state);
    }
}

/// Hashes one match entry, its value by its identity (see
/// [`HashableValue`]), so values `=` finds equal hash alike.
fn hash_entry<H: Hasher>(name: &str, value: &Value, state: &mut H) {
    name.hash(state);
    HashableValue::from(value).hash(state);
}

/// Hashes `properties` in their canonical form; one entry, the usual case,
/// needs no sort.
fn hash_properties<H: Hasher>(properties: &[(String, Value)], state: &mut H) {
    match properties {
        [] => state.write_usize(0),
        [(_, value)] if value.is_null() => state.write_usize(0),
        [(name, value)] => {
            state.write_usize(1);
            hash_entry(name, value, state);
        }
        _ => {
            let kept = canonical_properties(properties);
            state.write_usize(kept.len());
            for (name, value) in kept {
                hash_entry(name, value, state);
            }
        }
    }
}

impl<'a> PatternKeyBuilder<'a> {
    /// Adds a node the MERGE creates, with its labels and match properties.
    pub fn node(&mut self, labels: &'a [String], properties: &'a [(String, Value)]) -> &mut Self {
        self.claimable &= claimable(properties);
        self.elements.push(Element::Node { labels, properties });
        self
    }

    /// Adds a node the row holds.
    pub fn bound_node(&mut self, id: NodeId) -> &mut Self {
        self.elements.push(Element::Bound(id));
        self
    }

    /// Adds a relationship of `edge_type` with its match properties, in
    /// `direction` read left to right.
    pub fn relationship(
        &mut self,
        edge_type: &'a str,
        properties: &'a [(String, Value)],
        direction: ClaimDirection,
    ) -> &mut Self {
        self.claimable &= claimable(properties);
        self.elements.push(Element::Relationship {
            edge_type,
            properties,
            direction,
        });
        self
    }

    /// The key, `None` when a value of the pattern is not equal to itself.
    #[must_use]
    pub fn finish(self) -> Option<PatternKey<'a>> {
        self.claimable.then_some(PatternKey {
            elements: self.elements,
        })
    }
}

/// A key's elements as one orientation hashes them.
struct Oriented<'s, 'a> {
    graph: Option<&'s str>,
    elements: &'s [Element<'a>],
    /// Read right to left.
    reversed: bool,
    /// An undirected relationship read as created, left to right.
    as_created: bool,
}

impl Hash for Oriented<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.graph.hash(state);
        state.write_usize(self.elements.len());
        let mut write = |element: &Element<'_>| match *element {
            Element::Node { labels, properties } => {
                state.write_u8(0);
                hash_labels(labels, state);
                hash_properties(properties, state);
            }
            Element::Bound(id) => {
                state.write_u8(1);
                id.hash(state);
            }
            Element::Relationship {
                edge_type,
                properties,
                direction,
            } => {
                let mut direction = direction;
                if self.as_created && direction == ClaimDirection::Either {
                    direction = ClaimDirection::Out;
                }
                if self.reversed {
                    direction = direction.reversed();
                }
                state.write_u8(2);
                edge_type.hash(state);
                hash_properties(properties, state);
                direction.hash(state);
            }
        };
        if self.reversed {
            self.elements.iter().rev().for_each(&mut write);
        } else {
            self.elements.iter().for_each(&mut write);
        }
    }
}

impl<'a> PatternKey<'a> {
    /// A builder of a key, element by element in path order.
    #[must_use]
    pub fn builder() -> PatternKeyBuilder<'a> {
        PatternKeyBuilder {
            elements: SmallVec::new(),
            claimable: true,
        }
    }

    /// The key of a MERGE of one node with `labels` and the match
    /// properties `properties`; `None` when a value is not equal to itself.
    #[must_use]
    pub fn node(labels: &'a [String], properties: &'a [(String, Value)]) -> Option<Self> {
        claimable(properties).then(|| Self {
            elements: smallvec::smallvec![Element::Node { labels, properties }],
        })
    }

    /// The fingerprints the pattern claims in the graph with storage key
    /// `graph` (`None`: the default graph): one, and a second for the
    /// pattern as created when it has an undirected relationship. Each is
    /// the smaller hash of the path and its reverse, from one process-wide
    /// seed, so it is the same in every transaction of the process (a node
    /// alone reads the same both ways, and is hashed once).
    #[must_use]
    pub fn fingerprints(&self, graph: Option<&str>) -> SmallVec<[u64; 2]> {
        let fingerprint = |as_created: bool| {
            let oriented = |reversed: bool| {
                hash_one(&Oriented {
                    graph,
                    elements: &self.elements,
                    reversed,
                    as_created,
                })
            };
            if self.elements.len() > 1 {
                oriented(false).min(oriented(true))
            } else {
                oriented(false)
            }
        };
        let mut fingerprints = SmallVec::new();
        fingerprints.push(fingerprint(false));
        let undirected = self.elements.iter().any(|element| {
            matches!(
                element,
                Element::Relationship {
                    direction: ClaimDirection::Either,
                    ..
                }
            )
        });
        if undirected {
            fingerprints.push(fingerprint(true));
        }
        fingerprints
    }
}

/// Where an element of a MERGE pattern is, in path order, for the
/// [`CreateOperator`](super::CreateOperator) of the MERGE's create branch,
/// which claims the pattern before it creates (see
/// [`CreateOperator::with_claim`](super::CreateOperator::with_claim)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimSource {
    /// A node the row holds, in this input column.
    Bound(usize),
    /// The node this step creates, with its labels and properties.
    Node(usize),
    /// The edge this step creates, read in `direction` along the path.
    Relationship {
        /// The step.
        step: usize,
        /// Its direction, left to right.
        direction: ClaimDirection,
    },
}

/// Writes ` {name: value, ...}` for the canonical entries of `properties`,
/// when there are any.
fn write_properties(f: &mut fmt::Formatter<'_>, properties: &[(String, Value)]) -> fmt::Result {
    let kept = canonical_properties(properties);
    if kept.is_empty() {
        return Ok(());
    }
    write!(f, " {{")?;
    for (index, (name, value)) in kept.iter().enumerate() {
        if index > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{name}: {value:?}")?;
    }
    write!(f, "}}")
}

impl fmt::Display for PatternKey<'_> {
    /// The pattern as its key holds it, for an error message:
    /// `(:A {k: Int64(1)})-[:T]->(#3)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for element in &self.elements {
            match *element {
                Element::Node { labels, properties } => {
                    write!(f, "(")?;
                    for label in canonical_labels(labels) {
                        write!(f, ":{label}")?;
                    }
                    write_properties(f, properties)?;
                    write!(f, ")")?;
                }
                Element::Bound(id) => write!(f, "(#{})", id.as_u64())?,
                Element::Relationship {
                    edge_type,
                    properties,
                    direction,
                } => {
                    let (left, right) = match direction {
                        ClaimDirection::Out => ("-", "->"),
                        ClaimDirection::In => ("<-", "-"),
                        ClaimDirection::Either => ("-", "-"),
                    };
                    write!(f, "{left}[:{edge_type}")?;
                    write_properties(f, properties)?;
                    write!(f, "]{right}")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(entries: &[(&str, Value)]) -> Vec<(String, Value)> {
        entries
            .iter()
            .map(|(name, value)| ((*name).to_string(), value.clone()))
            .collect()
    }

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    fn node_key(names: &[&str], entries: &[(&str, Value)]) -> SmallVec<[u64; 2]> {
        PatternKey::node(&labels(names), &props(entries))
            .expect("claimable")
            .fingerprints(None)
    }

    #[test]
    fn labels_and_properties_are_read_as_sets_and_numbers_by_value() {
        assert_eq!(
            node_key(&["A", "B"], &[("k", Value::Int64(1))]),
            node_key(&["B", "A", "A"], &[("k", Value::Float64(1.0))])
        );
        assert_eq!(
            node_key(&["A"], &[("a", Value::Int64(1)), ("b", Value::Int64(2))]),
            node_key(&["A"], &[("b", Value::Int64(2)), ("a", Value::Int64(1))])
        );
        assert_eq!(
            node_key(&["A"], &[("k", Value::Null)]),
            node_key(&["A"], &[])
        );
        assert_ne!(node_key(&["A"], &[]), node_key(&["A", "B"], &[]));
        assert_ne!(
            node_key(&["A"], &[("k", Value::Int64(1))]),
            node_key(&["A"], &[("k", Value::from("1"))])
        );
    }

    #[test]
    fn a_value_not_equal_to_itself_claims_nothing() {
        assert!(
            PatternKey::node(&labels(&["A"]), &props(&[("k", Value::Float64(f64::NAN))])).is_none()
        );
        let list = Value::List(vec![Value::Int64(1), Value::Null].into());
        assert!(PatternKey::node(&labels(&["A"]), &props(&[("k", list)])).is_none());
    }

    #[test]
    fn a_path_and_its_reverse_are_one_key_and_graphs_apart() {
        let path = |first: u64, direction: ClaimDirection, second: u64| {
            let mut builder = PatternKey::builder();
            builder
                .bound_node(NodeId::new(first))
                .relationship("T", &[], direction)
                .bound_node(NodeId::new(second));
            builder.finish().expect("claimable")
        };
        let forward = path(1, ClaimDirection::Out, 2);
        let reverse = path(2, ClaimDirection::In, 1);
        assert_eq!(forward.fingerprints(None), reverse.fingerprints(None));
        assert_ne!(
            forward.fingerprints(None),
            path(2, ClaimDirection::Out, 1).fingerprints(None)
        );
        assert_ne!(forward.fingerprints(None), forward.fingerprints(Some("g")));

        // Undirected: one key for both orders, and the key of the direction
        // each creates, left to right.
        let undirected = path(1, ClaimDirection::Either, 2).fingerprints(None);
        let opposite = path(2, ClaimDirection::Either, 1).fingerprints(None);
        assert_eq!(undirected.len(), 2);
        assert_eq!(undirected[0], opposite[0]);
        assert_eq!(undirected[1], forward.fingerprints(None)[0]);
        assert_eq!(
            opposite[1],
            path(2, ClaimDirection::Out, 1).fingerprints(None)[0]
        );
    }

    #[test]
    fn the_key_reads_as_its_pattern() {
        let names = labels(&["B", "A"]);
        let entries = props(&[("k", Value::Int64(1))]);
        let mut builder = PatternKey::builder();
        builder
            .node(&names, &entries)
            .relationship("T", &[], ClaimDirection::Out)
            .bound_node(NodeId::new(3));
        let key = builder.finish().expect("claimable");
        assert_eq!(key.to_string(), "(:A:B {k: Int64(1)})-[:T]->(#3)");
    }
}
