//! Conversions between the in-memory catalog and the catalog records of
//! [`grafeo_common::storage::catalog_record`].
//!
//! The catalog section (version 2) holds the catalog as these records, and
//! the WAL is to log the same records: every conversion between an engine
//! definition and its record lives here, so both use the same one.
//!
//! A definition comes back from its record as it was, with two exceptions
//! that change nothing a query sees:
//!
//! - An edge type's lists of source and target node types are stored as
//!   single-type (source, target) pairs, their cross product
//!   ([`endpoint_pairs`]). A type listed twice is one endpoint: it comes
//!   back once.
//! - A vector index whose quantization is `Some(QuantizationType::None)`,
//!   which no code builds (no quantization builds a plain HNSW index), is
//!   stored as a plain index and comes back with no quantization.

use std::collections::HashSet;

use grafeo_common::storage::catalog_record::{
    CatalogRecord, ConstraintRecord, DistanceMetricRecord, EdgeTypeRecord, EndpointPair,
    GraphTypeRecord, IndexKindRecord, IndexNameKindRecord, IndexNameRecord, IndexRecord,
    MAX_CATALOG_RECORD_PAYLOAD, NamedConstraintKindRecord, NodeTypeRecord, ProcedureRecord,
    PropertyRecord, PropertyTypeRecord, QuantizationRecord, TypeConstraintRecord,
};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::index::vector::{DistanceMetric, QuantizationType};

use super::catalog_section::{GraphIndexes, IndexName, VectorIndexDefinition};
use crate::catalog::{
    ConstraintDefinition, ConstraintType, EdgeTypeDefinition, GraphTypeDefinition, IndexType,
    NodeTypeDefinition, ProcedureDefinition, PropertyDataType, TypeConstraint, TypedProperty,
};

// ── Property types and properties ───────────────────────────────────

/// The record of a property type. `TIMESTAMP` and `LOCAL DATETIME` stay two
/// types, as they are in the catalog.
///
/// The `LIST<...>` levels are counted, not recursed into. A type nested
/// deeper than a record holds converts, and its record then refuses to
/// encode.
#[must_use]
pub(crate) fn property_type_record(data_type: &PropertyDataType) -> PropertyTypeRecord {
    let mut levels = 0usize;
    let mut current = data_type;
    let element = loop {
        current = match current {
            PropertyDataType::ListTyped(inner) => {
                levels += 1;
                inner.as_ref()
            }
            PropertyDataType::String => break PropertyTypeRecord::String,
            PropertyDataType::Int64 => break PropertyTypeRecord::Int64,
            PropertyDataType::Float64 => break PropertyTypeRecord::Float64,
            PropertyDataType::Bool => break PropertyTypeRecord::Bool,
            PropertyDataType::Date => break PropertyTypeRecord::Date,
            PropertyDataType::Time => break PropertyTypeRecord::Time,
            PropertyDataType::Timestamp => break PropertyTypeRecord::Timestamp,
            PropertyDataType::LocalDatetime => break PropertyTypeRecord::LocalDatetime,
            PropertyDataType::ZonedDatetime => break PropertyTypeRecord::ZonedDatetime,
            PropertyDataType::Duration => break PropertyTypeRecord::Duration,
            PropertyDataType::List => break PropertyTypeRecord::List,
            PropertyDataType::Map => break PropertyTypeRecord::Map,
            PropertyDataType::Bytes => break PropertyTypeRecord::Bytes,
            PropertyDataType::Node => break PropertyTypeRecord::Node,
            PropertyDataType::Edge => break PropertyTypeRecord::Edge,
            PropertyDataType::Any => break PropertyTypeRecord::Any,
        };
    };
    (0..levels).fold(element, |inner, _| {
        PropertyTypeRecord::ListOf(Box::new(inner))
    })
}

/// The property type of a record (see [`property_type_record`]).
///
/// It builds a box per `LIST<...>` level while the record's own boxes still
/// live: the memory bound of the catalog records counts both, 32 bytes a
/// level, and caps the levels of a record
/// ([`MAX_LIST_LEVELS_PER_RECORD`](grafeo_common::storage::catalog_record::MAX_LIST_LEVELS_PER_RECORD)).
#[must_use]
pub(crate) fn property_data_type(record: &PropertyTypeRecord) -> PropertyDataType {
    let mut levels = 0usize;
    let mut current = record;
    let element = loop {
        current = match current {
            PropertyTypeRecord::ListOf(inner) => {
                levels += 1;
                inner.as_ref()
            }
            PropertyTypeRecord::String => break PropertyDataType::String,
            PropertyTypeRecord::Int64 => break PropertyDataType::Int64,
            PropertyTypeRecord::Float64 => break PropertyDataType::Float64,
            PropertyTypeRecord::Bool => break PropertyDataType::Bool,
            PropertyTypeRecord::Date => break PropertyDataType::Date,
            PropertyTypeRecord::Time => break PropertyDataType::Time,
            PropertyTypeRecord::Timestamp => break PropertyDataType::Timestamp,
            PropertyTypeRecord::LocalDatetime => break PropertyDataType::LocalDatetime,
            PropertyTypeRecord::ZonedDatetime => break PropertyDataType::ZonedDatetime,
            PropertyTypeRecord::Duration => break PropertyDataType::Duration,
            PropertyTypeRecord::List => break PropertyDataType::List,
            PropertyTypeRecord::Map => break PropertyDataType::Map,
            PropertyTypeRecord::Bytes => break PropertyDataType::Bytes,
            PropertyTypeRecord::Node => break PropertyDataType::Node,
            PropertyTypeRecord::Edge => break PropertyDataType::Edge,
            PropertyTypeRecord::Any => break PropertyDataType::Any,
        };
    };
    (0..levels).fold(element, |inner, _| {
        PropertyDataType::ListTyped(Box::new(inner))
    })
}

fn property_record(property: &TypedProperty) -> PropertyRecord {
    PropertyRecord {
        name: property.name.clone(),
        data_type: property_type_record(&property.data_type),
        nullable: property.nullable,
        default_value: property.default_value.clone(),
    }
}

fn typed_property(record: PropertyRecord) -> TypedProperty {
    TypedProperty {
        name: record.name,
        data_type: property_data_type(&record.data_type),
        nullable: record.nullable,
        default_value: record.default_value,
    }
}

fn type_constraint_record(constraint: &TypeConstraint) -> TypeConstraintRecord {
    match constraint {
        TypeConstraint::PrimaryKey(properties) => {
            TypeConstraintRecord::PrimaryKey(properties.clone())
        }
        TypeConstraint::Unique(properties) => TypeConstraintRecord::Unique(properties.clone()),
        TypeConstraint::NotNull(property) => TypeConstraintRecord::NotNull(property.clone()),
        TypeConstraint::Check { name, expression } => TypeConstraintRecord::Check {
            name: name.clone(),
            expression: expression.clone(),
        },
    }
}

fn type_constraint(record: TypeConstraintRecord) -> TypeConstraint {
    match record {
        TypeConstraintRecord::PrimaryKey(properties) => TypeConstraint::PrimaryKey(properties),
        TypeConstraintRecord::Unique(properties) => TypeConstraint::Unique(properties),
        TypeConstraintRecord::NotNull(property) => TypeConstraint::NotNull(property),
        TypeConstraintRecord::Check { name, expression } => {
            TypeConstraint::Check { name, expression }
        }
    }
}

// ── Node, edge and graph types ──────────────────────────────────────

/// The record of a node type.
#[must_use]
pub(crate) fn node_type_record(def: &NodeTypeDefinition) -> NodeTypeRecord {
    NodeTypeRecord {
        name: def.name.clone(),
        properties: def.properties.iter().map(property_record).collect(),
        constraints: def.constraints.iter().map(type_constraint_record).collect(),
        parent_types: def.parent_types.clone(),
        key_labels: def.key_labels.clone(),
    }
}

/// The node type of a record.
#[must_use]
pub(crate) fn node_type_definition(record: NodeTypeRecord) -> NodeTypeDefinition {
    NodeTypeDefinition {
        name: record.name,
        properties: record.properties.into_iter().map(typed_property).collect(),
        constraints: record
            .constraints
            .into_iter()
            .map(type_constraint)
            .collect(),
        parent_types: record.parent_types,
        key_labels: record.key_labels,
    }
}

/// The record of an edge type: its endpoint lists as single-type pairs
/// ([`endpoint_pairs`]).
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the edge type when its pairs
/// would not fit a catalog record, before a pair is built.
pub(crate) fn edge_type_record(def: &EdgeTypeDefinition) -> Result<EdgeTypeRecord> {
    Ok(EdgeTypeRecord {
        name: def.name.clone(),
        properties: def.properties.iter().map(property_record).collect(),
        constraints: def.constraints.iter().map(type_constraint_record).collect(),
        endpoints: endpoint_pairs(&def.name, &def.source_node_types, &def.target_node_types)?,
        key_labels: def.key_labels.clone(),
    })
}

/// The edge type of a record.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the edge type when its endpoint
/// pairs are not the product of two lists ([`endpoint_lists`]).
pub(crate) fn edge_type_definition(record: EdgeTypeRecord) -> Result<EdgeTypeDefinition> {
    let (source_node_types, target_node_types) = endpoint_lists(&record.name, &record.endpoints)?;
    Ok(EdgeTypeDefinition {
        name: record.name,
        properties: record.properties.into_iter().map(typed_property).collect(),
        constraints: record
            .constraints
            .into_iter()
            .map(type_constraint)
            .collect(),
        source_node_types,
        target_node_types,
        key_labels: record.key_labels,
    })
}

/// The single-type (source, target) pairs of an edge type's two lists:
/// their cross product, sources outermost, with `None` (any node type) for
/// an empty list. Two empty lists (any to any) are no pairs at all. A type
/// listed twice is one endpoint.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the edge type `name` when the
/// pairs would take more than the [`MAX_CATALOG_RECORD_PAYLOAD`] bytes of a
/// catalog record, counted from the lists before a pair is built: each
/// endpoint takes at least its option byte, and a type also its length (one
/// byte or more) and its name.
pub(crate) fn endpoint_pairs(
    name: &str,
    sources: &[String],
    targets: &[String],
) -> Result<Vec<EndpointPair>> {
    if sources.is_empty() && targets.is_empty() {
        return Ok(Vec::new());
    }
    let sources = endpoints(sources);
    let targets = endpoints(targets);
    let fewest_bytes = |list: &[Option<String>]| {
        list.iter().fold(0usize, |total, endpoint| {
            total.saturating_add(
                endpoint
                    .as_ref()
                    .map_or(1, |name| name.len().saturating_add(2)),
            )
        })
    };
    // Every source is paired with every target, and the other way round.
    let bytes = fewest_bytes(&sources)
        .saturating_mul(targets.len())
        .saturating_add(fewest_bytes(&targets).saturating_mul(sources.len()));
    let maximum = usize::try_from(MAX_CATALOG_RECORD_PAYLOAD).unwrap_or(usize::MAX);
    if bytes > maximum {
        return Err(Error::Serialization(format!(
            "edge type '{name}': its {} source and {} target node types make {} endpoint pairs, \
             at least {bytes} bytes, more than the {maximum} bytes a catalog record may hold",
            sources.len(),
            targets.len(),
            sources.len().saturating_mul(targets.len())
        )));
    }
    Ok(sources
        .iter()
        .flat_map(|source| {
            targets.iter().map(move |target| EndpointPair {
                source: source.clone(),
                target: target.clone(),
            })
        })
        .collect())
}

/// One list's endpoints: each type once, in list order, or `None` alone (any
/// node type) for an empty list.
fn endpoints(types: &[String]) -> Vec<Option<String>> {
    if types.is_empty() {
        return vec![None];
    }
    let mut seen = HashSet::with_capacity(types.len());
    types
        .iter()
        .filter(|name| seen.insert(name.as_str()))
        .map(|name| Some(name.clone()))
        .collect()
}

/// The two lists `pairs` are the cross product of, as [`endpoint_pairs`]
/// writes it: no pairs for two empty lists, else each source with each
/// target, sources outermost, each type once, and `None` only for an empty
/// list.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming `edge_type` when the pairs are
/// not such a product: an edge type family, which a later release may store,
/// or damaged data.
pub(crate) fn endpoint_lists(
    edge_type: &str,
    pairs: &[EndpointPair],
) -> Result<(Vec<String>, Vec<String>)> {
    let Some(first) = pairs.first() else {
        return Ok((Vec::new(), Vec::new()));
    };
    let not_a_product = || {
        Error::Serialization(format!(
            "edge type '{edge_type}': its {} endpoint pairs are not the product of a list of \
             source types and a list of target types, the only endpoints this release stores",
            pairs.len()
        ))
    };
    // The targets are those of the first source's pairs, and the sources
    // the first of each such row of pairs.
    let width = pairs
        .iter()
        .take_while(|pair| pair.source == first.source)
        .count();
    if !pairs.len().is_multiple_of(width) {
        return Err(not_a_product());
    }
    let row_targets: Vec<&Option<String>> =
        pairs[..width].iter().map(|pair| &pair.target).collect();
    let row_sources: Vec<&Option<String>> = pairs.chunks(width).map(|row| &row[0].source).collect();
    let in_order = pairs.chunks(width).zip(&row_sources).all(|(row, source)| {
        row.iter()
            .zip(&row_targets)
            .all(|(pair, target)| pair.source == **source && pair.target == **target)
    });
    let sources = endpoint_list(&row_sources).ok_or_else(not_a_product)?;
    let targets = endpoint_list(&row_targets).ok_or_else(not_a_product)?;
    if !in_order || (sources.is_empty() && targets.is_empty()) {
        return Err(not_a_product());
    }
    Ok((sources, targets))
}

/// The list one side of a product of pairs stands for: empty for `None`
/// alone, else the types, or `None` when `None` comes with types or a type
/// comes twice.
fn endpoint_list(endpoints: &[&Option<String>]) -> Option<Vec<String>> {
    if let [None] = endpoints {
        return Some(Vec::new());
    }
    let mut seen = HashSet::with_capacity(endpoints.len());
    endpoints
        .iter()
        .map(|endpoint| {
            endpoint
                .as_deref()
                .filter(|name| seen.insert(*name))
                .map(str::to_string)
        })
        .collect()
}

/// The record of a graph type.
#[must_use]
pub(crate) fn graph_type_record(def: &GraphTypeDefinition) -> GraphTypeRecord {
    GraphTypeRecord {
        name: def.name.clone(),
        node_types: def.allowed_node_types.clone(),
        edge_types: def.allowed_edge_types.clone(),
        open: def.open,
    }
}

/// The graph type of a record.
#[must_use]
pub(crate) fn graph_type_definition(record: GraphTypeRecord) -> GraphTypeDefinition {
    GraphTypeDefinition {
        name: record.name,
        allowed_node_types: record.node_types,
        allowed_edge_types: record.edge_types,
        open: record.open,
    }
}

// ── Constraints, procedures and index names ─────────────────────────

/// The record of a named constraint.
#[must_use]
pub(crate) fn constraint_record(def: &ConstraintDefinition) -> ConstraintRecord {
    ConstraintRecord {
        name: def.name.clone(),
        label: def.label.clone(),
        properties: def.properties.clone(),
        kind: match def.kind {
            ConstraintType::Unique => NamedConstraintKindRecord::Unique,
            ConstraintType::NodeKey => NamedConstraintKindRecord::NodeKey,
            ConstraintType::NotNull => NamedConstraintKindRecord::NotNull,
            ConstraintType::Exists => NamedConstraintKindRecord::Exists,
        },
    }
}

/// The named constraint of a record.
#[must_use]
pub(crate) fn constraint_definition(record: ConstraintRecord) -> ConstraintDefinition {
    ConstraintDefinition {
        name: record.name,
        label: record.label,
        properties: record.properties,
        kind: match record.kind {
            NamedConstraintKindRecord::Unique => ConstraintType::Unique,
            NamedConstraintKindRecord::NodeKey => ConstraintType::NodeKey,
            NamedConstraintKindRecord::NotNull => ConstraintType::NotNull,
            NamedConstraintKindRecord::Exists => ConstraintType::Exists,
        },
    }
}

/// The record of a stored procedure.
#[must_use]
pub(crate) fn procedure_record(def: &ProcedureDefinition) -> ProcedureRecord {
    ProcedureRecord {
        name: def.name.clone(),
        params: def.params.clone(),
        returns: def.returns.clone(),
        body: def.body.clone(),
    }
}

/// The stored procedure of a record.
#[must_use]
pub(crate) fn procedure_definition(record: ProcedureRecord) -> ProcedureDefinition {
    ProcedureDefinition {
        name: record.name,
        params: record.params,
        returns: record.returns,
        body: record.body,
    }
}

/// The record of an index name.
#[must_use]
pub(crate) fn index_name_record(name: &IndexName) -> IndexNameRecord {
    IndexNameRecord {
        name: name.name.clone(),
        label: name.label.clone(),
        property: name.property.clone(),
        kind: match name.index_type {
            IndexType::Hash => IndexNameKindRecord::Hash,
            IndexType::BTree => IndexNameKindRecord::BTree,
            IndexType::FullText => IndexNameKindRecord::FullText,
        },
    }
}

/// The index name of a record.
#[must_use]
pub(crate) fn index_name(record: IndexNameRecord) -> IndexName {
    IndexName {
        name: record.name,
        label: record.label,
        property: record.property,
        index_type: match record.kind {
            IndexNameKindRecord::Hash => IndexType::Hash,
            IndexNameKindRecord::BTree => IndexType::BTree,
            IndexNameKindRecord::FullText => IndexType::FullText,
        },
    }
}

// ── Index definitions ───────────────────────────────────────────────

/// The index records of the indexes of every graph, in the order of
/// `graphs` and, within a graph, property, vector and text indexes, each in
/// the order of its list.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the index when a vector index's
/// dimensions, links (`m`), candidate list size (`ef_construction`) or
/// product quantization subvectors exceed `u32::MAX`, or when it measures a
/// distance or quantizes in a way this release cannot store.
pub(crate) fn index_records(graphs: &[GraphIndexes]) -> Result<Vec<IndexRecord>> {
    let mut records = Vec::new();
    for graph in graphs {
        let record = |index| IndexRecord {
            graph: graph.graph.clone(),
            index,
        };
        for key in &graph.property {
            records.push(record(IndexKindRecord::Property { key: key.clone() }));
        }
        for def in &graph.vector {
            records.push(record(vector_index_record(def)?));
        }
        for (label, property) in &graph.text {
            records.push(record(IndexKindRecord::Text {
                label: label.clone(),
                property: property.clone(),
            }));
        }
    }
    Ok(records)
}

/// The index kind record of a vector index.
fn vector_index_record(def: &VectorIndexDefinition) -> Result<IndexKindRecord> {
    let unstorable = |what: String| {
        Error::Serialization(format!(
            "the vector index on :{}({}) {what}",
            def.label, def.property
        ))
    };
    let size = |what: &str, value: usize| {
        u32::try_from(value).map_err(|_| {
            unstorable(format!(
                "has {what} {value}, more than the {} a catalog record holds",
                u32::MAX
            ))
        })
    };
    let metric = match def.metric {
        DistanceMetric::Cosine => DistanceMetricRecord::Cosine,
        DistanceMetric::Euclidean => DistanceMetricRecord::Euclidean,
        DistanceMetric::DotProduct => DistanceMetricRecord::DotProduct,
        DistanceMetric::Manhattan => DistanceMetricRecord::Manhattan,
        other => {
            return Err(unstorable(format!(
                "measures {other:?}, a distance this release cannot store"
            )));
        }
    };
    let quantization = match def.quantization {
        None | Some(QuantizationType::None) => QuantizationRecord::None,
        Some(QuantizationType::Scalar) => QuantizationRecord::Scalar,
        Some(QuantizationType::Binary) => QuantizationRecord::Binary,
        Some(QuantizationType::Product { num_subvectors }) => QuantizationRecord::Product {
            num_subvectors: size("product quantization subvectors", num_subvectors)?,
        },
        Some(other) => {
            return Err(unstorable(format!(
                "quantizes as {other:?}, which this release cannot store"
            )));
        }
    };
    Ok(IndexKindRecord::Vector {
        label: def.label.clone(),
        property: def.property.clone(),
        dimensions: size("dimensions", def.dimensions)?,
        metric,
        m: size("m", def.m)?,
        ef_construction: size("ef_construction", def.ef_construction)?,
        quantization,
    })
}

/// The indexes of every graph that index records name, grouped by graph:
/// the default graph first, the others in the order their first record
/// comes, and each graph's indexes in record order.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming a vector index whose sizes do not
/// fit this platform.
pub(crate) fn graph_indexes(records: Vec<IndexRecord>) -> Result<Vec<GraphIndexes>> {
    let mut graphs: Vec<GraphIndexes> = Vec::new();
    for record in records {
        // Records come grouped by graph: the last graph is the usual one.
        let at = match graphs.iter().rposition(|graph| graph.graph == record.graph) {
            Some(at) => at,
            None => {
                graphs.push(GraphIndexes {
                    graph: record.graph,
                    ..GraphIndexes::default()
                });
                graphs.len() - 1
            }
        };
        let graph = &mut graphs[at];
        match record.index {
            IndexKindRecord::Property { key } => graph.property.push(key),
            IndexKindRecord::Vector {
                label,
                property,
                dimensions,
                metric,
                m,
                ef_construction,
                quantization,
            } => {
                let size = |what: &str, value: u32| {
                    usize::try_from(value).map_err(|_| {
                        Error::Serialization(format!(
                            "the vector index on :{label}({property}) has {what} {value}, more \
                             than this platform holds"
                        ))
                    })
                };
                let definition = VectorIndexDefinition {
                    dimensions: size("dimensions", dimensions)?,
                    metric: match metric {
                        DistanceMetricRecord::Cosine => DistanceMetric::Cosine,
                        DistanceMetricRecord::Euclidean => DistanceMetric::Euclidean,
                        DistanceMetricRecord::DotProduct => DistanceMetric::DotProduct,
                        DistanceMetricRecord::Manhattan => DistanceMetric::Manhattan,
                    },
                    m: size("m", m)?,
                    ef_construction: size("ef_construction", ef_construction)?,
                    quantization: match quantization {
                        QuantizationRecord::None => None,
                        QuantizationRecord::Scalar => Some(QuantizationType::Scalar),
                        QuantizationRecord::Binary => Some(QuantizationType::Binary),
                        QuantizationRecord::Product { num_subvectors } => {
                            Some(QuantizationType::Product {
                                num_subvectors: size(
                                    "product quantization subvectors",
                                    num_subvectors,
                                )?,
                            })
                        }
                    },
                    label,
                    property,
                };
                graph.vector.push(definition);
            }
            IndexKindRecord::Text { label, property } => graph.text.push((label, property)),
        }
    }
    if let Some(at) = graphs.iter().position(|graph| graph.graph.is_none()) {
        let default = graphs.remove(at);
        graphs.insert(0, default);
    }
    Ok(graphs)
}

// ── Naming records in errors ────────────────────────────────────────

/// A record as an error names it: its kind and what identifies it.
#[must_use]
pub(crate) fn record_name(record: &CatalogRecord) -> String {
    match record {
        CatalogRecord::Schema(record) => format!("schema '{}'", record.name),
        CatalogRecord::NodeType(record) => format!("node type '{}'", record.name),
        CatalogRecord::EdgeType(record) => format!("edge type '{}'", record.name),
        CatalogRecord::GraphType(record) => format!("graph type '{}'", record.name),
        CatalogRecord::GraphBinding(record) => format!(
            "the binding of graph '{}' to graph type '{}'",
            record.graph, record.graph_type
        ),
        CatalogRecord::Constraint(record) => format!("constraint '{}'", record.name),
        CatalogRecord::Index(record) => {
            let graph = record.graph.as_ref().map_or_else(
                || "the default graph".to_string(),
                |g| format!("graph '{g}'"),
            );
            match &record.index {
                IndexKindRecord::Property { key } => {
                    format!("the property index on '{key}' of {graph}")
                }
                IndexKindRecord::Vector {
                    label, property, ..
                } => format!("the vector index on :{label}({property}) of {graph}"),
                IndexKindRecord::Text { label, property } => {
                    format!("the text index on :{label}({property}) of {graph}")
                }
            }
        }
        CatalogRecord::IndexName(record) => format!(
            "index name '{}' on :{}({})",
            record.name, record.label, record.property
        ),
        CatalogRecord::Procedure(record) => format!("procedure '{}'", record.name),
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::storage::catalog_record::{
        EndpointPair, IndexKindRecord, IndexRecord, PropertyTypeRecord, QuantizationRecord,
    };
    use grafeo_common::types::{Timestamp, Value};
    use grafeo_core::index::vector::{DistanceMetric, QuantizationType};

    use super::*;
    use crate::catalog::{
        ConstraintDefinition, ConstraintType, EdgeTypeDefinition, GraphTypeDefinition,
        NodeTypeDefinition, ProcedureDefinition, PropertyDataType, TypeConstraint, TypedProperty,
    };
    use crate::database::catalog_section::{GraphIndexes, VectorIndexDefinition};

    fn names(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    fn pair(source: Option<&str>, target: Option<&str>) -> EndpointPair {
        EndpointPair {
            source: source.map(str::to_string),
            target: target.map(str::to_string),
        }
    }

    #[test]
    fn endpoint_lists_become_single_type_pairs_and_back() {
        for (sources, targets, pairs) in [
            (vec![], vec![], vec![]),
            (
                vec!["City"],
                vec!["City"],
                vec![pair(Some("City"), Some("City"))],
            ),
            (
                vec!["Person", "Museum"],
                vec!["City"],
                vec![
                    pair(Some("Person"), Some("City")),
                    pair(Some("Museum"), Some("City")),
                ],
            ),
            (
                vec!["Person"],
                vec!["City", "Museum"],
                vec![
                    pair(Some("Person"), Some("City")),
                    pair(Some("Person"), Some("Museum")),
                ],
            ),
            (vec![], vec!["City"], vec![pair(None, Some("City"))]),
            (vec!["Person"], vec![], vec![pair(Some("Person"), None)]),
        ] {
            assert_eq!(
                endpoint_pairs("ROUTE", &names(&sources), &names(&targets)).unwrap(),
                pairs,
                "{sources:?} to {targets:?}"
            );
            assert_eq!(
                endpoint_lists("ROUTE", &pairs).unwrap(),
                (names(&sources), names(&targets)),
                "{pairs:?}"
            );
        }
        let family = [
            pair(Some("Person"), Some("City")),
            pair(Some("Museum"), Some("Paris")),
        ];
        let error = endpoint_lists("ROUTE", &family).unwrap_err().to_string();
        assert!(
            error.contains("ROUTE") && error.contains("pairs"),
            "{error}"
        );
    }

    /// A type named twice in a list is one endpoint: the pairs hold it once,
    /// and the lists come back with it once.
    #[test]
    fn a_type_listed_twice_is_one_endpoint() {
        let pairs = endpoint_pairs(
            "ROUTE",
            &names(&["City", "City"]),
            &names(&["Paris", "Paris"]),
        )
        .unwrap();
        assert_eq!(pairs, [pair(Some("City"), Some("Paris"))]);
        assert_eq!(
            endpoint_lists("ROUTE", &pairs).unwrap(),
            (names(&["City"]), names(&["Paris"]))
        );
    }

    /// Pairs that are not the product of two lists, in the order
    /// `endpoint_pairs` writes it, are refused, naming the edge type.
    #[test]
    fn pairs_that_are_not_a_product_are_refused() {
        for pairs in [
            // "Any" next to a type, as source or as target.
            vec![pair(None, Some("City")), pair(Some("Person"), Some("City"))],
            vec![
                pair(Some("Person"), Some("City")),
                pair(Some("Person"), None),
            ],
            // A pair twice.
            vec![
                pair(Some("Person"), Some("City")),
                pair(Some("Person"), Some("City")),
            ],
            // The product of (Alix, Gus) and (Paris, Prague), in another order.
            vec![
                pair(Some("Alix"), Some("Paris")),
                pair(Some("Gus"), Some("Prague")),
                pair(Some("Alix"), Some("Prague")),
                pair(Some("Gus"), Some("Paris")),
            ],
            // Each source with the same targets, but in another order.
            vec![
                pair(Some("Alix"), Some("Paris")),
                pair(Some("Alix"), Some("Prague")),
                pair(Some("Gus"), Some("Prague")),
                pair(Some("Gus"), Some("Paris")),
            ],
            // A part of a product.
            vec![
                pair(Some("Alix"), Some("Paris")),
                pair(Some("Alix"), Some("Prague")),
                pair(Some("Gus"), Some("Paris")),
            ],
            // Any to any is no pair at all.
            vec![pair(None, None)],
        ] {
            let error = endpoint_lists("VISITED", &pairs).unwrap_err().to_string();
            assert!(
                error.contains("VISITED") && error.contains("pairs"),
                "{pairs:?}: {error}"
            );
        }
    }

    /// Every property type maps to a record and back; `TIMESTAMP` and
    /// `LOCAL DATETIME` stay two types.
    #[test]
    fn every_property_type_round_trips() {
        use PropertyDataType as T;
        let nested = |depth: usize, element: T| {
            (0..depth).fold(element, |inner, _| T::ListTyped(Box::new(inner)))
        };
        for data_type in [
            T::String,
            T::Int64,
            T::Float64,
            T::Bool,
            T::Date,
            T::Time,
            T::Timestamp,
            T::Duration,
            T::List,
            T::ListTyped(Box::new(T::Int64)),
            T::Map,
            T::Bytes,
            T::Node,
            T::Edge,
            T::Any,
            T::ZonedDatetime,
            T::LocalDatetime,
            nested(2, T::ZonedDatetime),
            nested(3, T::List),
            nested(PropertyDataType::MAX_LIST_DEPTH, T::LocalDatetime),
        ] {
            assert_eq!(
                property_data_type(&property_type_record(&data_type)),
                data_type
            );
        }
        assert_eq!(
            property_type_record(&T::Timestamp),
            PropertyTypeRecord::Timestamp
        );
        assert_eq!(
            property_type_record(&T::LocalDatetime),
            PropertyTypeRecord::LocalDatetime
        );
        assert_eq!(
            property_type_record(&T::ListTyped(Box::new(T::ZonedDatetime))),
            PropertyTypeRecord::ListOf(Box::new(PropertyTypeRecord::ZonedDatetime))
        );
    }

    /// The memory bound of the catalog records counts 32 bytes per `LIST`
    /// level: a 16-byte box in the record and another in the property type
    /// `property_data_type` builds from it.
    #[test]
    fn a_list_level_takes_a_16_byte_box_here_as_in_the_record() {
        use std::mem::size_of;

        assert_eq!(size_of::<PropertyDataType>(), 16);
        assert_eq!(size_of::<PropertyTypeRecord>(), 16);
        let record = PropertyTypeRecord::ListOf(Box::new(PropertyTypeRecord::ListOf(Box::new(
            PropertyTypeRecord::Int64,
        ))));
        assert_eq!(property_data_type(&record).list_levels(), 2);
    }

    fn city() -> NodeTypeDefinition {
        NodeTypeDefinition {
            name: "City".to_string(),
            properties: vec![
                TypedProperty {
                    name: "name".to_string(),
                    data_type: PropertyDataType::String,
                    nullable: false,
                    default_value: Some(Value::from("Amsterdam")),
                },
                TypedProperty {
                    name: "founded".to_string(),
                    data_type: PropertyDataType::LocalDatetime,
                    nullable: true,
                    // Microseconds, which the 0.5.x value encodings lose.
                    default_value: Some(Value::Timestamp(Timestamp::from_micros(
                        1_791_000_000_000_019,
                    ))),
                },
            ],
            constraints: vec![
                TypeConstraint::PrimaryKey(names(&["name"])),
                TypeConstraint::Unique(names(&["name", "founded"])),
                TypeConstraint::NotNull("name".to_string()),
                TypeConstraint::Check {
                    name: Some("named".to_string()),
                    expression: "size(name) > 3".to_string(),
                },
                TypeConstraint::Check {
                    name: None,
                    expression: "founded IS NOT NULL".to_string(),
                },
            ],
            parent_types: names(&["Place", "CityKey"]),
            key_labels: names(&["CityKey"]),
        }
    }

    /// Node, edge and graph types, constraints and procedures come back from
    /// their records as they were, key labels and defaults included.
    #[test]
    fn definitions_round_trip_through_records() {
        let node = node_type_definition(node_type_record(&city()));
        assert_eq!(format!("{node:?}"), format!("{:?}", city()));

        let route = EdgeTypeDefinition {
            name: "ROUTE".to_string(),
            properties: vec![TypedProperty {
                name: "km".to_string(),
                data_type: PropertyDataType::ListTyped(Box::new(PropertyDataType::Float64)),
                nullable: true,
                default_value: Some(Value::List(vec![Value::Float64(-0.0)].into())),
            }],
            constraints: vec![TypeConstraint::NotNull("km".to_string())],
            source_node_types: names(&["City", "Museum"]),
            target_node_types: names(&["City"]),
            key_labels: names(&["RouteKey"]),
        };
        let record = edge_type_record(&route).unwrap();
        assert_eq!(
            record.endpoints,
            [
                pair(Some("City"), Some("City")),
                pair(Some("Museum"), Some("City"))
            ]
        );
        let back = edge_type_definition(record).unwrap();
        assert_eq!(format!("{back:?}"), format!("{route:?}"));

        let atlas = GraphTypeDefinition {
            name: "Atlas".to_string(),
            allowed_node_types: names(&["City", "Museum"]),
            allowed_edge_types: names(&["ROUTE"]),
            open: true,
        };
        let back = graph_type_definition(graph_type_record(&atlas));
        assert_eq!(format!("{back:?}"), format!("{atlas:?}"));

        for kind in [
            ConstraintType::Unique,
            ConstraintType::NodeKey,
            ConstraintType::NotNull,
            ConstraintType::Exists,
        ] {
            let constraint = ConstraintDefinition {
                name: "city_name".to_string(),
                label: "City".to_string(),
                properties: names(&["name", "founded"]),
                kind,
            };
            assert_eq!(
                constraint_definition(constraint_record(&constraint)),
                constraint
            );
        }

        let procedure = ProcedureDefinition {
            name: "cities_near".to_string(),
            params: vec![("city".to_string(), "STRING".to_string())],
            returns: vec![("name".to_string(), "STRING".to_string())],
            body: "MATCH (c:City) RETURN c.name AS name".to_string(),
        };
        let back = procedure_definition(procedure_record(&procedure));
        assert_eq!(format!("{back:?}"), format!("{procedure:?}"));
    }

    /// Endpoint lists whose pairs would not fit a catalog record are refused
    /// from their sizes, naming the edge type; a product just below the
    /// record's maximum is built and its record encodes. Each pair of two
    /// eight-letter types takes 20 bytes.
    #[test]
    fn endpoint_pairs_past_a_record_are_refused_before_they_are_built() {
        let cities =
            |count: usize| -> Vec<String> { (0..count).map(|n| format!("City{n:04}")).collect() };
        let fits = EdgeTypeDefinition {
            name: "ROUTE".to_string(),
            properties: Vec::new(),
            constraints: Vec::new(),
            source_node_types: cities(320),
            target_node_types: cities(320),
            key_labels: Vec::new(),
        };
        let record = edge_type_record(&fits).unwrap();
        assert_eq!(record.endpoints.len(), 320 * 320);
        let mut out = Vec::new();
        CatalogRecord::EdgeType(record)
            .encode_framed(&mut out)
            .expect("102,400 pairs, 2,048,000 bytes, fit a record");

        let past = EdgeTypeDefinition {
            source_node_types: cities(330),
            ..fits
        };
        let error = edge_type_record(&past).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let error = error.to_string();
        assert!(
            error.contains("edge type 'ROUTE'")
                && error.contains("105600 endpoint pairs")
                && error.contains("2112000 bytes"),
            "{error}"
        );
    }

    /// An edge type whose stored pairs are not a product fails to convert,
    /// naming it.
    #[test]
    fn an_edge_type_whose_pairs_are_no_product_does_not_convert() {
        let mut record = edge_type_record(&EdgeTypeDefinition {
            name: "VISITED".to_string(),
            properties: Vec::new(),
            constraints: Vec::new(),
            source_node_types: names(&["Person"]),
            target_node_types: names(&["City"]),
            key_labels: Vec::new(),
        })
        .unwrap();
        record.endpoints.push(pair(Some("Museum"), Some("Paris")));
        let error = edge_type_definition(record).unwrap_err().to_string();
        assert!(error.contains("VISITED"), "{error}");
    }

    fn vector(label: &str, quantization: Option<QuantizationType>) -> VectorIndexDefinition {
        VectorIndexDefinition {
            label: label.to_string(),
            property: "emb".to_string(),
            dimensions: 88,
            metric: DistanceMetric::Euclidean,
            m: 19,
            ef_construction: 300,
            quantization,
        }
    }

    /// The indexes of every graph become index records, the default graph's
    /// first, and are grouped by graph again.
    #[test]
    fn index_definitions_round_trip_through_records() {
        let graphs = vec![
            GraphIndexes {
                graph: None,
                property: names(&["id", "name"]),
                vector: vec![
                    vector("Doc", None),
                    vector("Note", Some(QuantizationType::Scalar)),
                ],
                text: vec![("Doc".to_string(), "body".to_string())],
            },
            GraphIndexes {
                graph: Some("model".to_string()),
                property: Vec::new(),
                vector: vec![
                    vector("Doc", Some(QuantizationType::Binary)),
                    vector(
                        "Page",
                        Some(QuantizationType::Product { num_subvectors: 8 }),
                    ),
                ],
                text: Vec::new(),
            },
            GraphIndexes {
                graph: Some("trips".to_string()),
                property: names(&["km"]),
                vector: Vec::new(),
                text: vec![("Stop".to_string(), "notes".to_string())],
            },
        ];
        let records = index_records(&graphs).unwrap();
        assert_eq!(
            records.len(),
            9,
            "5 of the default graph, 2 of model, 2 of trips"
        );
        assert_eq!(
            records[0].index,
            IndexKindRecord::Property {
                key: "id".to_string()
            }
        );
        assert!(
            matches!(
                &records[6].index,
                IndexKindRecord::Vector {
                    quantization: QuantizationRecord::Product { num_subvectors: 8 },
                    dimensions: 88,
                    m: 19,
                    ef_construction: 300,
                    ..
                }
            ),
            "{:?}",
            records[6]
        );
        assert_eq!(graph_indexes(records).unwrap(), graphs);
        assert_eq!(graph_indexes(Vec::new()).unwrap(), Vec::new());
    }

    /// The default graph's indexes come first, whatever the order of the
    /// records; the other graphs in the order their first record comes.
    #[test]
    fn the_default_graph_comes_first() {
        let property = |graph: Option<&str>, key: &str| IndexRecord {
            graph: graph.map(str::to_string),
            index: IndexKindRecord::Property {
                key: key.to_string(),
            },
        };
        let graphs = graph_indexes(vec![
            property(Some("trips"), "km"),
            property(Some("model"), "id"),
            property(None, "name"),
            property(Some("trips"), "stops"),
        ])
        .unwrap();
        let summary: Vec<(Option<&str>, Vec<&str>)> = graphs
            .iter()
            .map(|graph| {
                (
                    graph.graph.as_deref(),
                    graph.property.iter().map(String::as_str).collect(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                (None, vec!["name"]),
                (Some("trips"), vec!["km", "stops"]),
                (Some("model"), vec!["id"]),
            ]
        );
    }

    /// A quantization of `None` is a plain HNSW index, stored as one.
    #[test]
    fn a_quantization_of_none_is_a_plain_index() {
        let graphs = [GraphIndexes {
            graph: None,
            vector: vec![vector("Doc", Some(QuantizationType::None))],
            ..GraphIndexes::default()
        }];
        assert_eq!(
            graph_indexes(index_records(&graphs).unwrap()).unwrap(),
            [GraphIndexes {
                graph: None,
                vector: vec![vector("Doc", None)],
                ..GraphIndexes::default()
            }]
        );
    }

    /// Sizes over `u32::MAX` are refused, naming the index, instead of being
    /// cut short.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn index_sizes_over_u32_are_refused() {
        let too_large = usize::try_from(u64::from(u32::MAX) + 3).unwrap();
        for change in [
            |def: &mut VectorIndexDefinition, value| def.dimensions = value,
            |def: &mut VectorIndexDefinition, value| def.m = value,
            |def: &mut VectorIndexDefinition, value| def.ef_construction = value,
            |def: &mut VectorIndexDefinition, value| {
                def.quantization = Some(QuantizationType::Product {
                    num_subvectors: value,
                });
            },
        ] {
            let mut def = vector("Doc", None);
            change(&mut def, too_large);
            let graphs = [GraphIndexes {
                graph: Some("model".to_string()),
                vector: vec![def],
                ..GraphIndexes::default()
            }];
            let error = index_records(&graphs).unwrap_err().to_string();
            assert!(
                error.contains("Doc") && error.contains("emb") && error.contains("4294967298"),
                "{error}"
            );
        }
    }
}
