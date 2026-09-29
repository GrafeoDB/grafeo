//! Replays schema records from the WAL into the catalog (#422).
//!
//! Replay makes the same catalog calls as the statements that wrote the
//! records. Kinds are parsed through the WAL's kind enums, so a kind this
//! version does not know fails the open instead of being skipped or guessed.

use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_storage::wal::{
    GraphTypeAlterationKind, NamedConstraintKind, PropertyAlterationKind, TypeConstraintKind,
    WalRecord,
};

use crate::catalog::{
    Catalog, CatalogError, ConstraintDefinition, EdgeTypeDefinition, GraphTypeDefinition,
    NodeTypeDefinition, ProcedureDefinition, PropertyDataType, TypeConstraint, TypedProperty,
};

/// Applies a schema record to `catalog`; other records are ignored.
///
/// # Errors
///
/// Fails when the record holds a kind this version does not know, or when the
/// catalog rejects it for a reason other than a repeat (see
/// [`tolerate_repeat`]).
pub(super) fn apply_schema_record(catalog: &Catalog, record: &WalRecord) -> Result<()> {
    match record {
        WalRecord::CreateNodeType {
            name,
            properties,
            constraints,
        } => {
            let def = NodeTypeDefinition {
                name: name.clone(),
                properties: typed_properties(properties),
                constraints: type_constraints(record, constraints)?,
                parent_types: Vec::new(),
            };
            tolerate_repeat(record, catalog.register_node_type(def))
        }
        WalRecord::DropNodeType { name } => tolerate_repeat(record, catalog.drop_node_type(name)),
        WalRecord::CreateEdgeType {
            name,
            properties,
            constraints,
        } => {
            let def = EdgeTypeDefinition {
                name: name.clone(),
                properties: typed_properties(properties),
                constraints: type_constraints(record, constraints)?,
                source_node_types: Vec::new(),
                target_node_types: Vec::new(),
            };
            tolerate_repeat(record, catalog.register_edge_type_def(def))
        }
        WalRecord::DropEdgeType { name } => {
            tolerate_repeat(record, catalog.drop_edge_type_def(name))
        }
        WalRecord::CreateGraphType {
            name,
            node_types,
            edge_types,
            open,
        } => {
            let def = GraphTypeDefinition {
                name: name.clone(),
                allowed_node_types: node_types.clone(),
                allowed_edge_types: edge_types.clone(),
                open: *open,
            };
            tolerate_repeat(record, catalog.register_graph_type(def))
        }
        WalRecord::DropGraphType { name } => tolerate_repeat(record, catalog.drop_graph_type(name)),
        WalRecord::CreateSchema { name } => {
            tolerate_repeat(record, catalog.register_schema_namespace(name.clone()))
        }
        WalRecord::DropSchema { name } => {
            tolerate_repeat(record, catalog.drop_schema_namespace(name))
        }
        WalRecord::AlterNodeType { name, alterations }
        | WalRecord::AlterEdgeType { name, alterations } => {
            let is_node_type = matches!(record, WalRecord::AlterNodeType { .. });
            for (action, property, type_name, nullable) in alterations {
                let result = match parse_kind(record, action, PropertyAlterationKind::parse)? {
                    PropertyAlterationKind::Add => {
                        let property = typed_property(property, type_name, *nullable);
                        if is_node_type {
                            catalog.alter_node_type_add_property(name, property)
                        } else {
                            catalog.alter_edge_type_add_property(name, property)
                        }
                    }
                    PropertyAlterationKind::Drop => {
                        if is_node_type {
                            catalog.alter_node_type_drop_property(name, property)
                        } else {
                            catalog.alter_edge_type_drop_property(name, property)
                        }
                    }
                };
                tolerate_repeat(record, result)?;
            }
            Ok(())
        }
        WalRecord::AlterGraphType { name, alterations } => {
            for (action, type_name) in alterations {
                let result = match parse_kind(record, action, GraphTypeAlterationKind::parse)? {
                    GraphTypeAlterationKind::AddNodeType => {
                        catalog.alter_graph_type_add_node_type(name, type_name.clone())
                    }
                    GraphTypeAlterationKind::DropNodeType => {
                        catalog.alter_graph_type_drop_node_type(name, type_name)
                    }
                    GraphTypeAlterationKind::AddEdgeType => {
                        catalog.alter_graph_type_add_edge_type(name, type_name.clone())
                    }
                    GraphTypeAlterationKind::DropEdgeType => {
                        catalog.alter_graph_type_drop_edge_type(name, type_name)
                    }
                };
                tolerate_repeat(record, result)?;
            }
            Ok(())
        }
        WalRecord::CreateProcedure {
            name,
            params,
            returns,
            body,
        } => {
            let def = ProcedureDefinition {
                name: name.clone(),
                params: params.clone(),
                returns: returns.clone(),
                body: body.clone(),
            };
            tolerate_repeat(record, catalog.register_procedure(def))
        }
        WalRecord::DropProcedure { name } => tolerate_repeat(record, catalog.drop_procedure(name)),
        WalRecord::CreateConstraint {
            name,
            label,
            properties,
            kind,
        } => {
            let kind = parse_kind(record, kind, NamedConstraintKind::parse)?;
            let def = ConstraintDefinition {
                name: name.clone(),
                label: label.clone(),
                properties: properties.clone(),
                kind: kind.into(),
            };
            tolerate_repeat(record, catalog.create_constraint(def))
        }
        WalRecord::DropConstraint { name } => {
            tolerate_repeat(record, catalog.drop_constraint(name))
        }
        _ => Ok(()),
    }
}

/// Replay can apply a record whose effect the checkpoint already holds (a
/// crash between writing the checkpoint and moving the WAL's checkpoint
/// marker), and the replayed WAL can start in the middle of a type's history.
/// So a create can find its type already there, and a drop or alter can find
/// it gone. Those outcomes leave the catalog as the records meant it, and are
/// logged at debug level so a recovery that went another way leaves a trace;
/// any other error fails the open.
fn tolerate_repeat(
    record: &WalRecord,
    result: std::result::Result<(), CatalogError>,
) -> Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err(
            error @ (CatalogError::TypeAlreadyExists(_)
            | CatalogError::TypeNotFound(_)
            | CatalogError::SchemaAlreadyExists(_)
            | CatalogError::SchemaNotFound(_)
            | CatalogError::ConstraintAlreadyExists
            | CatalogError::ConstraintNotFound(_)),
        ) => {
            grafeo_common::grafeo_debug!("WAL replay skipped {record:?}: {error}");
            Ok(())
        }
        Err(error) => Err(replay_failed(record, &error.to_string())),
    }
}

fn replay_failed(record: &WalRecord, reason: &str) -> Error {
    Error::Storage(StorageError::RecoveryFailed(format!(
        "cannot replay WAL record {record:?}: {reason}"
    )))
}

/// Parses a stored kind, failing the open for a kind this version does not
/// know rather than skipping the record or guessing what it meant.
fn parse_kind<K>(record: &WalRecord, text: &str, parse: fn(&str) -> Option<K>) -> Result<K> {
    parse(text).ok_or_else(|| replay_failed(record, &format!("unknown kind '{text}'")))
}

fn typed_property(name: &str, type_name: &str, nullable: bool) -> TypedProperty {
    TypedProperty {
        name: name.to_string(),
        data_type: PropertyDataType::from_type_name(type_name),
        nullable,
        default_value: None,
    }
}

fn typed_properties(properties: &[(String, String, bool)]) -> Vec<TypedProperty> {
    properties
        .iter()
        .map(|(name, type_name, nullable)| typed_property(name, type_name, *nullable))
        .collect()
}

fn type_constraints(
    record: &WalRecord,
    constraints: &[(String, Vec<String>)],
) -> Result<Vec<TypeConstraint>> {
    constraints
        .iter()
        .map(|(kind, properties)| {
            Ok(match parse_kind(record, kind, TypeConstraintKind::parse)? {
                TypeConstraintKind::Unique => TypeConstraint::Unique(properties.clone()),
                TypeConstraintKind::PrimaryKey => TypeConstraint::PrimaryKey(properties.clone()),
                TypeConstraintKind::NotNull => match properties.as_slice() {
                    [property] => TypeConstraint::NotNull(property.clone()),
                    _ => {
                        return Err(replay_failed(
                            record,
                            "a NOT NULL constraint names exactly one property",
                        ));
                    }
                },
            })
        })
        .collect()
}
