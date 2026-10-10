//! Schema metadata - what labels, properties, and indexes exist.
//!
//! The catalog is the "dictionary" of your database. When you write `(:Person)`,
//! the catalog maps "Person" to an internal LabelId. This indirection keeps
//! storage compact while names stay readable.
//!
//! | What it tracks | Why it matters |
//! | -------------- | -------------- |
//! | Labels | Maps "Person" → LabelId for efficient storage |
//! | Property keys | Maps "name" → PropertyKeyId |
//! | Edge types | Maps "KNOWS" → EdgeTypeId |
//! | Indexes | Which properties are indexed for fast lookups |

mod check_eval;

use check_eval::CheckExpression;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::{Mutex, RwLock};

use grafeo_common::collections::{GrafeoConcurrentMap, grafeo_concurrent_map};
use grafeo_common::storage::catalog_record::MAX_LIST_LEVELS_PER_RECORD;
use grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
use grafeo_common::types::{EdgeTypeId, IndexId, LabelId, PropertyKeyId, Value};

/// The database's schema dictionary - maps names to compact internal IDs.
///
/// You rarely interact with this directly. The query processor uses it to
/// resolve names like "Person" and "name" to internal IDs.
pub struct Catalog {
    /// Label name-to-ID mappings.
    labels: LabelCatalog,
    /// Property key name-to-ID mappings.
    property_keys: PropertyCatalog,
    /// Edge type name-to-ID mappings.
    edge_types: EdgeTypeCatalog,
    /// Index definitions.
    indexes: IndexCatalog,
    /// Optional schema constraints.
    schema: Option<SchemaCatalog>,
}

impl Catalog {
    /// Creates a new empty catalog with schema support enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            labels: LabelCatalog::new(),
            property_keys: PropertyCatalog::new(),
            edge_types: EdgeTypeCatalog::new(),
            indexes: IndexCatalog::new(),
            schema: Some(SchemaCatalog::new()),
        }
    }

    /// Creates a new catalog with schema constraints enabled.
    ///
    /// This is now equivalent to `new()` since schema is always enabled.
    #[must_use]
    pub fn with_schema() -> Self {
        Self::new()
    }

    // === Label Operations ===

    /// Gets or creates a label ID for the given label name.
    pub fn get_or_create_label(&self, name: &str) -> LabelId {
        self.labels.get_or_create(name)
    }

    /// Gets the label ID for a label name, if it exists.
    #[must_use]
    pub fn get_label_id(&self, name: &str) -> Option<LabelId> {
        self.labels.get_id(name)
    }

    /// Gets the label name for a label ID, if it exists.
    #[must_use]
    pub fn get_label_name(&self, id: LabelId) -> Option<Arc<str>> {
        self.labels.get_name(id)
    }

    /// Returns the number of distinct labels.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.labels.count()
    }

    /// Returns all label names.
    #[must_use]
    pub fn all_labels(&self) -> Vec<Arc<str>> {
        self.labels.all_names()
    }

    // === Property Key Operations ===

    /// Gets or creates a property key ID for the given property key name.
    pub fn get_or_create_property_key(&self, name: &str) -> PropertyKeyId {
        self.property_keys.get_or_create(name)
    }

    /// Gets the property key ID for a property key name, if it exists.
    #[must_use]
    pub fn get_property_key_id(&self, name: &str) -> Option<PropertyKeyId> {
        self.property_keys.get_id(name)
    }

    /// Gets the property key name for a property key ID, if it exists.
    #[must_use]
    pub fn get_property_key_name(&self, id: PropertyKeyId) -> Option<Arc<str>> {
        self.property_keys.get_name(id)
    }

    /// Returns the number of distinct property keys.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        self.property_keys.count()
    }

    /// Returns all property key names.
    #[must_use]
    pub fn all_property_keys(&self) -> Vec<Arc<str>> {
        self.property_keys.all_names()
    }

    // === Edge Type Operations ===

    /// Gets or creates an edge type ID for the given edge type name.
    pub fn get_or_create_edge_type(&self, name: &str) -> EdgeTypeId {
        self.edge_types.get_or_create(name)
    }

    /// Gets the edge type ID for an edge type name, if it exists.
    #[must_use]
    pub fn get_edge_type_id(&self, name: &str) -> Option<EdgeTypeId> {
        self.edge_types.get_id(name)
    }

    /// Gets the edge type name for an edge type ID, if it exists.
    #[must_use]
    pub fn get_edge_type_name(&self, id: EdgeTypeId) -> Option<Arc<str>> {
        self.edge_types.get_name(id)
    }

    /// Returns the number of distinct edge types.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.edge_types.count()
    }

    /// Returns all edge type names.
    #[must_use]
    pub fn all_edge_types(&self) -> Vec<Arc<str>> {
        self.edge_types.all_names()
    }

    // === Index Operations ===

    /// Creates a new index on a label and property key.
    pub fn create_index(
        &self,
        name: &str,
        label: LabelId,
        property_key: PropertyKeyId,
        index_type: IndexType,
    ) -> IndexId {
        self.indexes.create(name, label, property_key, index_type)
    }

    /// Drops an index by ID.
    pub fn drop_index(&self, id: IndexId) -> bool {
        self.indexes.drop(id)
    }

    /// Finds an index by its user-defined name.
    #[must_use]
    pub fn find_index_by_name(&self, name: &str) -> Option<IndexId> {
        self.indexes.find_by_name(name)
    }

    /// Gets the index definition for an index ID.
    #[must_use]
    pub fn get_index(&self, id: IndexId) -> Option<IndexDefinition> {
        self.indexes.get(id)
    }

    /// Finds indexes for a given label.
    #[must_use]
    pub fn indexes_for_label(&self, label: LabelId) -> Vec<IndexId> {
        self.indexes.for_label(label)
    }

    /// Finds indexes for a given label and property key.
    #[must_use]
    pub fn indexes_for_label_property(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Vec<IndexId> {
        self.indexes.for_label_property(label, property_key)
    }

    /// Returns all index definitions.
    #[must_use]
    pub fn all_indexes(&self) -> Vec<IndexDefinition> {
        self.indexes.all()
    }

    /// Returns the number of indexes.
    #[must_use]
    pub fn index_count(&self) -> usize {
        self.indexes.count()
    }

    // === Schema Operations ===

    /// Returns whether schema constraints are enabled.
    #[must_use]
    pub fn has_schema(&self) -> bool {
        self.schema.is_some()
    }

    /// Adds a uniqueness constraint.
    ///
    /// Returns an error if schema is not enabled or constraint already exists.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or a
    /// schema-specific error if the operation fails (e.g. duplicate constraint).
    pub fn add_unique_constraint(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.add_unique_constraint(label, property_key),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds a required property constraint (NOT NULL).
    ///
    /// Returns an error if schema is not enabled or constraint already exists.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or a
    /// schema-specific error if the operation fails (e.g. duplicate constraint).
    pub fn add_required_property(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.add_required_property(label, property_key),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Checks if a property is required for a label, through
    /// [`add_required_property`](Self::add_required_property) or a named
    /// constraint.
    #[must_use]
    pub fn is_property_required(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.schema.as_ref().is_some_and(|s| {
            s.is_property_required(label, property_key)
                || self.named_constraint_covers(s, label, property_key, ConstraintType::is_required)
        })
    }

    /// Checks if a property must be unique for a label, through
    /// [`add_unique_constraint`](Self::add_unique_constraint) or a named
    /// constraint.
    #[must_use]
    pub fn is_property_unique(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.schema.as_ref().is_some_and(|s| {
            s.is_property_unique(label, property_key)
                || self.named_constraint_covers(s, label, property_key, ConstraintType::is_unique)
        })
    }

    // === Type Definition Operations ===

    /// Returns a reference to the schema catalog.
    #[must_use]
    pub fn schema(&self) -> Option<&SchemaCatalog> {
        self.schema.as_ref()
    }

    /// Registers a node type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a type with the same name exists.
    /// * `CatalogError::InvalidCheck` if a CHECK expression does not parse.
    pub fn register_node_type(&self, def: NodeTypeDefinition) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.register_node_type(def),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Registers or replaces a node type definition.
    pub fn register_or_replace_node_type(&self, def: NodeTypeDefinition) {
        if let Some(schema) = &self.schema {
            schema.register_or_replace_node_type(def);
        }
    }

    /// Drops a node type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the type does not exist.
    pub fn drop_node_type(&self, name: &str) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.drop_node_type(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Gets a node type definition by name.
    #[must_use]
    pub fn get_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        self.schema.as_ref().and_then(|s| s.get_node_type(name))
    }

    /// Gets a resolved node type with inherited properties from parents.
    #[must_use]
    pub fn resolved_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        self.schema
            .as_ref()
            .and_then(|s| s.resolved_node_type(name))
    }

    /// Whether any node type is defined: without one, no property, NOT NULL
    /// or UNIQUE constraint applies to nodes.
    #[must_use]
    pub fn has_node_types(&self) -> bool {
        self.schema
            .as_ref()
            .is_some_and(SchemaCatalog::has_node_types)
    }

    /// Returns all registered node type names.
    #[must_use]
    pub fn all_node_type_names(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_node_types)
            .unwrap_or_default()
    }

    /// Returns all registered edge type definition names.
    #[must_use]
    pub fn all_edge_type_names(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_edge_types)
            .unwrap_or_default()
    }

    /// Registers an edge type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if an edge type with the same name exists.
    /// * `CatalogError::InvalidCheck` if a CHECK expression does not parse.
    pub fn register_edge_type_def(&self, def: EdgeTypeDefinition) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.register_edge_type(def),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Registers or replaces an edge type definition.
    pub fn register_or_replace_edge_type_def(&self, def: EdgeTypeDefinition) {
        if let Some(schema) = &self.schema {
            schema.register_or_replace_edge_type(def);
        }
    }

    /// Drops an edge type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    pub fn drop_edge_type_def(&self, name: &str) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.drop_edge_type(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Gets an edge type definition by name.
    #[must_use]
    pub fn get_edge_type_def(&self, name: &str) -> Option<EdgeTypeDefinition> {
        self.schema.as_ref().and_then(|s| s.get_edge_type(name))
    }

    /// Registers a graph type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a graph type with the same name exists.
    pub fn register_graph_type(&self, def: GraphTypeDefinition) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.register_graph_type(def),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Registers or replaces a graph type definition.
    pub fn register_or_replace_graph_type(&self, def: GraphTypeDefinition) {
        if let Some(schema) = &self.schema {
            schema.register_or_replace_graph_type(def);
        }
    }

    /// Drops a graph type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn drop_graph_type(&self, name: &str) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.drop_graph_type(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Returns all registered graph type names.
    #[must_use]
    pub fn all_graph_type_names(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_graph_types)
            .unwrap_or_default()
    }

    /// Gets a graph type definition by name.
    #[must_use]
    pub fn get_graph_type_def(&self, name: &str) -> Option<GraphTypeDefinition> {
        self.schema.as_ref().and_then(|s| s.get_graph_type(name))
    }

    /// Registers a schema namespace.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::SchemaAlreadyExists` if the namespace already exists.
    pub fn register_schema_namespace(&self, name: String) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.register_schema(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a schema namespace.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::SchemaNotFound` if the namespace does not exist.
    pub fn drop_schema_namespace(&self, name: &str) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.drop_schema(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Checks whether a schema namespace exists.
    #[must_use]
    pub fn schema_exists(&self, name: &str) -> bool {
        self.schema.as_ref().is_some_and(|s| s.schema_exists(name))
    }

    /// Returns all registered schema namespace names.
    #[must_use]
    pub fn schema_names(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(|s| s.schema_names())
            .unwrap_or_default()
    }

    /// Adds a constraint to an existing node type, creating a minimal type if needed.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::InvalidCheck` if the constraint is a CHECK whose
    ///   expression does not parse.
    pub fn add_constraint_to_type(
        &self,
        label: &str,
        constraint: TypeConstraint,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.add_constraint_to_type(label, constraint),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Whether `properties` (an entity's properties) satisfy the CHECK
    /// expression `expression`, which is parsed on its first use and kept:
    /// see [`CheckExpression::evaluate`].
    fn evaluate_check(
        &self,
        expression: &str,
        properties: &[(String, Value)],
    ) -> Result<bool, String> {
        match &self.schema {
            Some(schema) => schema.check_expression(expression)?.evaluate(properties),
            None => CheckExpression::parse(expression)?.evaluate(properties),
        }
    }

    /// Creates a named constraint: registers its name and adds the type
    /// constraints that enforce it. `CREATE CONSTRAINT` and WAL replay both
    /// call this.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::ConstraintAlreadyExists` if the name is taken; nothing
    ///   changes then.
    pub fn create_constraint(&self, def: ConstraintDefinition) -> Result<(), CatalogError> {
        self.schema
            .as_ref()
            .ok_or(CatalogError::SchemaNotEnabled)?
            .create_constraint(def)
    }

    /// Drops a named constraint and the type constraints it added.
    /// `DROP CONSTRAINT` and WAL replay both call this.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::ConstraintNotFound` if no constraint has this name.
    pub fn drop_constraint(&self, name: &str) -> Result<(), CatalogError> {
        self.schema
            .as_ref()
            .ok_or(CatalogError::SchemaNotEnabled)?
            .drop_constraint(name)
    }

    /// Calls `f` with the named constraints (sorted by name) while none can
    /// be created or dropped, so what `f` reads from the node types that
    /// enforce them is from the same moment (a checkpoint).
    pub fn with_constraints<R>(&self, f: impl FnOnce(Vec<ConstraintDefinition>) -> R) -> R {
        let Some(schema) = &self.schema else {
            return f(Vec::new());
        };
        let registry = schema.constraints.read();
        let mut constraints: Vec<ConstraintDefinition> = registry.values().cloned().collect();
        constraints.sort_by(|a, b| a.name.cmp(&b.name));
        let result = f(constraints);
        drop(registry);
        result
    }

    /// The named constraint called `name`, if any.
    #[must_use]
    pub fn constraint(&self, name: &str) -> Option<ConstraintDefinition> {
        self.schema
            .as_ref()
            .and_then(|schema| schema.constraints.read().get(name).cloned())
    }

    /// The named constraints, sorted by name.
    #[must_use]
    pub fn constraints(&self) -> Vec<ConstraintDefinition> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_constraints)
            .unwrap_or_default()
    }

    /// Registers the names of constraints whose type constraints the loaded
    /// node types already hold (loading a `.grafeo` catalog section).
    pub fn restore_constraint_names(&self, constraints: Vec<ConstraintDefinition>) {
        if let Some(schema) = &self.schema {
            let mut registry = schema.constraints.write();
            for def in constraints {
                registry.insert(def.name.clone(), def);
            }
        }
    }

    /// Whether a named constraint on `label` makes `property_key` pass `check`
    /// (see [`is_property_unique`](Self::is_property_unique)).
    fn named_constraint_covers(
        &self,
        schema: &SchemaCatalog,
        label: LabelId,
        property_key: PropertyKeyId,
        check: fn(ConstraintType) -> bool,
    ) -> bool {
        let (Some(label), Some(property)) = (
            self.get_label_name(label),
            self.get_property_key_name(property_key),
        ) else {
            return false;
        };
        schema.constraints.read().values().any(|def| {
            def.label == *label && check(def.kind) && def.properties.iter().any(|p| *p == *property)
        })
    }

    /// Adds a property to a node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the node type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    /// * `CatalogError::TooManyListLevels` if the type's property types would
    ///   nest more `LIST<...>` levels in all than its catalog record holds.
    pub fn alter_node_type_add_property(
        &self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.alter_node_type_add_property(type_name, property),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a property from a node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the node type or property does not exist.
    pub fn alter_node_type_drop_property(
        &self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.alter_node_type_drop_property(type_name, property_name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds a property to an edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    /// * `CatalogError::TooManyListLevels` if the type's property types would
    ///   nest more `LIST<...>` levels in all than its catalog record holds.
    pub fn alter_edge_type_add_property(
        &self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.alter_edge_type_add_property(type_name, property),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a property from an edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the edge type or property does not exist.
    pub fn alter_edge_type_drop_property(
        &self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.alter_edge_type_drop_property(type_name, property_name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds a node type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_add_node_type(
        &self,
        graph_type_name: &str,
        node_type: String,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.alter_graph_type_add_node_type(graph_type_name, node_type),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a node type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_drop_node_type(
        &self,
        graph_type_name: &str,
        node_type: &str,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.alter_graph_type_drop_node_type(graph_type_name, node_type),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Adds an edge type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_add_edge_type(
        &self,
        graph_type_name: &str,
        edge_type: String,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.alter_graph_type_add_edge_type(graph_type_name, edge_type),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops an edge type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled, or
    /// `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_drop_edge_type(
        &self,
        graph_type_name: &str,
        edge_type: &str,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.alter_graph_type_drop_edge_type(graph_type_name, edge_type),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Binds a graph instance to a graph type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn bind_graph_type(
        &self,
        graph_name: &str,
        graph_type: String,
    ) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => {
                // Verify the graph type exists
                if schema.get_graph_type(&graph_type).is_none() {
                    return Err(CatalogError::TypeNotFound(graph_type));
                }
                schema
                    .graph_type_bindings
                    .write()
                    .insert(graph_name.to_string(), graph_type);
                Ok(())
            }
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Gets the graph type binding for a graph instance.
    pub fn get_graph_type_binding(&self, graph_name: &str) -> Option<String> {
        self.schema
            .as_ref()?
            .graph_type_bindings
            .read()
            .get(graph_name)
            .cloned()
    }

    /// Registers a stored procedure.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeAlreadyExists` if a procedure with the same name exists.
    pub fn register_procedure(&self, def: ProcedureDefinition) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.register_procedure(def),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Replaces or creates a stored procedure.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled.
    pub fn replace_procedure(&self, def: ProcedureDefinition) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => {
                schema.replace_procedure(def);
                Ok(())
            }
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Drops a stored procedure.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the procedure does not exist.
    pub fn drop_procedure(&self, name: &str) -> Result<(), CatalogError> {
        match &self.schema {
            Some(schema) => schema.drop_procedure(name),
            None => Err(CatalogError::SchemaNotEnabled),
        }
    }

    /// Gets a stored procedure by name.
    pub fn get_procedure(&self, name: &str) -> Option<ProcedureDefinition> {
        self.schema.as_ref()?.get_procedure(name)
    }

    /// Returns all registered node type definitions.
    #[must_use]
    pub fn all_node_type_defs(&self) -> Vec<NodeTypeDefinition> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_node_type_defs)
            .unwrap_or_default()
    }

    /// Returns all registered edge type definitions.
    #[must_use]
    pub fn all_edge_type_defs(&self) -> Vec<EdgeTypeDefinition> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_edge_type_defs)
            .unwrap_or_default()
    }

    /// Returns all registered graph type definitions.
    #[must_use]
    pub fn all_graph_type_defs(&self) -> Vec<GraphTypeDefinition> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_graph_type_defs)
            .unwrap_or_default()
    }

    /// Returns all registered procedure definitions.
    #[must_use]
    pub fn all_procedure_defs(&self) -> Vec<ProcedureDefinition> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_procedure_defs)
            .unwrap_or_default()
    }

    /// Returns all graph type bindings (graph_name, type_name).
    #[must_use]
    pub fn all_graph_type_bindings(&self) -> Vec<(String, String)> {
        self.schema
            .as_ref()
            .map(SchemaCatalog::all_graph_type_bindings)
            .unwrap_or_default()
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

// === Label Catalog ===

/// Bidirectional mapping between label names and IDs.
///
/// Uses `DashMap` (shard-level locking) for `name_to_id` so concurrent
/// readers never block each other. A separate `Mutex` serializes the rare
/// create path to keep `id_to_name` consistent.
struct LabelCatalog {
    name_to_id: GrafeoConcurrentMap<Arc<str>, LabelId>,
    id_to_name: RwLock<Vec<Arc<str>>>,
    next_id: AtomicU32,
    create_lock: Mutex<()>,
}

impl LabelCatalog {
    fn new() -> Self {
        Self {
            name_to_id: grafeo_concurrent_map(),
            id_to_name: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
            create_lock: Mutex::new(()),
        }
    }

    fn get_or_create(&self, name: &str) -> LabelId {
        // Fast path: shard-level read (no global lock)
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }

        // Slow path: serialize creates to keep id_to_name consistent
        let _guard = self.create_lock.lock();
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }

        let id = LabelId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let name: Arc<str> = name.into();
        self.id_to_name.write().push(Arc::clone(&name));
        self.name_to_id.insert(name, id);
        id
    }

    fn get_id(&self, name: &str) -> Option<LabelId> {
        self.name_to_id.get(name).map(|r| *r)
    }

    fn get_name(&self, id: LabelId) -> Option<Arc<str>> {
        self.id_to_name.read().get(id.as_u32() as usize).cloned()
    }

    fn count(&self) -> usize {
        self.id_to_name.read().len()
    }

    fn all_names(&self) -> Vec<Arc<str>> {
        self.id_to_name.read().clone()
    }
}

// === Property Catalog ===

/// Bidirectional mapping between property key names and IDs.
struct PropertyCatalog {
    name_to_id: GrafeoConcurrentMap<Arc<str>, PropertyKeyId>,
    id_to_name: RwLock<Vec<Arc<str>>>,
    next_id: AtomicU32,
    create_lock: Mutex<()>,
}

impl PropertyCatalog {
    fn new() -> Self {
        Self {
            name_to_id: grafeo_concurrent_map(),
            id_to_name: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
            create_lock: Mutex::new(()),
        }
    }

    fn get_or_create(&self, name: &str) -> PropertyKeyId {
        // Fast path: shard-level read (no global lock)
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }

        // Slow path: serialize creates to keep id_to_name consistent
        let _guard = self.create_lock.lock();
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }

        let id = PropertyKeyId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let name: Arc<str> = name.into();
        self.id_to_name.write().push(Arc::clone(&name));
        self.name_to_id.insert(name, id);
        id
    }

    fn get_id(&self, name: &str) -> Option<PropertyKeyId> {
        self.name_to_id.get(name).map(|r| *r)
    }

    fn get_name(&self, id: PropertyKeyId) -> Option<Arc<str>> {
        self.id_to_name.read().get(id.as_u32() as usize).cloned()
    }

    fn count(&self) -> usize {
        self.id_to_name.read().len()
    }

    fn all_names(&self) -> Vec<Arc<str>> {
        self.id_to_name.read().clone()
    }
}

// === Edge Type Catalog ===

/// Bidirectional mapping between edge type names and IDs.
struct EdgeTypeCatalog {
    name_to_id: GrafeoConcurrentMap<Arc<str>, EdgeTypeId>,
    id_to_name: RwLock<Vec<Arc<str>>>,
    next_id: AtomicU32,
    create_lock: Mutex<()>,
}

impl EdgeTypeCatalog {
    fn new() -> Self {
        Self {
            name_to_id: grafeo_concurrent_map(),
            id_to_name: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
            create_lock: Mutex::new(()),
        }
    }

    fn get_or_create(&self, name: &str) -> EdgeTypeId {
        // Fast path: shard-level read (no global lock)
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }

        // Slow path: serialize creates to keep id_to_name consistent
        let _guard = self.create_lock.lock();
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }

        let id = EdgeTypeId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let name: Arc<str> = name.into();
        self.id_to_name.write().push(Arc::clone(&name));
        self.name_to_id.insert(name, id);
        id
    }

    fn get_id(&self, name: &str) -> Option<EdgeTypeId> {
        self.name_to_id.get(name).map(|r| *r)
    }

    fn get_name(&self, id: EdgeTypeId) -> Option<Arc<str>> {
        self.id_to_name.read().get(id.as_u32() as usize).cloned()
    }

    fn count(&self) -> usize {
        self.id_to_name.read().len()
    }

    fn all_names(&self) -> Vec<Arc<str>> {
        self.id_to_name.read().clone()
    }
}

// === Index Catalog ===

/// Type of index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum IndexType {
    /// Hash index for equality lookups.
    Hash,
    /// BTree index for range queries.
    BTree,
    /// Full-text index for text search.
    FullText,
}

/// Index definition.
///
/// Read, not built, outside this crate: later releases may add fields.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IndexDefinition {
    /// The index ID.
    pub id: IndexId,
    /// User-defined index name (e.g., `idx_person_name`).
    pub name: String,
    /// The label this index applies to.
    pub label: LabelId,
    /// The property key being indexed.
    pub property_key: PropertyKeyId,
    /// The type of index.
    pub index_type: IndexType,
}

/// Manages index definitions.
struct IndexCatalog {
    indexes: RwLock<HashMap<IndexId, IndexDefinition>>,
    label_indexes: RwLock<HashMap<LabelId, Vec<IndexId>>>,
    label_property_indexes: RwLock<HashMap<(LabelId, PropertyKeyId), Vec<IndexId>>>,
    name_index: RwLock<HashMap<String, IndexId>>,
    next_id: AtomicU32,
}

impl IndexCatalog {
    fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            label_indexes: RwLock::new(HashMap::new()),
            label_property_indexes: RwLock::new(HashMap::new()),
            name_index: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(0),
        }
    }

    fn create(
        &self,
        name: &str,
        label: LabelId,
        property_key: PropertyKeyId,
        index_type: IndexType,
    ) -> IndexId {
        let id = IndexId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let definition = IndexDefinition {
            id,
            name: name.to_string(),
            label,
            property_key,
            index_type,
        };

        let mut indexes = self.indexes.write();
        let mut label_indexes = self.label_indexes.write();
        let mut label_property_indexes = self.label_property_indexes.write();

        indexes.insert(id, definition);
        label_indexes.entry(label).or_default().push(id);
        label_property_indexes
            .entry((label, property_key))
            .or_default()
            .push(id);
        self.name_index.write().insert(name.to_string(), id);

        id
    }

    fn drop(&self, id: IndexId) -> bool {
        let mut indexes = self.indexes.write();
        let mut label_indexes = self.label_indexes.write();
        let mut label_property_indexes = self.label_property_indexes.write();

        if let Some(definition) = indexes.remove(&id) {
            // Remove from label index
            if let Some(ids) = label_indexes.get_mut(&definition.label) {
                ids.retain(|&i| i != id);
            }
            // Remove from label-property index
            if let Some(ids) =
                label_property_indexes.get_mut(&(definition.label, definition.property_key))
            {
                ids.retain(|&i| i != id);
            }
            // Remove from name index
            self.name_index.write().remove(&definition.name);
            true
        } else {
            false
        }
    }

    fn find_by_name(&self, name: &str) -> Option<IndexId> {
        self.name_index.read().get(name).copied()
    }

    fn get(&self, id: IndexId) -> Option<IndexDefinition> {
        self.indexes.read().get(&id).cloned()
    }

    fn for_label(&self, label: LabelId) -> Vec<IndexId> {
        self.label_indexes
            .read()
            .get(&label)
            .cloned()
            .unwrap_or_default()
    }

    fn for_label_property(&self, label: LabelId, property_key: PropertyKeyId) -> Vec<IndexId> {
        self.label_property_indexes
            .read()
            .get(&(label, property_key))
            .cloned()
            .unwrap_or_default()
    }

    fn count(&self) -> usize {
        self.indexes.read().len()
    }

    fn all(&self) -> Vec<IndexDefinition> {
        self.indexes.read().values().cloned().collect()
    }
}

// === Type Definitions ===

/// Data type for a typed property in a node or edge type definition.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum PropertyDataType {
    /// UTF-8 string.
    String,
    /// 64-bit signed integer.
    Int64,
    /// 64-bit floating point.
    Float64,
    /// Boolean.
    Bool,
    /// Calendar date.
    Date,
    /// Time of day.
    Time,
    /// Timestamp (date + time).
    Timestamp,
    /// Duration / interval.
    Duration,
    /// Ordered list of values (untyped).
    List,
    /// Typed list: `LIST<element_type>` (ISO sec 4.16.9).
    ListTyped(Box<PropertyDataType>),
    /// Key-value map.
    Map,
    /// Raw bytes.
    Bytes,
    /// Node reference type (ISO sec 4.15.1).
    Node,
    /// Edge reference type (ISO sec 4.15.1).
    Edge,
    /// Any type (no enforcement).
    Any,
    // The 0.5.x catalog layouts (catalog section v1, snapshot v4) are bincode
    // of this enum, which numbers the variants by position: new variants go
    // here, after the existing ones.
    /// Datetime with a fixed UTC offset (`ZONED DATETIME`), matching
    /// [`Value::ZonedDatetime`].
    ZonedDatetime,
    /// Datetime without a time zone (`LOCAL DATETIME`), matching
    /// [`Value::Timestamp`], which `local_datetime()` returns.
    LocalDatetime,
}

impl PropertyDataType {
    /// The most `LIST<...>` levels a property type nests: as deep as a
    /// property value may be ([`MAX_PROPERTY_VALUE_DEPTH`], a list of scalars
    /// being 1 deep), and as deep as the catalog records store.
    pub const MAX_LIST_DEPTH: usize = MAX_PROPERTY_VALUE_DEPTH;

    /// The most `LIST<...>` levels the property types of one node or edge
    /// type nest in all: as many as its catalog record holds
    /// ([`MAX_LIST_LEVELS_PER_RECORD`], 32,768, so 256 properties of the
    /// deepest type).
    pub const MAX_LIST_LEVELS_PER_TYPE: usize = MAX_LIST_LEVELS_PER_RECORD;

    /// The `LIST<...>` levels around the type inside them, counted, not
    /// recursed into.
    #[must_use]
    pub fn list_levels(&self) -> usize {
        let mut levels = 0;
        let mut current = self;
        while let Self::ListTyped(inner) = current {
            levels += 1;
            current = inner;
        }
        levels
    }

    /// Parses a type name string (case-insensitive) into a `PropertyDataType`.
    ///
    /// Reads every spelling [`Display`](std::fmt::Display) writes, including
    /// `ZONED DATETIME`, `LOCAL DATETIME` and nested `LIST<...>`, so the type
    /// names of WAL records and `SHOW` output read back as the same type.
    ///
    /// Unknown names are [`Any`](Self::Any): type DDL refuses them (the
    /// parsers check each name against
    /// [`PROPERTY_TYPE_NAMES`](grafeo_adapters::query::schema::PROPERTY_TYPE_NAMES)),
    /// so only a WAL record written by 0.5.x, which logged a type as it was
    /// written, holds one, and 0.5.x read it as `ANY` too.
    ///
    /// # Errors
    ///
    /// [`CatalogError::PropertyTypeTooDeep`] when the name nests more than
    /// [`MAX_LIST_DEPTH`](Self::MAX_LIST_DEPTH) `LIST<...>` levels. The levels
    /// are counted, not recursed into, so a name of any depth is refused
    /// without a stack overflow.
    pub fn from_type_name(name: &str) -> Result<Self, CatalogError> {
        let upper = name.to_uppercase();
        let mut element = upper.as_str();
        let mut levels = 0;
        while let Some(inner) = element
            .strip_prefix("LIST<")
            .and_then(|s| s.strip_suffix('>'))
        {
            levels += 1;
            if levels > Self::MAX_LIST_DEPTH {
                return Err(CatalogError::PropertyTypeTooDeep {
                    limit: Self::MAX_LIST_DEPTH,
                });
            }
            element = inner;
        }
        let mut data_type = Self::from_element_name(element);
        for _ in 0..levels {
            data_type = Self::ListTyped(Box::new(data_type));
        }
        Ok(data_type)
    }

    /// The type a name without `LIST<...>` around it names (in capitals).
    fn from_element_name(upper: &str) -> Self {
        match upper {
            "STRING" | "VARCHAR" | "TEXT" => Self::String,
            "INT" | "INT64" | "INTEGER" | "BIGINT" => Self::Int64,
            "FLOAT" | "FLOAT64" | "DOUBLE" | "REAL" => Self::Float64,
            "BOOL" | "BOOLEAN" => Self::Bool,
            "DATE" => Self::Date,
            "TIME" => Self::Time,
            "TIMESTAMP" | "DATETIME" => Self::Timestamp,
            "ZONED DATETIME" | "ZONED_DATETIME" | "ZONEDDATETIME" => Self::ZonedDatetime,
            "LOCAL DATETIME" | "LOCAL_DATETIME" | "LOCALDATETIME" => Self::LocalDatetime,
            "DURATION" | "INTERVAL" => Self::Duration,
            "LIST" | "ARRAY" => Self::List,
            "MAP" | "RECORD" => Self::Map,
            "BYTES" | "BINARY" | "BLOB" => Self::Bytes,
            "NODE" => Self::Node,
            "EDGE" | "RELATIONSHIP" => Self::Edge,
            // `ANY`, and a name only an older WAL holds (see `from_type_name`).
            _ => Self::Any,
        }
    }

    /// Checks whether a value conforms to this type.
    #[must_use]
    pub fn matches(&self, value: &Value) -> bool {
        match (self, value) {
            (Self::Any, _) | (_, Value::Null) => true,
            (Self::String, Value::String(_)) => true,
            (Self::Int64, Value::Int64(_)) => true,
            (Self::Float64, Value::Float64(_)) => true,
            (Self::Bool, Value::Bool(_)) => true,
            (Self::Date, Value::Date(_)) => true,
            (Self::Time, Value::Time(_)) => true,
            (Self::Timestamp | Self::LocalDatetime, Value::Timestamp(_)) => true,
            (Self::ZonedDatetime, Value::ZonedDatetime(_)) => true,
            (Self::Duration, Value::Duration(_)) => true,
            (Self::List, Value::List(_)) => true,
            (Self::ListTyped(elem_type), Value::List(items)) => {
                items.iter().all(|item| elem_type.matches(item))
            }
            (Self::Map, Value::Map(_)) => true,
            (Self::Bytes, Value::Bytes(_)) => true,
            // Node/Edge reference types match Map values (graph elements are
            // represented as maps with _id, _labels/_type, and properties)
            (Self::Node | Self::Edge, Value::Map(_)) => true,
            _ => false,
        }
    }

    /// The value a property of this type stores for `value` when `value` is
    /// not of the type but converts to it on assignment without losing
    /// anything, as ISO/IEC 39075 converts between numeric types and SQL's
    /// store assignment does:
    ///
    /// - an `INT64` to `FLOAT64`, when the float is exactly the integer (every
    ///   integer up to 2^53 in magnitude, and the larger ones a float holds);
    /// - a zoned datetime to its instant, for `TIMESTAMP` (`DATETIME`), which
    ///   compares with zoned values as an instant in UTC;
    /// - the items of a `LIST<...>`, each by these rules.
    ///
    /// A `LOCAL DATETIME` and a `ZONED DATETIME` take no other datetime: a
    /// local datetime has no offset to be an instant by.
    ///
    /// `None` when `value` is of the type already, or does not convert.
    #[must_use]
    pub fn convert(&self, value: &Value) -> Option<Value> {
        match (self, value) {
            (Self::Float64, Value::Int64(int)) => exact_float(*int).map(Value::Float64),
            (Self::Timestamp, Value::ZonedDatetime(zoned)) => {
                Some(Value::Timestamp(zoned.as_timestamp()))
            }
            (Self::ListTyped(item_type), Value::List(items)) if !self.matches(value) => items
                .iter()
                .map(|item| {
                    if item_type.matches(item) {
                        Some(item.clone())
                    } else {
                        item_type.convert(item)
                    }
                })
                .collect::<Option<Vec<Value>>>()
                .map(|items| Value::List(items.into())),
            _ => None,
        }
    }
}

/// `int` as a float, when the float is exactly `int`.
fn exact_float(int: i64) -> Option<f64> {
    // 2^63: the one float an i64 rounds to that is out of the i64 range
    // (i64::MAX rounds up to it).
    const BEYOND_I64: f64 = 9_223_372_036_854_775_808.0;
    #[allow(
        clippy::cast_precision_loss,
        reason = "the cast back below checks that the float is exact"
    )]
    let float = int as f64;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "`float` came from an i64 and is below 2^63, checked first, and a whole \
                  number (a float from an integer is), so the cast back is exact"
    )]
    let exact = float < BEYOND_I64 && float as i64 == int;
    exact.then_some(float)
}

/// The error for a property value that is not of the property's type.
fn type_mismatch(
    key: &str,
    owner: &str,
    data_type: &PropertyDataType,
    value: &Value,
) -> OperatorError {
    let detail = match (data_type, value) {
        (PropertyDataType::Float64, Value::Int64(int)) if exact_float(*int).is_none() => {
            ", which has no exact Float64 value (an integer beyond 2^53)"
        }
        _ => "",
    };
    OperatorError::ConstraintViolation(format!(
        "property '{key}' on :{owner} expects {data_type:?}, got {value:?}{detail}"
    ))
}

impl std::fmt::Display for PropertyDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String => write!(f, "STRING"),
            Self::Int64 => write!(f, "INT64"),
            Self::Float64 => write!(f, "FLOAT64"),
            Self::Bool => write!(f, "BOOLEAN"),
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
            Self::Duration => write!(f, "DURATION"),
            Self::List => write!(f, "LIST"),
            Self::ListTyped(elem) => write!(f, "LIST<{elem}>"),
            Self::Map => write!(f, "MAP"),
            Self::Bytes => write!(f, "BYTES"),
            Self::Node => write!(f, "NODE"),
            Self::Edge => write!(f, "EDGE"),
            Self::Any => write!(f, "ANY"),
            Self::ZonedDatetime => write!(f, "ZONED DATETIME"),
            Self::LocalDatetime => write!(f, "LOCAL DATETIME"),
        }
    }
}

/// A typed property within a node or edge type definition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TypedProperty {
    /// Property name.
    pub name: String,
    /// Expected data type.
    pub data_type: PropertyDataType,
    /// Whether NULL values are allowed.
    pub nullable: bool,
    /// Default value (used when property is not explicitly set).
    pub default_value: Option<Value>,
}

impl TypedProperty {
    /// Refuses the properties of one node or edge type when their types
    /// nest more `LIST<...>` levels in all than
    /// [`PropertyDataType::MAX_LIST_LEVELS_PER_TYPE`], more than the type's
    /// catalog record holds.
    ///
    /// # Errors
    ///
    /// [`CatalogError::TooManyListLevels`] with the levels they nest.
    pub(crate) fn check_list_levels<'a>(
        properties: impl IntoIterator<Item = &'a Self>,
    ) -> Result<(), CatalogError> {
        let levels = properties
            .into_iter()
            .map(|property| property.data_type.list_levels())
            .fold(0usize, usize::saturating_add);
        if levels > PropertyDataType::MAX_LIST_LEVELS_PER_TYPE {
            return Err(CatalogError::TooManyListLevels {
                levels,
                limit: PropertyDataType::MAX_LIST_LEVELS_PER_TYPE,
            });
        }
        Ok(())
    }
}

/// A constraint on a node or edge type.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum TypeConstraint {
    /// Primary key (implies UNIQUE + NOT NULL).
    PrimaryKey(Vec<String>),
    /// Uniqueness constraint on one or more properties.
    Unique(Vec<String>),
    /// NOT NULL constraint on a single property.
    NotNull(String),
    /// CHECK constraint with a named expression string.
    Check {
        /// Optional constraint name.
        name: Option<String>,
        /// Expression (stored as string for now).
        expression: String,
    },
}

/// What a named constraint (`CREATE CONSTRAINT`) requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ConstraintType {
    /// The properties are unique among nodes with the label.
    Unique,
    /// The properties are present and unique together.
    NodeKey,
    /// The properties are present (`NOT NULL`).
    NotNull,
    /// The properties are present (`EXISTS`, another spelling of `NOT NULL`).
    Exists,
}

impl ConstraintType {
    /// The name `SHOW CONSTRAINTS` reports.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Unique => "UNIQUE",
            Self::NodeKey => "NODE KEY",
            Self::NotNull => "NOT NULL",
            Self::Exists => "EXISTS",
        }
    }

    /// The last part of a constraint's default name, `Label_prop_unique`.
    #[must_use]
    pub const fn name_suffix(self) -> &'static str {
        match self {
            Self::Unique => "unique",
            Self::NodeKey => "node_key",
            Self::NotNull => "not_null",
            Self::Exists => "exists",
        }
    }

    fn is_unique(self) -> bool {
        matches!(self, Self::Unique | Self::NodeKey)
    }

    fn is_required(self) -> bool {
        matches!(self, Self::NodeKey | Self::NotNull | Self::Exists)
    }
}

#[cfg(feature = "wal")]
impl From<ConstraintType> for grafeo_storage::wal::NamedConstraintKind {
    fn from(kind: ConstraintType) -> Self {
        match kind {
            ConstraintType::Unique => Self::Unique,
            ConstraintType::NodeKey => Self::NodeKey,
            ConstraintType::NotNull => Self::NotNull,
            ConstraintType::Exists => Self::Exists,
        }
    }
}

#[cfg(feature = "wal")]
impl From<grafeo_storage::wal::NamedConstraintKind> for ConstraintType {
    fn from(kind: grafeo_storage::wal::NamedConstraintKind) -> Self {
        use grafeo_storage::wal::NamedConstraintKind;
        match kind {
            NamedConstraintKind::Unique => Self::Unique,
            NamedConstraintKind::NodeKey => Self::NodeKey,
            NamedConstraintKind::NotNull => Self::NotNull,
            NamedConstraintKind::Exists => Self::Exists,
        }
    }
}

/// A named constraint created with `CREATE CONSTRAINT` (#420). It is enforced
/// through the [`TypeConstraint`]s it adds to the label's node type.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ConstraintDefinition {
    /// The constraint's name.
    pub name: String,
    /// The node label it applies to.
    pub label: String,
    /// The constrained properties.
    pub properties: Vec<String>,
    /// What it requires.
    pub kind: ConstraintType,
}

impl ConstraintDefinition {
    /// The type constraints that enforce it on the label's node type.
    fn type_constraints(&self) -> Vec<TypeConstraint> {
        match self.kind {
            ConstraintType::Unique => vec![TypeConstraint::Unique(self.properties.clone())],
            ConstraintType::NodeKey => vec![TypeConstraint::PrimaryKey(self.properties.clone())],
            ConstraintType::NotNull | ConstraintType::Exists => self
                .properties
                .iter()
                .cloned()
                .map(TypeConstraint::NotNull)
                .collect(),
        }
    }
}

/// Definition of a node type (label schema).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeTypeDefinition {
    /// Type name (corresponds to a label).
    pub name: String,
    /// Typed property definitions.
    pub properties: Vec<TypedProperty>,
    /// Type-level constraints.
    pub constraints: Vec<TypeConstraint>,
    /// Parent type names for inheritance (GQL `EXTENDS`).
    pub parent_types: Vec<String>,
    /// The labels of the type's `KEY (...)` clause in a graph type. A node
    /// type declared inline also gets them as parent types.
    ///
    /// Not serialized: the 0.5.x catalog layouts (catalog section version 1,
    /// snapshot v4) are bincode of this struct and have no such field; the
    /// catalog records of version 2 hold it.
    #[serde(skip)]
    pub key_labels: Vec<String>,
}

/// Definition of an edge type (relationship type schema).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeTypeDefinition {
    /// Type name (corresponds to an edge type / relationship type).
    pub name: String,
    /// Typed property definitions.
    pub properties: Vec<TypedProperty>,
    /// Type-level constraints.
    pub constraints: Vec<TypeConstraint>,
    /// Allowed source node types (empty = any).
    pub source_node_types: Vec<String>,
    /// Allowed target node types (empty = any).
    pub target_node_types: Vec<String>,
    /// The labels of the type's `KEY (...)` clause in a graph type.
    ///
    /// Not serialized: the 0.5.x catalog layouts (catalog section version 1,
    /// snapshot v4) are bincode of this struct and have no such field; the
    /// catalog records of version 2 hold it.
    #[serde(skip)]
    pub key_labels: Vec<String>,
}

/// Definition of a graph type (constrains which node/edge types a graph allows).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphTypeDefinition {
    /// Graph type name.
    pub name: String,
    /// Allowed node types (empty = open).
    pub allowed_node_types: Vec<String>,
    /// Allowed edge types (empty = open).
    pub allowed_edge_types: Vec<String>,
    /// Whether unlisted types are permitted.
    pub open: bool,
}

/// Definition of a stored procedure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcedureDefinition {
    /// Procedure name.
    pub name: String,
    /// Parameter definitions: (name, type).
    pub params: Vec<(String, String)>,
    /// Return column definitions: (name, type).
    pub returns: Vec<(String, String)>,
    /// Raw GQL query body.
    pub body: String,
}

// === Schema Catalog ===

/// Adds `property` to the properties of the node or edge type `type_name`,
/// as `ALTER ... ADD PROPERTY` does: the statement checks it on a copy of
/// the type before anything changes, and the catalog applies it.
///
/// # Errors
///
/// * `CatalogError::TypeAlreadyExists` if the type has a property of that name.
/// * `CatalogError::TooManyListLevels` if the type's property types would
///   nest more `LIST<...>` levels in all than its catalog record holds.
///
/// Nothing changes on an error.
pub(crate) fn add_property(
    type_name: &str,
    properties: &mut Vec<TypedProperty>,
    property: TypedProperty,
) -> Result<(), CatalogError> {
    if properties.iter().any(|p| p.name == property.name) {
        return Err(CatalogError::TypeAlreadyExists(format!(
            "property {} on {}",
            property.name, type_name
        )));
    }
    TypedProperty::check_list_levels(properties.iter().chain([&property]))?;
    properties.push(property);
    Ok(())
}

/// Drops the property `property_name` from the properties of the node or
/// edge type `type_name`, as `ALTER ... DROP PROPERTY` does (see
/// [`add_property`]).
///
/// # Errors
///
/// `CatalogError::TypeNotFound` if the type has no property of that name;
/// nothing changes then.
pub(crate) fn drop_property(
    type_name: &str,
    properties: &mut Vec<TypedProperty>,
    property_name: &str,
) -> Result<(), CatalogError> {
    let len_before = properties.len();
    properties.retain(|p| p.name != property_name);
    if properties.len() == len_before {
        return Err(CatalogError::TypeNotFound(format!(
            "property {property_name} on {type_name}"
        )));
    }
    Ok(())
}

/// The name the parent type `parent` of the node type `child` is registered
/// under: a type created in a schema is named `schema/Type`, and names its
/// parents (`EXTENDS Base`) in that schema.
fn parent_type_key(child: &str, parent: &str) -> String {
    match child.rsplit_once('/') {
        Some((schema, _)) if !parent.contains('/') => format!("{schema}/{parent}"),
        _ => parent.to_string(),
    }
}

/// Schema constraints and type definitions.
pub struct SchemaCatalog {
    /// Properties that must be unique for a given label.
    unique_constraints: RwLock<HashSet<(LabelId, PropertyKeyId)>>,
    /// Properties that are required (NOT NULL) for a given label.
    required_properties: RwLock<HashSet<(LabelId, PropertyKeyId)>>,
    /// Registered node type definitions.
    node_types: RwLock<HashMap<String, NodeTypeDefinition>>,
    /// Registered edge type definitions.
    edge_types: RwLock<HashMap<String, EdgeTypeDefinition>>,
    /// Registered graph type definitions.
    graph_types: RwLock<HashMap<String, GraphTypeDefinition>>,
    /// Schema namespaces.
    schemas: RwLock<Vec<String>>,
    /// Graph instance to graph type bindings.
    graph_type_bindings: RwLock<HashMap<String, String>>,
    /// Stored procedure definitions.
    procedures: RwLock<HashMap<String, ProcedureDefinition>>,
    /// Named constraints (`CREATE CONSTRAINT`), by name.
    constraints: RwLock<HashMap<String, ConstraintDefinition>>,
    /// The CHECK expressions parsed so far, by their text: each is parsed
    /// once, not on every write it checks.
    checks: RwLock<HashMap<String, Arc<CheckExpression>>>,
}

impl SchemaCatalog {
    fn new() -> Self {
        Self {
            unique_constraints: RwLock::new(HashSet::new()),
            required_properties: RwLock::new(HashSet::new()),
            node_types: RwLock::new(HashMap::new()),
            edge_types: RwLock::new(HashMap::new()),
            graph_types: RwLock::new(HashMap::new()),
            schemas: RwLock::new(Vec::new()),
            graph_type_bindings: RwLock::new(HashMap::new()),
            procedures: RwLock::new(HashMap::new()),
            constraints: RwLock::new(HashMap::new()),
            checks: RwLock::new(HashMap::new()),
        }
    }

    /// The parsed CHECK expression `expression`, parsed on its first use.
    fn check_expression(&self, expression: &str) -> Result<Arc<CheckExpression>, String> {
        if let Some(parsed) = self.checks.read().get(expression) {
            return Ok(Arc::clone(parsed));
        }
        let parsed = Arc::new(CheckExpression::parse(expression)?);
        self.checks
            .write()
            .insert(expression.to_string(), Arc::clone(&parsed));
        Ok(parsed)
    }

    /// Parses the CHECK expressions among `constraints`, so one that does
    /// not parse is refused when it is declared, not by every write it
    /// would check.
    fn parse_checks<'a>(
        &self,
        constraints: impl IntoIterator<Item = &'a TypeConstraint>,
    ) -> Result<(), CatalogError> {
        for constraint in constraints {
            if let TypeConstraint::Check { expression, .. } = constraint {
                self.check_expression(expression)
                    .map_err(|reason| CatalogError::InvalidCheck {
                        expression: expression.clone(),
                        reason,
                    })?;
            }
        }
        Ok(())
    }

    // --- Node type operations ---

    /// Registers a new node type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::TypeAlreadyExists` if a type with the same name exists.
    /// * `CatalogError::InvalidCheck` if a CHECK expression does not parse.
    pub fn register_node_type(&self, def: NodeTypeDefinition) -> Result<(), CatalogError> {
        self.parse_checks(&def.constraints)?;
        let mut types = self.node_types.write();
        if types.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name));
        }
        types.insert(def.name.clone(), def);
        Ok(())
    }

    /// Registers or replaces a node type definition.
    pub fn register_or_replace_node_type(&self, def: NodeTypeDefinition) {
        self.node_types.write().insert(def.name.clone(), def);
    }

    /// Drops a node type definition by name.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if no type with the given name exists.
    pub fn drop_node_type(&self, name: &str) -> Result<(), CatalogError> {
        let mut types = self.node_types.write();
        if types.remove(name).is_none() {
            return Err(CatalogError::TypeNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Gets a node type definition by name.
    #[must_use]
    pub fn get_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        self.node_types.read().get(name).cloned()
    }

    /// Whether any node type is defined.
    #[must_use]
    pub fn has_node_types(&self) -> bool {
        !self.node_types.read().is_empty()
    }

    /// Gets a resolved node type with inherited properties and constraints from parents.
    ///
    /// Walks the parent chain depth-first, collecting properties and constraints.
    /// Detects cycles via a visited set. Child properties override parent ones
    /// with the same name.
    #[must_use]
    pub fn resolved_node_type(&self, name: &str) -> Option<NodeTypeDefinition> {
        let types = self.node_types.read();
        let base = types.get(name)?;
        if base.parent_types.is_empty() {
            return Some(base.clone());
        }
        let mut visited = HashSet::new();
        visited.insert(name.to_string());
        let mut all_properties = Vec::new();
        let mut all_constraints = Vec::new();
        Self::collect_inherited(
            &types,
            name,
            &mut visited,
            &mut all_properties,
            &mut all_constraints,
        );
        Some(NodeTypeDefinition {
            name: base.name.clone(),
            properties: all_properties,
            constraints: all_constraints,
            parent_types: base.parent_types.clone(),
            key_labels: base.key_labels.clone(),
        })
    }

    /// Recursively collects properties and constraints from a type and its parents.
    fn collect_inherited(
        types: &HashMap<String, NodeTypeDefinition>,
        name: &str,
        visited: &mut HashSet<String>,
        properties: &mut Vec<TypedProperty>,
        constraints: &mut Vec<TypeConstraint>,
    ) {
        let Some(def) = types.get(name) else { return };
        // Walk parents first (depth-first) so child properties override
        for parent in &def.parent_types {
            let parent = parent_type_key(name, parent);
            if visited.insert(parent.clone()) {
                Self::collect_inherited(types, &parent, visited, properties, constraints);
            }
        }
        // Add own properties, overriding parent ones with same name
        for prop in &def.properties {
            if let Some(pos) = properties.iter().position(|p| p.name == prop.name) {
                properties[pos] = prop.clone();
            } else {
                properties.push(prop.clone());
            }
        }
        // Append own constraints (no dedup, constraints are additive)
        constraints.extend(def.constraints.iter().cloned());
    }

    /// Returns all registered node type names.
    #[must_use]
    pub fn all_node_types(&self) -> Vec<String> {
        self.node_types.read().keys().cloned().collect()
    }

    /// Returns all registered node type definitions.
    #[must_use]
    pub fn all_node_type_defs(&self) -> Vec<NodeTypeDefinition> {
        self.node_types.read().values().cloned().collect()
    }

    // --- Edge type operations ---

    /// Registers a new edge type definition.
    ///
    /// # Errors
    ///
    /// * `CatalogError::TypeAlreadyExists` if an edge type with the same name exists.
    /// * `CatalogError::InvalidCheck` if a CHECK expression does not parse.
    pub fn register_edge_type(&self, def: EdgeTypeDefinition) -> Result<(), CatalogError> {
        self.parse_checks(&def.constraints)?;
        let mut types = self.edge_types.write();
        if types.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name));
        }
        types.insert(def.name.clone(), def);
        Ok(())
    }

    /// Registers or replaces an edge type definition.
    pub fn register_or_replace_edge_type(&self, def: EdgeTypeDefinition) {
        self.edge_types.write().insert(def.name.clone(), def);
    }

    /// Drops an edge type definition by name.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if no edge type with the given name exists.
    pub fn drop_edge_type(&self, name: &str) -> Result<(), CatalogError> {
        let mut types = self.edge_types.write();
        if types.remove(name).is_none() {
            return Err(CatalogError::TypeNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Gets an edge type definition by name.
    #[must_use]
    pub fn get_edge_type(&self, name: &str) -> Option<EdgeTypeDefinition> {
        self.edge_types.read().get(name).cloned()
    }

    /// Returns all registered edge type names.
    #[must_use]
    pub fn all_edge_types(&self) -> Vec<String> {
        self.edge_types.read().keys().cloned().collect()
    }

    /// Returns all registered edge type definitions.
    #[must_use]
    pub fn all_edge_type_defs(&self) -> Vec<EdgeTypeDefinition> {
        self.edge_types.read().values().cloned().collect()
    }

    // --- Graph type operations ---

    /// Registers a new graph type definition.
    ///
    /// The graph type types only the graphs bound to it from here on: a
    /// binding to its name already there is to a graph type of that name
    /// that was dropped (`DROP GRAPH TYPE` leaves it, binding nothing), and
    /// goes. It used to come back to life and type its graph by the new
    /// graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeAlreadyExists` if a graph type with the same name exists.
    pub fn register_graph_type(&self, def: GraphTypeDefinition) -> Result<(), CatalogError> {
        let mut types = self.graph_types.write();
        if types.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name));
        }
        self.graph_type_bindings
            .write()
            .retain(|_, graph_type| *graph_type != def.name);
        types.insert(def.name.clone(), def);
        Ok(())
    }

    /// Registers or replaces a graph type definition.
    pub fn register_or_replace_graph_type(&self, def: GraphTypeDefinition) {
        self.graph_types.write().insert(def.name.clone(), def);
    }

    /// Drops a graph type definition by name.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if no graph type with the given name exists.
    pub fn drop_graph_type(&self, name: &str) -> Result<(), CatalogError> {
        let mut types = self.graph_types.write();
        if types.remove(name).is_none() {
            return Err(CatalogError::TypeNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Gets a graph type definition by name.
    #[must_use]
    pub fn get_graph_type(&self, name: &str) -> Option<GraphTypeDefinition> {
        self.graph_types.read().get(name).cloned()
    }

    /// Returns all registered graph type names.
    #[must_use]
    pub fn all_graph_types(&self) -> Vec<String> {
        self.graph_types.read().keys().cloned().collect()
    }

    /// Returns all registered graph type definitions.
    #[must_use]
    pub fn all_graph_type_defs(&self) -> Vec<GraphTypeDefinition> {
        self.graph_types.read().values().cloned().collect()
    }

    // --- Schema namespace operations ---

    /// Registers a schema namespace.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaAlreadyExists` if the namespace already exists.
    pub fn register_schema(&self, name: String) -> Result<(), CatalogError> {
        let mut schemas = self.schemas.write();
        if schemas.contains(&name) {
            return Err(CatalogError::SchemaAlreadyExists(name));
        }
        schemas.push(name);
        Ok(())
    }

    /// Drops a schema namespace.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::SchemaNotFound` if the namespace does not exist.
    pub fn drop_schema(&self, name: &str) -> Result<(), CatalogError> {
        let mut schemas = self.schemas.write();
        if let Some(pos) = schemas.iter().position(|s| s == name) {
            schemas.remove(pos);
            Ok(())
        } else {
            Err(CatalogError::SchemaNotFound(name.to_string()))
        }
    }

    /// Checks whether a schema namespace exists.
    #[must_use]
    pub fn schema_exists(&self, name: &str) -> bool {
        self.schemas
            .read()
            .iter()
            .any(|s| s.eq_ignore_ascii_case(name))
    }

    /// Returns all registered schema namespace names.
    #[must_use]
    pub fn schema_names(&self) -> Vec<String> {
        self.schemas.read().clone()
    }

    // --- ALTER operations ---

    /// Adds a constraint to an existing node type, creating a minimal type if needed.
    ///
    /// # Errors
    ///
    /// `CatalogError::InvalidCheck` if the constraint is a CHECK whose
    /// expression does not parse; nothing changes then.
    pub fn add_constraint_to_type(
        &self,
        label: &str,
        constraint: TypeConstraint,
    ) -> Result<(), CatalogError> {
        self.parse_checks([&constraint])?;
        let mut types = self.node_types.write();
        if let Some(def) = types.get_mut(label) {
            def.constraints.push(constraint);
        } else {
            // Auto-create a minimal type definition for the label
            types.insert(
                label.to_string(),
                NodeTypeDefinition {
                    name: label.to_string(),
                    properties: Vec::new(),
                    constraints: vec![constraint],
                    parent_types: Vec::new(),
                    key_labels: Vec::new(),
                },
            );
        }
        Ok(())
    }

    /// Adds a property to an existing node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::TypeNotFound` if the node type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    /// * `CatalogError::TooManyListLevels` if the type's property types would
    ///   nest more `LIST<...>` levels in all than its catalog record holds.
    pub fn alter_node_type_add_property(
        &self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        let mut types = self.node_types.write();
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        add_property(type_name, &mut def.properties, property)
    }

    /// Drops a property from an existing node type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the node type or property does not exist.
    pub fn alter_node_type_drop_property(
        &self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        let mut types = self.node_types.write();
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        drop_property(type_name, &mut def.properties, property_name)
    }

    /// Adds a property to an existing edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    /// * `CatalogError::TooManyListLevels` if the type's property types would
    ///   nest more `LIST<...>` levels in all than its catalog record holds.
    pub fn alter_edge_type_add_property(
        &self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        let mut types = self.edge_types.write();
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        add_property(type_name, &mut def.properties, property)
    }

    /// Drops a property from an existing edge type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the edge type or property does not exist.
    pub fn alter_edge_type_drop_property(
        &self,
        type_name: &str,
        property_name: &str,
    ) -> Result<(), CatalogError> {
        let mut types = self.edge_types.write();
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        drop_property(type_name, &mut def.properties, property_name)
    }

    /// Adds a node type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_add_node_type(
        &self,
        graph_type_name: &str,
        node_type: String,
    ) -> Result<(), CatalogError> {
        let mut types = self.graph_types.write();
        let def = types
            .get_mut(graph_type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type_name.to_string()))?;
        if !def.allowed_node_types.contains(&node_type) {
            def.allowed_node_types.push(node_type);
        }
        Ok(())
    }

    /// Drops a node type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_drop_node_type(
        &self,
        graph_type_name: &str,
        node_type: &str,
    ) -> Result<(), CatalogError> {
        let mut types = self.graph_types.write();
        let def = types
            .get_mut(graph_type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type_name.to_string()))?;
        def.allowed_node_types.retain(|t| t != node_type);
        Ok(())
    }

    /// Adds an edge type to a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_add_edge_type(
        &self,
        graph_type_name: &str,
        edge_type: String,
    ) -> Result<(), CatalogError> {
        let mut types = self.graph_types.write();
        let def = types
            .get_mut(graph_type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type_name.to_string()))?;
        if !def.allowed_edge_types.contains(&edge_type) {
            def.allowed_edge_types.push(edge_type);
        }
        Ok(())
    }

    /// Drops an edge type from a graph type.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if the graph type does not exist.
    pub fn alter_graph_type_drop_edge_type(
        &self,
        graph_type_name: &str,
        edge_type: &str,
    ) -> Result<(), CatalogError> {
        let mut types = self.graph_types.write();
        let def = types
            .get_mut(graph_type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(graph_type_name.to_string()))?;
        def.allowed_edge_types.retain(|t| t != edge_type);
        Ok(())
    }

    // --- Procedure operations ---

    /// Registers a stored procedure.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeAlreadyExists` if a procedure with the same name exists.
    pub fn register_procedure(&self, def: ProcedureDefinition) -> Result<(), CatalogError> {
        let mut procs = self.procedures.write();
        if procs.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name.clone()));
        }
        procs.insert(def.name.clone(), def);
        Ok(())
    }

    /// Replaces or creates a stored procedure.
    pub fn replace_procedure(&self, def: ProcedureDefinition) {
        self.procedures.write().insert(def.name.clone(), def);
    }

    /// Drops a stored procedure.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeNotFound` if no procedure with the given name exists.
    pub fn drop_procedure(&self, name: &str) -> Result<(), CatalogError> {
        let mut procs = self.procedures.write();
        if procs.remove(name).is_none() {
            return Err(CatalogError::TypeNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Gets a stored procedure by name.
    pub fn get_procedure(&self, name: &str) -> Option<ProcedureDefinition> {
        self.procedures.read().get(name).cloned()
    }

    /// Returns all registered procedure definitions.
    #[must_use]
    pub fn all_procedure_defs(&self) -> Vec<ProcedureDefinition> {
        self.procedures.read().values().cloned().collect()
    }

    /// Returns all graph type bindings (graph_name, type_name).
    #[must_use]
    pub fn all_graph_type_bindings(&self) -> Vec<(String, String)> {
        self.graph_type_bindings
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn add_unique_constraint(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        let mut constraints = self.unique_constraints.write();
        let key = (label, property_key);
        if !constraints.insert(key) {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        Ok(())
    }

    fn add_required_property(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        let mut required = self.required_properties.write();
        let key = (label, property_key);
        if !required.insert(key) {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        Ok(())
    }

    fn is_property_required(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.required_properties
            .read()
            .contains(&(label, property_key))
    }

    fn is_property_unique(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.unique_constraints
            .read()
            .contains(&(label, property_key))
    }

    /// Registers a named constraint and adds its type constraints. The
    /// registry stays locked until both are done, so a reader holding it
    /// (a checkpoint) sees names and type constraints from one moment.
    fn create_constraint(&self, def: ConstraintDefinition) -> Result<(), CatalogError> {
        let mut constraints = self.constraints.write();
        if constraints.contains_key(&def.name) {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        for constraint in def.type_constraints() {
            self.add_constraint_to_type(&def.label, constraint)?;
        }
        constraints.insert(def.name.clone(), def);
        Ok(())
    }

    /// Removes a named constraint and its type constraints, under the same
    /// lock as [`create_constraint`](Self::create_constraint).
    fn drop_constraint(&self, name: &str) -> Result<(), CatalogError> {
        let mut constraints = self.constraints.write();
        let def = constraints
            .remove(name)
            .ok_or_else(|| CatalogError::ConstraintNotFound(name.to_string()))?;
        for constraint in def.type_constraints() {
            self.remove_constraint_from_type(&def.label, &constraint);
        }
        Ok(())
    }

    /// Removes one occurrence of `constraint` from the node type `label`:
    /// another named constraint may have added the same one.
    fn remove_constraint_from_type(&self, label: &str, constraint: &TypeConstraint) {
        if let Some(def) = self.node_types.write().get_mut(label)
            && let Some(position) = def.constraints.iter().position(|c| c == constraint)
        {
            def.constraints.remove(position);
        }
    }

    fn all_constraints(&self) -> Vec<ConstraintDefinition> {
        let mut constraints: Vec<ConstraintDefinition> =
            self.constraints.read().values().cloned().collect();
        constraints.sort_by(|a, b| a.name.cmp(&b.name));
        constraints
    }
}

// === Errors ===

/// Catalog-related errors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CatalogError {
    /// Schema constraints are not enabled.
    SchemaNotEnabled,
    /// No constraint with this name exists.
    ConstraintNotFound(String),
    /// The constraint already exists.
    ConstraintAlreadyExists,
    /// The label does not exist.
    LabelNotFound(String),
    /// The property key does not exist.
    PropertyKeyNotFound(String),
    /// The edge type does not exist.
    EdgeTypeNotFound(String),
    /// The index does not exist.
    IndexNotFound(IndexId),
    /// A type with this name already exists.
    TypeAlreadyExists(String),
    /// No type with this name exists.
    TypeNotFound(String),
    /// A schema with this name already exists.
    SchemaAlreadyExists(String),
    /// No schema with this name exists.
    SchemaNotFound(String),
    /// A property type nests more `LIST<...>` levels than `limit`
    /// ([`PropertyDataType::MAX_LIST_DEPTH`]).
    PropertyTypeTooDeep {
        /// The most levels a property type nests.
        limit: usize,
    },
    /// The property types of a node or edge type nest more `LIST<...>`
    /// levels in all than `limit`
    /// ([`PropertyDataType::MAX_LIST_LEVELS_PER_TYPE`]).
    TooManyListLevels {
        /// The levels they nest.
        levels: usize,
        /// The most levels they may nest.
        limit: usize,
    },
    /// The expression of a CHECK constraint does not parse.
    InvalidCheck {
        /// The expression.
        expression: String,
        /// What is wrong with it.
        reason: String,
    },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaNotEnabled => write!(f, "Schema constraints are not enabled"),
            Self::ConstraintNotFound(name) => write!(f, "Constraint not found: {name}"),
            Self::ConstraintAlreadyExists => write!(f, "Constraint already exists"),
            Self::LabelNotFound(name) => write!(f, "Label not found: {name}"),
            Self::PropertyKeyNotFound(name) => write!(f, "Property key not found: {name}"),
            Self::EdgeTypeNotFound(name) => write!(f, "Edge type not found: {name}"),
            Self::IndexNotFound(id) => write!(f, "Index not found: {id}"),
            Self::TypeAlreadyExists(name) => write!(f, "Type already exists: {name}"),
            Self::TypeNotFound(name) => write!(f, "Type not found: {name}"),
            Self::SchemaAlreadyExists(name) => write!(f, "Schema already exists: {name}"),
            Self::SchemaNotFound(name) => write!(f, "Schema not found: {name}"),
            Self::PropertyTypeTooDeep { limit } => {
                write!(f, "A property type nests at most {limit} LIST<...> levels")
            }
            Self::TooManyListLevels { levels, limit } => write!(
                f,
                "The property types of a node or edge type nest at most {limit} LIST<...> \
                 levels in all, not {levels}"
            ),
            Self::InvalidCheck { expression, reason } => {
                write!(f, "Invalid CHECK constraint ({expression}): {reason}")
            }
        }
    }
}

impl std::error::Error for CatalogError {}

// === Constraint Validator ===

use grafeo_core::execution::operators::ConstraintValidator;
use grafeo_core::execution::operators::OperatorError;

/// Validates schema constraints during mutation operations using the Catalog.
///
/// Checks type definitions, NOT NULL constraints, and UNIQUE constraints
/// against registered node/edge type definitions.
pub struct CatalogConstraintValidator {
    catalog: Arc<Catalog>,
    /// The closed graph type of the graph written to, if it has one.
    closed_type: Option<ClosedGraphType>,
    /// Optional graph store for UNIQUE constraint enforcement via index lookup
    /// and the dimensions of vector indexes.
    store: Option<Arc<dyn grafeo_core::graph::GraphStoreSearch>>,
    /// Optional maximum property value size in bytes.
    max_property_size: Option<usize>,
    /// The writing transaction: its uncommitted writes count for UNIQUE.
    transaction: Option<(
        grafeo_common::types::EpochId,
        grafeo_common::types::TransactionId,
    )>,
    /// The schema written in, whose node and edge types check the writes
    /// (`None`: no schema).
    schema: Option<String>,
}

impl CatalogConstraintValidator {
    /// Creates a new validator wrapping the given catalog.
    pub fn new(catalog: Arc<Catalog>) -> Self {
        Self {
            catalog,
            closed_type: None,
            store: None,
            max_property_size: None,
            transaction: None,
            schema: None,
        }
    }

    /// Checks the writes with the node and edge types of `schema`, the schema
    /// of the graph written to (`None`: no schema). A type created while a
    /// session works in a schema is registered under `schema/Type` and
    /// checks the nodes with the label `Type` (or the edges of that type) in
    /// that schema only, as `SHOW NODE TYPES` lists the types of the current
    /// schema only.
    #[must_use]
    pub fn with_schema(mut self, schema: Option<&str>) -> Self {
        self.schema = schema.map(ToString::to_string);
        self
    }

    /// The node type, with what it inherits, that checks nodes with `label`
    /// in the schema written in.
    fn node_type(&self, label: &str) -> Option<NodeTypeDefinition> {
        match &self.schema {
            Some(schema) => self
                .catalog
                .resolved_node_type(&format!("{schema}/{label}")),
            None => self.catalog.resolved_node_type(label),
        }
    }

    /// The edge type that checks edges of `edge_type` in the schema written
    /// in.
    fn edge_type(&self, edge_type: &str) -> Option<EdgeTypeDefinition> {
        match &self.schema {
            Some(schema) => self
                .catalog
                .get_edge_type_def(&format!("{schema}/{edge_type}")),
            None => self.catalog.get_edge_type_def(edge_type),
        }
    }

    /// Checks as `transaction_id` sees the data, reading at `epoch`: a UNIQUE
    /// value the transaction already wrote counts as taken.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        epoch: grafeo_common::types::EpochId,
        transaction_id: Option<grafeo_common::types::TransactionId>,
    ) -> Self {
        self.transaction = transaction_id.map(|id| (epoch, id));
        self
    }

    /// The nodes with `label` whose `key` holds `value`, as the writing
    /// transaction sees them: committed nodes and its own writes.
    ///
    /// Without a property index the candidates come from the label index,
    /// which holds the transaction's pending nodes; the property scan only
    /// sees committed nodes and would miss a value written earlier in the
    /// same transaction.
    fn nodes_with_value(
        &self,
        store: &dyn grafeo_core::graph::GraphStoreSearch,
        label: &str,
        key: &str,
        value: &Value,
    ) -> Vec<grafeo_core::graph::lpg::Node> {
        let candidates = if store.has_property_index(key) {
            store.find_nodes_by_property(key, value)
        } else {
            store.nodes_by_label(label)
        };
        let key = grafeo_common::types::PropertyKey::from(key);
        candidates
            .into_iter()
            .filter_map(|id| match self.transaction {
                Some((epoch, transaction_id)) => {
                    store.get_node_versioned(id, epoch, transaction_id)
                }
                None => store.get_node(id),
            })
            .filter(|node| {
                node.labels.iter().any(|l| l.as_str() == label)
                    && node.properties.get(&key) == Some(value)
            })
            .collect()
    }

    /// Checks writes to the graph with storage key `name` (`schema/graph` in
    /// a schema) against the closed graph type it is bound to, if any: the
    /// labels, edge types and properties it declares.
    #[must_use]
    pub fn with_graph_name(mut self, name: &str) -> Self {
        self.closed_type = ClosedGraphType::of_graph(&self.catalog, name);
        self
    }

    /// Attaches a graph store for UNIQUE constraint enforcement and vector
    /// index dimensions.
    pub fn with_store(mut self, store: Arc<dyn grafeo_core::graph::GraphStoreSearch>) -> Self {
        self.store = Some(store);
        self
    }

    /// Sets the maximum property value size in bytes.
    pub fn with_max_property_size(mut self, limit: Option<usize>) -> Self {
        self.max_property_size = limit;
        self
    }
}

impl ConstraintValidator for CatalogConstraintValidator {
    fn validate_node_property(
        &self,
        labels: &[String],
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError> {
        // A vector index on the property fixes the vector's size, and takes
        // only values it can measure a distance to (#593).
        #[cfg(feature = "vector-index")]
        if let (Value::Vector(vector), Some(store)) = (value, &self.store) {
            for label in labels {
                let Some(config) = store.vector_index_config(label, key) else {
                    continue;
                };
                if vector.len() != config.dimensions {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "property '{key}' on :{label} has a vector index of {} dimensions, got a vector of {}",
                        config.dimensions,
                        vector.len()
                    )));
                }
                if let Some((position, value)) =
                    grafeo_core::index::vector::first_non_finite(vector)
                {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "property '{key}' on :{label} has a vector index, which cannot measure \
                         {value} (at position {position})"
                    )));
                }
            }
        }
        if let Some(limit) = self.max_property_size {
            let size = value.estimated_size_bytes();
            if size > limit {
                return Err(property_size_error(key, size, limit));
            }
        }
        for label in labels {
            let Some(type_def) = self.node_type(label) else {
                continue;
            };
            if let Some(typed_prop) = type_def.properties.iter().find(|p| p.name == key) {
                // Check NOT NULL
                if !typed_prop.nullable && *value == Value::Null {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "property '{key}' on :{label} is NOT NULL, cannot set to null"
                    )));
                }
                // Check type compatibility
                if *value != Value::Null && !typed_prop.data_type.matches(value) {
                    return Err(type_mismatch(key, label, &typed_prop.data_type, value));
                }
            }
            // A null removes the property (`SET n.p = NULL`, `REMOVE n.p`),
            // which a NOT NULL or NODE KEY constraint forbids.
            let required = type_def
                .constraints
                .iter()
                .any(|constraint| match constraint {
                    TypeConstraint::NotNull(property) => property == key,
                    TypeConstraint::PrimaryKey(properties) => properties.iter().any(|p| p == key),
                    _ => false,
                });
            if required && value.is_null() {
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' on :{label} is required by a NOT NULL constraint, \
                     cannot remove it or set it to null"
                )));
            }
        }
        Ok(())
    }

    fn validate_node_complete(
        &self,
        labels: &[String],
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError> {
        let prop_names: std::collections::HashSet<&str> =
            properties.iter().map(|(n, _)| n.as_str()).collect();

        for label in labels {
            if let Some(type_def) = self.node_type(label) {
                // Check that all NOT NULL properties are present
                for typed_prop in &type_def.properties {
                    if !typed_prop.nullable
                        && typed_prop.default_value.is_none()
                        && !prop_names.contains(typed_prop.name.as_str())
                    {
                        return Err(OperatorError::ConstraintViolation(format!(
                            "missing required property '{}' on :{label}",
                            typed_prop.name
                        )));
                    }
                }
                // Check type-level constraints
                for constraint in &type_def.constraints {
                    match constraint {
                        TypeConstraint::NotNull(prop_name) => {
                            if !prop_names.contains(prop_name.as_str()) {
                                return Err(OperatorError::ConstraintViolation(format!(
                                    "missing required property '{prop_name}' on :{label} (NOT NULL constraint)"
                                )));
                            }
                        }
                        TypeConstraint::PrimaryKey(key_props) => {
                            for pk in key_props {
                                if !prop_names.contains(pk.as_str()) {
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "missing primary key property '{pk}' on :{label}"
                                    )));
                                }
                            }
                        }
                        TypeConstraint::Check { name, expression } => {
                            let constraint_name = name.as_deref().unwrap_or("unnamed");
                            match self.catalog.evaluate_check(expression, properties) {
                                Ok(true) => {}
                                Ok(false) => {
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "CHECK constraint '{constraint_name}' violated on :{label}"
                                    )));
                                }
                                Err(err) => {
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "CHECK constraint '{constraint_name}' on :{label} \
                                         cannot be evaluated: {err}"
                                    )));
                                }
                            }
                        }
                        TypeConstraint::Unique(_) => {}
                    }
                }
            }
        }
        Ok(())
    }

    fn check_unique_node_property(
        &self,
        labels: &[String],
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError> {
        // Skip uniqueness check for NULL values (NULLs are never duplicates)
        if *value == Value::Null {
            return Ok(());
        }
        for label in labels {
            if let Some(type_def) = self.node_type(label) {
                for constraint in &type_def.constraints {
                    // A constraint on several properties holds for the
                    // combination of values: see `check_unique_node`.
                    let is_unique = match constraint {
                        TypeConstraint::Unique(props) | TypeConstraint::PrimaryKey(props) => {
                            matches!(props.as_slice(), [only] if only == key)
                        }
                        _ => false,
                    };
                    if is_unique
                        && let Some(ref store) = self.store
                        && !self
                            .nodes_with_value(store.as_ref(), label, key, value)
                            .is_empty()
                    {
                        return Err(OperatorError::ConstraintViolation(format!(
                            "UNIQUE constraint violation: property '{key}' \
                             with value {value:?} already exists on :{label}"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_unique_node(
        &self,
        labels: &[String],
        properties: &[(String, Value)],
        node: Option<grafeo_common::types::NodeId>,
    ) -> Result<(), OperatorError> {
        let Some(ref store) = self.store else {
            return Ok(());
        };
        let value_of = |key: &str| {
            properties
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value)
                .filter(|value| !value.is_null())
        };
        for label in labels {
            let Some(type_def) = self.node_type(label) else {
                continue;
            };
            for constraint in &type_def.constraints {
                let (TypeConstraint::Unique(keys) | TypeConstraint::PrimaryKey(keys)) = constraint
                else {
                    continue;
                };
                if keys.len() < 2 {
                    continue;
                }
                // NULLs are never duplicates: a combination with a missing
                // value cannot collide.
                let Some(values) = keys
                    .iter()
                    .map(|key| value_of(key))
                    .collect::<Option<Vec<&Value>>>()
                else {
                    continue;
                };
                let duplicate = self
                    .nodes_with_value(store.as_ref(), label, &keys[0], values[0])
                    .into_iter()
                    .filter(|other| Some(other.id) != node)
                    .any(|other| {
                        keys.iter().zip(&values).all(|(key, value)| {
                            other
                                .properties
                                .get(&grafeo_common::types::PropertyKey::from(key.as_str()))
                                == Some(*value)
                        })
                    });
                if duplicate {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "UNIQUE constraint violation: properties ({}) with values {values:?} \
                         already exist on :{label}",
                        keys.join(", ")
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_edge_property(
        &self,
        edge_type: &str,
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError> {
        if let Some(limit) = self.max_property_size {
            let size = value.estimated_size_bytes();
            if size > limit {
                return Err(property_size_error(key, size, limit));
            }
        }
        if let Some(type_def) = self.edge_type(edge_type)
            && let Some(typed_prop) = type_def.properties.iter().find(|p| p.name == key)
        {
            // Check NOT NULL
            if !typed_prop.nullable && *value == Value::Null {
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' on :{edge_type} is NOT NULL, cannot set to null"
                )));
            }
            // Check type compatibility
            if *value != Value::Null && !typed_prop.data_type.matches(value) {
                return Err(type_mismatch(key, edge_type, &typed_prop.data_type, value));
            }
        }
        Ok(())
    }

    fn validate_edge_complete(
        &self,
        edge_type: &str,
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError> {
        if let Some(type_def) = self.edge_type(edge_type) {
            let prop_names: std::collections::HashSet<&str> =
                properties.iter().map(|(n, _)| n.as_str()).collect();

            for typed_prop in &type_def.properties {
                if !typed_prop.nullable
                    && typed_prop.default_value.is_none()
                    && !prop_names.contains(typed_prop.name.as_str())
                {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "missing required property '{}' on :{edge_type}",
                        typed_prop.name
                    )));
                }
            }

            for constraint in &type_def.constraints {
                if let TypeConstraint::Check { name, expression } = constraint {
                    let constraint_name = name.as_deref().unwrap_or("unnamed");
                    match self.catalog.evaluate_check(expression, properties) {
                        Ok(true) => {}
                        Ok(false) => {
                            return Err(OperatorError::ConstraintViolation(format!(
                                "CHECK constraint '{constraint_name}' violated on :{edge_type}"
                            )));
                        }
                        Err(err) => {
                            return Err(OperatorError::ConstraintViolation(format!(
                                "CHECK constraint '{constraint_name}' on :{edge_type} \
                                 cannot be evaluated: {err}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_node_labels_allowed(&self, labels: &[String]) -> Result<(), OperatorError> {
        let Some(closed) = &self.closed_type else {
            return Ok(());
        };
        let Some(node_types) = &closed.node_types else {
            return Ok(());
        };
        if labels.is_empty() {
            return Err(OperatorError::ConstraintViolation(format!(
                "a node without a label is not allowed by closed graph type '{}': \
                 give it the label of one of its node types",
                closed.name
            )));
        }
        match labels.iter().find(|label| !node_types.contains_key(*label)) {
            Some(label) => Err(OperatorError::ConstraintViolation(format!(
                "label '{label}' is not a node type of closed graph type '{}'",
                closed.name
            ))),
            None => Ok(()),
        }
    }

    fn validate_edge_type_allowed(&self, edge_type: &str) -> Result<(), OperatorError> {
        let Some(closed) = &self.closed_type else {
            return Ok(());
        };
        match &closed.edge_types {
            Some(edge_types) if !edge_types.contains_key(edge_type) => {
                Err(OperatorError::ConstraintViolation(format!(
                    "edge type '{edge_type}' is not an edge type of closed graph type '{}'",
                    closed.name
                )))
            }
            _ => Ok(()),
        }
    }

    fn validate_node_properties_declared(
        &self,
        labels: &[String],
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError> {
        let Some(closed) = &self.closed_type else {
            return Ok(());
        };
        let Some(node_types) = &closed.node_types else {
            return Ok(());
        };
        // The properties the node's node types declare; `None` when one of
        // them has no definition to check against.
        let mut declared: Vec<&HashSet<String>> = Vec::new();
        for label in labels {
            match node_types.get(label) {
                Some(Some(properties)) => declared.push(properties),
                Some(None) => return Ok(()),
                // A label the graph type does not declare is refused by
                // `validate_node_labels_allowed`.
                None => {}
            }
        }
        if declared.is_empty() {
            return Ok(());
        }
        let undeclared = properties
            .iter()
            .find(|(key, value)| !value.is_null() && !declared.iter().any(|d| d.contains(key)));
        match undeclared {
            Some((key, _)) => Err(OperatorError::ConstraintViolation(format!(
                "property '{key}' is not declared by node type {} of closed graph type '{}'",
                labels
                    .iter()
                    .filter(|label| node_types.contains_key(*label))
                    .map(|label| format!("'{label}'"))
                    .collect::<Vec<_>>()
                    .join(", "),
                closed.name
            ))),
            None => Ok(()),
        }
    }

    fn validate_edge_properties_declared(
        &self,
        edge_type: &str,
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError> {
        let Some(closed) = &self.closed_type else {
            return Ok(());
        };
        let Some(Some(Some(declared))) = closed
            .edge_types
            .as_ref()
            .map(|edge_types| edge_types.get(edge_type))
        else {
            return Ok(());
        };
        let undeclared = properties
            .iter()
            .find(|(key, value)| !value.is_null() && !declared.contains(key));
        match undeclared {
            Some((key, _)) => Err(OperatorError::ConstraintViolation(format!(
                "property '{key}' is not declared by edge type '{edge_type}' of closed graph \
                 type '{}'",
                closed.name
            ))),
            None => Ok(()),
        }
    }

    fn validate_edge_endpoints(
        &self,
        edge_type: &str,
        source_labels: &[String],
        target_labels: &[String],
    ) -> Result<(), OperatorError> {
        let Some(type_def) = self.edge_type(edge_type) else {
            return Ok(());
        };
        if !type_def.source_node_types.is_empty() {
            let source_ok = source_labels
                .iter()
                .any(|l| type_def.source_node_types.iter().any(|s| s == l));
            if !source_ok {
                return Err(OperatorError::ConstraintViolation(format!(
                    "source node labels {source_labels:?} are not allowed for edge type '{edge_type}', \
                     expected one of {:?}",
                    type_def.source_node_types
                )));
            }
        }
        if !type_def.target_node_types.is_empty() {
            let target_ok = target_labels
                .iter()
                .any(|l| type_def.target_node_types.iter().any(|t| t == l));
            if !target_ok {
                return Err(OperatorError::ConstraintViolation(format!(
                    "target node labels {target_labels:?} are not allowed for edge type '{edge_type}', \
                     expected one of {:?}",
                    type_def.target_node_types
                )));
            }
        }
        Ok(())
    }

    fn constrains_edge_endpoints(&self, edge_type: &str) -> bool {
        self.edge_type(edge_type).is_some_and(|def| {
            !def.source_node_types.is_empty() || !def.target_node_types.is_empty()
        })
    }

    fn constrains_node_property(&self, _key: &str, value: &Value) -> bool {
        // A vector index on (label, key) fixes a vector's size; the node
        // types and a closed graph type hold every other constraint.
        matches!(value, Value::Vector(_))
            || self.catalog.has_node_types()
            || self
                .closed_type
                .as_ref()
                .is_some_and(|closed| closed.node_types.is_some())
    }

    fn convert_node_property(&self, labels: &[String], key: &str, value: &Value) -> Option<Value> {
        if value.is_null() || !self.catalog.has_node_types() {
            return None;
        }
        labels.iter().find_map(|label| {
            let type_def = self.node_type(label)?;
            let typed = type_def.properties.iter().find(|p| p.name == key)?;
            typed.data_type.convert(value)
        })
    }

    fn convert_edge_property(&self, edge_type: &str, key: &str, value: &Value) -> Option<Value> {
        if value.is_null() {
            return None;
        }
        let type_def = self.edge_type(edge_type)?;
        let typed = type_def.properties.iter().find(|p| p.name == key)?;
        typed.data_type.convert(value)
    }

    fn inject_defaults(&self, labels: &[String], properties: &mut Vec<(String, Value)>) {
        for label in labels {
            if let Some(type_def) = self.node_type(label) {
                for typed_prop in &type_def.properties {
                    if let Some(ref default) = typed_prop.default_value {
                        let already_set = properties.iter().any(|(n, _)| n == &typed_prop.name);
                        if !already_set {
                            properties.push((typed_prop.name.clone(), default.clone()));
                        }
                    }
                }
            }
        }
    }

    fn inject_edge_defaults(&self, edge_type: &str, properties: &mut Vec<(String, Value)>) {
        let Some(type_def) = self.edge_type(edge_type) else {
            return;
        };
        for typed_prop in &type_def.properties {
            if let Some(default) = &typed_prop.default_value
                && !properties.iter().any(|(name, _)| name == &typed_prop.name)
            {
                properties.push((typed_prop.name.clone(), default.clone()));
            }
        }
    }
}

/// What a closed graph type lets a graph hold (ISO/IEC 39075:2024 4.13): the
/// labels of its node types, its edge types, and the properties each of them
/// declares.
struct ClosedGraphType {
    /// The graph type's name, for messages.
    name: String,
    /// The labels a node may have, each with the properties its node type
    /// declares (`None` when the node type has no definition to check
    /// against). `None` when the graph type lists no node types: it does not
    /// restrict them.
    node_types: Option<HashMap<String, Option<HashSet<String>>>>,
    /// The edge types an edge may have, with their properties like
    /// `node_types`.
    edge_types: Option<HashMap<String, Option<HashSet<String>>>>,
}

impl ClosedGraphType {
    /// The closed graph type the graph with storage key `graph` is bound to;
    /// `None` for a graph without one, or bound to an open graph type.
    fn of_graph(catalog: &Catalog, graph: &str) -> Option<Self> {
        let type_name = catalog.get_graph_type_binding(graph)?;
        let def = catalog.schema()?.get_graph_type(&type_name)?;
        if def.open {
            return None;
        }
        let declared = |properties: &[TypedProperty]| -> HashSet<String> {
            properties.iter().map(|p| p.name.clone()).collect()
        };
        let node_types = (!def.allowed_node_types.is_empty()).then(|| {
            let mut labels = HashMap::new();
            for name in &def.allowed_node_types {
                // A node type of a graph type in a schema is named
                // `schema/Type`; its nodes carry the label `Type`.
                let resolved = catalog.resolved_node_type(name);
                let properties = resolved.as_ref().map(|t| declared(&t.properties));
                if let Some(node_type) = &resolved {
                    for key_label in &node_type.key_labels {
                        labels
                            .entry(key_label.clone())
                            .or_insert_with(|| properties.clone());
                    }
                }
                labels.insert(unqualified(name).to_string(), properties);
            }
            labels
        });
        let edge_types = (!def.allowed_edge_types.is_empty()).then(|| {
            def.allowed_edge_types
                .iter()
                .map(|name| {
                    let properties = catalog
                        .get_edge_type_def(name)
                        .map(|t| declared(&t.properties));
                    (unqualified(name).to_string(), properties)
                })
                .collect()
        });
        Some(Self {
            name: def.name,
            node_types,
            edge_types,
        })
    }
}

/// A type name without the schema a graph type in a schema prefixes it with
/// (`schema/Type`).
fn unqualified(name: &str) -> &str {
    name.rsplit_once('/').map_or(name, |(_, name)| name)
}

/// The error for a property value over the size limit.
fn property_size_error(key: &str, size: usize, limit: usize) -> OperatorError {
    let limit_display = if limit >= 1024 * 1024 && limit.is_multiple_of(1024 * 1024) {
        format!("{} MiB", limit / (1024 * 1024))
    } else if limit >= 1024 && limit.is_multiple_of(1024) {
        format!("{} KiB", limit / 1024)
    } else {
        format!("{limit} bytes")
    };
    OperatorError::ConstraintViolation(format!(
        "property '{key}' value exceeds maximum size of {limit_display} ({size} bytes); \
         raise it with Config::with_max_property_size() or disable it with \
         Config::without_max_property_size()"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// A graph type created under the name of a dropped one types no graph:
    /// the binding the drop left goes, while the other bindings stay.
    #[test]
    fn a_new_graph_type_takes_over_no_binding() {
        let catalog = Catalog::new();
        let graph_type = |name: &str| GraphTypeDefinition {
            name: name.to_string(),
            allowed_node_types: vec!["City".to_string()],
            allowed_edge_types: Vec::new(),
            open: false,
        };
        catalog.register_graph_type(graph_type("atlas")).unwrap();
        catalog.register_graph_type(graph_type("globe")).unwrap();
        catalog
            .bind_graph_type("europe", "atlas".to_string())
            .unwrap();
        catalog
            .bind_graph_type("world", "globe".to_string())
            .unwrap();

        catalog.drop_graph_type("atlas").unwrap();
        catalog.register_graph_type(graph_type("atlas")).unwrap();
        assert_eq!(catalog.get_graph_type_binding("europe"), None);
        assert_eq!(
            catalog.all_graph_type_bindings(),
            [("world".to_string(), "globe".to_string())]
        );
        // A graph type that already exists is not registered again, and
        // keeps its graphs.
        catalog
            .register_graph_type(graph_type("globe"))
            .unwrap_err();
        assert_eq!(
            catalog.get_graph_type_binding("world").as_deref(),
            Some("globe")
        );
    }

    /// Dropping one of two named constraints on a property removes only its
    /// own type constraint, and keeps the unique and required markers that
    /// the other one still needs.
    /// A marker set through `add_unique_constraint` or
    /// `add_required_property` is not a named constraint: dropping one on
    /// the same property leaves it.
    #[test]
    fn dropping_a_constraint_keeps_markers_set_directly() {
        let catalog = Catalog::new();
        let person = catalog.get_or_create_label("Person");
        let email = catalog.get_or_create_property_key("email");
        catalog.add_unique_constraint(person, email).unwrap();
        catalog.add_required_property(person, email).unwrap();
        catalog
            .create_constraint(ConstraintDefinition {
                name: "email_key".to_string(),
                label: "Person".to_string(),
                properties: vec!["email".to_string()],
                kind: ConstraintType::NodeKey,
            })
            .unwrap();

        catalog.drop_constraint("email_key").unwrap();
        assert!(catalog.is_property_unique(person, email));
        assert!(catalog.is_property_required(person, email));
    }

    #[test]
    fn dropping_a_constraint_keeps_what_another_needs() {
        let catalog = Catalog::new();
        let email = |name: &str, kind| ConstraintDefinition {
            name: name.to_string(),
            label: "Person".to_string(),
            properties: vec!["email".to_string()],
            kind,
        };
        catalog
            .create_constraint(email("unique_email", ConstraintType::Unique))
            .unwrap();
        catalog
            .create_constraint(email("email_key", ConstraintType::NodeKey))
            .unwrap();
        let person = catalog.get_or_create_label("Person");
        let email_key = catalog.get_or_create_property_key("email");

        catalog.drop_constraint("unique_email").unwrap();
        assert!(catalog.is_property_unique(person, email_key), "the key");
        assert!(catalog.is_property_required(person, email_key));
        assert_eq!(
            catalog.get_node_type("Person").unwrap().constraints,
            vec![TypeConstraint::PrimaryKey(vec!["email".to_string()])]
        );

        catalog.drop_constraint("email_key").unwrap();
        assert!(!catalog.is_property_unique(person, email_key));
        assert!(!catalog.is_property_required(person, email_key));
        assert!(
            catalog
                .get_node_type("Person")
                .unwrap()
                .constraints
                .is_empty(),
            "expected no constraints"
        );
        assert_eq!(
            catalog.drop_constraint("email_key"),
            Err(CatalogError::ConstraintNotFound("email_key".to_string()))
        );
    }

    #[test]
    fn test_catalog_labels() {
        let catalog = Catalog::new();

        // Get or create labels
        let person_id = catalog.get_or_create_label("Person");
        let company_id = catalog.get_or_create_label("Company");

        // IDs should be different
        assert_ne!(person_id, company_id);

        // Getting the same label should return the same ID
        assert_eq!(catalog.get_or_create_label("Person"), person_id);

        // Should be able to look up by name
        assert_eq!(catalog.get_label_id("Person"), Some(person_id));
        assert_eq!(catalog.get_label_id("Company"), Some(company_id));
        assert_eq!(catalog.get_label_id("Unknown"), None);

        // Should be able to look up by ID
        assert_eq!(catalog.get_label_name(person_id).as_deref(), Some("Person"));
        assert_eq!(
            catalog.get_label_name(company_id).as_deref(),
            Some("Company")
        );

        // Count should be correct
        assert_eq!(catalog.label_count(), 2);
    }

    #[test]
    fn test_catalog_property_keys() {
        let catalog = Catalog::new();

        let name_id = catalog.get_or_create_property_key("name");
        let age_id = catalog.get_or_create_property_key("age");

        assert_ne!(name_id, age_id);
        assert_eq!(catalog.get_or_create_property_key("name"), name_id);
        assert_eq!(catalog.get_property_key_id("name"), Some(name_id));
        assert_eq!(
            catalog.get_property_key_name(name_id).as_deref(),
            Some("name")
        );
        assert_eq!(catalog.property_key_count(), 2);
    }

    #[test]
    fn test_catalog_edge_types() {
        let catalog = Catalog::new();

        let knows_id = catalog.get_or_create_edge_type("KNOWS");
        let works_at_id = catalog.get_or_create_edge_type("WORKS_AT");

        assert_ne!(knows_id, works_at_id);
        assert_eq!(catalog.get_or_create_edge_type("KNOWS"), knows_id);
        assert_eq!(catalog.get_edge_type_id("KNOWS"), Some(knows_id));
        assert_eq!(
            catalog.get_edge_type_name(knows_id).as_deref(),
            Some("KNOWS")
        );
        assert_eq!(catalog.edge_type_count(), 2);
    }

    #[test]
    fn test_catalog_indexes() {
        let catalog = Catalog::new();

        let person_id = catalog.get_or_create_label("Person");
        let name_id = catalog.get_or_create_property_key("name");
        let age_id = catalog.get_or_create_property_key("age");

        // Create indexes
        let idx1 = catalog.create_index("idx_person_name", person_id, name_id, IndexType::Hash);
        let idx2 = catalog.create_index("idx_person_age", person_id, age_id, IndexType::BTree);

        assert_ne!(idx1, idx2);
        assert_eq!(catalog.index_count(), 2);

        // Look up by label
        let label_indexes = catalog.indexes_for_label(person_id);
        assert_eq!(label_indexes.len(), 2);
        assert!(label_indexes.contains(&idx1));
        assert!(label_indexes.contains(&idx2));

        // Look up by label and property
        let name_indexes = catalog.indexes_for_label_property(person_id, name_id);
        assert_eq!(name_indexes.len(), 1);
        assert_eq!(name_indexes[0], idx1);

        // Get definition
        let def = catalog.get_index(idx1).unwrap();
        assert_eq!(def.label, person_id);
        assert_eq!(def.property_key, name_id);
        assert_eq!(def.index_type, IndexType::Hash);

        // Drop index
        assert!(catalog.drop_index(idx1));
        assert_eq!(catalog.index_count(), 1);
        assert!(catalog.get_index(idx1).is_none());
        assert_eq!(catalog.indexes_for_label(person_id).len(), 1);
    }

    #[test]
    fn test_catalog_schema_constraints() {
        let catalog = Catalog::with_schema();

        let person_id = catalog.get_or_create_label("Person");
        let email_id = catalog.get_or_create_property_key("email");
        let name_id = catalog.get_or_create_property_key("name");

        // Add constraints
        assert!(catalog.add_unique_constraint(person_id, email_id).is_ok());
        assert!(catalog.add_required_property(person_id, name_id).is_ok());

        // Check constraints
        assert!(catalog.is_property_unique(person_id, email_id));
        assert!(!catalog.is_property_unique(person_id, name_id));
        assert!(catalog.is_property_required(person_id, name_id));
        assert!(!catalog.is_property_required(person_id, email_id));

        // Duplicate constraint should fail
        assert_eq!(
            catalog.add_unique_constraint(person_id, email_id),
            Err(CatalogError::ConstraintAlreadyExists)
        );
    }

    #[test]
    fn test_catalog_schema_always_enabled() {
        // Catalog::new() always enables schema
        let catalog = Catalog::new();
        assert!(catalog.has_schema());

        let person_id = catalog.get_or_create_label("Person");
        let email_id = catalog.get_or_create_property_key("email");

        // Should succeed with schema enabled
        assert_eq!(catalog.add_unique_constraint(person_id, email_id), Ok(()));
    }

    // === Additional tests for comprehensive coverage ===

    #[test]
    fn test_catalog_default() {
        let catalog = Catalog::default();
        assert!(catalog.has_schema());
        assert_eq!(catalog.label_count(), 0);
        assert_eq!(catalog.property_key_count(), 0);
        assert_eq!(catalog.edge_type_count(), 0);
        assert_eq!(catalog.index_count(), 0);
    }

    #[test]
    fn test_catalog_all_labels() {
        let catalog = Catalog::new();

        catalog.get_or_create_label("Person");
        catalog.get_or_create_label("Company");
        catalog.get_or_create_label("Product");

        let all = catalog.all_labels();
        assert_eq!(all.len(), 3);
        assert!(all.iter().any(|l| l.as_ref() == "Person"));
        assert!(all.iter().any(|l| l.as_ref() == "Company"));
        assert!(all.iter().any(|l| l.as_ref() == "Product"));
    }

    #[test]
    fn test_catalog_all_property_keys() {
        let catalog = Catalog::new();

        catalog.get_or_create_property_key("name");
        catalog.get_or_create_property_key("age");
        catalog.get_or_create_property_key("email");

        let all = catalog.all_property_keys();
        assert_eq!(all.len(), 3);
        assert!(all.iter().any(|k| k.as_ref() == "name"));
        assert!(all.iter().any(|k| k.as_ref() == "age"));
        assert!(all.iter().any(|k| k.as_ref() == "email"));
    }

    #[test]
    fn test_catalog_all_edge_types() {
        let catalog = Catalog::new();

        catalog.get_or_create_edge_type("KNOWS");
        catalog.get_or_create_edge_type("WORKS_AT");
        catalog.get_or_create_edge_type("LIVES_IN");

        let all = catalog.all_edge_types();
        assert_eq!(all.len(), 3);
        assert!(all.iter().any(|t| t.as_ref() == "KNOWS"));
        assert!(all.iter().any(|t| t.as_ref() == "WORKS_AT"));
        assert!(all.iter().any(|t| t.as_ref() == "LIVES_IN"));
    }

    #[test]
    fn test_catalog_invalid_id_lookup() {
        let catalog = Catalog::new();

        // Create one label to ensure IDs are allocated
        let _ = catalog.get_or_create_label("Person");

        // Try to look up non-existent IDs
        let invalid_label = LabelId::new(999);
        let invalid_property = PropertyKeyId::new(999);
        let invalid_edge_type = EdgeTypeId::new(999);
        let invalid_index = IndexId::new(999);

        assert!(catalog.get_label_name(invalid_label).is_none());
        assert!(catalog.get_property_key_name(invalid_property).is_none());
        assert!(catalog.get_edge_type_name(invalid_edge_type).is_none());
        assert!(catalog.get_index(invalid_index).is_none());
    }

    #[test]
    fn test_catalog_drop_nonexistent_index() {
        let catalog = Catalog::new();
        let invalid_index = IndexId::new(999);
        assert!(!catalog.drop_index(invalid_index));
    }

    #[test]
    fn test_catalog_indexes_for_nonexistent_label() {
        let catalog = Catalog::new();
        let invalid_label = LabelId::new(999);
        let invalid_property = PropertyKeyId::new(999);

        assert!(
            catalog.indexes_for_label(invalid_label).is_empty(),
            "expected empty"
        );
        assert!(
            catalog
                .indexes_for_label_property(invalid_label, invalid_property)
                .is_empty(),
            "expected no indexes"
        );
    }

    #[test]
    fn test_catalog_multiple_indexes_same_property() {
        let catalog = Catalog::new();

        let person_id = catalog.get_or_create_label("Person");
        let name_id = catalog.get_or_create_property_key("name");

        // Create multiple indexes on the same property with different types
        let hash_idx = catalog.create_index("idx_hash", person_id, name_id, IndexType::Hash);
        let btree_idx = catalog.create_index("idx_btree", person_id, name_id, IndexType::BTree);
        let fulltext_idx =
            catalog.create_index("idx_fulltext", person_id, name_id, IndexType::FullText);

        assert_eq!(catalog.index_count(), 3);

        let indexes = catalog.indexes_for_label_property(person_id, name_id);
        assert_eq!(indexes.len(), 3);
        assert!(indexes.contains(&hash_idx));
        assert!(indexes.contains(&btree_idx));
        assert!(indexes.contains(&fulltext_idx));

        // Verify each has the correct type
        assert_eq!(
            catalog.get_index(hash_idx).unwrap().index_type,
            IndexType::Hash
        );
        assert_eq!(
            catalog.get_index(btree_idx).unwrap().index_type,
            IndexType::BTree
        );
        assert_eq!(
            catalog.get_index(fulltext_idx).unwrap().index_type,
            IndexType::FullText
        );
    }

    #[test]
    fn test_catalog_schema_required_property_duplicate() {
        let catalog = Catalog::with_schema();

        let person_id = catalog.get_or_create_label("Person");
        let name_id = catalog.get_or_create_property_key("name");

        // First should succeed
        assert!(catalog.add_required_property(person_id, name_id).is_ok());

        // Duplicate should fail
        assert_eq!(
            catalog.add_required_property(person_id, name_id),
            Err(CatalogError::ConstraintAlreadyExists)
        );
    }

    #[test]
    fn test_catalog_schema_check_without_constraints() {
        let catalog = Catalog::new();

        let person_id = catalog.get_or_create_label("Person");
        let name_id = catalog.get_or_create_property_key("name");

        // Without schema enabled, these should return false
        assert!(!catalog.is_property_unique(person_id, name_id));
        assert!(!catalog.is_property_required(person_id, name_id));
    }

    #[test]
    fn test_catalog_has_schema() {
        // Both new() and with_schema() enable schema by default
        let catalog = Catalog::new();
        assert!(catalog.has_schema());

        let with_schema = Catalog::with_schema();
        assert!(with_schema.has_schema());
    }

    #[test]
    fn test_catalog_error_display() {
        assert_eq!(
            CatalogError::SchemaNotEnabled.to_string(),
            "Schema constraints are not enabled"
        );
        assert_eq!(
            CatalogError::ConstraintAlreadyExists.to_string(),
            "Constraint already exists"
        );
        assert_eq!(
            CatalogError::LabelNotFound("Person".to_string()).to_string(),
            "Label not found: Person"
        );
        assert_eq!(
            CatalogError::PropertyKeyNotFound("name".to_string()).to_string(),
            "Property key not found: name"
        );
        assert_eq!(
            CatalogError::EdgeTypeNotFound("KNOWS".to_string()).to_string(),
            "Edge type not found: KNOWS"
        );
        let idx = IndexId::new(42);
        assert!(CatalogError::IndexNotFound(idx).to_string().contains("42"));
    }

    #[test]
    fn test_catalog_concurrent_label_creation() {
        use std::sync::Arc;

        let catalog = Arc::new(Catalog::new());
        let mut handles = vec![];

        // Spawn multiple threads trying to create the same labels
        for i in 0..10 {
            let catalog = Arc::clone(&catalog);
            handles.push(thread::spawn(move || {
                let label_name = format!("Label{}", i % 3); // Only 3 unique labels
                catalog.get_or_create_label(&label_name)
            }));
        }

        let mut ids: Vec<LabelId> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        ids.sort_by_key(|id| id.as_u32());
        ids.dedup();

        // Should only have 3 unique label IDs
        assert_eq!(ids.len(), 3);
        assert_eq!(catalog.label_count(), 3);
    }

    #[test]
    fn test_catalog_concurrent_property_key_creation() {
        use std::sync::Arc;

        let catalog = Arc::new(Catalog::new());
        let mut handles = vec![];

        for i in 0..10 {
            let catalog = Arc::clone(&catalog);
            handles.push(thread::spawn(move || {
                let key_name = format!("key{}", i % 4);
                catalog.get_or_create_property_key(&key_name)
            }));
        }

        let mut ids: Vec<PropertyKeyId> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        ids.sort_by_key(|id| id.as_u32());
        ids.dedup();

        assert_eq!(ids.len(), 4);
        assert_eq!(catalog.property_key_count(), 4);
    }

    #[test]
    fn test_catalog_concurrent_index_operations() {
        use std::sync::Arc;

        let catalog = Arc::new(Catalog::new());
        let label = catalog.get_or_create_label("Node");

        let mut handles = vec![];

        // Create indexes concurrently
        for i in 0..5 {
            let catalog = Arc::clone(&catalog);
            handles.push(thread::spawn(move || {
                let prop = PropertyKeyId::new(i);
                catalog.create_index(&format!("idx_{i}"), label, prop, IndexType::Hash)
            }));
        }

        let ids: Vec<IndexId> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(ids.len(), 5);
        assert_eq!(catalog.index_count(), 5);
    }

    #[test]
    fn test_catalog_special_characters_in_names() {
        let catalog = Catalog::new();

        // Test with various special characters
        let label1 = catalog.get_or_create_label("Label With Spaces");
        let label2 = catalog.get_or_create_label("Label-With-Dashes");
        let label3 = catalog.get_or_create_label("Label_With_Underscores");
        let label4 = catalog.get_or_create_label("LabelWithUnicode\u{00E9}");

        assert_ne!(label1, label2);
        assert_ne!(label2, label3);
        assert_ne!(label3, label4);

        assert_eq!(
            catalog.get_label_name(label1).as_deref(),
            Some("Label With Spaces")
        );
        assert_eq!(
            catalog.get_label_name(label4).as_deref(),
            Some("LabelWithUnicode\u{00E9}")
        );
    }

    #[test]
    fn test_catalog_empty_names() {
        let catalog = Catalog::new();

        // Empty names should be valid (edge case)
        let empty_label = catalog.get_or_create_label("");
        let empty_prop = catalog.get_or_create_property_key("");
        let empty_edge = catalog.get_or_create_edge_type("");

        assert_eq!(catalog.get_label_name(empty_label).as_deref(), Some(""));
        assert_eq!(
            catalog.get_property_key_name(empty_prop).as_deref(),
            Some("")
        );
        assert_eq!(catalog.get_edge_type_name(empty_edge).as_deref(), Some(""));

        // Calling again should return same ID
        assert_eq!(catalog.get_or_create_label(""), empty_label);
    }

    #[test]
    fn test_catalog_large_number_of_entries() {
        let catalog = Catalog::new();

        // Create many labels
        for i in 0..1000 {
            catalog.get_or_create_label(&format!("Label{}", i));
        }

        assert_eq!(catalog.label_count(), 1000);

        // Verify we can retrieve them all
        let all = catalog.all_labels();
        assert_eq!(all.len(), 1000);

        // Verify a specific one
        let id = catalog.get_label_id("Label500").unwrap();
        assert_eq!(catalog.get_label_name(id).as_deref(), Some("Label500"));
    }

    #[test]
    fn test_index_definition_debug() {
        let def = IndexDefinition {
            id: IndexId::new(1),
            name: "test_index".to_string(),
            label: LabelId::new(2),
            property_key: PropertyKeyId::new(3),
            index_type: IndexType::Hash,
        };

        // Should be able to debug print
        let debug_str = format!("{:?}", def);
        assert!(debug_str.contains("IndexDefinition"));
        assert!(debug_str.contains("Hash"));
    }

    #[test]
    fn test_index_type_equality() {
        assert_eq!(IndexType::Hash, IndexType::Hash);
        assert_ne!(IndexType::Hash, IndexType::BTree);
        assert_ne!(IndexType::BTree, IndexType::FullText);

        // Clone
        let t = IndexType::Hash;
        let t2 = t;
        assert_eq!(t, t2);
    }

    #[test]
    fn test_catalog_error_equality() {
        assert_eq!(
            CatalogError::SchemaNotEnabled,
            CatalogError::SchemaNotEnabled
        );
        assert_eq!(
            CatalogError::ConstraintAlreadyExists,
            CatalogError::ConstraintAlreadyExists
        );
        assert_eq!(
            CatalogError::LabelNotFound("X".to_string()),
            CatalogError::LabelNotFound("X".to_string())
        );
        assert_ne!(
            CatalogError::LabelNotFound("X".to_string()),
            CatalogError::LabelNotFound("Y".to_string())
        );
    }

    /// `SHOW` prints a property type with `Display` and WAL replay reads the
    /// statement's spelling with `from_type_name`: every type, nested in
    /// lists too, reads back as itself (#569).
    #[test]
    fn every_property_type_reads_back_from_its_name() {
        use PropertyDataType as T;

        let list = |element: T| T::ListTyped(Box::new(element));
        let types = [
            T::String,
            T::Int64,
            T::Float64,
            T::Bool,
            T::Date,
            T::Time,
            T::Timestamp,
            T::Duration,
            T::List,
            list(T::Int64),
            T::Map,
            T::Bytes,
            T::Node,
            T::Edge,
            T::Any,
            T::ZonedDatetime,
            T::LocalDatetime,
            list(T::ZonedDatetime),
            list(list(T::LocalDatetime)),
        ];
        for data_type in types {
            let name = data_type.to_string();
            assert_eq!(T::from_type_name(&name), Ok(data_type.clone()), "{name}");
            assert_eq!(
                T::from_type_name(&name.to_lowercase()),
                Ok(data_type),
                "{name} in lowercase"
            );
        }
        assert_eq!(T::ZonedDatetime.to_string(), "ZONED DATETIME");
        assert_eq!(T::LocalDatetime.to_string(), "LOCAL DATETIME");
        for spelling in ["zoned_datetime", "ZonedDateTime"] {
            assert_eq!(
                T::from_type_name(spelling),
                Ok(T::ZonedDatetime),
                "{spelling}"
            );
        }
        for spelling in ["local_datetime", "LocalDateTime"] {
            assert_eq!(
                T::from_type_name(spelling),
                Ok(T::LocalDatetime),
                "{spelling}"
            );
        }
        assert_eq!(
            T::from_type_name("DATETIME"),
            Ok(T::Timestamp),
            "a plain DATETIME stays a TIMESTAMP"
        );
    }

    /// Type DDL takes the names of `PROPERTY_TYPE_NAMES` (grafeo-adapters)
    /// and refuses every other one. The catalog reads each of them as a type
    /// of the kind the parsers give it, so a `DEFAULT` a parser lets through
    /// is a value of the type, and `ANY` only where the parsers say `ANY`
    /// (an unknown name reads as `ANY` too); every name `SHOW` lists is one
    /// of them.
    #[test]
    fn type_ddl_names_are_the_catalog_types() {
        use PropertyDataType as T;
        use grafeo_adapters::query::schema::{
            PROPERTY_TYPE_NAMES, PropertyTypeKind as K, property_type_kind,
        };

        for (name, kind) in PROPERTY_TYPE_NAMES {
            let data_type = T::from_type_name(name).unwrap();
            let read_kind = match data_type {
                T::String => K::String,
                T::Int64 => K::Integer,
                T::Float64 => K::Float,
                T::Bool => K::Boolean,
                T::Any => K::Any,
                _ => K::Other,
            };
            assert_eq!(*kind, read_kind, "{name} reads as {data_type}");
        }
        let listed = [
            T::String,
            T::Int64,
            T::Float64,
            T::Bool,
            T::Date,
            T::Time,
            T::Timestamp,
            T::Duration,
            T::List,
            T::Map,
            T::Bytes,
            T::Node,
            T::Edge,
            T::Any,
            T::ZonedDatetime,
            T::LocalDatetime,
        ];
        for data_type in listed {
            assert!(
                property_type_kind(&data_type.to_string()).is_some(),
                "type DDL refuses {data_type}, which SHOW lists"
            );
        }
    }

    /// A type name nests at most 128 `LIST<...>` levels; a deeper one, however
    /// deep, is refused with the limit named, without a stack overflow (the
    /// levels are counted, not recursed into).
    #[test]
    fn property_type_names_nest_at_most_128_lists() {
        let nested = |levels: usize| {
            format!(
                "{}local datetime{}",
                "list<".repeat(levels),
                ">".repeat(levels)
            )
        };
        let mut expected = PropertyDataType::LocalDatetime;
        for _ in 0..128 {
            expected = PropertyDataType::ListTyped(Box::new(expected));
        }
        assert_eq!(PropertyDataType::MAX_LIST_DEPTH, 128);
        assert_eq!(PropertyDataType::from_type_name(&nested(128)), Ok(expected));

        for levels in [129, 100_000] {
            let error = PropertyDataType::from_type_name(&nested(levels)).unwrap_err();
            assert_eq!(
                error,
                CatalogError::PropertyTypeTooDeep { limit: 128 },
                "{levels} levels"
            );
            assert_eq!(
                error.to_string(),
                "A property type nests at most 128 LIST<...> levels"
            );
        }
    }

    /// `ZONED DATETIME` holds only zoned datetimes and `LOCAL DATETIME` only
    /// local ones (`Value::Timestamp`); both take a null, as every type does.
    #[test]
    fn zoned_and_local_datetimes_match_only_their_own_values() {
        use grafeo_common::types::{Timestamp, ZonedDatetime};

        let zoned =
            Value::ZonedDatetime(ZonedDatetime::parse("2026-10-05T10:30:00+02:00").unwrap());
        let local = Value::Timestamp(Timestamp::from_secs(1_791_000_000));
        let zoned_type = PropertyDataType::ZonedDatetime;
        let local_type = PropertyDataType::LocalDatetime;

        assert!(zoned_type.matches(&zoned));
        assert!(!zoned_type.matches(&local), "a local datetime is not zoned");
        assert!(!zoned_type.matches(&Value::from("2026-10-05T10:30:00+02:00")));
        assert!(local_type.matches(&local));
        assert!(!local_type.matches(&zoned), "a zoned datetime is not local");
        assert!(!local_type.matches(&Value::Int64(88)));
        assert!(zoned_type.matches(&Value::Null) && local_type.matches(&Value::Null));

        let zoned_list = PropertyDataType::ListTyped(Box::new(zoned_type));
        assert!(zoned_list.matches(&Value::List(vec![zoned.clone(), Value::Null].into())));
        assert!(!zoned_list.matches(&Value::List(vec![zoned, local].into())));
    }

    /// `MAP` (and its spelling `RECORD`) holds every map, the empty one and
    /// nested ones included, and nothing else; a `LIST<MAP>` holds lists of
    /// maps.
    #[test]
    fn a_map_type_matches_maps_only() {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;

        let map = |entries: &[(&str, Value)]| {
            Value::Map(std::sync::Arc::new(
                entries
                    .iter()
                    .map(|(key, value)| (PropertyKey::from(*key), value.clone()))
                    .collect::<BTreeMap<_, _>>(),
            ))
        };
        let settings = map(&[("mode", Value::from("fast")), ("level", Value::Int64(3))]);
        let nested = map(&[("inner", settings.clone())]);
        for name in ["MAP", "map", "RECORD"] {
            let map_type = PropertyDataType::from_type_name(name).unwrap();
            assert_eq!(map_type, PropertyDataType::Map, "{name}");
            for value in [&settings, &nested, &map(&[]), &Value::Null] {
                assert!(map_type.matches(value), "{name} takes {value:?}");
            }
            for value in [
                Value::from("fast"),
                Value::Int64(19),
                Value::List(vec![settings.clone()].into()),
            ] {
                assert!(!map_type.matches(&value), "{name} refuses {value:?}");
            }
        }
        let maps = PropertyDataType::from_type_name("LIST<MAP>").unwrap();
        assert!(maps.matches(&Value::List(vec![settings.clone(), nested].into())));
        assert!(!maps.matches(&Value::List(vec![settings, Value::Int64(88)].into())));
    }
}
