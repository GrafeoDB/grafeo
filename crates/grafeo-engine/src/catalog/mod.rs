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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::{Mutex, RwLock};

use grafeo_common::collections::{GrafeoConcurrentMap, grafeo_concurrent_map};
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

    /// Checks if a property is required for a label.
    #[must_use]
    pub fn is_property_required(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.schema
            .as_ref()
            .is_some_and(|s| s.is_property_required(label, property_key))
    }

    /// Checks if a property must be unique for a label.
    #[must_use]
    pub fn is_property_unique(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.schema
            .as_ref()
            .is_some_and(|s| s.is_property_unique(label, property_key))
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
    /// Returns `CatalogError::SchemaNotEnabled` if schema is disabled.
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
        let schema = self.schema.as_ref().ok_or(CatalogError::SchemaNotEnabled)?;
        schema.insert_constraint(def.clone())?;
        self.mark_constrained_properties(schema, &def);
        for constraint in def.type_constraints() {
            schema.add_constraint_to_type(&def.label, constraint)?;
        }
        Ok(())
    }

    /// Drops a named constraint and the type constraints it added.
    /// `DROP CONSTRAINT` and WAL replay both call this.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::ConstraintNotFound` if no constraint has this name.
    pub fn drop_constraint(&self, name: &str) -> Result<(), CatalogError> {
        let schema = self.schema.as_ref().ok_or(CatalogError::SchemaNotEnabled)?;
        let def = schema.remove_constraint(name)?;
        for constraint in def.type_constraints() {
            schema.remove_constraint_from_type(&def.label, &constraint);
        }
        // Keep the unique and required markers that another constraint on
        // the same label and property still needs.
        let remaining = schema.all_constraints();
        let label = self.get_or_create_label(&def.label);
        for property in &def.properties {
            let key = (label, self.get_or_create_property_key(property));
            let covers = |check: fn(ConstraintType) -> bool| {
                remaining.iter().any(|other| {
                    other.label == def.label
                        && check(other.kind)
                        && other.properties.contains(property)
                })
            };
            if def.kind.is_unique() && !covers(ConstraintType::is_unique) {
                schema.unique_constraints.write().remove(&key);
            }
            if def.kind.is_required() && !covers(ConstraintType::is_required) {
                schema.required_properties.write().remove(&key);
            }
        }
        Ok(())
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
            for def in constraints {
                self.mark_constrained_properties(schema, &def);
                schema.constraints.write().insert(def.name.clone(), def);
            }
        }
    }

    /// Marks the label's properties unique or required for
    /// [`is_property_unique`](Self::is_property_unique) and
    /// [`is_property_required`](Self::is_property_required).
    fn mark_constrained_properties(&self, schema: &SchemaCatalog, def: &ConstraintDefinition) {
        let label = self.get_or_create_label(&def.label);
        for property in &def.properties {
            let key = (label, self.get_or_create_property_key(property));
            if def.kind.is_unique() {
                schema.unique_constraints.write().insert(key);
            }
            if def.kind.is_required() {
                schema.required_properties.write().insert(key);
            }
        }
    }

    /// Adds a property to a node type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::SchemaNotEnabled` if schema is disabled.
    /// * `CatalogError::TypeNotFound` if the node type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone)]
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
}

impl PropertyDataType {
    /// Parses a type name string (case-insensitive) into a `PropertyDataType`.
    #[must_use]
    pub fn from_type_name(name: &str) -> Self {
        let upper = name.to_uppercase();
        // Handle parameterized LIST<element_type>
        if let Some(inner) = upper
            .strip_prefix("LIST<")
            .and_then(|s| s.strip_suffix('>'))
        {
            return Self::ListTyped(Box::new(Self::from_type_name(inner)));
        }
        match upper.as_str() {
            "STRING" | "VARCHAR" | "TEXT" => Self::String,
            "INT" | "INT64" | "INTEGER" | "BIGINT" => Self::Int64,
            "FLOAT" | "FLOAT64" | "DOUBLE" | "REAL" => Self::Float64,
            "BOOL" | "BOOLEAN" => Self::Bool,
            "DATE" => Self::Date,
            "TIME" => Self::Time,
            "TIMESTAMP" | "DATETIME" => Self::Timestamp,
            "DURATION" | "INTERVAL" => Self::Duration,
            "LIST" | "ARRAY" => Self::List,
            "MAP" | "RECORD" => Self::Map,
            "BYTES" | "BINARY" | "BLOB" => Self::Bytes,
            "NODE" => Self::Node,
            "EDGE" | "RELATIONSHIP" => Self::Edge,
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
            (Self::Timestamp, Value::Timestamp(_)) => true,
            (Self::Duration, Value::Duration(_)) => true,
            (Self::List, Value::List(_)) => true,
            (Self::ListTyped(elem_type), Value::List(items)) => {
                items.iter().all(|item| elem_type.matches(item))
            }
            (Self::Bytes, Value::Bytes(_)) => true,
            // Node/Edge reference types match Map values (graph elements are
            // represented as maps with _id, _labels/_type, and properties)
            (Self::Node | Self::Edge, Value::Map(_)) => true,
            _ => false,
        }
    }
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
        }
    }

    // --- Node type operations ---

    /// Registers a new node type definition.
    ///
    /// # Errors
    ///
    /// Returns `CatalogError::TypeAlreadyExists` if a type with the same name exists.
    pub fn register_node_type(&self, def: NodeTypeDefinition) -> Result<(), CatalogError> {
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
            if visited.insert(parent.clone()) {
                Self::collect_inherited(types, parent, visited, properties, constraints);
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
    /// Returns `CatalogError::TypeAlreadyExists` if an edge type with the same name exists.
    pub fn register_edge_type(&self, def: EdgeTypeDefinition) -> Result<(), CatalogError> {
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
    /// # Errors
    ///
    /// Returns `CatalogError::TypeAlreadyExists` if a graph type with the same name exists.
    pub fn register_graph_type(&self, def: GraphTypeDefinition) -> Result<(), CatalogError> {
        let mut types = self.graph_types.write();
        if types.contains_key(&def.name) {
            return Err(CatalogError::TypeAlreadyExists(def.name));
        }
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
    /// Currently infallible, but returns `Result` for forward compatibility.
    pub fn add_constraint_to_type(
        &self,
        label: &str,
        constraint: TypeConstraint,
    ) -> Result<(), CatalogError> {
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
    pub fn alter_node_type_add_property(
        &self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        let mut types = self.node_types.write();
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        if def.properties.iter().any(|p| p.name == property.name) {
            return Err(CatalogError::TypeAlreadyExists(format!(
                "property {} on {}",
                property.name, type_name
            )));
        }
        def.properties.push(property);
        Ok(())
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
        let len_before = def.properties.len();
        def.properties.retain(|p| p.name != property_name);
        if def.properties.len() == len_before {
            return Err(CatalogError::TypeNotFound(format!(
                "property {} on {}",
                property_name, type_name
            )));
        }
        Ok(())
    }

    /// Adds a property to an existing edge type.
    ///
    /// # Errors
    ///
    /// * `CatalogError::TypeNotFound` if the edge type does not exist.
    /// * `CatalogError::TypeAlreadyExists` if the property already exists on the type.
    pub fn alter_edge_type_add_property(
        &self,
        type_name: &str,
        property: TypedProperty,
    ) -> Result<(), CatalogError> {
        let mut types = self.edge_types.write();
        let def = types
            .get_mut(type_name)
            .ok_or_else(|| CatalogError::TypeNotFound(type_name.to_string()))?;
        if def.properties.iter().any(|p| p.name == property.name) {
            return Err(CatalogError::TypeAlreadyExists(format!(
                "property {} on {}",
                property.name, type_name
            )));
        }
        def.properties.push(property);
        Ok(())
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
        let len_before = def.properties.len();
        def.properties.retain(|p| p.name != property_name);
        if def.properties.len() == len_before {
            return Err(CatalogError::TypeNotFound(format!(
                "property {} on {}",
                property_name, type_name
            )));
        }
        Ok(())
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

    /// Registers a constraint name, failing when it is taken.
    fn insert_constraint(&self, def: ConstraintDefinition) -> Result<(), CatalogError> {
        let mut constraints = self.constraints.write();
        if constraints.contains_key(&def.name) {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        constraints.insert(def.name.clone(), def);
        Ok(())
    }

    fn remove_constraint(&self, name: &str) -> Result<ConstraintDefinition, CatalogError> {
        self.constraints
            .write()
            .remove(name)
            .ok_or_else(|| CatalogError::ConstraintNotFound(name.to_string()))
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
    /// Optional graph name for graph-type-bound validation.
    graph_name: Option<String>,
    /// Optional graph store for UNIQUE constraint enforcement via index lookup.
    store: Option<Arc<dyn grafeo_core::graph::GraphStore>>,
    /// Optional maximum property value size in bytes.
    max_property_size: Option<usize>,
}

impl CatalogConstraintValidator {
    /// Creates a new validator wrapping the given catalog.
    pub fn new(catalog: Arc<Catalog>) -> Self {
        Self {
            catalog,
            graph_name: None,
            store: None,
            max_property_size: None,
        }
    }

    /// Sets the graph name for graph-type-bound validation.
    pub fn with_graph_name(mut self, name: String) -> Self {
        self.graph_name = Some(name);
        self
    }

    /// Attaches a graph store for UNIQUE constraint enforcement.
    pub fn with_store(mut self, store: Arc<dyn grafeo_core::graph::GraphStore>) -> Self {
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
        if let Some(limit) = self.max_property_size {
            let size = value.estimated_size_bytes();
            if size > limit {
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' value exceeds maximum size of {} MiB ({size} bytes)",
                    limit / (1024 * 1024)
                )));
            }
        }
        for label in labels {
            if let Some(type_def) = self.catalog.resolved_node_type(label)
                && let Some(typed_prop) = type_def.properties.iter().find(|p| p.name == key)
            {
                // Check NOT NULL
                if !typed_prop.nullable && *value == Value::Null {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "property '{key}' on :{label} is NOT NULL, cannot set to null"
                    )));
                }
                // Check type compatibility
                if *value != Value::Null && !typed_prop.data_type.matches(value) {
                    return Err(OperatorError::ConstraintViolation(format!(
                        "property '{key}' on :{label} expects {:?}, got {:?}",
                        typed_prop.data_type, value
                    )));
                }
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
            if let Some(type_def) = self.catalog.resolved_node_type(label) {
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
                            match check_eval::evaluate_check(expression, properties) {
                                Ok(true) => {}
                                Ok(false) => {
                                    let constraint_name = name.as_deref().unwrap_or("unnamed");
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "CHECK constraint '{constraint_name}' violated on :{label}"
                                    )));
                                }
                                Err(err) => {
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "CHECK constraint evaluation error: {err}"
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
            if let Some(type_def) = self.catalog.resolved_node_type(label) {
                for constraint in &type_def.constraints {
                    let is_unique = match constraint {
                        TypeConstraint::Unique(props) => props.iter().any(|p| p == key),
                        TypeConstraint::PrimaryKey(props) => props.iter().any(|p| p == key),
                        _ => false,
                    };
                    if is_unique && let Some(ref store) = self.store {
                        let existing = store.find_nodes_by_property(key, value);
                        for node_id in existing {
                            if let Some(node) = store.get_node(node_id) {
                                let has_label = node.labels.iter().any(|l| l.as_str() == label);
                                if has_label {
                                    return Err(OperatorError::ConstraintViolation(format!(
                                        "UNIQUE constraint violation: property '{key}' \
                                             with value {value:?} already exists on :{label}"
                                    )));
                                }
                            }
                        }
                    }
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
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' value exceeds maximum size of {} MiB ({size} bytes)",
                    limit / (1024 * 1024)
                )));
            }
        }
        if let Some(type_def) = self.catalog.get_edge_type_def(edge_type)
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
                return Err(OperatorError::ConstraintViolation(format!(
                    "property '{key}' on :{edge_type} expects {:?}, got {:?}",
                    typed_prop.data_type, value
                )));
            }
        }
        Ok(())
    }

    fn validate_edge_complete(
        &self,
        edge_type: &str,
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError> {
        if let Some(type_def) = self.catalog.get_edge_type_def(edge_type) {
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
                    match check_eval::evaluate_check(expression, properties) {
                        Ok(true) => {}
                        Ok(false) => {
                            let constraint_name = name.as_deref().unwrap_or("unnamed");
                            return Err(OperatorError::ConstraintViolation(format!(
                                "CHECK constraint '{constraint_name}' violated on :{edge_type}"
                            )));
                        }
                        Err(err) => {
                            return Err(OperatorError::ConstraintViolation(format!(
                                "CHECK constraint evaluation error: {err}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_node_labels_allowed(&self, labels: &[String]) -> Result<(), OperatorError> {
        let Some(ref graph_name) = self.graph_name else {
            return Ok(());
        };
        let Some(type_name) = self.catalog.get_graph_type_binding(graph_name) else {
            return Ok(());
        };
        let Some(gt) = self
            .catalog
            .schema()
            .and_then(|s| s.get_graph_type(&type_name))
        else {
            return Ok(());
        };
        if !gt.open && !gt.allowed_node_types.is_empty() {
            let allowed = labels
                .iter()
                .any(|l| gt.allowed_node_types.iter().any(|a| a == l));
            if !allowed {
                return Err(OperatorError::ConstraintViolation(format!(
                    "node labels {labels:?} are not allowed by graph type '{}'",
                    gt.name
                )));
            }
        }
        Ok(())
    }

    fn validate_edge_type_allowed(&self, edge_type: &str) -> Result<(), OperatorError> {
        let Some(ref graph_name) = self.graph_name else {
            return Ok(());
        };
        let Some(type_name) = self.catalog.get_graph_type_binding(graph_name) else {
            return Ok(());
        };
        let Some(gt) = self
            .catalog
            .schema()
            .and_then(|s| s.get_graph_type(&type_name))
        else {
            return Ok(());
        };
        if !gt.open && !gt.allowed_edge_types.is_empty() {
            let allowed = gt.allowed_edge_types.iter().any(|a| a == edge_type);
            if !allowed {
                return Err(OperatorError::ConstraintViolation(format!(
                    "edge type '{edge_type}' is not allowed by graph type '{}'",
                    gt.name
                )));
            }
        }
        Ok(())
    }

    fn validate_edge_endpoints(
        &self,
        edge_type: &str,
        source_labels: &[String],
        target_labels: &[String],
    ) -> Result<(), OperatorError> {
        let Some(type_def) = self.catalog.get_edge_type_def(edge_type) else {
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

    fn inject_defaults(&self, labels: &[String], properties: &mut Vec<(String, Value)>) {
        for label in labels {
            if let Some(type_def) = self.catalog.resolved_node_type(label) {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Dropping one of two named constraints on a property removes only its
    /// own type constraint, and keeps the unique and required markers that
    /// the other one still needs.
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
                .is_empty()
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

        assert!(catalog.indexes_for_label(invalid_label).is_empty());
        assert!(
            catalog
                .indexes_for_label_property(invalid_label, invalid_property)
                .is_empty()
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
}
