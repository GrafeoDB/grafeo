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

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::RwLock;

use grafeo_common::types::{EdgeTypeId, IndexId, LabelId, PropertyKeyId};

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
    /// Creates a new empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self {
            labels: LabelCatalog::new(),
            property_keys: PropertyCatalog::new(),
            edge_types: EdgeTypeCatalog::new(),
            indexes: IndexCatalog::new(),
            schema: None,
        }
    }

    /// Creates a new catalog with schema constraints enabled.
    #[must_use]
    pub fn with_schema() -> Self {
        Self {
            labels: LabelCatalog::new(),
            property_keys: PropertyCatalog::new(),
            edge_types: EdgeTypeCatalog::new(),
            indexes: IndexCatalog::new(),
            schema: Some(SchemaCatalog::new()),
        }
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
        label: LabelId,
        property_key: PropertyKeyId,
        index_type: IndexType,
    ) -> IndexId {
        self.indexes.create(label, property_key, index_type)
    }

    /// Drops an index by ID.
    pub fn drop_index(&self, id: IndexId) -> bool {
        self.indexes.drop(id)
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
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

// === Label Catalog ===

/// Bidirectional mapping between label names and IDs.
struct LabelCatalog {
    name_to_id: RwLock<HashMap<Arc<str>, LabelId>>,
    id_to_name: RwLock<Vec<Arc<str>>>,
    next_id: AtomicU32,
}

impl LabelCatalog {
    fn new() -> Self {
        Self {
            name_to_id: RwLock::new(HashMap::new()),
            id_to_name: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    fn get_or_create(&self, name: &str) -> LabelId {
        // Fast path: check if already exists
        {
            let name_to_id = self.name_to_id.read();
            if let Some(&id) = name_to_id.get(name) {
                return id;
            }
        }

        // Slow path: create new entry
        let mut name_to_id = self.name_to_id.write();
        let mut id_to_name = self.id_to_name.write();

        // Double-check after acquiring write lock
        if let Some(&id) = name_to_id.get(name) {
            return id;
        }

        let id = LabelId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let name: Arc<str> = name.into();
        name_to_id.insert(Arc::clone(&name), id);
        id_to_name.push(name);
        id
    }

    fn get_id(&self, name: &str) -> Option<LabelId> {
        self.name_to_id.read().get(name).copied()
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
    name_to_id: RwLock<HashMap<Arc<str>, PropertyKeyId>>,
    id_to_name: RwLock<Vec<Arc<str>>>,
    next_id: AtomicU32,
}

impl PropertyCatalog {
    fn new() -> Self {
        Self {
            name_to_id: RwLock::new(HashMap::new()),
            id_to_name: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    fn get_or_create(&self, name: &str) -> PropertyKeyId {
        // Fast path: check if already exists
        {
            let name_to_id = self.name_to_id.read();
            if let Some(&id) = name_to_id.get(name) {
                return id;
            }
        }

        // Slow path: create new entry
        let mut name_to_id = self.name_to_id.write();
        let mut id_to_name = self.id_to_name.write();

        // Double-check after acquiring write lock
        if let Some(&id) = name_to_id.get(name) {
            return id;
        }

        let id = PropertyKeyId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let name: Arc<str> = name.into();
        name_to_id.insert(Arc::clone(&name), id);
        id_to_name.push(name);
        id
    }

    fn get_id(&self, name: &str) -> Option<PropertyKeyId> {
        self.name_to_id.read().get(name).copied()
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
    name_to_id: RwLock<HashMap<Arc<str>, EdgeTypeId>>,
    id_to_name: RwLock<Vec<Arc<str>>>,
    next_id: AtomicU32,
}

impl EdgeTypeCatalog {
    fn new() -> Self {
        Self {
            name_to_id: RwLock::new(HashMap::new()),
            id_to_name: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    fn get_or_create(&self, name: &str) -> EdgeTypeId {
        // Fast path: check if already exists
        {
            let name_to_id = self.name_to_id.read();
            if let Some(&id) = name_to_id.get(name) {
                return id;
            }
        }

        // Slow path: create new entry
        let mut name_to_id = self.name_to_id.write();
        let mut id_to_name = self.id_to_name.write();

        // Double-check after acquiring write lock
        if let Some(&id) = name_to_id.get(name) {
            return id;
        }

        let id = EdgeTypeId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let name: Arc<str> = name.into();
        name_to_id.insert(Arc::clone(&name), id);
        id_to_name.push(name);
        id
    }

    fn get_id(&self, name: &str) -> Option<EdgeTypeId> {
        self.name_to_id.read().get(name).copied()
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
    next_id: AtomicU32,
}

impl IndexCatalog {
    fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            label_indexes: RwLock::new(HashMap::new()),
            label_property_indexes: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(0),
        }
    }

    fn create(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
        index_type: IndexType,
    ) -> IndexId {
        let id = IndexId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let definition = IndexDefinition {
            id,
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
            true
        } else {
            false
        }
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
}

// === Schema Catalog ===

/// Schema constraints.
struct SchemaCatalog {
    /// Properties that must be unique for a given label.
    unique_constraints: RwLock<HashMap<(LabelId, PropertyKeyId), ()>>,
    /// Properties that are required (NOT NULL) for a given label.
    required_properties: RwLock<HashMap<(LabelId, PropertyKeyId), ()>>,
}

impl SchemaCatalog {
    fn new() -> Self {
        Self {
            unique_constraints: RwLock::new(HashMap::new()),
            required_properties: RwLock::new(HashMap::new()),
        }
    }

    fn add_unique_constraint(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        let mut constraints = self.unique_constraints.write();
        let key = (label, property_key);
        if constraints.contains_key(&key) {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        constraints.insert(key, ());
        Ok(())
    }

    fn add_required_property(
        &self,
        label: LabelId,
        property_key: PropertyKeyId,
    ) -> Result<(), CatalogError> {
        let mut required = self.required_properties.write();
        let key = (label, property_key);
        if required.contains_key(&key) {
            return Err(CatalogError::ConstraintAlreadyExists);
        }
        required.insert(key, ());
        Ok(())
    }

    fn is_property_required(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.required_properties
            .read()
            .contains_key(&(label, property_key))
    }

    fn is_property_unique(&self, label: LabelId, property_key: PropertyKeyId) -> bool {
        self.unique_constraints
            .read()
            .contains_key(&(label, property_key))
    }
}

// === Errors ===

/// Catalog-related errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogError {
    /// Schema constraints are not enabled.
    SchemaNotEnabled,
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
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaNotEnabled => write!(f, "Schema constraints are not enabled"),
            Self::ConstraintAlreadyExists => write!(f, "Constraint already exists"),
            Self::LabelNotFound(name) => write!(f, "Label not found: {name}"),
            Self::PropertyKeyNotFound(name) => write!(f, "Property key not found: {name}"),
            Self::EdgeTypeNotFound(name) => write!(f, "Edge type not found: {name}"),
            Self::IndexNotFound(id) => write!(f, "Index not found: {id}"),
        }
    }
}

impl std::error::Error for CatalogError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

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
        let idx1 = catalog.create_index(person_id, name_id, IndexType::Hash);
        let idx2 = catalog.create_index(person_id, age_id, IndexType::BTree);

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
    fn test_catalog_no_schema() {
        let catalog = Catalog::new();

        let person_id = catalog.get_or_create_label("Person");
        let email_id = catalog.get_or_create_property_key("email");

        // Should fail without schema
        assert_eq!(
            catalog.add_unique_constraint(person_id, email_id),
            Err(CatalogError::SchemaNotEnabled)
        );
    }

    // === Additional tests for comprehensive coverage ===

    #[test]
    fn test_catalog_default() {
        let catalog = Catalog::default();
        assert!(!catalog.has_schema());
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
        let hash_idx = catalog.create_index(person_id, name_id, IndexType::Hash);
        let btree_idx = catalog.create_index(person_id, name_id, IndexType::BTree);
        let fulltext_idx = catalog.create_index(person_id, name_id, IndexType::FullText);

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
        let without_schema = Catalog::new();
        assert!(!without_schema.has_schema());

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
                catalog.create_index(label, prop, IndexType::Hash)
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
