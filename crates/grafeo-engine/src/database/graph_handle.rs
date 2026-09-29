//! Handles on one named graph, for working in several graphs at once.

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

use super::{GrafeoDB, QueryResult};
use crate::session::{Session, graph_storage_key};

/// A handle on one named graph of a database.
///
/// Every call opens a session in the handle's graph, so a handle neither
/// reads nor changes the database's current graph
/// ([`set_current_graph`](GrafeoDB::set_current_graph)): handles on
/// different graphs can be used side by side, and one handle from several
/// threads at once. Each query through [`execute`](Self::execute) and each
/// direct write through [`session`](Self::session) is a transaction of its
/// own; begin a transaction on a session to group several.
///
/// A handle keeps the schema that was current when it was made. Every call
/// fails if the graph no longer exists, instead of working in another graph.
///
/// # Example
///
/// ```
/// # use grafeo_engine::GrafeoDB;
/// # use grafeo_common::types::Value;
/// let db = GrafeoDB::new_in_memory();
/// db.execute("CREATE GRAPH model")?;
/// let model = db.graph("model")?;
///
/// model
///     .session()?
///     .create_node_with_props(&["Component"], [("name", Value::from("Billing"))])?;
///
/// let count = "MATCH (n) RETURN count(n)";
/// assert_eq!(model.execute(count)?.rows()[0][0], Value::Int64(1));
/// assert_eq!(db.execute(count)?.rows()[0][0], Value::Int64(0));
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
#[derive(Clone)]
pub struct GraphHandle<'db> {
    db: &'db GrafeoDB,
    schema: Option<String>,
    name: String,
}

impl std::fmt::Debug for GraphHandle<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphHandle")
            .field("schema", &self.schema)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl GrafeoDB {
    /// Returns a handle on the named graph of the current schema.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph does not exist.
    pub fn graph(&self, name: &str) -> Result<GraphHandle<'_>> {
        let schema = self.current_schema();
        self.graph_in(schema.as_deref(), name)
    }

    /// Returns a handle on the named graph of `schema` (`None`: no schema).
    ///
    /// # Errors
    ///
    /// Returns an error if the graph does not exist.
    pub fn graph_in(&self, schema: Option<&str>, name: &str) -> Result<GraphHandle<'_>> {
        let handle = GraphHandle {
            db: self,
            schema: schema.map(ToString::to_string),
            name: name.to_string(),
        };
        handle.check_exists()?;
        Ok(handle)
    }
}

impl GraphHandle<'_> {
    /// The graph's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The schema the graph belongs to, if any.
    #[must_use]
    pub fn schema(&self) -> Option<&str> {
        self.schema.as_deref()
    }

    /// Opens a session working in this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists.
    pub fn session(&self) -> Result<Session> {
        self.check_exists()?;
        let session = self.db.session();
        match &self.schema {
            Some(schema) => session.set_schema(schema),
            None => session.reset_schema(),
        }
        session.use_graph(&self.name);
        Ok(session)
    }

    /// Runs a GQL query on this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the query fails.
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        self.session()?.execute(query)
    }

    /// Runs a GQL query with parameters on this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or the query fails.
    pub fn execute_with_params(
        &self,
        query: &str,
        params: HashMap<String, Value>,
    ) -> Result<QueryResult> {
        self.session()?.execute_with_params(query, params)
    }

    /// Runs a query in the named language (`"gql"`, `"cypher"`, ...) on this
    /// graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists, the language is not
    /// available or the query fails.
    pub fn execute_language(
        &self,
        query: &str,
        language: &str,
        params: Option<HashMap<String, Value>>,
    ) -> Result<QueryResult> {
        self.session()?.execute_language(query, language, params)
    }

    fn check_exists(&self) -> Result<()> {
        let exists = match graph_storage_key(self.schema.as_deref(), Some(&self.name)) {
            // The default graph always exists.
            None => true,
            Some(_) if self.name.eq_ignore_ascii_case("default") => true,
            Some(key) => self
                .db
                .store
                .as_ref()
                .is_some_and(|store| store.graph(&key).is_some()),
        };
        if exists {
            Ok(())
        } else {
            Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("Graph '{}' does not exist", self.name),
            )))
        }
    }
}
