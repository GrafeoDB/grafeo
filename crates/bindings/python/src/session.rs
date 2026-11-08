//! `db.session(read_only=False, external_access=True)`: a session whose limits bind every
//! statement run through it, its auto-commit statements, its transactions and its lazy results
//! alike.

use std::sync::Arc;

use grafeo_engine::auth::{Identity, Role};
use grafeo_engine::database::GrafeoDB;
use parking_lot::RwLock;
use pyo3::prelude::*;

use crate::database::{PyTransaction, extract_isolation_level};
use crate::error::PyGrafeoError;
use crate::gil::without_gil;
use crate::query::PyQueryResult;

/// A session on a database: its statements run with the limits it was opened with.
///
/// `read_only` refuses writes. `external_access=False` refuses every statement that reads or
/// writes outside the database (`LOAD DATA`, `LOAD CSV`, the load procedures, SPARQL `LOAD`),
/// before any file is opened. The two are independent: `read_only=True` keeps file access.
/// Nothing run on the session can lift a limit. Use it in a `with` block, or call `close()`.
///
/// Example:
///     with db.session(external_access=False) as session:
///         session.execute("MATCH (p:Person) RETURN p.name")
#[pyclass(name = "Session")]
pub struct PySession {
    db: Arc<RwLock<GrafeoDB>>,
    /// What the session may do: each transaction it begins gets the same.
    identity: Identity,
    /// The session's statements run here; `None` once closed.
    session: parking_lot::Mutex<Option<grafeo_engine::session::Session>>,
}

impl PySession {
    /// A session on `db` with the given limits.
    pub(crate) fn open(db: Arc<RwLock<GrafeoDB>>, read_only: bool, external_access: bool) -> Self {
        // Explicit, so the arguments decide, not the role's default.
        let identity = if read_only {
            Identity::new("anonymous", [Role::ReadOnly])
        } else {
            Identity::anonymous()
        }
        .with_external_access(external_access);
        let session = db.read().session_with_identity(identity.clone());
        Self {
            db,
            identity,
            session: parking_lot::Mutex::new(Some(session)),
        }
    }

    fn closed() -> PyGrafeoError {
        PyGrafeoError::database("the session is closed")
    }

    fn run(
        &self,
        py: Python<'_>,
        language: &str,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyQueryResult> {
        let params = crate::direct::params(params)?;
        let session = &self.session;
        let result = without_gil(py, || match session.lock().as_ref() {
            Some(session) => crate::direct::run(session, language, query, params),
            None => Err(Self::closed()),
        });
        crate::direct::surfaced(py, result)
    }
}

#[pymethods]
impl PySession {
    /// Runs a GQL statement on the session.
    #[pyo3(signature = (query, params=None))]
    fn execute(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        self.run(py, "gql", query, params)
    }

    /// Runs a Cypher statement on the session.
    #[cfg(feature = "cypher")]
    #[pyo3(signature = (query, params=None))]
    fn execute_cypher(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        self.run(py, "cypher", query, params)
    }

    /// Runs a SQL/PGQ statement on the session.
    #[cfg(feature = "sql-pgq")]
    #[pyo3(signature = (query, params=None))]
    fn execute_sql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        self.run(py, "sql", query, params)
    }

    /// Runs a Gremlin statement on the session.
    #[cfg(feature = "gremlin")]
    #[pyo3(signature = (query, params=None))]
    fn execute_gremlin(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        self.run(py, "gremlin", query, params)
    }

    /// Runs a GraphQL statement on the session.
    #[cfg(feature = "graphql")]
    #[pyo3(signature = (query, params=None))]
    fn execute_graphql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        self.run(py, "graphql", query, params)
    }

    /// Runs a SPARQL statement on the session.
    #[cfg(feature = "sparql")]
    #[pyo3(signature = (query, params=None))]
    fn execute_sparql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        self.run(py, "sparql", query, params)
    }

    /// Runs a statement in a named language (e.g. `"graphql-rdf"`) on the session.
    #[pyo3(signature = (language, query, params=None))]
    fn execute_language(
        &self,
        language: &str,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        self.run(py, language, query, params)
    }

    /// Runs a read-only GQL query on the session and returns a lazy iterator, as
    /// `GrafeoDB.execute_lazy` does.
    #[cfg(feature = "gql")]
    fn execute_lazy(&self, query: &str, py: Python<'_>) -> PyResult<crate::stream::PyResultStream> {
        let session = &self.session;
        let stream = without_gil(py, || match session.lock().as_ref() {
            Some(session) => Ok(session.execute_streaming_owned(query)?),
            None => Err(Self::closed()),
        })?;
        let mut stream = crate::stream::PyResultStream::new(Arc::clone(&self.db), stream);
        stream.warn_new(py)?;
        Ok(stream)
    }

    /// Begins a transaction with the session's limits, used as one from
    /// `GrafeoDB.begin_transaction` (the isolation level as there).
    #[pyo3(signature = (isolation_level=None))]
    fn begin_transaction(
        &self,
        isolation_level: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyTransaction> {
        if self.session.lock().is_none() {
            return Err(Self::closed().into());
        }
        let level = extract_isolation_level(isolation_level)?;
        let session = self.db.read().session_with_identity(self.identity.clone());
        PyTransaction::begin(Arc::clone(&self.db), session, level.as_deref())
    }

    /// Closes the session: its statements raise from now on; the database and other sessions are
    /// unaffected. Closing a closed session does nothing.
    fn close(&self) {
        drop(self.session.lock().take());
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.close();
        false
    }
}
