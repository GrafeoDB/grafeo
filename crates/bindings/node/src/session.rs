//! `db.session({ readOnly, externalAccess })`: a session whose limits bind every statement run
//! through it, its auto-commit statements, its transactions and its streams alike.

use std::sync::Arc;

use grafeo_engine::auth::{Identity, Role};
use grafeo_engine::database::GrafeoDB;
use napi::JsString;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::{Mutex, RwLock};

use crate::error::NodeGrafeoError;
use crate::query::QueryResult;
use crate::transaction::Transaction;
use crate::types;

/// The options `db.session` reads: any other key is refused, never ignored.
const OPTIONS: [&str; 2] = ["readOnly", "externalAccess"];

/// A session on a database: its statements run with the limits it was opened with.
///
/// `readOnly` refuses writes. `externalAccess: false` refuses every statement that reads or
/// writes outside the database (`LOAD DATA`, `LOAD CSV`, the load procedures, SPARQL `LOAD`),
/// before any file is opened. The two are independent: `readOnly: true` keeps file access.
/// Nothing run on the session can lift a limit. `close()` it when done.
#[napi(js_name = "Session")]
pub struct JsSession {
    db: Arc<RwLock<GrafeoDB>>,
    /// What the session may do: each transaction it begins gets the same.
    identity: Identity,
    /// The session's statements run here; `None` once closed.
    session: Arc<Mutex<Option<grafeo_engine::session::Session>>>,
}

/// The error of a statement on a closed session.
fn closed() -> napi::Error {
    NodeGrafeoError::Database("the session is closed".into()).into()
}

impl JsSession {
    /// A session on `db` with the limits of `options` (`{ readOnly, externalAccess }`, either
    /// optional); an unknown key, or a value that is not a boolean, is an error.
    pub(crate) fn open(db: Arc<RwLock<GrafeoDB>>, options: Option<&Object<'_>>) -> Result<Self> {
        let (mut read_only, mut external_access) = (false, true);
        if let Some(options) = options {
            let keys = options.get_property_names()?;
            for i in 0..keys.get_array_length()? {
                let key: JsString = keys.get_element(i)?;
                let key = key.into_utf8()?.into_owned()?;
                let value: Option<bool> = options.get_named_property(&key).map_err(|_| {
                    NodeGrafeoError::InvalidArgument(format!(
                        "session option {key} must be a boolean"
                    ))
                })?;
                match key.as_str() {
                    "readOnly" => read_only = value.unwrap_or(false),
                    "externalAccess" => external_access = value.unwrap_or(true),
                    other => {
                        return Err(NodeGrafeoError::InvalidArgument(format!(
                            "unknown session option {other}: the options are {}",
                            OPTIONS.join(" and ")
                        ))
                        .into());
                    }
                }
            }
        }
        // Explicit, so the options decide, not the role's default.
        let identity = if read_only {
            Identity::new("anonymous", [Role::ReadOnly])
        } else {
            Identity::anonymous()
        }
        .with_external_access(external_access);
        let session = db.read().session_with_identity(identity.clone());
        Ok(Self {
            db,
            identity,
            session: Arc::new(Mutex::new(Some(session))),
        })
    }

    async fn run(
        &self,
        language: &'static str,
        query: String,
        params: Option<types::Params>,
    ) -> Result<QueryResult> {
        let params = params.map(types::Params::into_map).transpose()?;
        let session = Arc::clone(&self.session);
        let result = tokio::task::spawn_blocking(move || match session.lock().as_ref() {
            Some(session) => session
                .execute_language(&query, language, params)
                .map_err(|error| napi::Error::from(NodeGrafeoError::from(error))),
            None => Err(closed()),
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;
        Ok(crate::database::js_result(result))
    }
}

#[napi]
impl JsSession {
    /// Execute a GQL statement on the session. Returns a Promise<QueryResult>.
    #[napi]
    pub async fn execute(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<types::Params>,
    ) -> Result<QueryResult> {
        self.run("gql", query, params).await
    }

    /// Execute a statement in a named language (e.g. `"graphql-rdf"`) on the session.
    #[napi(js_name = "executeLanguage")]
    pub async fn execute_language(
        &self,
        language: String,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<types::Params>,
    ) -> Result<QueryResult> {
        let params = params.map(types::Params::into_map).transpose()?;
        let session = Arc::clone(&self.session);
        let result = tokio::task::spawn_blocking(move || match session.lock().as_ref() {
            Some(session) => session
                .execute_language(&query, &language, params)
                .map_err(|error| napi::Error::from(NodeGrafeoError::from(error))),
            None => Err(closed()),
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;
        Ok(crate::database::js_result(result))
    }

    /// Run a read-only GQL query on the session and return an async cursor, as
    /// `GrafeoDB.executeStream` does.
    #[napi(js_name = "executeStream")]
    pub async fn execute_stream(&self, query: String) -> Result<crate::stream::JsResultStream> {
        let session = Arc::clone(&self.session);
        let stream = tokio::task::spawn_blocking(move || match session.lock().as_ref() {
            Some(session) => session
                .execute_streaming_owned(&query)
                .map_err(|error| napi::Error::from(NodeGrafeoError::from(error))),
            None => Err(closed()),
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;
        Ok(crate::stream::JsResultStream::new(
            Arc::clone(&self.db),
            stream,
        ))
    }

    /// Begin a transaction with the session's limits, used as one from
    /// `GrafeoDB.beginTransaction` (the isolation level as there).
    #[napi(js_name = "beginTransaction")]
    pub fn begin_transaction(&self, isolation_level: Option<String>) -> Result<Transaction> {
        if self.session.lock().is_none() {
            return Err(closed());
        }
        let session = self.db.read().session_with_identity(self.identity.clone());
        Transaction::begin(Arc::clone(&self.db), session, isolation_level.as_deref())
    }

    /// Close the session: its statements fail from now on; the database and other sessions are
    /// unaffected. Closing a closed session does nothing.
    #[napi]
    pub fn close(&self) {
        drop(self.session.lock().take());
    }
}

#[cfg(feature = "cypher")]
#[napi]
impl JsSession {
    /// Execute a Cypher statement on the session.
    #[napi(js_name = "executeCypher")]
    pub async fn execute_cypher(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<types::Params>,
    ) -> Result<QueryResult> {
        self.run("cypher", query, params).await
    }
}

#[cfg(feature = "sql-pgq")]
#[napi]
impl JsSession {
    /// Execute a SQL/PGQ statement on the session.
    #[napi(js_name = "executeSql")]
    pub async fn execute_sql(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<types::Params>,
    ) -> Result<QueryResult> {
        self.run("sql", query, params).await
    }
}

#[cfg(feature = "gremlin")]
#[napi]
impl JsSession {
    /// Execute a Gremlin statement on the session.
    #[napi(js_name = "executeGremlin")]
    pub async fn execute_gremlin(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<types::Params>,
    ) -> Result<QueryResult> {
        self.run("gremlin", query, params).await
    }
}

#[cfg(feature = "graphql")]
#[napi]
impl JsSession {
    /// Execute a GraphQL statement on the session.
    #[napi(js_name = "executeGraphql")]
    pub async fn execute_graphql(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<types::Params>,
    ) -> Result<QueryResult> {
        self.run("graphql", query, params).await
    }
}

#[cfg(feature = "sparql")]
#[napi]
impl JsSession {
    /// Execute a SPARQL statement on the session.
    #[napi(js_name = "executeSparql")]
    pub async fn execute_sparql(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<types::Params>,
    ) -> Result<QueryResult> {
        self.run("sparql", query, params).await
    }
}
