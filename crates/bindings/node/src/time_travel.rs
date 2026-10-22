//! Time travel: reads of the database as it was at an earlier epoch or time.
//!
//! A point in time is a `bigint` (an epoch), a `Date` (a time) or
//! `{ ago: milliseconds }` (that long before now). A plain `number` is a
//! `TypeError`: it does not say whether it is an epoch or a time.

use std::sync::Arc;
use std::time::Duration;

use napi::bindgen_prelude::*;
use napi::{JsDate, JsValue, ValueType};
use napi_derive::napi;
use parking_lot::{Mutex, RwLock};

use grafeo_common::types::{EpochId, Timestamp};
use grafeo_engine::database::GrafeoDB;
use grafeo_engine::{AsOfView, HistoryRetention, TimePoint};

use crate::error::NodeGrafeoError;
use crate::graph::{JsEdge, JsNode};
use crate::query::QueryResult;

/// A `TypeError` with `message`, thrown at once; the returned error tells
/// napi it is pending.
fn type_error(env: &Env, message: &str) -> Error {
    // SAFETY: `env` is the environment of the call in progress.
    unsafe { JsTypeError::from(Error::new(Status::InvalidArg, message)).throw_into(env.raw()) };
    Error::new(Status::PendingException, message)
}

const POINT_TYPES: &str =
    "a point in time is a bigint (an epoch), a Date (a time) or { ago: milliseconds }";

/// The point in time `point` names: a `bigint` an epoch, a `Date` a time,
/// `{ ago: ms }` that long before now.
///
/// # Errors
///
/// A `TypeError` for a `number` (which names neither) and any other type,
/// and for a negative or too large epoch, an invalid `Date` or a negative
/// `ago`.
pub(crate) fn time_point(env: Env, point: Unknown<'_>) -> Result<TimePoint> {
    match point.get_type()? {
        ValueType::BigInt => {
            // SAFETY: the type was checked as a BigInt.
            let epoch: BigInt = unsafe { point.cast()? };
            let value = match epoch.words.as_slice() {
                [] => Some(0),
                [word] if !epoch.sign_bit || *word == 0 => Some(*word),
                _ => None,
            };
            let Some(value) = value else {
                return Err(type_error(&env, "an epoch is a bigint from 0 to 2^64 - 1"));
            };
            Ok(TimePoint::Epoch(EpochId::new(value)))
        }
        ValueType::Number => Err(type_error(
            &env,
            &format!(
                "{POINT_TYPES}: a number could be either an epoch or a time, so pass \
                 BigInt(epoch) or new Date(ms)"
            ),
        )),
        ValueType::Object => {
            // SAFETY: the type was checked as an Object.
            let object: Object<'_> = unsafe { point.cast()? };
            if object.is_date()? {
                // SAFETY: `is_date` holds, and the value lives for this call.
                let date: JsDate = unsafe { point.cast()? };
                let millis = date.value_of()?;
                if !millis.is_finite() {
                    return Err(type_error(&env, "an invalid Date names no time"));
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "a valid Date is within 8.64e15 ms of 1970, which fits an i64"
                )]
                let millis = millis.floor() as i64;
                return Ok(TimePoint::Timestamp(Timestamp::from_millis(millis)));
            }
            let ago: Option<Unknown<'_>> = object.get_named_property("ago").ok();
            match ago {
                Some(ago) if ago.get_type()? == ValueType::Number => {
                    let millis = ago.coerce_to_number()?.get_double()?;
                    if !(millis.is_finite() && millis >= 0.0) {
                        return Err(type_error(
                            &env,
                            "{ ago } is a number of milliseconds of at least 0",
                        ));
                    }
                    Ok(TimePoint::Ago(Duration::from_secs_f64(millis / 1000.0)))
                }
                _ => Err(type_error(&env, POINT_TYPES)),
            }
        }
        _ => Err(type_error(&env, POINT_TYPES)),
    }
}

/// The history retention `retention` names: `"off"`, `"forever"` or a
/// number of milliseconds (how long before now the history reaches).
pub(crate) fn history_retention(retention: Either<String, f64>) -> Result<HistoryRetention> {
    match retention {
        Either::A(name) => match name.to_ascii_lowercase().as_str() {
            "off" => Ok(HistoryRetention::Off),
            "forever" => Ok(HistoryRetention::Forever),
            _ => Err(NodeGrafeoError::InvalidArgument(format!(
                "historyRetention is 'off', 'forever' or a number of milliseconds, not '{name}'"
            ))
            .into()),
        },
        Either::B(millis) if millis.is_finite() && millis >= 0.0 => Ok(HistoryRetention::Duration(
            Duration::from_secs_f64(millis / 1000.0),
        )),
        Either::B(millis) => Err(NodeGrafeoError::InvalidArgument(format!(
            "historyRetention in milliseconds is at least 0: {millis}"
        ))
        .into()),
    }
}

/// An epoch as a `bigint`.
pub(crate) fn epoch_bigint(epoch: EpochId) -> BigInt {
    BigInt::from(epoch.as_u64())
}

/// A read-only view of the database as it was at an earlier epoch, from
/// `db.asOf(point)`. Its queries and reads see the committed state of
/// `epoch`; a write is rejected. It keeps the history of its epoch while it
/// is open: `close()` it when done.
#[napi(js_name = "AsOfView")]
pub struct JsAsOfView {
    /// Keeps the database open while the view lives.
    _db: Arc<RwLock<GrafeoDB>>,
    view: Arc<Mutex<Option<AsOfView>>>,
    epoch: EpochId,
}

impl JsAsOfView {
    pub(crate) fn new(db: Arc<RwLock<GrafeoDB>>, view: AsOfView) -> Self {
        let epoch = view.epoch();
        Self {
            _db: db,
            view: Arc::new(Mutex::new(Some(view))),
            epoch,
        }
    }

    fn closed() -> NodeGrafeoError {
        NodeGrafeoError::InvalidArgument(
            "the AsOfView is closed: open another with db.asOf(point)".to_string(),
        )
    }

    fn with_view<T>(&self, read: impl FnOnce(&AsOfView) -> T) -> Result<T> {
        let view = self.view.lock();
        Ok(read(view.as_ref().ok_or_else(Self::closed)?))
    }
}

#[napi]
impl JsAsOfView {
    /// The epoch the view reads at.
    #[napi(getter)]
    pub fn epoch(&self) -> BigInt {
        epoch_bigint(self.epoch)
    }

    /// Runs a GQL query as the database was at the view's epoch. Returns a
    /// Promise<QueryResult>; a query that writes is rejected.
    #[napi]
    pub async fn execute(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
    ) -> Result<QueryResult> {
        let params = params.map(crate::types::Params::into_map).transpose()?;
        let view = Arc::clone(&self.view);
        let result = tokio::task::spawn_blocking(move || {
            let view = view.lock();
            let view = view.as_ref().ok_or_else(Self::closed)?;
            match params {
                Some(params) => view.execute_with_params(&query, params),
                None => view.execute(&query),
            }
            .map_err(NodeGrafeoError::from)
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))??;
        Ok(crate::database::js_result(result))
    }

    /// The node as it was at the view's epoch, or null.
    #[napi(js_name = "getNode")]
    pub fn get_node(&self, id: f64) -> Result<Option<JsNode>> {
        let node_id = crate::database::validate_node_id(id)?;
        let node = self
            .with_view(|view| view.get_node(node_id))?
            .map_err(NodeGrafeoError::from)?;
        Ok(node.map(|node| crate::database::js_node(node_id, node)))
    }

    /// The edge as it was at the view's epoch, or null.
    #[napi(js_name = "getEdge")]
    pub fn get_edge(&self, id: f64) -> Result<Option<JsEdge>> {
        let edge_id = crate::database::validate_edge_id(id)?;
        let edge = self
            .with_view(|view| view.get_edge(edge_id))?
            .map_err(NodeGrafeoError::from)?;
        Ok(edge.map(|edge| crate::database::js_edge(edge_id, edge)))
    }

    /// A node's value of `key` as it was at the view's epoch, or null.
    #[napi(js_name = "getNodeProperty")]
    pub fn get_node_property(&self, env: Env, id: f64, key: String) -> Result<Unknown<'_>> {
        let node_id = crate::database::validate_node_id(id)?;
        let value = self
            .with_view(|view| view.get_node_property(node_id, &key))?
            .map_err(NodeGrafeoError::from)?
            .unwrap_or(grafeo_common::types::Value::Null);
        let raw = crate::types::value_to_napi(env.raw(), &value)?;
        // SAFETY: `raw` is a value `value_to_napi` just made in this call's
        // environment.
        Ok(unsafe { Unknown::from_raw_unchecked(env.raw(), raw) })
    }

    /// Releases the view and the history it keeps; later calls throw.
    /// Closing twice is fine.
    #[napi]
    pub fn close(&self) {
        self.view.lock().take();
    }
}

/// `db.executeAsOf(query, point, params)`: the query runs off the event
/// loop, at the epoch the point named when the call was made.
pub struct ExecuteAsOf {
    pub(crate) db: Arc<RwLock<GrafeoDB>>,
    pub(crate) query: String,
    pub(crate) point: TimePoint,
    pub(crate) params: Option<crate::types::Params>,
}

impl Task for ExecuteAsOf {
    type Output = grafeo_engine::database::QueryResult;
    type JsValue = QueryResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let params = self
            .params
            .take()
            .map(crate::types::Params::into_map)
            .transpose()?;
        let db = self.db.read();
        let epoch = db.epoch_as_of(self.point).map_err(NodeGrafeoError::from)?;
        db.session()
            .execute_at_epoch_with_params(&self.query, epoch, params)
            .map_err(|e| NodeGrafeoError::from(e).into())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(crate::database::js_result(output))
    }
}
