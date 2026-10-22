//! Time travel: reads of the database as it was at an earlier epoch or time.
//!
//! A point in time is a `bigint` (an epoch), a `Date` (a time) or
//! `{ ago: milliseconds }` (that long before now). A plain `number` is a
//! `TypeError`: it does not say whether it is an epoch or a time.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use wasm_bindgen::prelude::*;

use grafeo_common::types::{EpochId, Timestamp};
use grafeo_engine::{AsOfView as EngineAsOfView, GrafeoDB, HistoryRetention, TimePoint};

use crate::{Database, database_error, js_error, raw_result_to_js, rows_to_js};

const POINT_TYPES: &str =
    "a point in time is a bigint (an epoch), a Date (a time) or { ago: milliseconds }";

fn type_error(message: &str) -> JsValue {
    js_sys::TypeError::new(message).into()
}

/// The point in time `point` names: a `bigint` an epoch, a `Date` a time,
/// `{ ago: ms }` that long before now.
///
/// # Errors
///
/// A `TypeError` for a `number` (which names neither) and any other type,
/// and for a negative or too large epoch, an invalid `Date` or a negative
/// `ago`.
pub(crate) fn time_point(point: &JsValue) -> Result<TimePoint, JsValue> {
    if point.is_bigint() {
        let epoch = u64::try_from(point.clone())
            .map_err(|_| type_error("an epoch is a bigint from 0 to 2^64 - 1"))?;
        return Ok(TimePoint::Epoch(EpochId::new(epoch)));
    }
    if point.as_f64().is_some() {
        return Err(type_error(&format!(
            "{POINT_TYPES}: a number could be either an epoch or a time, so pass \
             BigInt(epoch) or new Date(ms)"
        )));
    }
    if let Some(date) = point.dyn_ref::<js_sys::Date>() {
        let millis = date.get_time();
        if !millis.is_finite() {
            return Err(type_error("an invalid Date names no time"));
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a valid Date is within 8.64e15 ms of 1970, which fits an i64"
        )]
        let millis = millis.floor() as i64;
        return Ok(TimePoint::Timestamp(Timestamp::from_millis(millis)));
    }
    if point.is_object() {
        let ago = js_sys::Reflect::get(point, &JsValue::from_str("ago"))?;
        return match ago.as_f64() {
            Some(millis) if millis.is_finite() && millis >= 0.0 => {
                Ok(TimePoint::Ago(Duration::from_secs_f64(millis / 1000.0)))
            }
            Some(_) => Err(type_error(
                "{ ago } is a number of milliseconds of at least 0",
            )),
            None => Err(type_error(POINT_TYPES)),
        };
    }
    Err(type_error(POINT_TYPES))
}

/// The history retention of the constructor's `options.historyRetention`:
/// `'off'`, `'forever'` or a number of milliseconds; `None` when not given.
pub(crate) fn history_retention(options: &JsValue) -> Result<Option<HistoryRetention>, JsValue> {
    if options.is_undefined() || options.is_null() {
        return Ok(None);
    }
    let retention = js_sys::Reflect::get(options, &JsValue::from_str("historyRetention"))?;
    if retention.is_undefined() {
        return Ok(None);
    }
    if let Some(name) = retention.as_string() {
        return match name.to_ascii_lowercase().as_str() {
            "off" => Ok(Some(HistoryRetention::Off)),
            "forever" => Ok(Some(HistoryRetention::Forever)),
            _ => Err(js_error(&format!(
                "historyRetention is 'off', 'forever' or a number of milliseconds, not '{name}'"
            ))),
        };
    }
    match retention.as_f64() {
        Some(millis) if millis.is_finite() && millis >= 0.0 => Ok(Some(
            HistoryRetention::Duration(Duration::from_secs_f64(millis / 1000.0)),
        )),
        _ => Err(js_error(
            "historyRetention is 'off', 'forever' or a number of milliseconds of at least 0",
        )),
    }
}

#[wasm_bindgen]
impl Database {
    /// The epoch of the last commit, as a `bigint`.
    ///
    /// # Errors
    ///
    /// After `close()`.
    #[wasm_bindgen(js_name = "currentEpoch")]
    pub fn current_epoch(&self) -> Result<u64, JsValue> {
        self.check_open()?;
        Ok(self.db().current_epoch().as_u64())
    }

    /// The epoch a point in time reads at, as a `bigint`: the epoch itself
    /// (the current one for a later one), or the last commit at or before a
    /// time.
    ///
    /// # Errors
    ///
    /// A `TypeError` for a point that is not a `bigint`, a `Date` or
    /// `{ ago }`; an error for a point before the history the database keeps.
    #[wasm_bindgen(js_name = "epochAsOf")]
    pub fn epoch_as_of(&self, point: JsValue) -> Result<u64, JsValue> {
        self.check_open()?;
        let point = time_point(&point)?;
        let epoch = self.db().epoch_as_of(point).map_err(database_error)?;
        Ok(epoch.as_u64())
    }

    /// A read-only view of the database as it was at a point in time. It
    /// keeps the history of its epoch until `close()`.
    ///
    /// ```js
    /// const view = db.asOf({ ago: 5 * 60 * 1000 });
    /// const rows = view.execute("MATCH (p:Person) RETURN count(p) AS n");
    /// view.close();
    /// ```
    ///
    /// # Errors
    ///
    /// As `epochAsOf`.
    #[wasm_bindgen(js_name = "asOf")]
    pub fn as_of(&self, point: JsValue) -> Result<AsOfView, JsValue> {
        self.check_open()?;
        let point = time_point(&point)?;
        let view = self.db().as_of(point).map_err(database_error)?;
        Ok(AsOfView::new(Rc::clone(&self.inner), view))
    }

    /// Runs one GQL query, with optional parameters, as the database was at
    /// a point in time; rows as objects, as `execute`. A query that writes
    /// throws.
    ///
    /// # Errors
    ///
    /// As `epochAsOf`, and the query's errors.
    #[wasm_bindgen(js_name = "executeAsOf")]
    pub fn execute_as_of(
        &self,
        query: &str,
        point: JsValue,
        params: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.check_open()?;
        let point = time_point(&point)?;
        let params = Self::convert_params(Some(params))?;
        let db = self.db();
        let epoch = db.epoch_as_of(point).map_err(database_error)?;
        let result = db
            .session()
            .execute_at_epoch_with_params(query, epoch, params)
            .map_err(database_error)?;
        Ok(rows_to_js(&result))
    }
}

/// A read-only view of the database as it was at an earlier epoch, from
/// `db.asOf(point)`. Its queries see the committed state of `epoch`; a query
/// that writes throws. It keeps the history of its epoch until `close()`.
#[wasm_bindgen]
pub struct AsOfView {
    /// Keeps the database alive while the view lives.
    _db: Rc<RefCell<GrafeoDB>>,
    view: RefCell<Option<EngineAsOfView>>,
    epoch: u64,
}

impl AsOfView {
    fn new(db: Rc<RefCell<GrafeoDB>>, view: EngineAsOfView) -> Self {
        let epoch = view.epoch().as_u64();
        Self {
            _db: db,
            view: RefCell::new(Some(view)),
            epoch,
        }
    }

    fn run(
        &self,
        query: &str,
        params: Option<JsValue>,
    ) -> Result<grafeo_engine::database::QueryResult, JsValue> {
        let params = Database::convert_params(params)?;
        let view = self.view.borrow();
        let view = view
            .as_ref()
            .ok_or_else(|| js_error("the AsOfView is closed: open another with db.asOf(point)"))?;
        match params {
            Some(params) => view.execute_with_params(query, params),
            None => view.execute(query),
        }
        .map_err(database_error)
    }
}

#[wasm_bindgen]
impl AsOfView {
    /// The epoch the view reads at, as a `bigint`.
    #[wasm_bindgen(getter)]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Runs a GQL query as the database was at the view's epoch; rows as
    /// objects, as `Database.execute`.
    ///
    /// # Errors
    ///
    /// The query's errors, an error for a query that writes, and after
    /// `close()`.
    pub fn execute(&self, query: &str) -> Result<JsValue, JsValue> {
        Ok(rows_to_js(&self.run(query, None)?))
    }

    /// Runs a GQL query with parameters at the view's epoch.
    ///
    /// # Errors
    ///
    /// As `execute`, and for parameters that are not an object.
    #[wasm_bindgen(js_name = "executeWithParams")]
    pub fn execute_with_params(&self, query: &str, params: JsValue) -> Result<JsValue, JsValue> {
        Ok(rows_to_js(&self.run(query, Some(params))?))
    }

    /// Runs a GQL query at the view's epoch and returns raw columns, rows and
    /// metadata, as `Database.executeRaw`.
    ///
    /// # Errors
    ///
    /// As `execute`.
    #[wasm_bindgen(js_name = "executeRaw")]
    pub fn execute_raw(&self, query: &str) -> Result<JsValue, JsValue> {
        Ok(raw_result_to_js(&self.run(query, None)?))
    }

    /// Releases the view and the history it keeps; later calls throw.
    /// Closing twice is fine.
    pub fn close(&self) {
        self.view.borrow_mut().take();
    }
}
