// The entry module of @grafeo-db/js, written by hand over the loader napi
// generates (index.js, rewritten on every build).
//
// An error from the database carries its GQLSTATUS (`22012` for a division by
// zero, `42001` for a syntax error), its Grafeo code (`GRAFEO-Q006`) and
// whether a retry can succeed. napi builds a thrown or rejected error from its
// status, message and `cause` only, so the native side sends them as a `cause`
// whose message is `GQLSTATUS <status> <code> <1 or 0>`. This module wraps
// every method of the exported classes once, moves them to `err.gqlStatus`,
// `err.errorCode` and `err.isRetryable` and removes that `cause`, for a sync
// throw and for a rejected promise alike. Any other error, and any other
// cause, passes through as is.
//
// It also gives Transaction a `[Symbol.dispose]()` that rolls back a
// transaction still active, for `using tx = db.beginTransaction()`.

'use strict'

const { wrapExports } = require('./error-status.js')

const wrapped = wrapExports(require('./index.js'))

const { Transaction } = wrapped
if (typeof Symbol.dispose === 'symbol' && Transaction && !(Symbol.dispose in Transaction.prototype)) {
  Object.defineProperty(Transaction.prototype, Symbol.dispose, {
    value: function dispose() {
      if (this.isActive) this.rollback()
    },
    writable: true,
    configurable: true,
  })
}

module.exports = wrapped
// Each export assigned by name as well, so Node finds the names when an ES
// module imports this file (`import { GrafeoDB } from '@grafeo-db/js'`): it
// reads them from these assignments, not from the object above. Keep the
// list in step with the exports of index.js.
module.exports.AsOfView = wrapped.AsOfView
module.exports.JsAsOfView = wrapped.JsAsOfView
module.exports.GrafeoDB = wrapped.GrafeoDB
module.exports.JsGrafeoDB = wrapped.JsGrafeoDB
module.exports.GraphHandle = wrapped.GraphHandle
module.exports.JsGraphHandle = wrapped.JsGraphHandle
module.exports.JsEdge = wrapped.JsEdge
module.exports.JsNode = wrapped.JsNode
module.exports.QueryResult = wrapped.QueryResult
module.exports.ResultStream = wrapped.ResultStream
module.exports.JsResultStream = wrapped.JsResultStream
module.exports.Transaction = wrapped.Transaction
module.exports.simdSupport = wrapped.simdSupport
module.exports.version = wrapped.version
