// The entry module of @grafeo-db/js, written by hand over the loader napi
// generates (index.js, rewritten on every build).
//
// An error from the database carries its GQLSTATUS (`22012` for a division by
// zero, `42001` for a syntax error). napi builds a thrown or rejected error
// from its status, message and `cause` only, so the native side sends the
// status as a `cause` whose message is `GQLSTATUS <code>`. This module wraps
// every method of the exported classes once, moves that status to
// `err.gqlStatus` and removes its `cause`, for a sync throw and for a rejected
// promise alike. Any other error, and any other cause, passes through as is.

'use strict'

const { wrapExports } = require('./error-status.js')

module.exports = wrapExports(require('./index.js'))
