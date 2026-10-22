// Moves the GQLSTATUS of a database error to `err.gqlStatus` (see grafeo.js).
// Internal to @grafeo-db/js; not part of its API.

'use strict'

// The same text as GQLSTATUS_MARKER in src/error.rs.
const MARKER = 'GQLSTATUS '

function annotate(error) {
  if (error instanceof Error) {
    const cause = error.cause
    if (cause instanceof Error && typeof cause.message === 'string' && cause.message.startsWith(MARKER)) {
      error.gqlStatus = cause.message.slice(MARKER.length)
      delete error.cause
    }
  }
  return error
}

function wrap(fn) {
  return function (...args) {
    let result
    try {
      result = fn.apply(this, args)
    } catch (error) {
      throw annotate(error)
    }
    if (result !== null && typeof result === 'object' && typeof result.then === 'function') {
      return result.then(undefined, (error) => {
        throw annotate(error)
      })
    }
    return result
  }
}

function wrapMembers(target, skip) {
  for (const key of Reflect.ownKeys(target)) {
    if (skip.has(key)) continue
    const descriptor = Object.getOwnPropertyDescriptor(target, key)
    if (!descriptor || !descriptor.configurable) continue
    if (typeof descriptor.value === 'function') {
      descriptor.value = wrap(descriptor.value)
    } else if (typeof descriptor.get === 'function') {
      descriptor.get = wrap(descriptor.get)
    } else {
      continue
    }
    Object.defineProperty(target, key, descriptor)
  }
}

const STATIC_SKIP = new Set(['length', 'name', 'prototype'])
const PROTOTYPE_SKIP = new Set(['constructor'])

/** Wraps every static and prototype member of the classes `exports` holds, once. */
function wrapExports(exports) {
  for (const value of new Set(Object.values(exports))) {
    if (typeof value !== 'function' || typeof value.prototype !== 'object' || value.prototype === null) continue
    wrapMembers(value, STATIC_SKIP)
    wrapMembers(value.prototype, PROTOTYPE_SKIP)
  }
  return exports
}

module.exports = { MARKER, annotate, wrap, wrapExports }
