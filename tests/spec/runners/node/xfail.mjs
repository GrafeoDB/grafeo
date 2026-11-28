/**
 * Expected failures (`xfail:`), shared by the Node.js and WASM runners.
 *
 * A case marked `xfail:` passes as an expected failure only when it fails the way a spec case
 * fails: on an assertion (a `SpecFailure` of the comparator, a vitest `AssertionError`) or on an
 * error the binding raises (one with a GQLSTATUS). A case that passes fails the run (XPASS,
 * naming the spec and the issue), and any other error is a fault of the runner and fails it as
 * it is.
 */

import { isAbsolute, resolve } from 'path'

/** The repository root: tests/spec/runners/node. */
const REPO_ROOT = resolve(import.meta.dirname, '..', '..', '..', '..')

/** The failure of a spec case's assertion. */
export class SpecFailure extends Error {
  constructor(message) {
    super(message)
    this.name = 'SpecFailure'
  }
}

/** Whether `err` is a spec case's failure, the only one an `xfail:` absorbs. */
export function isExpectedFailure(err) {
  return err instanceof SpecFailure || err?.name === 'AssertionError' || err?.gqlStatus != null
}

/** The run's counts for the `gtest summary` line. */
export const summary = { skipped: 0, xfailed: 0 }

/**
 * Skips the case of the test context `ctx`, counting it in the summary: every skip counts,
 * whichever check skips the case (`skip:`, a runner gap, a language or feature the build
 * lacks), so a runner that skips a whole language shows it.
 */
export function skipCase(ctx) {
  summary.skipped++
  return ctx.skip()
}

/** The `gtest summary` line. */
export function summaryLine() {
  return `gtest summary: skipped=${summary.skipped} xfailed=${summary.xfailed}`
}

/**
 * Runs `run`, the case `tc`: as it is without an `xfail:`; with one, a spec failure is the
 * expected failure, a pass an XPASS, and any other error goes on.
 */
export async function runMarked(tc, run, runner = 'node') {
  // An xfail that names this runner in `passes_in` does not hold here: the case runs
  if (tc.xfail == null || tc.xfail.passesIn?.includes(runner)) return run()
  try {
    await run()
  } catch (err) {
    if (!isExpectedFailure(err)) throw err
    summary.xfailed++
    return
  }
  throw new Error(
    `XPASS: the case passes, but it is marked xfail against ${tc.xfail.spec} ` +
      `(${tc.xfail.issue}); take the mark off`,
  )
}

/** `runMarked` for a runner whose cases run synchronously (WASM). */
export function runMarkedSync(tc, run, runner = 'wasm') {
  if (tc.xfail == null || tc.xfail.passesIn?.includes(runner)) return run()
  try {
    run()
  } catch (err) {
    if (!isExpectedFailure(err)) throw err
    summary.xfailed++
    return
  }
  throw new Error(
    `XPASS: the case passes, but it is marked xfail against ${tc.xfail.spec} ` +
      `(${tc.xfail.issue}); take the mark off`,
  )
}

/**
 * The directory whose .gtest files the run reads: the one `GRAFEO_SPEC_ROOT` names (the
 * runner probes), relative to the repository or absolute, else `specDir`.
 */
export function specRoot(specDir) {
  const root = process.env.GRAFEO_SPEC_ROOT
  if (!root) return specDir
  return isAbsolute(root) ? root : resolve(REPO_ROOT, root)
}

/** A probe of the runner itself: it fails as the runner fails, with its ordinary error. */
export function runProbe(tc) {
  if (tc.probe === 'runner_error') {
    throw new TypeError('runner error probe: the runner fails, not the case')
  }
}
