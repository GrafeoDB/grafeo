// The types of @grafeo-db/js: the generated declarations (index.d.ts) and the
// error shape the entry module (grafeo.js) adds.

export * from './index'

/**
 * An error the database raised. Errors thrown or rejected by a call of this
 * package are `Error` instances; those that come from the database, and those
 * of a transaction that ended (`25000`), also carry their GQLSTATUS (ISO/IEC
 * 39075), for example `22012` for a division by zero, `22003` for an integer
 * overflow and `42001` for a syntax error. An error the
 * binding raises before it calls the database (a wrong argument type) has no
 * `gqlStatus`.
 *
 * ```ts
 * try {
 *   await db.execute('RETURN 1 / 0')
 * } catch (error) {
 *   if ((error as GrafeoError).gqlStatus === '22012') {
 *     // division by zero
 *   }
 * }
 * ```
 */
export interface GrafeoError extends Error {
  /** The GQLSTATUS of an error from the database: five characters, such as `22012`. */
  gqlStatus?: string
  /**
   * The Grafeo code of an error from the database, such as `GRAFEO-T001` for a
   * write conflict or `GRAFEO-S004` for a database file open elsewhere.
   */
  errorCode?: string
  /**
   * Whether running the same work again can succeed, such as a transaction that
   * lost a write conflict (`false` for an error that has a `gqlStatus` and
   * cannot).
   */
  isRetryable?: boolean
}

declare module './index' {
  interface Transaction {
    /**
     * Rolls the transaction back if it is still active, so
     * `using tx = db.beginTransaction()` cleans up where `using` is supported.
     */
    [Symbol.dispose](): void
  }
}
