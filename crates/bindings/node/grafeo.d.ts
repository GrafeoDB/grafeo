// The types of @grafeo-db/js: the generated declarations (index.d.ts) and the
// error shape the entry module (grafeo.js) adds.

export * from './index'

/**
 * An error the database raised. Errors thrown or rejected by a call of this
 * package are `Error` instances; those that come from the database also carry
 * their GQLSTATUS (ISO/IEC 39075), for example `22012` for a division by zero,
 * `22003` for an integer overflow and `42001` for a syntax error. An error the
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
}
