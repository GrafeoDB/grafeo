import { emitStatements } from './emit-statements.mjs';

/** GQL statements, one per line, no trailing semicolon. */
export const emitGql = (source) =>
  emitStatements(source, { insert: 'INSERT', dateLiteral: (d) => `DATE '${d}'` });
