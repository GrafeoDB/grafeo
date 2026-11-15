import { emitStatements } from './emit-statements.mjs';

/** Cypher statements, one per line, no trailing semicolon. */
export const emitCypher = (source) =>
  emitStatements(source, { insert: 'CREATE', dateLiteral: (d) => `date('${d}')` });
