/**
 * One formatter for approximate numbers in every output. A literal with an exponent is a float in GQL and
 * Cypher (a plain decimal such as 5.2 reads as DECIMAL in GQL but float in Cypher, and the engine refuses
 * plain decimals over 38 digits) and an xsd:double in Turtle.
 */

/** `1.5e-7`, never `1.5e+7`: the shortest digits of `text`, an exponent string from `toExponential`. */
const tidy = (text) => text.replace('e+', 'e');

/** The shortest exponent-form literal that parses back to the same double. */
export function floatLiteral(x) {
  if (!Number.isFinite(x)) throw new Error(`not a finite number: ${x}`);
  return tidy(x.toExponential());
}

/** The shortest exponent-form literal that parses back to the same float32. */
export function f32Literal(x) {
  const f = Math.fround(x);
  if (!Number.isFinite(f)) throw new Error(`not a finite float32: ${x}`);
  for (let digits = 1; digits <= 9; digits++) {
    const s = f.toExponential(digits - 1);
    if (Math.fround(Number(s)) === f) return tidy(s);
  }
  return tidy(f.toExponential(8));
}

/** A property number: integers stay plain unless their text has an exponent (1e21 and up); others are floats. */
export const numberLiteral = (x) => (Number.isInteger(x) && !/e/i.test(String(x)) ? String(x) : floatLiteral(x));
