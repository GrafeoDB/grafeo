# openCypher TCK

The `.gtest` files here are the scenarios of the openCypher Technology Compatibility
Kit (TCK), converted by `scripts/convert_opencypher_tck.py`. Every spec runner runs
them like the other `.gtest` files.

## Source and licence

The cases come from [opencypher/openCypher](https://github.com/opencypher/openCypher)
at tag `2024.3` (commit `677cbafabb8c3c5eed458fd3b1ec0daec8d67d23`). Each file's
header names it as well.

The TCK is licensed under the Apache License, Version 2.0, with the openCypher
attribution notice; each converted file keeps the licence header and the notice of
its feature file. Only the generated files are kept here; the TCK's own files are
not.

## Regenerating

Check out the TCK at the pinned tag (the folder is ignored by git), then convert:

```bash
git clone --depth 1 --branch 2024.3 https://github.com/opencypher/openCypher.git tests/spec/lpg/cypher/tck/openCypher
python scripts/convert_opencypher_tck.py --tck tests/spec/lpg/cypher/tck/openCypher/tck
```

The output is the same on every run and machine. To move to a newer tag, check it
out, convert, review the differences, and update the tag above and `TAG` in the
converter.

A scenario becomes one case (an outline, one case per example row; a control query,
a second case). The TCK's graphs and `having executed` steps become `setup`, its
test procedures GQL `CREATE PROCEDURE` statements in `setup_gql`, its result tables
`values:` (type-exact, in the TCK's own notation) and its side effects
`side_effects:`. An error becomes the GQLSTATUS class of its condition: `42` for an
error the TCK raises at compile time, which GQL's Syntax Rules catch, otherwise the
class of the TCK's error type (`22` for a type or argument error). Scenarios tagged
`@ignore` upstream are left out and named in the file's header.

Cases that build or return a temporal value carry a `runner_gap` for the bindings
that do not yet hand temporals back without loss.

## Known failures

`known_failures.toml` lists, by feature, the cases Grafeo does not pass:

- `{ issue = "...", note = "..." }`: the case becomes a strict expected failure
  (`xfail:`) held against its TCK scenario, naming the release that fixes it.
  `passes_in` names runners where it passes all the same. Remove the entry once
  fixed and convert again.
- `{ reason = "out-of-scope", note = "..." }`: the case is left out and named in
  the file's header.
- `{ gql_class = "22", note = "..." }`: GQL decides the condition by a General Rule
  at run time, so the case expects GQL's class instead of the TCK's `42`, and says
  so in a comment (a negative SKIP or LIMIT, or a non-integer one given as a
  parameter).

The converter stops when an entry names no converted case, so the list cannot go
stale.
