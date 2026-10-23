# Contributing to Grafeo

Thanks for wanting to help out! Here's what you need to know.

## Setup

```bash
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
cargo build --workspace
```

You'll need **Rust 1.99.0+** and optionally **Python 3.12+** / **Node.js 20+** for the bindings.

## Branching

Each release has its own branch, `release/<version>` (for example `release/0.5.44`), and work for
that release lands there. `main` receives the release branch when the release ships. The current
release branch belongs to the lowest open [milestone](https://github.com/GrafeoDB/grafeo/milestones).

- Branch from the current release branch: `fix/<issue>-<description>` for bug fixes,
  `feat/<issue>-<description>` for features.
- Open your pull request against that release branch, not `main`.

## Making Changes

1. Create a branch: `git switch -c fix/123-short-name origin/release/0.5.44`
2. Write code and tests
3. Run checks: `./scripts/ci-local.sh` (or `.\scripts\ci-local.ps1` on Windows), and the policy
   checks on your changes: `python scripts/check_policy.py diff --base origin/release/0.5.44`
4. Push and open a pull request against the release branch

You can also run checks individually:

```bash
cargo fmt --all              # Format
cargo clippy --all-targets --all-features -- -D warnings  # Lint
cargo test --all-features --workspace     # Test
```

### Commit Messages

We use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`.

## Pull Request Eligibility

Every pull request runs the **PR Policy** check (`scripts/check_policy.py pr`). For contributor
pull requests the rules below are required; for maintainers most of them are warnings. Editing
the description or the labels runs the check again.

| Rule | What it asks | How to satisfy it |
| ---- | ------------ | ----------------- |
| Target branch | The pull request targets the current `release/<version>` branch | Change the base branch of the pull request |
| Planned issue | The description links an issue (`Fixes #123`) that has a milestone or the `help wanted` or `good first issue` label | Open an issue or a discussion first and wait until it is planned |
| AI assistance | AI help is welcome and declared: when the description names AI tools (`AI tools used: ...`) or a commit or the description credits one ("Co-authored-by", "Generated with", "Made with"), the ownership box in the description is ticked | Tick "I have read every line of this change, I understand it, and I can explain and defend it in review". AI co-author lines are not kept in the history: such changes are landed as one squashed commit |
| New dependencies | No new Rust, npm, Python, Go, NuGet or Dart dependency | Agree on it in the issue; a maintainer adds `approved: deps` |
| Infrastructure | No changes to `.github/`, `scripts/`, the root `Cargo.toml` or other root configuration | Only when the issue asks for it; a maintainer adds `approved: infra` |
| Structure | No new crate, feature flag or `GraphStore` wrapper | Agree on the design in the issue; a maintainer adds `approved: arch` |
| Tests | A fix (a `fix:` title or a linked bug) changes or adds a test | Add a regression test; a maintainer can add `no-test-needed` |
| Changelog (warning) | Changes under `crates/` come with a `CHANGELOG.md` entry | Add one line under the unreleased version |
| Size (warning) | At most about 1,500 added lines outside tests | Split the change into smaller pull requests |

The check also applies these rules to the lines your pull request adds (existing code is never
flagged): a new `#[allow(...)]` states a `reason = "..."`; docs and code comments use no em or en
dashes; public text does not reference internal planning notes; crash-injection tests are not
marked `#[ignore]`; WAL and recovery code does not drop results with `let _ =`. Run them locally
before you push:

```bash
python scripts/check_policy.py diff --base origin/release/0.5.44
```

## Architecture

| Crate | What it does |
| ----- | ------------ |
| `grafeo` | Top-level facade, re-exports public API |
| `grafeo-common` | Foundation types, memory, utilities |
| `grafeo-core` | Graph storage, indexes, execution |
| `grafeo-storage` | Persistence: WAL, `.grafeo` container, crash safety |
| `grafeo-adapters` | Query parsers (GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ) |
| `grafeo-engine` | Database facade, sessions, transactions |
| `grafeo-cli` | CLI with interactive shell, query execution, import/export, backup, WAL management |
| `grafeo-bindings-common` | Shared library for all language bindings |
| `grafeo-python` | Python bindings (PyO3) |
| `grafeo-node` | Node.js/TypeScript bindings (napi-rs) |
| `grafeo-c` | C FFI layer (also used by Go via CGO) |
| `grafeo-wasm` | WebAssembly bindings (wasm-bindgen) |
| `grafeo-csharp` | C# / .NET 8 bindings (P/Invoke, wraps grafeo-c) |
| `grafeo-dart` | Dart bindings (dart:ffi, wraps grafeo-c) |

## Spec Tests (gtests)

Declarative, cross-language integration tests live in `tests/spec/` as `.gtest` files:

```text
tests/spec/
├── lpg/           # Labeled Property Graph tests
│   ├── gql/       # GQL (ISO 39075)
│   ├── cypher/    # openCypher
│   ├── gremlin/   # TinkerPop Gremlin
│   ├── graphql/   # GraphQL over LPG
│   └── sql_pgq/   # SQL/PGQ (SQL:2023)
├── rdf/           # RDF model tests
│   ├── sparql/    # SPARQL 1.1
│   └── graphql/   # GraphQL over RDF
├── common/        # Language-agnostic tests
├── datasets/      # Shared test fixtures (.setup files)
├── regression/    # Issue-mapped regression tests
└── rosetta/       # Cross-language equivalence tests
```

Each `.gtest` file is YAML-like with a `meta:` header and `tests:` list. A build script generates Rust `#[test]` functions at compile time. Run them with:

```bash
cargo test -p grafeo-spec-tests                          # All spec tests
cargo test -p grafeo-spec-tests -- gremlin               # Filter by keyword
cargo test -p grafeo-spec-tests -- rdf_sparql             # SPARQL tests only
```

### Writing a spec test

```yaml
meta:
  language: gql
  model: lpg
  section: "my-feature"
  title: My Feature Tests
  dataset: social_network

tests:
  - name: basic_query
    query: MATCH (p:Person) RETURN p.name
    expect:
      rows:
        - [Mia]
        - [Jules]
        - [Vincent]
```

Tests can use `skip: "reason"` to mark known gaps, `expect: { count: N }` for row count checks, `expect: { ordered: true }` for order-sensitive assertions, and `expect: { error: "substring" }` for expected errors.

### Persistent variants for index-dependent cases

Any `.gtest` file or case that declares `requires: [text-index]` or `requires: [vector-index]` automatically gets a second generated test, suffixed `_persistent`, that opens `GrafeoDB::open(tempdir)` instead of `GrafeoDB::new_in_memory()`. This exercises the WAL-wrapped read path that every on-disk session uses, so regressions in wrapper-layer delegation fail here instead of slipping past the in-memory suite.

Contributors don't need to do anything special: write the test once, name it as you would any other case, and the harness emits both variants. If a test case passes in-memory but fails `_persistent`, the bug is almost certainly in a store wrapper (WAL, CDC, Layered) missing a delegation.

## Code Style

- Standard Rust conventions: `rustfmt` and `clippy` are enforced in CI
- Use `thiserror` for error types
- Tests go in the same file under `#[cfg(test)]`
- Descriptive test names: `test_<function>_<scenario>`

## Python Bindings

```bash
cd crates/bindings/python
maturin develop
pytest tests/ -v --ignore=tests/benchmark_phases.py
```

## Node.js Bindings

```bash
cd crates/bindings/node
npm install
npm run build
npm test
```

## Versioning

Grafeo follows the rules in [Versioning and Compatibility](https://grafeo.dev/versioning/). In short, for a
pull request:

- A breaking change to the stable surface (the `grafeo` crate and its re-exports, the bindings, the CLI, the file
  format) or a higher minimum Rust version goes into a minor release (`0.6` to `0.7`). Mark its CHANGELOG entry
  **Breaking**; patch releases have none.
- Changes inside the implementation crates (`grafeo-common`, `grafeo-core`, `grafeo-storage`, `grafeo-adapters`,
  `grafeo-engine`) that do not reach the stable surface can go into any release.
- Bug fixes go into patch releases, also when they change query results.
- Deprecate before removing: name the replacement, and remove it at the earliest in the next minor release.

## Ecosystem Projects

These companion projects live in separate repositories under the [GrafeoDB](https://github.com/GrafeoDB) organization:

| Project | Description |
| ------- | ----------- |
| [grafeo-server](https://github.com/GrafeoDB/grafeo-server) | HTTP server & web UI |
| [grafeo-web](https://github.com/GrafeoDB/grafeo-web) | Browser-based Grafeo (WASM) |
| [gwp](https://github.com/GrafeoDB/gql-wire-protocol) | GQL Wire Protocol (gRPC) |
| [boltr](https://github.com/GrafeoDB/boltr) | Bolt v5.x Wire Protocol |
| [grafeo-memory](https://github.com/GrafeoDB/grafeo-memory) | AI memory layer for LLM applications |
| [grafeo-langchain](https://github.com/GrafeoDB/grafeo-langchain) | LangChain graph + vector store |
| [grafeo-llamaindex](https://github.com/GrafeoDB/grafeo-llamaindex) | LlamaIndex PropertyGraphStore |
| [grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp) | MCP server for LLM agents |
| [anywidget-graph](https://github.com/GrafeoDB/anywidget-graph) | Graph visualization widget |
| [anywidget-vector](https://github.com/GrafeoDB/anywidget-vector) | Vector visualization widget |
| [graph-bench](https://github.com/GrafeoDB/graph-bench) | Benchmark suite |
| [ann-benchmarks](https://github.com/GrafeoDB/ann-benchmarks) | Vector search benchmarking |

## Benchmarks and Performance Regressions

PRs to `main` and pushes to `main` are benchmarked on [CodSpeed](https://codspeed.io/)
(`.github/workflows/codspeed.yml`), which posts a comment with the changes against `main`. PRs from forks
are skipped: the CodSpeed token isn't exposed to fork workflows; after an initial review a maintainer can
push the branch to this repository to trigger a run.

### What runs where, and why

| Job | Instrument | Runner | Benchmarks |
| --- | --- | --- | --- |
| `simulation` | CPU simulation: instructions and simulated cache misses, system calls left out | `ubuntu-24.04` (one image, so base and head run on the same runner type) | `grafeo-common` arena_bench; `grafeo-core` index_bench, store_apply; `grafeo-engine` query_bench, serialization_bench, regression_bench, topk_e2e, and write_path_bench's in-memory cases, reads and transaction manager (`GRAFEO_BENCH_STORAGE=mem`) |
| `walltime` | wall time on a bare-metal CodSpeed Macro Runner | `codspeed-macro-arm64-graviton-ubuntu-22-04` | write_path_bench's file-backed cases and checkpoints (`GRAFEO_BENCH_STORAGE=file`): the WAL, fsync, and a checkpoint thread beside the writers |

- **Simulation** is deterministic for CPU work, so it suits the in-memory benchmarks. It runs each
  benchmark once, so a case that times one call of a few nanoseconds measures a cold call: such cases
  time a batch per iteration (the `/x64` and `/x1024` cases), sized to a whole cycle of the periodic work
  they do (a collect, the transaction manager's cleanup), so every iteration does the same work.
- **The simulation build is deterministic**: `--cfg grafeo_codspeed` fixes the hash tables' seed (a random
  seed per run sends hash joins, DISTINCT and GROUP BY through other collision chains, and moves their
  counts) and takes the SIMD kernels from the build's target features, which `-C target-cpu=x86-64-v3`
  fixes, instead of the runner's CPU. The run is on one CPU (`taskset -c 0`, `RAYON_NUM_THREADS=1`), so
  every pool has one thread: simulation runs threads one at a time, and parallelism would only add
  scheduling. A shipped build never sets the cfg: there the random seed guards against hash flooding.
- **No duplicates**: a case that times a path another case already times (the same query at a larger
  size, the same pattern in two suites) is left out, which keeps the run short.
- **Walltime** measures what simulation cannot: system calls and the scheduling of several threads. It
  needs a CodSpeed Macro Runner, a machine of its own, and is off until the organization allows the macro
  runners for this public repository (Organization settings, Actions, Runner groups, Default) and the
  repository variable `CODSPEED_MACRO_RUNNERS` is `true`. The free plan holds 600 macro-runner minutes a
  month, so on a pull request it runs only with the label `bench: walltime`; it always runs on `main`.
- **Not on CodSpeed:** memory_bench (its `iter_custom` cases are skipped by CodSpeed, and it measures
  memory, not time), morsel_pipeline (it times thread pools, which simulation runs one thread at a time),
  core top_k (covered end to end by topk_e2e) and the storage crate's wal_bench (it times the 0.5.x WAL
  writer, which the engine no longer uses).

### Reading CodSpeed's report

- **"Base and head ran on different runner types"**: the two runs were on different hardware, so part of
  each change may be the machine's (simulation models the runner's caches). Pinning one image keeps the
  runner type stable; after a change to the workflow, `main` needs one run on the new setup before PRs
  compare against it. The simulation job prints the runner's CPU, its caches and SIMD features: a change
  on unchanged code that follows a CPU change there is the machine's.
- **"N benchmarks are dominated by system calls"**: simulation leaves system calls out, so such a
  benchmark understates its cost. The file-backed cases run in the walltime job for that reason; a case
  that still shows the warning belongs there too.
- **Skipped benchmarks** ran on the base but not on the head (renamed or removed): archive them in the
  CodSpeed app so the report stops carrying them.
- **An A/A run** checks the setup: run the workflow twice on one commit (Actions, CodSpeed, Run workflow,
  on `main`), or open a PR to `main` that changes only a comment. Both runs must report no change; a
  change there names a benchmark that depends on the machine or on timing.

### Paired benchmarks on your machine

CodSpeed tells you that something moved; a local paired run tells you how much and whether it is noise.
`scripts/bench-paired.sh` builds the benchmarks of two revisions in release, runs them interleaved for
several rounds and prints each case's fastest round per side, the change, and whether the rounds overlap:

```bash
scripts/bench-paired.sh -r 3 main HEAD \
  'grafeo-core:store_apply:in_degree|expand_outgoing' \
  'grafeo-engine:query_bench:query_1hop_pattern'
GRAFEO_BENCH_SCALE=10 GRAFEO_BENCH_STORAGE=file scripts/bench-paired.sh main HEAD \
  'grafeo-engine:write_path_bench:cypher_create_rel'
```

Run it on a quiet machine: another build at the same time makes rounds of one side differ by a factor of
two. A change whose rounds overlap the other side's is within the noise.

To reproduce a CodSpeed run itself (Linux, with valgrind):

```bash
cargo install cargo-codspeed
cargo codspeed build -m simulation --all-features -p grafeo-core --bench store_apply
GRAFEO_BENCH_STORAGE=mem cargo codspeed run -m simulation
```

If a PR flags a >5% regression on a hot path, include a brief explanation in the PR description. The
trade-off is often acceptable (e.g. a correctness fix that costs some throughput), but the maintainer
should know it was deliberate rather than accidental.

Adding a new Criterion bench: import `use criterion::{...}` as usual. The workspace renames `criterion`
to the `codspeed-criterion-compat` package (see root `Cargo.toml`), so there is nothing to change per
crate: add the `[[bench]]` entry and any features in the owning crate's `Cargo.toml`, and add the suite to
the job of `.github/workflows/codspeed.yml` that fits it (simulation for CPU work, walltime for system
calls and threads). Build the case's input outside the measured closure (`iter_batched`, not
`iter_custom`, which CodSpeed skips), and measure a batch when one call takes nanoseconds.

## Pre-commit Hooks (Optional)

```bash
cargo install prek
prek install
```

This runs format, lint, typo and policy checks on the staged changes before each commit. It also
keeps AI co-author lines out of commit messages: name the AI tools you used in the pull request
description instead (see Pull Request Eligibility).

## Links

- [Repository](https://github.com/GrafeoDB/grafeo)
- [Issues](https://github.com/GrafeoDB/grafeo/issues)
- [Documentation](https://grafeo.dev)

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
