# Join the GrafeoDB Community

We're building a modern graph database ecosystem in Rust, and we'd love your help.

## Why Contribute?

**Learn by doing**: Work with cutting-edge tech including Rust, WebAssembly, Arrow/Polars vectorization, MVCC transactions, and multiple query language parsers.

**Shape the project**: We're early-stage and open to ideas. Your contributions can influence the direction of the entire ecosystem.

**Build your portfolio**: Graph databases are in demand. Contributing here gives you real experience with database internals, query optimization, and systems programming.

## Ways to Get Involved

### Code Contributions

| Area | Skills | Projects |
|------|--------|----------|
| **Core Database** | Rust, database internals | [grafeo](https://github.com/GrafeoDB/grafeo) |
| **Query Languages** | Parsing, compilers | GQL, Cypher, SPARQL, Gremlin, GraphQL support |
| **Graph Algorithms** | Algorithms, math | PageRank, community detection, centrality |
| **Python Bindings** | Rust + Python, PyO3 | [grafeo](https://github.com/GrafeoDB/grafeo) |
| **Browser Runtime** | TypeScript, WebAssembly | [grafeo-web](https://github.com/GrafeoDB/grafeo-web) |
| **Visualization** | Three.js, Sigma.js | [anywidget-graph](https://github.com/GrafeoDB/anywidget-graph), [anywidget-vector](https://github.com/GrafeoDB/anywidget-vector) |
| **Server** | Rust, Axum, REST APIs | [grafeo-server](https://github.com/GrafeoDB/grafeo-server) |
| **Benchmarking** | Python, data analysis | [graph-bench](https://github.com/GrafeoDB/graph-bench) |

### Non-Code Contributions

- **Documentation**: Tutorials, examples, API docs
- **Testing**: Bug reports, edge cases, stress testing
- **Design**: Logo, diagrams, UI/UX for widgets
- **Community**: Answer questions, write blog posts, give talks

## Getting Started

1. **Pick a project** that matches your interests
2. **Read the README** and try running it locally
3. **Browse issues** labeled `good first issue` or `help wanted`
4. **Ask questions** by opening a discussion or issue

### Good First Issues

Look for issues tagged with:
- `good first issue`: Beginner-friendly tasks
- `help wanted`: We'd appreciate help here
- `documentation`: Docs improvements needed
- `testing`: Test coverage improvements

## Development Setup

Most projects use similar tooling:

```bash
# Rust projects
cargo build --workspace
cargo test --workspace

# Python projects
uv sync
uv run pytest
```

See each project's CONTRIBUTING.md for specific instructions.

## Our Stack

| Layer | Technology |
|-------|------------|
| Core | Rust, Arrow, Polars |
| Python | PyO3, maturin |
| Server | Axum, Tower, Docker |
| Browser | WebAssembly, IndexedDB |
| Visualization | Three.js, Sigma.js, anywidget |
| Build | Cargo, uv, hatch |
| CI | GitHub Actions |

## Communication

- **Issues**: Bug reports and feature requests
- **Discussions**: Questions and ideas
- **Pull Requests**: Code contributions

We aim to respond within a few days. Be patient with us, and we'll be patient with you.

## Recognition

Contributors are recognized in:
- Release notes
- Project documentation
- This file (once we have more contributors!)

## Current Maintainers

- **S.T. Grond** ([@stgrond](https://github.com/StevenBtw)): Architect

## License

All contributions are licensed under Apache-2.0.

---

**Ready to contribute?** Pick a repo, find an issue, and send a PR. We're excited to have you.
