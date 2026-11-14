# @grafeo-db/cli

Command-line interface for [Grafeo](https://grafeo.dev) graph database, distributed via npm.

## Installation

```bash
npm install -g @grafeo-db/cli
# or
npx @grafeo-db/cli --help
```

The correct platform-specific binary is installed automatically via optional dependencies.

## Supported Platforms

| Platform | Package |
|----------|---------|
| Linux x64 | `@grafeo-db/cli-linux-x64` |
| Linux ARM64 | `@grafeo-db/cli-linux-arm64` |
| macOS x64 | `@grafeo-db/cli-darwin-x64` |
| macOS ARM64 | `@grafeo-db/cli-darwin-arm64` |
| Windows x64 | `@grafeo-db/cli-win32-x64` |

## Features

The pre-built binaries (and `cargo install grafeo-cli`) open every database the default
`grafeo` library writes: GQL, persistent storage, vector, text and hybrid search indexes, change
data capture, graph algorithms and JSON Lines import. `grafeo version` lists the features of the
installed build.

Other features, such as RDF databases (`triple-store`) or more query languages, need a build
from source:

```bash
cargo install grafeo-cli --features triple-store
cargo install grafeo-cli --features full    # every language, RDF and the defaults
```

A database that holds data the build cannot read is refused before anything on disk changes, and
the error names the `cargo install` line that adds the feature. The full list:
[Command-Line Interface](https://grafeo.dev/getting-started/cli/#features).

## Usage

```bash
# Database management
grafeo info ./my-db
grafeo stats ./my-db
grafeo validate ./my-db

# Query execution
grafeo query ./my-db "MATCH (n:Person) RETURN n.name"

# Interactive shell
grafeo shell ./my-db

# Create a new database
grafeo init ./new-db
```

## Alternative Installation

```bash
# Via Rust
cargo install grafeo-cli

# Via pip/uv
uv add grafeo-cli

# Direct download
# https://github.com/GrafeoDB/grafeo/releases
```

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)

## License

Apache-2.0
