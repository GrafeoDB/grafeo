# grafeo-cli

Command-line interface for [Grafeo](https://grafeo.dev) graph database.

This is a thin Python launcher package that runs the pre-built Grafeo CLI binary.

## Installation

```bash
uv add grafeo-cli
# or: pip install grafeo-cli
```

The package looks for the `grafeo` binary in this order:
1. Bundled with the wheel (platform-specific wheels)
2. In the virtualenv `bin/` or `Scripts/` directory
3. On your system `PATH`

If no binary is found, install it separately:

```bash
# Via cargo
cargo install grafeo-cli

# Or download from GitHub releases
# https://github.com/GrafeoDB/grafeo/releases
```

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

# Shell completions
grafeo completions bash > ~/.local/share/bash-completion/completions/grafeo
```

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Grafeo Python Library](https://pypi.org/project/grafeo/) (the database engine, not this CLI tool)

## License

Apache-2.0
