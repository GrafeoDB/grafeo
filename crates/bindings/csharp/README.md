# Grafeo C# Bindings

C# bindings for the [Grafeo](https://grafeo.dev) graph database.

## Quick Start

```csharp
using Grafeo;

// Create an in-memory database
await using var db = GrafeoDB.Memory();

// Execute a GQL query
db.Execute("INSERT (:Person {name: 'Alix', age: 30})");

// Query with parameters
var result = db.ExecuteWithParams(
    "MATCH (p:Person) WHERE p.name = $name RETURN p.name, p.age",
    new Dictionary<string, object?> { ["name"] = "Alix" });

foreach (var row in result.Rows)
    Console.WriteLine($"{row["p.name"]}: {row["p.age"]}");

// Async execution
var asyncResult = await db.ExecuteAsync("MATCH (p:Person) RETURN p");

// ACID transactions with auto-rollback
using var tx = db.BeginTransaction();
tx.Execute("INSERT (:Person {name: 'Gus'})");
tx.Execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Jules'})");
tx.Commit(); // rolls back automatically if not reached
```

## Building from Source

1. Build the Grafeo C library:
   ```bash
   cargo build --release -p grafeo-c
   ```

2. Copy the native library to the test directory:
   - Windows: `copy target\release\grafeo_c.dll crates\bindings\csharp\tests\Grafeo.Tests\`
   - macOS: `cp target/release/libgrafeo_c.dylib crates/bindings/csharp/tests/Grafeo.Tests/`
   - Linux: `cp target/release/libgrafeo_c.so crates/bindings/csharp/tests/Grafeo.Tests/`

3. Build and test:
   ```bash
   cd crates/bindings/csharp
   dotnet build
   dotnet test
   ```

## Requirements

- .NET 8.0 or later
- `grafeo_c` native library (built from the `grafeo-c` crate)

## License

Apache-2.0
