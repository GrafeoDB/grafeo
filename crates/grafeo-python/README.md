# grafeo

Python bindings for Grafeo, a pure-Rust, high-performance, embeddable graph database.

## Installation

```bash
uv add grafeo
```

## Usage

```python
import grafeo

# Create an in-memory database
db = grafeo.GrafeoDB()

# Create nodes using GQL
db.execute("INSERT (:Person {name: 'Alice', age: 30})")
db.execute("INSERT (:Person {name: 'Bob', age: 25})")

# Query the graph
result = db.execute("MATCH (p:Person) RETURN p.name, p.age")
for row in result:
    print(row)
```

## License

Apache-2.0
