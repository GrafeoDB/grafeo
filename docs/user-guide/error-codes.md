# Error Codes

Every error Grafeo raises carries a machine-readable code of the form
`GRAFEO-<Category><Number>`, and its message starts with that code
(`GRAFEO-Q002: semantic error: ...`), so every binding can tell the kind of
an error from its message. The code is stable across releases: once a
number is assigned to a failure mode, it does not move.

In Python, catch `grafeo.GrafeoError` (subclass of `RuntimeError`) to inspect
the structured attributes:

```python
import grafeo

db = grafeo.GrafeoDB()
try:
    db.execute("NOT VALID GQL")
except grafeo.GrafeoError as e:
    print(e.error_code)    # "GRAFEO-Q001"
    print(e.is_retryable)  # False
```

Categories:

| Prefix | Category    | Meaning                                      |
| ------ | ----------- | -------------------------------------------- |
| `Q`    | Query       | Parser, planner, or executor rejected a query|
| `T`    | Transaction | Transaction lifecycle or isolation violation |
| `S`    | Storage     | On-disk state, WAL, or recovery              |
| `V`    | Validation  | Input or reference does not resolve          |
| `X`    | Internal    | Bug, I/O, or serialization                   |

## Query (Q)

| Code          | Name                | Retryable | Meaning                                                    |
| ------------- | ------------------- | --------- | ---------------------------------------------------------- |
| `GRAFEO-Q001` | QuerySyntax         | no        | Parser rejected the query. Check the reported line/column. |
| `GRAFEO-Q002` | QuerySemantic       | no        | Parsed but invalid: unknown label, graph or procedure, a variable nothing binds (in a stored procedure's body too), a call with another number of arguments, an unknown `YIELD` column, type mismatch in a function call, etc. |
| `GRAFEO-Q003` | QueryTimeout        | **yes**   | Query exceeded its deadline. Raise `query_timeout` or narrow the pattern. |
| `GRAFEO-Q004` | QueryUnsupported    | no        | The database or this build cannot run the statement: a feature the build leaves out (vector or text indexes, procedures, RDF, an import format), a language of another graph model (GQL on an RDF database), or an operation the database cannot do (a backup of an in-memory database). |
| `GRAFEO-Q005` | QueryOptimization   | no        | Optimizer could not produce a plan. Report with the query text if you hit this. |
| `GRAFEO-Q006` | QueryExecution      | no        | A valid statement failed while it ran, on its data: a file to load that cannot be read, a SPARQL graph that exists already or does not exist, a write to a node the transaction deleted. |

## Transaction (T)

| Code          | Name                     | Retryable | Meaning                                                    |
| ------------- | ------------------------ | --------- | ---------------------------------------------------------- |
| `GRAFEO-T001` | TransactionConflict      | **yes**   | Write-write conflict with another transaction. Retry the whole transaction. |
| `GRAFEO-T002` | TransactionTimeout       | **yes**   | Transaction exceeded its TTL. |
| `GRAFEO-T003` | TransactionReadOnly      | no        | Attempted a write inside `START TRANSACTION READ ONLY`. |
| `GRAFEO-T004` | TransactionInvalidState  | no        | `COMMIT` / `ROLLBACK` without an active transaction, or a transaction command outside GQL. |
| `GRAFEO-T005` | TransactionSerialization | **yes**   | SSI validation rejected the commit. Retry under a fresh snapshot. |
| `GRAFEO-T006` | TransactionDeadlock      | **yes**   | Lock manager detected a cycle. Retry. |
| `GRAFEO-T007` | DatabaseClosed           | no        | A write to a database whose `close()` started (persistent databases), or a checkpoint, backup or save after it. Open it again to write. Python raises `DatabaseClosedError`, a subclass of `GrafeoError`; the C API returns `GRAFEO_ERROR_DATABASE`. |
| `GRAFEO-T008` | IncompleteCommit         | no        | An earlier commit did not complete, so nothing commits, checkpoints, saves or copies the database (`to_memory()`, snapshots) until it is reopened (reads still work). Python raises `GrafeoError`; the C API returns `GRAFEO_ERROR_DATABASE`. |

## Storage (S)

| Code          | Name                  | Retryable | Meaning                                                    |
| ------------- | --------------------- | --------- | ---------------------------------------------------------- |
| `GRAFEO-S001` | StorageFull           | no        | Buffer budget, disk, or memory limit reached. |
| `GRAFEO-S002` | StorageCorrupted      | no        | A file Grafeo wrote is damaged: a checksum, a header, a section or a WAL record that does not read back as written. The message names the file and, when known, the byte. Python raises `GrafeoCorruptionError`, a subclass of `GrafeoError`; the C API returns `GRAFEO_ERROR_STORAGE`. The database may need a restore from a backup. |
| `GRAFEO-S003` | StorageRecoveryFailed | no        | WAL replay failed during `GrafeoDB::open`. Inspect the logs. |

## Validation (V)

| Code          | Name             | Retryable | Meaning                                                    |
| ------------- | ---------------- | --------- | ---------------------------------------------------------- |
| `GRAFEO-V001` | InvalidInput     | no        | A value or name a call gives does not fit: wrong vector dimensionality, oversized property, a missing or ill-typed procedure argument, an index, named graph or embedding model that does not exist, an invalid setting, an import line or snapshot that is not valid. |
| `GRAFEO-V002` | NodeNotFound     | no        | `NodeId` does not exist in the current graph (a direct write to it, or an edge to it). |
| `GRAFEO-V003` | EdgeNotFound     | no        | `EdgeId` does not exist in the current graph (a direct write to it). |
| `GRAFEO-V004` | PropertyNotFound | no        | Referenced property key is not declared for this label/type. |
| `GRAFEO-V005` | LabelNotFound    | no        | Referenced label is not declared. |
| `GRAFEO-V006` | TypeMismatch     | no        | Value type does not match the schema declaration. |

## Internal (X)

| Code          | Name                | Retryable | Meaning                                                    |
| ------------- | ------------------- | --------- | ---------------------------------------------------------- |
| `GRAFEO-X001` | Internal            | no        | A bug in Grafeo: an invariant of the engine does not hold. A mistake in a query, a call or its input never has this code. Please file a bug with the query and stack trace. |
| `GRAFEO-X002` | SerializationError  | no        | Could not encode/decode a value (snapshot, WAL, or binding boundary). |
| `GRAFEO-X003` | IoError             | no        | Underlying I/O call failed. |

## Retry guidance

Only codes flagged **yes** above are worth retrying automatically.
Everything else signals a logical or structural issue that retrying will
not fix. In Python:

```python
import time

for attempt in range(3):
    try:
        db.execute(my_query)
        break
    except grafeo.GrafeoError as e:
        if not e.is_retryable or attempt == 2:
            raise
        time.sleep(0.05 * (2 ** attempt))
```

## Stability

The codes above are stable API. New codes will be added for new failure
modes; existing codes will not be renamed or reassigned. `ErrorCode` in
Rust is `#[non_exhaustive]`, and the Python string form is the canonical
identifier to match against.
