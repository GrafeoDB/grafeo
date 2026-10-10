# Error Codes

Every error Grafeo raises carries a machine-readable code of the form
`GRAFEO-<Category><Number>`. The code is stable across releases: once a
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
| `GRAFEO-Q002` | QuerySemantic       | no        | Parsed but invalid: unknown label, unknown graph, type mismatch in a function call, etc. |
| `GRAFEO-Q003` | QueryTimeout        | **yes**   | Query exceeded its deadline. Raise `query_timeout` or narrow the pattern. |
| `GRAFEO-Q004` | QueryUnsupported    | no        | Feature not implemented for this query language (e.g. a SPARQL-only function used from GQL). |
| `GRAFEO-Q005` | QueryOptimization   | no        | Optimizer could not produce a plan. Report with the query text if you hit this. |
| `GRAFEO-Q006` | QueryExecution      | no        | A physical operator failed at runtime. |

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
| `GRAFEO-V001` | InvalidInput     | no        | Argument or property value failed validation (e.g. wrong vector dimensionality, oversized property). |
| `GRAFEO-V002` | NodeNotFound     | no        | `NodeId` does not exist in the current graph. |
| `GRAFEO-V003` | EdgeNotFound     | no        | `EdgeId` does not exist in the current graph. |
| `GRAFEO-V004` | PropertyNotFound | no        | Referenced property key is not declared for this label/type. |
| `GRAFEO-V005` | LabelNotFound    | no        | Referenced label is not declared. |
| `GRAFEO-V006` | TypeMismatch     | no        | Value type does not match the schema declaration. |

## Internal (X)

| Code          | Name                | Retryable | Meaning                                                    |
| ------------- | ------------------- | --------- | ---------------------------------------------------------- |
| `GRAFEO-X001` | Internal            | no        | Unexpected internal error. Please file a bug with the query and stack trace. |
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
