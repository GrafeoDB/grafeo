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
    print(e.gql_status)    # "42001", see GQLSTATUS below
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
| `GRAFEO-Q002` | QuerySemantic       | no        | Parsed but invalid: unknown label, graph or procedure, a variable nothing binds (in a stored procedure's body too), a call with another number of arguments, an unknown `YIELD` column, etc. GQLSTATUS `42002` for a name that resolves to nothing, `42000` for the rest. |
| `GRAFEO-Q003` | QueryTimeout        | **yes**   | Query exceeded its deadline. Raise `query_timeout` or narrow the pattern. |
| `GRAFEO-Q004` | QueryUnsupported    | no        | The database or this build cannot run the statement: a feature the build leaves out (vector or text indexes, procedures, RDF, an import format), a language of another graph model (GQL on an RDF database), or an operation the database cannot do (a backup of an in-memory database). |
| `GRAFEO-Q005` | QueryOptimization   | no        | Optimizer could not produce a plan. Report with the query text if you hit this. |
| `GRAFEO-Q006` | QueryExecution      | no        | A valid statement failed while it ran, on its data: a file to load that cannot be read, a SPARQL graph that exists already or does not exist, a write to a node the transaction deleted, an integer or decimal result out of range or a division by zero (GQLSTATUS `22003` and `22012`). |

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
| `GRAFEO-T009` | LogWriteFailed           | **yes**   | The commit could not be written to the write-ahead log (a full disk, an I/O error) and was rolled back: nothing of it is in the log or the database. Retry the transaction once the cause is gone. |
| `GRAFEO-T010` | CommitOutcomeUnknown     | no        | The commit was written to the write-ahead log, but the log could not be made durable. Nobody sees the commit, and whether it exists is known only after the database is reopened; nothing commits until then. Do not retry before the reopen: the commit may already be there. The C API returns `GRAFEO_ERROR_DATABASE`. |
| `GRAFEO-T011` | LogUnavailable           | no        | The write-ahead log takes no more writes after an earlier failure. Nothing of this commit was written, and nothing commits until the database is reopened. The C API returns `GRAFEO_ERROR_DATABASE`. |

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

## GQLSTATUS

Every error also carries its GQLSTATUS, the five-character status of ISO GQL (section 23): a class of two
characters and a subclass of three. The Grafeo code says which part of Grafeo refused the statement, the
GQLSTATUS says what kind of condition it is, as SQL's SQLSTATE does:

| GQLSTATUS | Condition                      | Example                                          |
| --------- | ------------------------------ | ------------------------------------------------ |
| `22003`   | numeric value out of range     | `RETURN 9223372036854775807 + 1`, a `sum` past 64 bits |
| `22012`   | division by zero               | `RETURN 10 / 0`, `RETURN 10 % 0`                 |
| `22G02`   | negative limit value           | `LIMIT $k` with `k = -1`                         |
| `22G03`   | invalid value type             | `LIMIT $k` with `k = 'three'`, a procedure argument of another type |
| `22000`   | data exception                 | other failures on the statement's data           |
| `02000`   | no data                        | a node or edge that does not exist               |
| `25000`   | invalid transaction state      | a write to a closed database                     |
| `25G03`   | read-only transaction          | a write or DDL in a read-only transaction or database |
| `40000`   | transaction rollback           | a write conflict                                 |
| `42001`   | syntax error                   | `MATCH (n RETURN n`                              |
| `42002`   | invalid reference              | a name that resolves to nothing: an undefined variable, an unknown function, procedure, parameter, graph or type |
| `42000`   | syntax error or access rule violation | any other rule a statement breaks beyond the grammar (an aggregate in a `WHERE`, a wrong number of arguments, a pattern `MERGE` cannot take), a feature Grafeo lacks, and a permission the user does not have |

`42000` is the code ISO GQL gives a violated syntax or access rule that names no condition of its own; the
message says which rule it is.

Python has it as `GrafeoError.gql_status`, WASM as the `gqlStatus` property of the thrown `Error`, C as
`grafeo_last_error_gql_status()` (next to `grafeo_last_error()`), Go as the `GQLStatus` field of `*grafeo.Error`
(get it with `errors.As`), C# as `GrafeoException.GqlStatus` and Dart as `GrafeoException.gqlStatus`. A
float operation keeps its IEEE result (`toFloat(1) / 0` is infinity), and SPARQL leaves a variable unbound
when its expression fails, as SPARQL 1.1 specifies, instead of raising one of these errors.

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
