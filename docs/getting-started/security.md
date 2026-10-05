---
title: Security Best Practices
description: Security considerations for Grafeo deployments.
tags:
  - security
  - best-practices
---

# Security Best Practices

Grafeo is an embedded database without built-in authentication. Security depends on how it is deployed and used.

---

## Understanding Grafeo's Security Model

Grafeo is designed as an **embedded library**, not a network-accessible server:

- **Role-based access control** - Sessions can be scoped to `Admin`, `ReadWrite` or `ReadOnly` roles
- **Per-graph grants** - Identities can be restricted to specific named graphs
- **No built-in authentication** - The caller is trusted to assign roles; no credentials or crypto at this layer
- **No network protocol** - No TCP/HTTP ports to secure
- **Optional encryption at rest**: a `.grafeo` database and its WAL can be encrypted (see [Encryption at Rest](#encryption-at-rest))
- **File-based access control** - Database files rely on filesystem permissions

This model is appropriate for:

- Single-user applications
- Microservices with internal graph state
- Data science environments
- Multi-tenant applications that assign roles based on their own authentication

---

## Role-Based Access Control

Grafeo provides session-level role-based access control (RBAC). Each session can be scoped to a role that restricts which operations are allowed. Permission checks run after parsing but before execution, across all query languages.

### Roles

| Role | Reads | Writes | Schema DDL |
|------|-------|--------|------------|
| `Admin` | Yes | Yes | Yes |
| `ReadWrite` | Yes | Yes | No |
| `ReadOnly` | Yes | No | No |

### Creating Role-Scoped Sessions (Rust)

```rust
use grafeo::{GrafeoDB, auth::{Identity, Role, Grant}};

let db = GrafeoDB::new_in_memory();

// Convenience: session with a specific role
let reader = db.session_with_role(Role::ReadOnly);

// Full control: session with an identity
let identity = Identity::new("api-user", [Role::ReadWrite]);
let writer = db.session_with_identity(identity);
```

### Per-Graph Access Grants

Identities can be restricted to specific named graphs. When grants are present, only the listed graphs are accessible. Empty grants means unrestricted access (backward compatible).

```rust
use grafeo::auth::{Identity, Role, Grant};

let identity = Identity::new("analyst", [Role::ReadWrite])
    .with_grants([
        Grant::new("social", Role::ReadWrite),
        Grant::new("public", Role::ReadOnly),
    ]);

let session = db.session_with_identity(identity);
// This session can write to "social", read from "public", and nothing else
```

### GQL Syntax

Graph projections and named graph operations respect grants:

```sql
-- These are enforced when the session has grants:
USE GRAPH social;
CREATE GRAPH analytics;
DROP GRAPH old_data;
```

!!! note "No credentials at this layer"
    Grafeo does not handle authentication (passwords, tokens, certificates). The caller is trusted to assign the correct role. Use your application's auth layer to map users to Grafeo identities.

---

## Encryption at Rest

Grafeo can encrypt a single-file database (`.grafeo`) and its WAL with AES-256-GCM. Enable the `encryption` feature of `grafeo-engine` and set `Config::encryption` to a key chain built from a 32-byte master key:

```rust
use std::sync::Arc;

use grafeo_common::encryption::KeyChain;
use grafeo_engine::config::EncryptionConfig;
use grafeo_engine::{Config, GrafeoDB};

// 32 bytes from your key management (a KMS, a secrets manager, an HSM).
let master_key: [u8; 32] = load_master_key();

let mut config = Config::persistent("social.grafeo");
config.encryption = Some(EncryptionConfig {
    key_chain: Arc::new(KeyChain::new(master_key)),
});
let db = GrafeoDB::with_config(config)?;
```

Grafeo stores no key material: keep the master key safe, as the database cannot be opened without it. To derive the master key from a passphrase, `PasswordKeyProvider::derive_with_salt` (Argon2id) gives the same key for the same passphrase and salt; store the salt with the database.

### What Is Encrypted

| Encrypted | Not encrypted |
|-----------|---------------|
| The `.grafeo` file: every section (data, schema, indexes) and the directory that lists them | The file header (format version, the encrypted flag, the database id, the creation time and Grafeo version) and the two database headers (checkpoint counters and time, node and edge counts) |
| The sidecar WAL (`<file>.wal/`): every record | WAL bookkeeping files (the checkpoint marker, the backup cursor) |
| A copy written by `save()` to a `.grafeo` path | The bytes `export_snapshot()` returns: plaintext, to be stored as safely as the data |
| Full backups (`backup_full()`, a copy of the encrypted file) and incremental segments (`backup_incremental()`, encrypted WAL records) | The backup manifest (segment names, epochs, sizes, checksums) |
| A database restored with `restore_to_epoch_with()`, and the WAL it writes next to it | An in-memory copy made with `to_memory()`: it has no key, so a copy saved from it is plaintext (call `save()` on the encrypted database instead) |
| | The `.pre-0.6` files a migration keeps of a database written by 0.5.x: the file and, if they existed, its WAL and a pending checkpoint image |

An encrypted database writes no spill files: spill files are not encrypted, so it gets no spill path (`<file>.spill/` for other databases), and `Config::validate` refuses an explicit `spill_path`, or a section pinned to `TierOverride::ForceDisk`, together with `encryption`. In 0.6.0 a memory limit therefore cannot move the data of an encrypted database to disk.

### Keys

The key chain derives the keys with HKDF-SHA256, one per database and component: `"grafeo-container"` for the file and `"grafeo-wal"` for its WAL, each with the database id from the file header. Two databases configured with the same master key therefore have different keys, and a copy made with `save()` gets a new database id and keys of its own.

### Opening an Encrypted Database

- A database created with a key is encrypted, one created without a key is not, and that stays so: changing the key, or encrypting an existing 0.6 database, is not supported yet.
- Opening an encrypted database without a key fails with "the database is encrypted and needs its key". With another master key it fails while reading the file ("wrong key or corrupted data"). Neither changes the file.
- Opening an unencrypted database with a key fails with "the database is not encrypted", so plaintext data is never taken for encrypted data.
- A read-only open (`Config::read_only`) needs the key as well, and works with it.
- A database written by 0.5.x is never encrypted. A read-write open with a key migrates it into an encrypted 0.6 file. The migration keeps the old files unencrypted: the file as `<file>.pre-0.6`, its WAL as `<file>.pre-0.6.wal` and, if present, a checkpoint 0.5.44 left pending as `<file>.pre-0.6.checkpoint`. Remove all of them once you no longer need to go back to 0.5.x. A read-only open with a key fails with "the database is not encrypted".
- Encryption needs a persistent single-file database: `Config::validate` refuses it for an in-memory database, and opening a WAL-directory database with a key fails.
- `GrafeoDB::open_in_memory` takes no key: open an encrypted database with its key and call `to_memory()` instead.

### Backups of an Encrypted Database

`backup_full()` and `backup_incremental()` of an encrypted database write encrypted backups, and need no key. To restore them, pass the same key chain to `restore_to_epoch_with`:

```rust
use std::path::Path;
use std::sync::Arc;

use grafeo_common::encryption::KeyChain;
use grafeo_engine::config::EncryptionConfig;
use grafeo_engine::GrafeoDB;

let encryption = EncryptionConfig {
    key_chain: Arc::new(KeyChain::new(master_key)),
};
let output = Path::new("restored.grafeo");
GrafeoDB::restore_to_epoch_with(backup_dir, target_epoch, output, &encryption)?;
```

The key is checked against the full backup before anything is written. The restored database is encrypted with the backup's keys: open it with the same key chain. `restore_to_epoch` without a key refuses an encrypted backup when it has incremental segments to replay; a restore that needs only the full backup copies the encrypted file.

---

## Securing a Deployment

### 1. File System Permissions

Protect database files with appropriate permissions:

=== "Linux/macOS"
    ```bash
    # Create directory with restricted permissions
    mkdir -p /var/lib/myapp/data
    chmod 700 /var/lib/myapp/data
    chown myapp:myapp /var/lib/myapp/data

    # Set umask for new files
    umask 077
    ```

=== "Windows"
    ```powershell
    # Create directory
    New-Item -ItemType Directory -Path "C:\ProgramData\MyApp\Data"

    # Set permissions (restrict to current user)
    $acl = Get-Acl "C:\ProgramData\MyApp\Data"
    $acl.SetAccessRuleProtection($true, $false)
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $env:USERNAME, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
    )
    $acl.AddAccessRule($rule)
    Set-Acl "C:\ProgramData\MyApp\Data" $acl
    ```

### 2. Input Validation

**Always use parameterized queries** to prevent injection:

```python
# DANGEROUS - SQL injection risk
user_input = request.form["name"]
db.execute(f"MATCH (n:Person {{name: '{user_input}'}}) RETURN n")  # DON'T DO THIS

# SAFE - Parameterized query
user_input = request.form["name"]
db.execute("MATCH (n:Person {name: $name}) RETURN n", {"name": user_input})  # DO THIS
```

### 3. Validate Property Values

Sanitize data before storing:

```python
def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input."""
    if not isinstance(value, str):
        raise ValueError("Expected string")
    # Limit length
    value = value[:max_length]
    # Remove null bytes
    value = value.replace("\x00", "")
    return value

def create_user(db, name: str, email: str):
    """Create user with validated input."""
    name = sanitize_string(name, max_length=100)
    email = sanitize_string(email, max_length=255)

    # Validate email format
    if "@" not in email or "." not in email:
        raise ValueError("Invalid email format")

    return db.create_node(["User"], {"name": name, "email": email})
```

### 4. Limit Query Complexity

Prevent denial-of-service via expensive queries:

```python
def safe_execute(db, query: str, params: dict = None, max_results: int = 10000):
    """Execute query with result limit."""
    # Add LIMIT if not present
    if "LIMIT" not in query.upper():
        query = f"{query} LIMIT {max_results}"

    return db.execute(query, params)

# Usage
result = safe_execute(db, "MATCH (n) RETURN n")  # Limited to 10000 results
```

### 5. Audit Logging

Log database operations for security auditing:

```python
import logging
from datetime import datetime
from functools import wraps

logger = logging.getLogger("grafeo.audit")

def audit_query(func):
    """Decorator to audit database queries."""
    @wraps(func)
    def wrapper(self, query: str, params: dict = None, *args, **kwargs):
        start = datetime.now()
        try:
            result = func(self, query, params, *args, **kwargs)
            logger.info(
                "QUERY",
                extra={
                    "query": query[:500],  # Truncate long queries
                    "params": str(params)[:200] if params else None,
                    "duration_ms": (datetime.now() - start).total_seconds() * 1000,
                    "result_count": len(result) if hasattr(result, "__len__") else None,
                }
            )
            return result
        except Exception as e:
            logger.error(
                "QUERY_ERROR",
                extra={
                    "query": query[:500],
                    "error": str(e),
                }
            )
            raise
    return wrapper
```

---

## Sensitive Data Handling

### Don't Store Secrets in Properties

```python
# BAD - Storing plaintext password
db.create_node(["User"], {"email": "user@example.com", "password": "secret123"})

# GOOD - Store only hashed password
import hashlib
password_hash = hashlib.sha256(b"secret123").hexdigest()
db.create_node(["User"], {"email": "user@example.com", "password_hash": password_hash})
```

### Mask Sensitive Data in Logs

```python
def mask_sensitive(data: dict, sensitive_keys: set = {"password", "token", "secret"}):
    """Mask sensitive values in dictionaries."""
    return {
        k: "***MASKED***" if k.lower() in sensitive_keys else v
        for k, v in data.items()
    }

# Usage in logging
logger.info(f"Creating user: {mask_sensitive(user_data)}")
```

### Consider Encryption for Sensitive Properties

```python
from cryptography.fernet import Fernet

# Generate key (store securely!)
key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_value(value: str) -> str:
    return cipher.encrypt(value.encode()).decode()

def decrypt_value(encrypted: str) -> str:
    return cipher.decrypt(encrypted.encode()).decode()

# Store encrypted
ssn_encrypted = encrypt_value("123-45-6789")
db.create_node(["Person"], {"name": "Alix", "ssn_encrypted": ssn_encrypted})

# Retrieve and decrypt
node = db.get_node(node_id)
ssn = decrypt_value(node.properties["ssn_encrypted"])
```

---

## Network Security

If exposing Grafeo through an API:

### 1. Add Authentication Layer

```python
from flask import Flask, request, jsonify
from functools import wraps
from grafeo import GrafeoDB

app = Flask(__name__)
db = GrafeoDB("./mydb")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if api_key != os.environ["API_KEY"]:
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/query", methods=["POST"])
@require_api_key
def query():
    data = request.json
    result = db.execute(data["query"], data.get("params"))
    return jsonify(result.to_list())
```

### 2. Use HTTPS

Always use TLS when exposing over network:

```python
# Use gunicorn with SSL
# gunicorn --certfile cert.pem --keyfile key.pem app:app
```

### 3. Rate Limiting

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.headers.get("X-API-Key"))

@app.route("/query", methods=["POST"])
@limiter.limit("100/minute")
@require_api_key
def query():
    ...
```

---

## Backup Security

### Secure Backup Storage

```python
import shutil
import os

def secure_backup(db_path: str, backup_path: str):
    """Create a secure backup."""
    # Create backup
    db.save(backup_path)

    # Set restrictive permissions
    os.chmod(backup_path, 0o600)

    # Optionally encrypt
    # gpg --encrypt --recipient admin@example.com backup_path
```

### Secure Backup Transfer

```bash
# Encrypt before transfer
gpg --encrypt --recipient admin@example.com backup.db

# Transfer encrypted file
scp backup.db.gpg backup-server:/backups/
```

---

## Security Checklist

Before deploying:

- [ ] Sessions use appropriate roles (`ReadOnly` for read paths, `ReadWrite` for mutations)
- [ ] Per-graph grants restrict multi-tenant access where needed
- [ ] Database files have restricted permissions (700 or 600)
- [ ] Databases holding sensitive data are encrypted at rest (`Config::encryption`) and the master key is kept outside the database directory
- [ ] All queries use parameterization (no string interpolation)
- [ ] Input validation on all user-provided data
- [ ] Query results are limited to prevent DoS
- [ ] Sensitive data is encrypted or hashed
- [ ] Audit logging is enabled
- [ ] API endpoints require authentication
- [ ] HTTPS is enabled for network access
- [ ] Rate limiting is configured
- [ ] Backups are encrypted and access-controlled
- [ ] Error messages don't expose internal details

---

## Reporting Security Issues

To report a security vulnerability:

1. **Do not** open a public GitHub issue
2. Email security concerns to security@grafeo.dev
3. Include steps to reproduce
4. Allow time for a fix before public disclosure

Security issues are taken seriously and will receive a prompt response.
