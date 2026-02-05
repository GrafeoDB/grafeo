---
title: Security Best Practices
description: Security considerations for Grafeo deployments.
tags:
  - security
  - best-practices
---

# Security Best Practices

Grafeo is an embedded database without built-in authentication. Security depends on how you deploy and use it.

---

## Understanding Grafeo's Security Model

Grafeo is designed as an **embedded library**, not a network-accessible server:

- **No authentication** - Anyone with access to your application can access the database
- **No network protocol** - No TCP/HTTP ports to secure
- **No encryption at rest** - Database files are not encrypted
- **File-based access control** - Security relies on filesystem permissions

This model is appropriate for:

- Single-user applications
- Microservices with internal graph state
- Data science environments
- Applications that implement their own access control

---

## Securing Your Deployment

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
db.create_node(["Person"], {"name": "Alice", "ssn_encrypted": ssn_encrypted})

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

- [ ] Database files have restricted permissions (700 or 600)
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

If you discover a security vulnerability:

1. **Do not** open a public GitHub issue
2. Email security concerns to security@grafeo.dev
3. Include steps to reproduce
4. Allow time for a fix before public disclosure

We take security seriously and will respond promptly.
