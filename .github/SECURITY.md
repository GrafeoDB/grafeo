# Security Policy

## Supported Versions

Grafeo is currently in active development (pre-1.0). Security updates are provided for the latest minor release only.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| 0.1.x   | :x:                |
| < 0.1   | :x:                |

## Security Model

Grafeo is an **embeddable** graph database designed to run within your application process. It does not include:

- Network listeners or remote access protocols
- Built-in authentication or authorization
- Multi-tenant isolation

**Security is the responsibility of the embedding application.** If you expose Grafeo through a network service, you must implement appropriate authentication, authorization, and input validation in your application layer.

### Query Injection

Like SQL databases, Grafeo query languages (GQL, Cypher, SPARQL, etc.) can be vulnerable to injection attacks if user input is concatenated directly into queries. Always use parameterized queries when accepting user input:

```python
# UNSAFE - vulnerable to injection
db.execute(f"MATCH (n:User {{name: '{user_input}'}}) RETURN n")

# SAFE - use parameters
db.execute("MATCH (n:User {name: $name}) RETURN n", {"name": user_input})
```

### File System Access

When using persistent storage (WAL, file-backed databases), Grafeo reads and writes to the file system. Ensure:

- Database files are stored in directories with appropriate permissions
- The application has minimal required file system access
- Backup files are secured appropriately

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**For sensitive security issues**, please email: **<security@grafeo.dev>**

Do NOT create a public GitHub issue for security vulnerabilities.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: We aim to release patches within 30 days for confirmed vulnerabilities
- **Credit**: We'll acknowledge your contribution in the release notes (unless you prefer anonymity)

### Scope

The following are in scope for security reports:

- Memory safety issues (buffer overflows, use-after-free, etc.)
- Query injection vulnerabilities in the parsers
- Denial of service through malformed input
- Data corruption or integrity issues
- Unsafe deserialization

The following are **out of scope**:

- Issues requiring physical access to the machine
- Social engineering attacks
- Issues in dependencies (report these to the respective projects)
- Missing security features in the embedding application

## Security Best Practices

When embedding Grafeo in your application:

1. **Validate all user input** before constructing queries
2. **Use parameterized queries** to prevent injection
3. **Limit file system permissions** for database directories
4. **Keep Grafeo updated** to the latest supported version
5. **Review the `unsafe` code** if modifying core internals (we minimize unsafe usage)
6. **Enable security audits** in your CI pipeline:

   ```bash
   cargo audit
   ```

## Acknowledgments

We thank the security researchers who help keep Grafeo safe. Contributors will be listed here (with permission) after vulnerabilities are patched.
