---
title: Pull Requests
description: PR guidelines and process.
tags:
  - contributing
---

# Pull Requests

## Before Submitting

- [ ] Tests pass: `cargo test --workspace`
- [ ] No clippy warnings: `cargo clippy --workspace`
- [ ] Code formatted: `cargo fmt --all`
- [ ] Documentation updated

## Commit Messages

Use conventional commits:

```
feat: Add node caching
fix: Handle null properties correctly
docs: Update installation guide
test: Add hash index property tests
```

## PR Description

Include:

1. **What** - What does this PR do?
2. **Why** - Why is this change needed?
3. **How** - How was it implemented?
4. **Testing** - How was it tested?

## Review Process

1. Create PR from feature branch
2. CI runs automatically
3. Maintainer reviews
4. Address feedback
5. Merge when approved

## Checklist Template

```markdown
## Summary
Brief description of changes.

## Changes
- Added X
- Fixed Y
- Updated Z

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing performed
```
