# Deprecations

A deprecated query form still runs and returns a warning with its result: a GQLSTATUS of class
01 and a message that names its replacement and the release that refuses it. Each binding shows
the warnings with the result; Python also raises a `DeprecationWarning`.

| Code | Form | Since | Removed in | Replacement |
| --- | --- | --- | --- | --- |
| 01X01 | `ZONED DATETIME '...'` | 0.6.2 | 0.7.0 | `DATETIME '...'` with an offset |
| 01X02 | `+` on two lists in GQL and SQL/PGQ | 0.6.2 | 0.7.0 | `\|\|` |
| 01X03 | `ZONED TIME '...'` | 0.6.2 | 0.7.0 | `TIME '...'` with an offset |
