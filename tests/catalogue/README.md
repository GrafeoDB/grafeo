# Catalogues

One file per enum that a class test covers (class test 0, 2026-11-05). Each lists the enum's variants, one
per line; `#` lines are comments. The enum's crate holds a `#[cfg(test)]` module whose exhaustive `match`
names every variant, so a new variant fails `cargo test` of that crate until it is named there and here.
The class tests read these files and key their rows by the names: each name needs a row, or "not
reachable" with a reason.
