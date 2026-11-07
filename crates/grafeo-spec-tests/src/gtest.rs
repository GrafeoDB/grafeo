// The `.gtest` file model and its parser, shared by the build script, which
// `include!`s this file to generate the Rust tests, and the library (`pub mod gtest`),
// whose tests check the parse rules. No inner doc comment and no `use` of a name the
// build script imports itself: both would break the `include!`.

/// A parsed `.gtest` file: its `meta:` block and its cases.
#[derive(Debug)]
pub struct GtestFile {
    /// The file's `meta:` block.
    pub meta: Meta,
    /// The file's cases, in order.
    pub tests: Vec<TestCase>,
}

/// A file's `meta:` block.
#[derive(Debug, Default)]
pub struct Meta {
    /// The language of its cases (`gql` when not given).
    pub language: String,
    /// The graph model (`lpg`, `rdf`).
    pub model: String,
    /// The spec section it covers.
    pub section: String,
    /// Its title.
    pub title: String,
    /// The dataset its cases load (`empty` when not given).
    pub dataset: String,
    /// The features its cases need.
    pub requires: Vec<String>,
    /// Its tags.
    pub tags: Vec<String>,
    /// The ISO references it covers.
    pub iso: Vec<String>,
}

/// One case of a `.gtest` file.
#[derive(Debug)]
pub struct TestCase {
    /// The case's name.
    pub name: String,
    /// The query whose result `expect` checks.
    pub query: Option<String>,
    /// Statements run in order, the last one checked.
    pub statements: Vec<String>,
    /// Statements run before the case.
    pub setup: Vec<String>,
    /// The query's parameters.
    pub params: std::collections::HashMap<String, ParamText>,
    /// The case's tags.
    pub tags: Vec<String>,
    /// The features the case needs.
    pub requires: Vec<String>,
    /// The ISO references the case covers.
    pub iso: Vec<String>,
    /// A language of its own, over the file's.
    pub language: Option<String>,
    /// A dataset of its own, over the file's.
    pub dataset: Option<String>,
    /// Why the case is skipped, when it is (`skip:`).
    pub skip: Option<String>,
    /// The spec and issue the case is expected to fail against, when it is (`xfail:`).
    pub xfail: Option<Xfail>,
    /// A probe of the runner itself (`probe:`), only in a file under `spec-probes`.
    pub probe: Option<Probe>,
    /// What the case must give.
    pub expect: Expect,
    /// One query per language, each a case of its own (rosetta files).
    pub variants: std::collections::HashMap<String, String>,
}

/// A case expected to fail (`xfail:`): it passes as an expected failure only when it fails on
/// one of its assertions or on an error of one of its statements, and a pass fails the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xfail {
    /// The spec the expected result is held against (`spec:`).
    pub spec: String,
    /// The issue that fixes it: `#N` or `fix in X.Y.Z` (`issue:`).
    pub issue: String,
}

/// A probe of the runner itself, which only a file under a `spec-probes` directory may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// The case fails as the runner itself fails (not an assertion and not a statement
    /// error), which an `xfail:` must not take for the expected failure.
    RunnerError,
}

/// A parameter value as the .gtest file writes it. Whether it was quoted
/// decides its type, so the quotes are not simply dropped: see
/// `param_value_code` for the rule every runner follows.
#[derive(Debug)]
pub enum ParamText {
    /// Written in quotes: a string, whatever it reads like. Holds the text
    /// without the quotes, YAML-level escapes resolved.
    Quoted(String),
    /// Written bare: JSON, a number, a boolean or a string.
    Bare(String),
}

/// What a case must give (`expect:`).
#[derive(Debug, Default)]
pub struct Expect {
    /// The rows, each cell as its text.
    pub rows: Vec<Vec<String>>,
    /// Whether the rows must come in this order.
    pub ordered: bool,
    /// The number of rows.
    pub count: Option<usize>,
    /// Whether there are no rows.
    pub empty: bool,
    /// Text the error's message must contain.
    pub error: Option<String>,
    /// The GQLSTATUS the error must have (`22012`), alone or next to `error`.
    pub gql_status: Option<String>,
    /// The MD5 of the sorted, pipe-delimited rows.
    pub hash: Option<String>,
    /// The decimals a float cell is compared to.
    pub precision: Option<u32>,
    /// The column names.
    pub columns: Vec<String>,
    /// Rows of RDF terms in N-Triples form, each cell compared as the exact
    /// term (kind, lexical form, datatype and language); see `assert_terms`.
    pub terms: Vec<Vec<String>>,
    /// The codes of the warnings the statement raises, in the result's order
    /// (by code): none without the key, so every test that runs to a result
    /// also checks that it raises no other warning.
    pub warnings: Vec<String>,
}

/// Parses the `.gtest` file at `path` (used in errors, and to tell a `spec-probes` file) whose
/// content is `text`.
///
/// # Errors
///
/// The first thing the file gets wrong, naming the case where there is one: a missing
/// `meta:` or `tests:` block, an unknown `expect:` key, a block-scalar param, an `xfail:`
/// without its `spec:` or `issue:`, a case both `skip:` and `xfail:`, or a `probe:` outside a
/// `spec-probes` directory.
pub fn parse_gtest_text(path: &str, text: &str) -> Result<GtestFile, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut idx = 0;
    skip_blank_and_comments(&lines, &mut idx);
    let meta = parse_meta(&lines, &mut idx)?;
    let tests = parse_tests(path, &lines, &mut idx)?;
    Ok(GtestFile { meta, tests })
}

/// Whether `path` lies under a directory named exactly `spec-probes`.
fn in_spec_probes(path: &str) -> bool {
    path.split(['/', '\\'])
        .any(|component| component == "spec-probes")
}

fn parse_meta(lines: &[&str], idx: &mut usize) -> Result<Meta, String> {
    let mut meta = Meta::default();

    // Expect "meta:"
    skip_blank_and_comments(lines, idx);
    if *idx >= lines.len() || lines[*idx].trim() != "meta:" {
        return Err(format!(
            "Expected 'meta:' at line {}, found: '{}'",
            *idx + 1,
            lines.get(*idx).unwrap_or(&"<EOF>")
        ));
    }
    *idx += 1;

    // Parse meta fields (indented key: value pairs)
    while *idx < lines.len() {
        let line = lines[*idx];
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            *idx += 1;
            continue;
        }
        // If not indented, we've left the meta block
        if !line.starts_with(' ') && !line.starts_with('\t') {
            break;
        }
        if let Some((key, value)) = parse_kv(trimmed) {
            match key {
                "language" => meta.language = value.to_string(),
                "model" => meta.model = value.to_string(),
                "section" => meta.section = unquote(value),
                "title" => meta.title = value.to_string(),
                "dataset" => meta.dataset = value.to_string(),
                "requires" => meta.requires = parse_yaml_list(value),
                "tags" => meta.tags = parse_yaml_list(value),
                "iso" => meta.iso = parse_yaml_list(value),
                _ => {} // ignore unknown keys
            }
        }
        *idx += 1;
    }

    // Default language
    if meta.language.is_empty() {
        meta.language = "gql".to_string();
    }
    if meta.dataset.is_empty() {
        meta.dataset = "empty".to_string();
    }

    Ok(meta)
}

fn parse_tests(path: &str, lines: &[&str], idx: &mut usize) -> Result<Vec<TestCase>, String> {
    let mut tests = Vec::new();

    skip_blank_and_comments(lines, idx);

    // Expect "tests:"
    if *idx >= lines.len() || lines[*idx].trim() != "tests:" {
        return Err(format!(
            "Expected 'tests:' at line {}, found: '{}'",
            *idx + 1,
            lines.get(*idx).unwrap_or(&"<EOF>")
        ));
    }
    *idx += 1;

    // Parse list of test cases
    while *idx < lines.len() {
        skip_blank_and_comments(lines, idx);
        if *idx >= lines.len() {
            break;
        }

        let trimmed = lines[*idx].trim();

        // Each test starts with "- name: <test_name>"
        if trimmed.starts_with("- name:") {
            let test = parse_single_test(path, lines, idx)?;
            tests.push(test);
        } else {
            break;
        }
    }

    Ok(tests)
}

fn parse_single_test(path: &str, lines: &[&str], idx: &mut usize) -> Result<TestCase, String> {
    let mut tc = TestCase {
        name: String::new(),
        query: None,
        statements: Vec::new(),
        setup: Vec::new(),
        params: std::collections::HashMap::new(),
        tags: Vec::new(),
        requires: Vec::new(),
        iso: Vec::new(),
        language: None,
        dataset: None,
        skip: None,
        xfail: None,
        probe: None,
        expect: Expect::default(),
        variants: std::collections::HashMap::new(),
    };

    // First line: "- name: xxx"
    let first = lines[*idx].trim();
    if let Some((_, name)) = parse_kv(&first[2..]) {
        // strip leading "- "
        tc.name = unquote(name.trim());
    }
    *idx += 1;

    // Parse remaining fields until next "- name:" or end
    while *idx < lines.len() {
        let line = lines[*idx];
        let trimmed = line.trim();

        // Skip comments
        if trimmed.starts_with('#') {
            *idx += 1;
            continue;
        }

        // End of this test: next test item or end of list
        if trimmed.starts_with("- name:") {
            break;
        }

        // Blank lines are OK
        if trimmed.is_empty() {
            *idx += 1;
            continue;
        }

        if let Some((key, value)) = parse_kv(trimmed) {
            match key {
                "query" => {
                    if value == "|" {
                        tc.query = Some(parse_block_scalar(lines, idx));
                    } else {
                        tc.query = Some(unquote(value));
                        *idx += 1;
                    }
                }
                "skip" => {
                    tc.skip = Some(unquote(value));
                    *idx += 1;
                }
                "xfail" => {
                    tc.xfail = Some(
                        parse_xfail(lines, idx).map_err(|e| format!("test '{}': {e}", tc.name))?,
                    );
                }
                "probe" => {
                    if !in_spec_probes(path) {
                        return Err(format!(
                            "test '{}': `probe:` is for the runner probes only, under a \
                             `spec-probes` directory, not in {path}",
                            tc.name
                        ));
                    }
                    tc.probe = Some(match unquote(value).as_str() {
                        "runner_error" => Probe::RunnerError,
                        other => {
                            return Err(format!(
                                "test '{}': unknown probe '{other}' (known: runner_error)",
                                tc.name
                            ));
                        }
                    });
                    *idx += 1;
                }
                "setup" => {
                    *idx += 1;
                    tc.setup = parse_string_list(lines, idx);
                }
                "statements" => {
                    *idx += 1;
                    tc.statements = parse_string_list(lines, idx);
                }
                "tags" => {
                    tc.tags = parse_yaml_list(value);
                    *idx += 1;
                }
                "requires" => {
                    tc.requires = parse_yaml_list(value);
                    *idx += 1;
                }
                "iso" => {
                    tc.iso = parse_yaml_list(value);
                    *idx += 1;
                }
                "language" => {
                    tc.language = Some(unquote(value));
                    *idx += 1;
                }
                "dataset" => {
                    tc.dataset = Some(unquote(value));
                    *idx += 1;
                }
                "params" => {
                    *idx += 1;
                    tc.params =
                        parse_params(lines, idx).map_err(|e| format!("test '{}': {e}", tc.name))?;
                }
                "expect" => {
                    *idx += 1;
                    tc.expect = parse_expect(lines, idx)?;
                }
                "variants" => {
                    *idx += 1;
                    tc.variants = parse_variants(lines, idx);
                }
                _ => {
                    *idx += 1;
                }
            }
        } else {
            *idx += 1;
        }
    }

    if tc.skip.is_some() && tc.xfail.is_some() {
        return Err(format!(
            "test '{}': a case is either skipped or expected to fail, not both",
            tc.name
        ));
    }
    Ok(tc)
}

/// The `spec:` and `issue:` of an `xfail:` whose line is `lines[*idx]`, both required; leaves
/// `idx` past them.
fn parse_xfail(lines: &[&str], idx: &mut usize) -> Result<Xfail, String> {
    let indent = |line: &str| line.len() - line.trim_start().len();
    let own = indent(lines[*idx]);
    *idx += 1;
    let (mut spec, mut issue) = (None, None);
    while *idx < lines.len() {
        let line = lines[*idx];
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            *idx += 1;
            continue;
        }
        if indent(line) <= own {
            break;
        }
        match parse_kv(trimmed) {
            Some(("spec", value)) => spec = Some(unquote(value)),
            Some(("issue", value)) => issue = Some(unquote(value)),
            _ => {
                return Err(format!(
                    "unknown xfail key at line {}: '{trimmed}'",
                    *idx + 1
                ));
            }
        }
        *idx += 1;
    }
    let named = |value: Option<String>, key: &str| {
        value
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("an xfail names its `{key}:`"))
    };
    Ok(Xfail {
        spec: named(spec, "spec")?,
        issue: named(issue, "issue")?,
    })
}

fn parse_expect(lines: &[&str], idx: &mut usize) -> Result<Expect, String> {
    let mut expect = Expect::default();
    // The indent of the expectation's own keys: a line indented less is the
    // test's next key (a `skip:`, `params:` or `note:` after `expect:`).
    let mut child_indent: Option<usize> = None;

    while *idx < lines.len() {
        let trimmed = lines[*idx].trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            *idx += 1;
            continue;
        }

        // End if we hit a non-indented line or next test
        if trimmed.starts_with("- name:") {
            break;
        }
        let indent = lines[*idx].len() - lines[*idx].trim_start().len();
        if indent < *child_indent.get_or_insert(indent) {
            break;
        }

        // Check if we're at a peer-level key (not indented enough for expect)
        if !lines[*idx].starts_with(' ') && !lines[*idx].starts_with('\t') {
            break;
        }

        if let Some((key, value)) = parse_kv(trimmed) {
            match key {
                "ordered" => {
                    expect.ordered = value == "true";
                    *idx += 1;
                }
                "count" => {
                    expect.count = value.parse().ok();
                    *idx += 1;
                }
                "empty" => {
                    expect.empty = value == "true";
                    *idx += 1;
                }
                "error" => {
                    expect.error = Some(unquote(value));
                    *idx += 1;
                }
                "gql_status" => {
                    expect.gql_status = Some(unquote(value));
                    *idx += 1;
                }
                "hash" => {
                    expect.hash = Some(unquote(value));
                    *idx += 1;
                }
                "precision" => {
                    expect.precision = value.parse().ok();
                    *idx += 1;
                }
                "columns" => {
                    expect.columns = parse_yaml_list(value);
                    *idx += 1;
                }
                "rows" => {
                    *idx += 1;
                    expect.rows = parse_rows(lines, idx);
                }
                "terms" => {
                    *idx += 1;
                    expect.terms = parse_rows(lines, idx);
                }
                "warnings" => {
                    expect.warnings = parse_yaml_list(value);
                    *idx += 1;
                }
                // A key no runner reads would leave a test that checks
                // nothing and passes.
                other => return Err(format!("unknown expect key '{other}'")),
            }
        } else {
            // Could be a continuation or unknown
            break;
        }
    }

    Ok(expect)
}

fn parse_rows(lines: &[&str], idx: &mut usize) -> Vec<Vec<String>> {
    let mut rows = Vec::new();

    while *idx < lines.len() {
        let trimmed = lines[*idx].trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            *idx += 1;
            continue;
        }

        // A row line starts with "- ["
        if trimmed.starts_with("- [") {
            let inner = &trimmed[2..]; // strip "- "
            let values = parse_yaml_inline_list(inner);
            rows.push(values);
            *idx += 1;
        } else {
            break;
        }
    }

    rows
}

fn parse_string_list(lines: &[&str], idx: &mut usize) -> Vec<String> {
    let mut items = Vec::new();

    while *idx < lines.len() {
        let trimmed = lines[*idx].trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            *idx += 1;
            continue;
        }

        if let Some(value) = trimmed.strip_prefix("- ") {
            if value == "|" {
                items.push(parse_block_scalar(lines, idx));
            } else {
                items.push(unquote(value));
                *idx += 1;
            }
        } else {
            break;
        }
    }

    items
}

/// Parses the `params` map. A block scalar (`|`, `>` and their variants) is
/// an error: a param is written on one line, and its lines would otherwise
/// be read as more params (`Amsterdam: 3` as a param `Amsterdam`).
fn parse_params(
    lines: &[&str],
    idx: &mut usize,
) -> Result<std::collections::HashMap<String, ParamText>, String> {
    let mut params = std::collections::HashMap::new();

    while *idx < lines.len() {
        let trimmed = lines[*idx].trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            *idx += 1;
            continue;
        }

        if let Some((key, value)) = parse_kv(trimmed) {
            // Check indent level: params entries are deeply nested (indent >= 6)
            let indent = lines[*idx].len() - lines[*idx].trim_start().len();
            if indent >= 6 {
                // Still in params
                if is_block_scalar_indicator(value) {
                    return Err(format!(
                        "param '{key}' at line {} is a block scalar (`{value}`), which params do \
                         not support: write the value on one line, as JSON for a list or map",
                        *idx + 1
                    ));
                }
                let text = if is_quoted(value) {
                    ParamText::Quoted(unquote(value))
                } else {
                    ParamText::Bare(value.to_string())
                };
                params.insert(key.to_string(), text);
                *idx += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    Ok(params)
}

/// `|` or `>`, with an optional chomping indicator (`+`, `-`) and
/// indentation digit: the value of a YAML block scalar starts on the next
/// line.
fn is_block_scalar_indicator(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some('|' | '>'))
        && chars.all(|c| c == '+' || c == '-' || c.is_ascii_digit())
}

fn parse_variants(lines: &[&str], idx: &mut usize) -> std::collections::HashMap<String, String> {
    let mut variants = std::collections::HashMap::new();

    while *idx < lines.len() {
        let trimmed = lines[*idx].trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            *idx += 1;
            continue;
        }

        if let Some((key, value)) = parse_kv(trimmed) {
            let indent = lines[*idx].len() - lines[*idx].trim_start().len();
            if indent >= 6 {
                if value == "|" {
                    variants.insert(key.to_string(), parse_block_scalar(lines, idx));
                } else {
                    variants.insert(key.to_string(), unquote(value));
                    *idx += 1;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }

    variants
}

// ---------------------------------------------------------------------------
// YAML primitives
// ---------------------------------------------------------------------------

fn parse_kv(s: &str) -> Option<(&str, &str)> {
    // Find first colon not inside quotes
    let mut in_single = false;
    let mut in_double = false;
    for (i, c) in s.char_indices() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            ':' if !in_single && !in_double => {
                let key = s[..i].trim();
                let value = s[i + 1..].trim();
                if !key.is_empty() {
                    return Some((key, value));
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `s` is written in single or double quotes.
fn is_quoted(s: &str) -> bool {
    let s = s.trim();
    s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if is_quoted(s) {
        let inner = &s[1..s.len() - 1];
        // Only unescape YAML-level escapes (quotes and backslashes).
        // Do NOT process \n or \t here: those are GQL string escapes
        // that the GQL parser handles via unescape_string().
        inner
            .replace("\\\\", "\x00")
            .replace("\\\"", "\"")
            .replace("\\'", "'")
            .replace('\x00', "\\")
    } else {
        s.to_string()
    }
}

fn parse_yaml_list(s: &str) -> Vec<String> {
    let s = s.trim();
    if s == "[]" || s.is_empty() {
        return Vec::new();
    }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        inner
            .split(',')
            .map(|v| unquote(v.trim()))
            .filter(|v| !v.is_empty())
            .collect()
    } else {
        vec![unquote(s)]
    }
}

fn parse_yaml_inline_list(s: &str) -> Vec<String> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return vec![unquote(s)];
    }
    let inner = &s[1..s.len() - 1];

    // Split by commas, but respect nested brackets and quotes
    let mut items = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut in_single = false;
    let mut in_double = false;

    for c in inner.chars() {
        match c {
            '\'' if !in_double && depth == 0 => {
                in_single = !in_single;
                current.push(c);
            }
            '"' if !in_single && depth == 0 => {
                in_double = !in_double;
                current.push(c);
            }
            '[' | '{' if !in_single && !in_double => {
                depth += 1;
                current.push(c);
            }
            ']' | '}' if !in_single && !in_double => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 && !in_single && !in_double => {
                items.push(unquote(current.trim()));
                current.clear();
            }
            _ => current.push(c),
        }
    }
    let last = current.trim();
    if !last.is_empty() {
        items.push(unquote(last));
    }

    items
}

fn parse_block_scalar(lines: &[&str], idx: &mut usize) -> String {
    // Current line ends with "|", advance past it
    *idx += 1;

    // Determine indent of block content
    let block_indent = if *idx < lines.len() {
        let line = lines[*idx];
        line.len() - line.trim_start().len()
    } else {
        return String::new();
    };

    let mut result = String::new();
    while *idx < lines.len() {
        let line = lines[*idx];
        let trimmed = line.trim();

        if trimmed.is_empty() {
            result.push('\n');
            *idx += 1;
            continue;
        }

        let current_indent = line.len() - line.trim_start().len();
        if current_indent < block_indent {
            break;
        }

        if !result.is_empty() {
            result.push('\n');
        }
        // Strip the block indent
        if line.len() >= block_indent {
            result.push_str(&line[block_indent..]);
        } else {
            result.push_str(trimmed);
        }
        *idx += 1;
    }

    result.trim_end().to_string()
}

fn skip_blank_and_comments(lines: &[&str], idx: &mut usize) {
    while *idx < lines.len() {
        let trimmed = lines[*idx].trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            *idx += 1;
        } else {
            break;
        }
    }
}
