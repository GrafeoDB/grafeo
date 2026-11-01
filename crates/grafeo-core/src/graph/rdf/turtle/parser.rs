//! Turtle parser.
//!
//! Parses [W3C Turtle](https://www.w3.org/TR/turtle/) format into Grafeo RDF triples.
//! Supports `@prefix`, `@base`, predicate lists (`;`), object lists (`,`),
//! blank node syntax (`_:label`, `[]`), and the `a` shorthand for `rdf:type`.

use crate::graph::rdf::sink::{TripleSink, VecSink};
use crate::graph::rdf::term::{Literal, Term, unescape_iri};
use crate::graph::rdf::triple::Triple;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique per-parser import prefixes.
/// Each `TurtleParser` instance gets a unique prefix so that blank node IDs
/// from different imports never collide.
static IMPORT_COUNTER: AtomicU64 = AtomicU64::new(0);

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";

/// Error from parsing a Turtle document.
#[derive(Debug)]
pub struct TurtleError {
    /// 1-based line number where the error occurred.
    pub line: usize,
    /// 1-based column number where the error occurred.
    pub column: usize,
    /// Description of the error.
    pub message: String,
}

impl std::fmt::Display for TurtleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Turtle parse error at {}:{}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for TurtleError {}

/// Parses Turtle text into RDF triples.
///
/// The parser supports two modes:
/// - [`parse`](Self::parse): collects all triples into a `Vec` (convenience wrapper)
/// - [`parse_into`](Self::parse_into): streams triples into any [`TripleSink`]
///
/// Use `parse_into` with a [`BatchInsertSink`](crate::graph::rdf::sink::BatchInsertSink)
/// for memory-bounded loading of large Turtle files.
pub struct TurtleParser {
    /// Prefix mappings (prefix -> namespace IRI).
    prefixes: HashMap<String, String>,
    /// Base IRI for resolving relative IRIs.
    base: Option<String>,
    /// Counter for generating anonymous blank node IDs.
    blank_counter: usize,
    /// Unique prefix for this parser instance, used to scope blank node IDs
    /// so that `_:b0` in one import does not collide with `_:b0` in another.
    import_prefix: String,
    /// Current position in the input.
    pos: usize,
    /// Input bytes.
    input: Vec<u8>,
    /// Current line number (1-based).
    line: usize,
    /// Current column number (1-based).
    column: usize,
}

impl TurtleParser {
    /// Creates a new parser.
    ///
    /// Each parser instance gets a unique import prefix so that blank node IDs
    /// (e.g. `_:b0`) from different imports never collide.
    #[must_use]
    pub fn new() -> Self {
        let counter = IMPORT_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            prefixes: HashMap::new(),
            base: None,
            blank_counter: 0,
            import_prefix: format!("imp{counter}_"),
            pos: 0,
            input: Vec::new(),
            line: 1,
            column: 1,
        }
    }

    /// Sets the base IRI for resolving relative IRIs.
    #[must_use]
    pub fn with_base(mut self, base: &str) -> Self {
        self.base = Some(base.to_string());
        self
    }

    /// Parses a Turtle string and returns the resulting triples.
    ///
    /// This is a convenience wrapper around [`parse_into`](Self::parse_into) using a
    /// [`VecSink`]. For large files, use `parse_into` with a
    /// [`BatchInsertSink`](crate::graph::rdf::sink::BatchInsertSink) to keep memory bounded.
    ///
    /// # Errors
    ///
    /// Returns a `TurtleError` if the input is malformed.
    pub fn parse(mut self, input: &str) -> Result<Vec<Triple>, TurtleError> {
        let mut sink = VecSink::new();
        self.parse_into(input, &mut sink)?;
        Ok(sink.into_triples())
    }

    /// Parses a Turtle string, streaming each triple into the provided sink.
    ///
    /// Triples are emitted as soon as they are parsed, so the sink can process
    /// them incrementally (e.g., flush to a store in batches).
    ///
    /// # Errors
    ///
    /// Returns a `TurtleError` if the input is malformed or if the sink returns
    /// an error.
    pub fn parse_into(
        &mut self,
        input: &str,
        sink: &mut dyn TripleSink,
    ) -> Result<(), TurtleError> {
        self.input = input.as_bytes().to_vec();
        self.pos = 0;
        self.line = 1;
        self.column = 1;

        while self.pos < self.input.len() {
            self.skip_ws_and_comments();
            if self.pos >= self.input.len() {
                break;
            }

            if self.starts_with(b"@prefix") {
                self.parse_prefix_directive()?;
            } else if self.starts_with(b"@base") {
                self.parse_base_directive()?;
            } else if self.starts_with(b"PREFIX") || self.starts_with(b"prefix") {
                self.parse_sparql_prefix()?;
            } else if self.starts_with(b"BASE") || self.starts_with(b"base") {
                self.parse_sparql_base()?;
            } else {
                self.parse_triples_block(sink)?;
            }
        }

        sink.finish().map_err(|e| self.error(e))?;
        Ok(())
    }

    // =========================================================================
    // Directive parsing
    // =========================================================================

    fn parse_prefix_directive(&mut self) -> Result<(), TurtleError> {
        self.advance(7); // @prefix
        self.skip_ws();
        let prefix = self.read_prefix_label()?;
        self.skip_ws();
        let iri = self.read_iri_ref()?;
        self.skip_ws();
        self.expect_byte(b'.')?;
        self.prefixes.insert(prefix, iri);
        Ok(())
    }

    fn parse_base_directive(&mut self) -> Result<(), TurtleError> {
        self.advance(5); // @base
        self.skip_ws();
        let iri = self.read_iri_ref()?;
        self.skip_ws();
        self.expect_byte(b'.')?;
        self.base = Some(iri);
        Ok(())
    }

    fn parse_sparql_prefix(&mut self) -> Result<(), TurtleError> {
        self.advance(6); // PREFIX
        self.skip_ws();
        let prefix = self.read_prefix_label()?;
        self.skip_ws();
        let iri = self.read_iri_ref()?;
        self.prefixes.insert(prefix, iri);
        Ok(())
    }

    fn parse_sparql_base(&mut self) -> Result<(), TurtleError> {
        self.advance(4); // BASE
        self.skip_ws();
        let iri = self.read_iri_ref()?;
        self.base = Some(iri);
        Ok(())
    }

    // =========================================================================
    // Triples parsing
    // =========================================================================

    fn parse_triples_block(&mut self, sink: &mut dyn TripleSink) -> Result<(), TurtleError> {
        let subject = self.read_subject(sink)?;
        self.skip_ws_and_comments();
        self.parse_predicate_object_list(&subject, sink)?;
        self.skip_ws_and_comments();
        self.expect_byte(b'.')?;
        Ok(())
    }

    fn parse_predicate_object_list(
        &mut self,
        subject: &Term,
        sink: &mut dyn TripleSink,
    ) -> Result<(), TurtleError> {
        loop {
            self.skip_ws_and_comments();
            if self.pos >= self.input.len() {
                break;
            }

            // Check for end of predicate-object list.
            if self.peek() == Some(b'.') || self.peek() == Some(b']') {
                break;
            }

            let predicate = self.read_predicate()?;
            self.skip_ws_and_comments();

            // Object list (comma-separated).
            loop {
                let object = self.read_object(sink)?;
                sink.emit(Triple::new(subject.clone(), predicate.clone(), object))
                    .map_err(|e| self.error(e))?;

                self.skip_ws_and_comments();
                if self.peek() == Some(b',') {
                    self.advance(1);
                    self.skip_ws_and_comments();
                } else {
                    break;
                }
            }

            self.skip_ws_and_comments();
            if self.peek() == Some(b';') {
                self.advance(1);
                self.skip_ws_and_comments();
                // Allow trailing semicolon before `.` or `]`.
            } else {
                break;
            }
        }
        Ok(())
    }

    // =========================================================================
    // Term reading
    // =========================================================================

    fn read_subject(&mut self, sink: &mut dyn TripleSink) -> Result<Term, TurtleError> {
        match self.peek() {
            Some(b'<') => self.read_iri_ref().map(Term::iri),
            Some(b'_') => self.read_blank_node(),
            Some(b'[') => self.read_blank_node_property_list(sink),
            Some(b'(') => self.read_collection(sink),
            _ => self.read_prefixed_name().map(Term::iri),
        }
    }

    fn read_predicate(&mut self) -> Result<Term, TurtleError> {
        // `a` shorthand for rdf:type.
        if self.peek() == Some(b'a') && self.is_keyword_boundary(1) {
            self.advance(1);
            return Ok(Term::iri(RDF_TYPE));
        }

        match self.peek() {
            Some(b'<') => self.read_iri_ref().map(Term::iri),
            _ => self.read_prefixed_name().map(Term::iri),
        }
    }

    fn read_object(&mut self, sink: &mut dyn TripleSink) -> Result<Term, TurtleError> {
        match self.peek() {
            Some(b'<') => self.read_iri_ref().map(Term::iri),
            Some(b'_') => self.read_blank_node(),
            Some(b'[') => self.read_blank_node_property_list(sink),
            Some(b'(') => self.read_collection(sink),
            Some(b'"') | Some(b'\'') => self.read_literal(),
            Some(b't') | Some(b'f') => self.try_read_boolean(),
            Some(c) if c == b'+' || c == b'-' || c.is_ascii_digit() => self.read_numeric_literal(),
            _ => self.read_prefixed_name().map(Term::iri),
        }
    }

    fn read_iri_ref(&mut self) -> Result<String, TurtleError> {
        self.expect_byte(b'<')?;
        let start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos] != b'>' {
            if self.input[self.pos] == b'\\' {
                // An escape at the end of the input ends the IRI, which then
                // has no closing '>'.
                self.pos = (self.pos + 2).min(self.input.len());
            } else {
                self.advance_byte();
            }
        }
        let written = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| self.error("invalid UTF-8 in IRI"))?;
        // Decoded before the closing '>' is checked, as an escape at the end
        // says more than the missing '>'.
        let iri_str = unescape_iri(written)
            .map_err(|error| self.error(format!("IRI: {error}")))?
            .into_owned();
        self.expect_byte(b'>')?;
        Ok(self.resolve_iri(iri_str))
    }

    fn read_prefixed_name(&mut self) -> Result<String, TurtleError> {
        let prefix = self.read_prefix_part()?;
        self.expect_byte(b':')?;
        let local = self.read_local_name();

        let namespace = self
            .prefixes
            .get(&prefix)
            .ok_or_else(|| self.error(format!("undefined prefix: {prefix}")))?
            .clone();

        Ok(format!("{namespace}{local}"))
    }

    fn read_prefix_part(&mut self) -> Result<String, TurtleError> {
        let start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos] != b':' {
            self.advance_byte();
        }
        let s = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| self.error("invalid UTF-8 in prefix"))?;
        Ok(s.to_string())
    }

    fn read_prefix_label(&mut self) -> Result<String, TurtleError> {
        let label = self.read_prefix_part()?;
        self.expect_byte(b':')?;
        Ok(label)
    }

    fn read_local_name(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            let b = self.input[self.pos];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' || b == b':' {
                // Percent-encoded characters.
                self.advance_byte();
            } else if b == b'%' && self.pos + 2 < self.input.len() {
                self.advance_byte();
                self.advance_byte();
                self.advance_byte();
            } else if b == b'\\' && self.pos + 1 < self.input.len() {
                self.advance_byte();
                self.advance_byte();
            } else {
                break;
            }
        }
        // Trim trailing dots (not valid at end of local name).
        let mut end = self.pos;
        while end > start && self.input[end - 1] == b'.' {
            end -= 1;
            self.pos = end;
            self.column -= 1;
        }
        std::str::from_utf8(&self.input[start..end])
            .unwrap_or("")
            .to_string()
    }

    fn read_blank_node(&mut self) -> Result<Term, TurtleError> {
        self.expect_byte(b'_')?;
        self.expect_byte(b':')?;
        let start = self.pos;
        while self.pos < self.input.len() {
            let b = self.input[self.pos];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' {
                self.advance_byte();
            } else {
                break;
            }
        }
        // Trim trailing dots.
        while self.pos > start && self.input[self.pos - 1] == b'.' {
            self.pos -= 1;
            self.column -= 1;
        }
        let id = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| self.error("invalid UTF-8 in blank node ID"))?;
        if id.is_empty() {
            return Err(self.error("empty blank node identifier"));
        }
        Ok(Term::blank(format!("{}{}", self.import_prefix, id)))
    }

    fn read_blank_node_property_list(
        &mut self,
        sink: &mut dyn TripleSink,
    ) -> Result<Term, TurtleError> {
        self.expect_byte(b'[')?;
        self.skip_ws_and_comments();

        let id = self.fresh_blank_id();
        let subject = Term::blank(id.as_str());

        // Empty `[]` is just an anonymous blank node.
        if self.peek() == Some(b']') {
            self.advance(1);
            return Ok(subject);
        }

        self.parse_predicate_object_list(&subject, sink)?;
        self.skip_ws_and_comments();
        self.expect_byte(b']')?;
        Ok(subject)
    }

    fn read_collection(&mut self, sink: &mut dyn TripleSink) -> Result<Term, TurtleError> {
        self.expect_byte(b'(')?;
        self.skip_ws_and_comments();

        let mut items = Vec::new();
        while self.peek() != Some(b')') {
            let item = self.read_object(sink)?;
            items.push(item);
            self.skip_ws_and_comments();
        }
        self.expect_byte(b')')?;

        if items.is_empty() {
            return Ok(Term::iri(RDF_NIL));
        }

        // Build rdf:first/rdf:rest chain.
        let first_id = self.fresh_blank_id();
        let head = Term::blank(first_id.as_str());
        let mut current = Term::blank(first_id.as_str());
        let last_idx = items.len() - 1;

        for (i, item) in items.into_iter().enumerate() {
            sink.emit(Triple::new(current.clone(), Term::iri(RDF_FIRST), item))
                .map_err(|e| self.error(e))?;

            let rest = if i == last_idx {
                Term::iri(RDF_NIL)
            } else {
                let next_id = self.fresh_blank_id();
                Term::blank(next_id.as_str())
            };

            sink.emit(Triple::new(current, Term::iri(RDF_REST), rest.clone()))
                .map_err(|e| self.error(e))?;
            current = rest;
        }

        Ok(head)
    }

    fn read_literal(&mut self) -> Result<Term, TurtleError> {
        let value = self.read_string_literal()?;

        // Check for language tag or datatype.
        if self.peek() == Some(b'@') {
            self.advance(1);
            let lang = self.read_language_tag()?;
            Ok(Term::lang_literal(value, lang))
        } else if self.peek() == Some(b'^')
            && self.pos + 1 < self.input.len()
            && self.input[self.pos + 1] == b'^'
        {
            self.advance(2); // ^^
            let datatype = if self.peek() == Some(b'<') {
                self.read_iri_ref()?
            } else {
                self.read_prefixed_name()?
            };
            Ok(Term::typed_literal(value, datatype))
        } else {
            Ok(Term::literal(value))
        }
    }

    fn read_string_literal(&mut self) -> Result<String, TurtleError> {
        let quote = self.input[self.pos];

        // Check for long string (triple-quoted).
        if self.pos + 2 < self.input.len()
            && self.input[self.pos + 1] == quote
            && self.input[self.pos + 2] == quote
        {
            return self.read_long_string(quote);
        }

        // Short string.
        self.advance(1); // opening quote
        let mut value = String::new();
        while self.pos < self.input.len() && self.input[self.pos] != quote {
            if self.input[self.pos] == b'\\' {
                value.push(self.read_escape()?);
            } else {
                let start = self.pos;
                self.advance_byte();
                // Handle multi-byte UTF-8.
                while self.pos < self.input.len() && self.input[self.pos] & 0xC0 == 0x80 {
                    self.pos += 1;
                }
                let s = std::str::from_utf8(&self.input[start..self.pos])
                    .map_err(|_| self.error("invalid UTF-8 in string literal"))?;
                value.push_str(s);
            }
        }
        if self.pos >= self.input.len() {
            return Err(self.error("unterminated string literal"));
        }
        self.advance(1); // closing quote
        Ok(value)
    }

    fn read_long_string(&mut self, quote: u8) -> Result<String, TurtleError> {
        self.advance(3); // opening triple-quote
        let mut value = String::new();
        loop {
            if self.pos >= self.input.len() {
                return Err(self.error("unterminated long string literal"));
            }
            if self.input[self.pos] == quote
                && self.pos + 2 < self.input.len()
                && self.input[self.pos + 1] == quote
                && self.input[self.pos + 2] == quote
            {
                self.advance(3); // closing triple-quote
                return Ok(value);
            }
            if self.input[self.pos] == b'\\' {
                value.push(self.read_escape()?);
            } else if self.input[self.pos] == b'\n' {
                value.push('\n');
                self.pos += 1;
                self.line += 1;
                self.column = 1;
            } else if self.input[self.pos] == b'\r' {
                self.pos += 1;
                if self.pos < self.input.len() && self.input[self.pos] == b'\n' {
                    self.pos += 1;
                }
                value.push('\n');
                self.line += 1;
                self.column = 1;
            } else {
                let start = self.pos;
                self.advance_byte();
                while self.pos < self.input.len() && self.input[self.pos] & 0xC0 == 0x80 {
                    self.pos += 1;
                }
                let s = std::str::from_utf8(&self.input[start..self.pos])
                    .map_err(|_| self.error("invalid UTF-8 in long string"))?;
                value.push_str(s);
            }
        }
    }

    fn read_escape(&mut self) -> Result<char, TurtleError> {
        self.advance(1); // backslash
        if self.pos >= self.input.len() {
            return Err(self.error("unexpected end of input in escape sequence"));
        }
        let ch = self.input[self.pos];
        self.advance(1);
        match ch {
            b'n' => Ok('\n'),
            b'r' => Ok('\r'),
            b't' => Ok('\t'),
            b'b' => Ok('\x08'),
            b'f' => Ok('\x0c'),
            b'\\' => Ok('\\'),
            b'"' => Ok('"'),
            b'\'' => Ok('\''),
            b'u' => self.read_unicode_escape(4),
            b'U' => self.read_unicode_escape(8),
            _ => Err(self.error(format!("unknown escape: \\{}", ch as char))),
        }
    }

    fn read_unicode_escape(&mut self, digits: usize) -> Result<char, TurtleError> {
        if self.pos + digits > self.input.len() {
            return Err(self.error("incomplete unicode escape"));
        }
        let hex = std::str::from_utf8(&self.input[self.pos..self.pos + digits])
            .map_err(|_| self.error("invalid UTF-8 in unicode escape"))?;
        // Only hexadecimal digits: `from_str_radix` also takes a leading sign.
        if !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(self.error(format!("invalid hex in unicode escape: {hex}")));
        }
        let code_point = u32::from_str_radix(hex, 16)
            .map_err(|_| self.error(format!("invalid hex in unicode escape: {hex}")))?;
        self.pos += digits;
        self.column += digits;
        char::from_u32(code_point).ok_or_else(|| self.error("invalid unicode code point"))
    }

    fn read_language_tag(&mut self) -> Result<String, TurtleError> {
        let start = self.pos;
        while self.pos < self.input.len() {
            let b = self.input[self.pos];
            if b.is_ascii_alphanumeric() || b == b'-' {
                self.advance_byte();
            } else {
                break;
            }
        }
        let tag = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| self.error("invalid UTF-8 in language tag"))?;
        if tag.is_empty() {
            return Err(self.error("empty language tag"));
        }
        Ok(tag.to_string())
    }

    fn try_read_boolean(&mut self) -> Result<Term, TurtleError> {
        if self.starts_with(b"true") && self.is_keyword_boundary(4) {
            self.advance(4);
            return Ok(Term::typed_literal("true", Literal::XSD_BOOLEAN));
        }
        if self.starts_with(b"false") && self.is_keyword_boundary(5) {
            self.advance(5);
            return Ok(Term::typed_literal("false", Literal::XSD_BOOLEAN));
        }
        // Not a boolean, try as prefixed name.
        self.read_prefixed_name().map(Term::iri)
    }

    fn read_numeric_literal(&mut self) -> Result<Term, TurtleError> {
        let start = self.pos;
        let mut is_double = false;
        let mut is_decimal = false;

        // Optional sign.
        if self.peek() == Some(b'+') || self.peek() == Some(b'-') {
            self.advance_byte();
        }

        // Integer part.
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.advance_byte();
        }

        // Decimal point.
        if self.peek() == Some(b'.')
            && self.pos + 1 < self.input.len()
            && self.input[self.pos + 1].is_ascii_digit()
        {
            is_decimal = true;
            self.advance_byte(); // .
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.advance_byte();
            }
        }

        // Exponent.
        if self.peek() == Some(b'e') || self.peek() == Some(b'E') {
            is_double = true;
            self.advance_byte();
            if self.peek() == Some(b'+') || self.peek() == Some(b'-') {
                self.advance_byte();
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.advance_byte();
            }
        }

        let value = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| self.error("invalid UTF-8 in numeric literal"))?;

        if value.is_empty() || value == "+" || value == "-" {
            return Err(self.error("invalid numeric literal"));
        }

        let datatype = if is_double {
            Literal::XSD_DOUBLE
        } else if is_decimal {
            Literal::XSD_DECIMAL
        } else {
            Literal::XSD_INTEGER
        };

        Ok(Term::typed_literal(value, datatype))
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    fn fresh_blank_id(&mut self) -> String {
        self.blank_counter += 1;
        format!("{}_g{}", self.import_prefix, self.blank_counter)
    }

    fn resolve_iri(&self, iri: String) -> String {
        if iri.contains(':') || self.base.is_none() {
            return iri;
        }
        // Simple base resolution: prepend base.
        if let Some(ref base) = self.base {
            if iri.is_empty() {
                return base.clone();
            }
            if iri.starts_with('#') {
                return format!("{base}{iri}");
            }
            // Strip base path and append relative.
            if let Some(last_slash) = base.rfind('/') {
                return format!("{}{iri}", &base[..=last_slash]);
            }
        }
        iri
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn starts_with(&self, prefix: &[u8]) -> bool {
        self.input[self.pos..].starts_with(prefix)
    }

    fn is_keyword_boundary(&self, offset: usize) -> bool {
        let next_pos = self.pos + offset;
        if next_pos >= self.input.len() {
            return true;
        }
        let b = self.input[next_pos];
        !b.is_ascii_alphanumeric() && b != b'_' && b != b'-'
    }

    fn advance(&mut self, count: usize) {
        for _ in 0..count {
            if self.pos < self.input.len() {
                if self.input[self.pos] == b'\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
                self.pos += 1;
            }
        }
    }

    fn advance_byte(&mut self) {
        if self.pos < self.input.len() {
            self.column += 1;
            self.pos += 1;
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\r' => self.advance(1),
                b'\n' => self.advance(1),
                _ => break,
            }
        }
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            self.skip_ws();
            if self.pos < self.input.len() && self.input[self.pos] == b'#' {
                // Skip to end of line.
                while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), TurtleError> {
        if self.pos >= self.input.len() {
            return Err(self.error(format!(
                "expected '{}', found end of input",
                expected as char
            )));
        }
        if self.input[self.pos] != expected {
            return Err(self.error(format!(
                "expected '{}', found '{}'",
                expected as char, self.input[self.pos] as char
            )));
        }
        self.advance(1);
        Ok(())
    }

    fn error(&self, message: impl Into<String>) -> TurtleError {
        TurtleError {
            line: self.line,
            column: self.column,
            message: message.into(),
        }
    }
}

impl Default for TurtleParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_triples() {
        let input = r#"
            <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
            <http://example.org/gus> <http://xmlns.com/foaf/0.1/name> "Gus" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].subject(), &Term::iri("http://example.org/alix"));
        assert_eq!(triples[0].object(), &Term::literal("Alix"));
    }

    #[test]
    fn test_parse_prefixed_names() {
        let input = r#"
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix ex: <http://example.org/> .

            ex:alix foaf:name "Alix" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject(), &Term::iri("http://example.org/alix"));
        assert_eq!(
            triples[0].predicate(),
            &Term::iri("http://xmlns.com/foaf/0.1/name")
        );
    }

    #[test]
    fn test_parse_a_shorthand() {
        let input = r#"
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            <http://example.org/alix> a foaf:Person .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].predicate(), &Term::iri(RDF_TYPE));
        assert_eq!(
            triples[0].object(),
            &Term::iri("http://xmlns.com/foaf/0.1/Person")
        );
    }

    #[test]
    fn test_parse_predicate_list() {
        let input = r#"
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix ex: <http://example.org/> .

            ex:alix foaf:name "Alix" ;
                    foaf:age 30 .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].object(), &Term::literal("Alix"));
        assert_eq!(
            triples[1].object(),
            &Term::typed_literal("30", Literal::XSD_INTEGER)
        );
    }

    #[test]
    fn test_parse_object_list() {
        let input = r#"
            @prefix ex: <http://example.org/> .
            ex:alix ex:knows ex:gus, ex:jules .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].object(), &Term::iri("http://example.org/gus"));
        assert_eq!(triples[1].object(), &Term::iri("http://example.org/jules"));
    }

    #[test]
    fn test_parse_blank_node() {
        let input = r#"
            _:b0 <http://xmlns.com/foaf/0.1/name> "Alix" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(triples[0].subject().is_blank_node());
        // The blank node ID is prefixed with the import prefix but ends with the original ID.
        let bn_id = triples[0].subject().as_blank_node().unwrap().id();
        assert!(bn_id.ends_with("b0"), "expected suffix 'b0', got: {bn_id}");
    }

    #[test]
    fn test_parse_typed_literal() {
        let input = r#"
            <http://example.org/s> <http://example.org/p> "42"^^<http://www.w3.org/2001/XMLSchema#integer> .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(
            triples[0].object(),
            &Term::typed_literal("42", Literal::XSD_INTEGER)
        );
    }

    #[test]
    fn test_parse_language_tag() {
        let input = r#"
            <http://example.org/s> <http://example.org/p> "Bonjour"@fr .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples[0].object(), &Term::lang_literal("Bonjour", "fr"));
    }

    #[test]
    fn test_parse_numeric_literals() {
        let input = r#"
            @prefix ex: <http://example.org/> .
            ex:s ex:int 42 .
            ex:s ex:decimal 3.14 .
            ex:s ex:double 1.5e10 .
            ex:s ex:negative -7 .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 4);
        assert_eq!(
            triples[0].object(),
            &Term::typed_literal("42", Literal::XSD_INTEGER)
        );
        assert_eq!(
            triples[1].object(),
            &Term::typed_literal("3.14", Literal::XSD_DECIMAL)
        );
        assert_eq!(
            triples[2].object(),
            &Term::typed_literal("1.5e10", Literal::XSD_DOUBLE)
        );
        assert_eq!(
            triples[3].object(),
            &Term::typed_literal("-7", Literal::XSD_INTEGER)
        );
    }

    #[test]
    fn test_parse_boolean() {
        let input = r#"
            <http://example.org/s> <http://example.org/active> true .
            <http://example.org/s> <http://example.org/deleted> false .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(
            triples[0].object(),
            &Term::typed_literal("true", Literal::XSD_BOOLEAN)
        );
        assert_eq!(
            triples[1].object(),
            &Term::typed_literal("false", Literal::XSD_BOOLEAN)
        );
    }

    #[test]
    fn test_parse_comments() {
        let input = r#"
            # This is a comment
            @prefix ex: <http://example.org/> .
            ex:s ex:p "value" . # inline comment
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_parse_anonymous_blank_node() {
        let input = r#"
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            [ foaf:name "Alix" ] foaf:knows <http://example.org/gus> .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        // Should produce: _:g1 foaf:name "Alix" ; _:g1 foaf:knows <...gus>
        assert_eq!(triples.len(), 2);
        assert!(triples[0].subject().is_blank_node());
        assert_eq!(triples[0].object(), &Term::literal("Alix"));
        assert_eq!(triples[0].subject(), triples[1].subject());
    }

    #[test]
    fn test_parse_escape_sequences() {
        let input = r#"
            <http://example.org/s> <http://example.org/p> "line1\nline2\ttab \"quoted\"" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(
            triples[0].object(),
            &Term::literal("line1\nline2\ttab \"quoted\"")
        );
    }

    #[test]
    fn test_parse_sparql_style_prefix() {
        let input = r#"
            PREFIX ex: <http://example.org/>
            ex:s ex:p "value" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject(), &Term::iri("http://example.org/s"));
    }

    #[test]
    fn test_parse_trailing_semicolon() {
        let input = r#"
            @prefix ex: <http://example.org/> .
            ex:s ex:p1 "a" ;
                 ex:p2 "b" ;
            .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_parse_error_position() {
        let input = "invalid content here";
        let err = TurtleParser::new().parse(input).unwrap_err();
        assert_eq!(err.line, 1);
        assert!(err.column > 0);
    }

    /// An IRI's `\u` and `\U` escapes (UCHAR in the Turtle grammar) stand for
    /// their characters, as in N-Triples; no other escape is allowed in an
    /// IRI.
    #[test]
    fn iris_decode_their_unicode_escapes() {
        // `~` stands for a backslash.
        let input = concat!(
            "<http://example.org/~u0041lix> <http://example.org/~U0001F6B2> ",
            "\"7\"^^<http://example.org/Krak~u00F3w> .",
        )
        .replace('~', "\\");
        let triples = TurtleParser::new().parse(&input).unwrap();
        assert_eq!(
            triples,
            vec![Triple::new(
                Term::iri("http://example.org/Alix"),
                Term::iri("http://example.org/🚲"),
                Term::typed_literal("7", "http://example.org/Kraków"),
            )]
        );
        for (iri, reason) in [
            ("<http://example.org/~n>", "an IRI has only"),
            ("<http://example.org/~u00F>", "needs 4 hexadecimal digits"),
            ("<http://example.org/~u+0F3>", "needs 4 hexadecimal digits"),
        ] {
            let text = format!("{iri} <http://example.org/p> \"Gus\" .").replace('~', "\\");
            let error = TurtleParser::new().parse(&text).unwrap_err();
            assert!(error.message.contains(reason), "{text}: {error}");
        }
        // An escape at the end of the input ran the IRI reader past it.
        let error = TurtleParser::new()
            .parse(&"<http://example.org/~".replace('~', "\\"))
            .unwrap_err();
        assert!(
            error.message.contains("an escape without its character"),
            "{error}"
        );
    }

    /// Turtle's string escapes (ECHAR) are those of N-Triples: `\b` and `\f`
    /// among them.
    #[test]
    fn strings_decode_the_backspace_and_form_feed_escapes() {
        // `~` stands for a backslash.
        let input = concat!(
            "<http://example.org/s> <http://example.org/p> \"Gus~bMia~fVincent\" .\n",
            "<http://example.org/s> <http://example.org/q> '''Jules~b~f''' .\n",
        )
        .replace('~', "\\");
        let triples = TurtleParser::new().parse(&input).unwrap();
        assert_eq!(triples[0].object(), &Term::literal("Gus\x08Mia\x0cVincent"));
        assert_eq!(triples[1].object(), &Term::literal("Jules\x08\x0c"));
    }

    /// A unicode escape is exactly 4 or 8 hexadecimal digits: a sign, which
    /// a radix parse would take, is no digit.
    #[test]
    fn a_unicode_escape_with_a_sign_is_refused() {
        // `~` stands for a backslash.
        for literal in ["\"~u+0F3\"", "\"~U+001F6B2\"", "\"~u-0F3\""] {
            let text = format!("<http://example.org/s> <http://example.org/p> {literal} .")
                .replace('~', "\\");
            let error = TurtleParser::new().parse(&text).unwrap_err();
            assert!(error.message.contains("hex"), "{text}: {error}");
        }
    }

    #[test]
    fn test_parse_unicode_escape_in_string() {
        // \u0048 is 'H'
        let input = r#"
            <http://example.org/s> <http://example.org/p> "\u0048ello" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].object(), &Term::literal("Hello"));
    }

    #[test]
    fn test_parse_long_string_literal() {
        let input = "<http://example.org/s> <http://example.org/p> \"\"\"Hello\nWorld\"\"\" .";
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].object(), &Term::literal("Hello\nWorld"));
    }

    #[test]
    fn test_parse_collection_syntax() {
        let input = r#"
            @prefix ex: <http://example.org/> .
            ex:s ex:p (1 2 3) .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();

        // A collection (1 2 3) produces 6 triples:
        //   _:g1 rdf:first 1 ; rdf:rest _:g2 .
        //   _:g2 rdf:first 2 ; rdf:rest _:g3 .
        //   _:g3 rdf:first 3 ; rdf:rest rdf:nil .
        // Plus the top-level: ex:s ex:p _:g1 .
        // Total: 7
        assert_eq!(triples.len(), 7);

        // The top-level triple should link subject to the collection head.
        let top = &triples[6];
        assert_eq!(top.subject(), &Term::iri("http://example.org/s"));
        assert_eq!(top.predicate(), &Term::iri("http://example.org/p"));
        assert!(top.object().is_blank_node());

        // Verify rdf:first chain values.
        let rdf_first = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#first");
        let rdf_rest = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest");
        let rdf_nil = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil");

        let first_values: Vec<_> = triples
            .iter()
            .filter(|t| t.predicate() == &rdf_first)
            .map(|t| t.object().clone())
            .collect();
        assert_eq!(first_values.len(), 3);
        assert_eq!(
            first_values[0],
            Term::typed_literal("1", Literal::XSD_INTEGER)
        );
        assert_eq!(
            first_values[1],
            Term::typed_literal("2", Literal::XSD_INTEGER)
        );
        assert_eq!(
            first_values[2],
            Term::typed_literal("3", Literal::XSD_INTEGER)
        );

        // The last rdf:rest should point to rdf:nil.
        let rest_objects: Vec<_> = triples
            .iter()
            .filter(|t| t.predicate() == &rdf_rest)
            .map(|t| t.object().clone())
            .collect();
        assert_eq!(rest_objects.len(), 3);
        assert_eq!(rest_objects[2], rdf_nil);
    }

    #[test]
    fn test_parse_escape_cr_backslash_single_quote() {
        let input = r#"
            <http://example.org/s> <http://example.org/p1> "line1\rline2" .
            <http://example.org/s> <http://example.org/p2> "back\\slash" .
            <http://example.org/s> <http://example.org/p3> "single\'quote" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 3);
        assert_eq!(triples[0].object(), &Term::literal("line1\rline2"));
        assert_eq!(triples[1].object(), &Term::literal("back\\slash"));
        assert_eq!(triples[2].object(), &Term::literal("single'quote"));
    }

    #[test]
    fn test_parse_language_tag_with_subtag() {
        let input = r#"
            <http://example.org/s> <http://example.org/p1> "chat"@fr .
            <http://example.org/s> <http://example.org/p2> "hello"@en-US .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].object(), &Term::lang_literal("chat", "fr"));
        assert_eq!(triples[1].object(), &Term::lang_literal("hello", "en-US"));
    }

    #[test]
    fn test_roundtrip() {
        let input = r#"
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix ex: <http://example.org/> .

            ex:alix a foaf:Person ;
                foaf:name "Alix" ;
                foaf:knows ex:gus .

            ex:gus foaf:name "Gus" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 4);

        // Serialize back.
        let arcs: Vec<std::sync::Arc<Triple>> =
            triples.into_iter().map(std::sync::Arc::new).collect();
        let serializer = crate::graph::rdf::turtle::TurtleSerializer::new()
            .with_prefix("ex", "http://example.org/");
        let output = serializer.to_string(&arcs).unwrap();

        // Re-parse the output.
        let reparsed = TurtleParser::new().parse(&output).unwrap();
        assert_eq!(reparsed.len(), 4);
    }

    #[test]
    fn test_blank_node_prefix_isolation_across_imports() {
        // Two separate parsers parsing the same blank node `_:b0` must produce
        // different blank node IDs so imports do not collide.
        let input = r#"
            _:b0 <http://xmlns.com/foaf/0.1/name> "Alix" .
        "#;
        let triples_a = TurtleParser::new().parse(input).unwrap();
        let triples_b = TurtleParser::new().parse(input).unwrap();

        let id_a = triples_a[0]
            .subject()
            .as_blank_node()
            .unwrap()
            .id()
            .to_string();
        let id_b = triples_b[0]
            .subject()
            .as_blank_node()
            .unwrap()
            .id()
            .to_string();
        assert_ne!(
            id_a, id_b,
            "blank node IDs from different parsers must differ"
        );
        // Both should still end with the original local name.
        assert!(id_a.ends_with("b0"));
        assert!(id_b.ends_with("b0"));
    }

    #[test]
    fn test_blank_node_prefix_preserves_internal_links() {
        // Blank nodes within a single import must still reference each other correctly.
        let input = r#"
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            _:alix foaf:name "Alix" ;
                   foaf:knows _:gus .
            _:gus foaf:name "Gus" .
        "#;
        let triples = TurtleParser::new().parse(input).unwrap();
        assert_eq!(triples.len(), 3);

        // The object of "knows" should be the same blank node as the subject of "Gus".
        let knows_object = triples[1].object();
        let gus_subject = triples[2].subject();
        assert_eq!(
            knows_object, gus_subject,
            "internal blank node references must match"
        );
    }

    #[test]
    fn test_anonymous_blank_node_prefix_isolation() {
        // Anonymous blank nodes from `[]` syntax should also get prefixed.
        let input = r#"
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            [ foaf:name "Alix" ] foaf:knows <http://example.org/gus> .
        "#;
        let triples_a = TurtleParser::new().parse(input).unwrap();
        let triples_b = TurtleParser::new().parse(input).unwrap();

        let subj_a = triples_a[0]
            .subject()
            .as_blank_node()
            .unwrap()
            .id()
            .to_string();
        let subj_b = triples_b[0]
            .subject()
            .as_blank_node()
            .unwrap()
            .id()
            .to_string();
        assert_ne!(
            subj_a, subj_b,
            "anonymous blank nodes from different parsers must differ"
        );
    }
}
