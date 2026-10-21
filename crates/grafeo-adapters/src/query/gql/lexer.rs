//! GQL Lexer.

use grafeo_common::utils::error::SourceSpan;

/// A token in the GQL language.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The token kind.
    pub kind: TokenKind,
    /// The source text.
    pub text: String,
    /// Source span.
    pub span: SourceSpan,
}

/// Token kinds in GQL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    // Keywords
    /// MATCH keyword.
    Match,
    /// RETURN keyword.
    Return,
    /// WHERE keyword.
    Where,
    /// AND keyword.
    And,
    /// OR keyword.
    Or,
    /// NOT keyword.
    Not,
    /// XOR keyword.
    Xor,
    /// CAST keyword.
    Cast,
    /// UNION keyword.
    Union,
    /// EXCEPT keyword.
    Except,
    /// INTERSECT keyword.
    Intersect,
    /// ALL keyword.
    All,
    /// OTHERWISE keyword.
    Otherwise,
    /// WALK keyword (path mode).
    Walk,
    /// TRAIL keyword (path mode).
    Trail,
    /// SIMPLE keyword (path mode).
    Simple,
    /// ACYCLIC keyword (path mode).
    Acyclic,
    /// FILTER keyword (standalone WHERE).
    Filter,
    /// GROUP keyword (for GROUP BY).
    Group,
    /// INSERT keyword.
    Insert,
    /// DELETE keyword.
    Delete,
    /// SET keyword.
    Set,
    /// REMOVE keyword.
    Remove,
    /// CREATE keyword.
    Create,
    /// NODE keyword.
    Node,
    /// EDGE keyword.
    Edge,
    /// TYPE keyword.
    Type,
    /// AS keyword.
    As,
    /// DISTINCT keyword.
    Distinct,
    /// ORDER keyword.
    Order,
    /// BY keyword.
    By,
    /// ASC keyword.
    Asc,
    /// DESC keyword.
    Desc,
    /// SKIP keyword.
    Skip,
    /// LIMIT keyword.
    Limit,
    /// NULL keyword.
    Null,
    /// TRUE keyword.
    True,
    /// FALSE keyword.
    False,
    /// DETACH keyword.
    Detach,
    /// CALL keyword.
    Call,
    /// YIELD keyword.
    Yield,
    /// IN keyword.
    In,
    /// LIKE keyword.
    Like,
    /// IS keyword.
    Is,
    /// CASE keyword.
    Case,
    /// WHEN keyword.
    When,
    /// THEN keyword.
    Then,
    /// ELSE keyword.
    Else,
    /// END keyword.
    End,
    /// OPTIONAL keyword.
    Optional,
    /// WITH keyword.
    With,
    /// EXISTS keyword for subquery expressions.
    Exists,
    /// UNWIND keyword.
    Unwind,
    /// MERGE keyword.
    Merge,
    /// HAVING keyword (for filtering aggregate results).
    Having,
    /// ON keyword (for MERGE ON CREATE/MATCH).
    On,
    /// STARTS keyword (for STARTS WITH).
    Starts,
    /// ENDS keyword (for ENDS WITH).
    Ends,
    /// CONTAINS keyword.
    Contains,
    /// FOR keyword (GQL standard list iteration).
    For,
    /// ORDINALITY keyword (FOR ... WITH ORDINALITY, 1-based index).
    Ordinality,
    /// OFFSET keyword (FOR ... WITH OFFSET, 0-based index).
    Offset,
    /// SHORTEST keyword (for path search prefix).
    Shortest,
    /// GROUPS keyword (for SHORTEST k GROUPS).
    Groups,
    /// VECTOR keyword (for vector index and type).
    Vector,
    /// INDEX keyword (for CREATE INDEX).
    Index,
    /// DIMENSION keyword (for vector dimensions).
    Dimension,
    /// METRIC keyword (for distance metric).
    Metric,
    /// NODETACH keyword (explicit non-detach delete).
    Nodetach,
    /// FETCH keyword (for FETCH FIRST n ROWS ONLY).
    Fetch,
    /// FIRST keyword (for FETCH FIRST n ROWS ONLY).
    First,
    /// ROWS keyword (for FETCH FIRST n ROWS ONLY).
    Rows,
    /// ONLY keyword (for FETCH FIRST n ROWS ONLY).
    Only,
    /// NEXT keyword (for FETCH NEXT n ROWS ONLY).
    Next,
    /// ROW keyword (for FETCH FIRST 1 ROW ONLY).
    Row,

    // Literals
    /// Integer literal.
    Integer,
    /// Approximate number literal: with an exponent or the suffix `F` or `D`.
    Float,
    /// Exact number literal: common notation with a point (`1.5`), or the
    /// suffix `M` (`1.5M`, `2M`).
    Decimal,
    /// String literal.
    String,

    // Identifiers
    /// Identifier.
    Identifier,
    /// Backtick-quoted identifier (e.g., `rdf:type`).
    QuotedIdentifier,

    // Operators
    /// = operator.
    Eq,
    /// <> operator.
    Ne,
    /// < operator.
    Lt,
    /// <= operator.
    Le,
    /// > operator.
    Gt,
    /// >= operator.
    Ge,
    /// + operator.
    Plus,
    /// - operator.
    Minus,
    /// * operator.
    Star,
    /// / operator.
    Slash,
    /// % operator.
    Percent,
    /// ^ operator (exponentiation).
    Caret,
    /// || operator.
    Concat,
    /// | pipe (label disjunction).
    Pipe,
    /// |+| multiset alternation.
    PipePlusPipe,
    /// & ampersand (label conjunction).
    Ampersand,
    /// ! exclamation (label negation).
    Exclamation,
    /// ~ tilde (undirected edge).
    Tilde,
    /// ? question mark (questioned path).
    QuestionMark,
    /// `;` statement terminator.
    Semicolon,

    // Punctuation
    /// ( punctuation.
    LParen,
    /// ) punctuation.
    RParen,
    /// [ punctuation.
    LBracket,
    /// ] punctuation.
    RBracket,
    /// { punctuation.
    LBrace,
    /// } punctuation.
    RBrace,
    /// : punctuation.
    Colon,
    /// , punctuation.
    Comma,
    /// . punctuation.
    Dot,
    /// .. range operator (for list slicing).
    DotDot,
    /// -> arrow.
    Arrow,
    /// <- arrow.
    LeftArrow,
    /// -- double dash.
    DoubleDash,

    /// Parameter ($name).
    Parameter,

    /// End of input.
    Eof,

    /// Error token.
    Error,
}

/// GQL Lexer.
pub struct Lexer<'a> {
    input: &'a str,
    position: usize,
    line: u32,
    column: u32,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given input.
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Returns the next token.
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let start = self.position;
        let start_line = self.line;
        let start_column = self.column;

        if self.position >= self.input.len() {
            return Token {
                kind: TokenKind::Eof,
                text: String::new(),
                span: SourceSpan::new(start, start, start_line, start_column),
            };
        }

        let ch = self.current_char();

        let kind = match ch {
            '(' => {
                self.advance();
                TokenKind::LParen
            }
            ')' => {
                self.advance();
                TokenKind::RParen
            }
            '[' => {
                self.advance();
                TokenKind::LBracket
            }
            ']' => {
                self.advance();
                TokenKind::RBracket
            }
            '{' => {
                self.advance();
                TokenKind::LBrace
            }
            '}' => {
                self.advance();
                TokenKind::RBrace
            }
            ':' => {
                self.advance();
                TokenKind::Colon
            }
            ',' => {
                self.advance();
                TokenKind::Comma
            }
            // `.5`: a number with no digit before its point.
            '.' if self.peek_char().is_ascii_digit() => self.scan_number(),
            '.' => {
                self.advance();
                if self.current_char() == '.' {
                    self.advance();
                    TokenKind::DotDot
                } else {
                    TokenKind::Dot
                }
            }
            '+' => {
                self.advance();
                TokenKind::Plus
            }
            '*' => {
                self.advance();
                TokenKind::Star
            }
            '/' => {
                self.advance();
                TokenKind::Slash
            }
            '%' => {
                self.advance();
                TokenKind::Percent
            }
            '^' => {
                self.advance();
                TokenKind::Caret
            }
            '=' => {
                self.advance();
                TokenKind::Eq
            }
            '<' => {
                self.advance();
                if self.current_char() == '>' {
                    self.advance();
                    TokenKind::Ne
                } else if self.current_char() == '=' {
                    self.advance();
                    TokenKind::Le
                } else if self.current_char() == '-' {
                    self.advance();
                    TokenKind::LeftArrow
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    TokenKind::Ge
                } else {
                    TokenKind::Gt
                }
            }
            '-' => {
                self.advance();
                if self.current_char() == '>' {
                    self.advance();
                    TokenKind::Arrow
                } else if self.current_char() == '-' {
                    self.advance();
                    TokenKind::DoubleDash
                } else {
                    TokenKind::Minus
                }
            }
            '|' => {
                self.advance();
                if self.current_char() == '|' {
                    self.advance();
                    TokenKind::Concat
                } else if self.current_char() == '+' && self.peek_char() == '|' {
                    self.advance(); // '+'
                    self.advance(); // '|'
                    TokenKind::PipePlusPipe
                } else {
                    TokenKind::Pipe
                }
            }
            '&' => {
                self.advance();
                TokenKind::Ampersand
            }
            '!' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    TokenKind::Ne
                } else {
                    TokenKind::Exclamation
                }
            }
            '~' => {
                self.advance();
                TokenKind::Tilde
            }
            '?' => {
                self.advance();
                TokenKind::QuestionMark
            }
            ';' => {
                self.advance();
                TokenKind::Semicolon
            }
            '\'' | '"' => self.scan_string(),
            '`' => self.scan_quoted_identifier(),
            '$' => self.scan_parameter(),
            _ if ch.is_ascii_digit() => self.scan_number(),
            _ if ch.is_alphabetic() || ch == '_' => self.scan_identifier(),
            _ => {
                self.advance();
                TokenKind::Error
            }
        };

        let mut text = self.input[start..self.position].to_string();
        // A number's digit separators are no part of its value.
        if matches!(
            kind,
            TokenKind::Integer | TokenKind::Decimal | TokenKind::Float
        ) {
            text.retain(|ch| ch != '_');
        }
        Token {
            kind,
            text,
            span: SourceSpan::new(start, self.position, start_line, start_column),
        }
    }

    fn skip_whitespace(&mut self) {
        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_whitespace() {
                if ch == '\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
                self.position += ch.len_utf8();
            } else if self.rest().starts_with("/*") {
                // Block comment: skip to */
                self.position += 2;
                self.column += 2;
                while self.position < self.input.len() {
                    if self.rest().starts_with("*/") {
                        self.position += 2;
                        self.column += 2;
                        break;
                    }
                    let c = self.current_char();
                    if c == '\n' {
                        self.line += 1;
                        self.column = 1;
                    } else {
                        self.column += 1;
                    }
                    self.position += c.len_utf8();
                }
            } else {
                // Line comment: "-- " (with space/tab) to disambiguate from undirected edge "--"
                let rest = self.rest();
                let is_line_comment = rest.starts_with("-- ")
                    || rest.starts_with("--\t")
                    || rest.starts_with("--\n")
                    || rest.starts_with("--\r");
                if is_line_comment {
                    while self.position < self.input.len() && self.current_char() != '\n' {
                        self.position += self.current_char().len_utf8();
                    }
                } else {
                    break;
                }
            }
        }
    }

    /// Returns the remaining input from the current position.
    fn rest(&self) -> &str {
        &self.input[self.position..]
    }

    fn current_char(&self) -> char {
        self.input[self.position..].chars().next().unwrap_or('\0')
    }

    fn peek_char(&self) -> char {
        let ch = self.current_char();
        let next_pos = self.position + ch.len_utf8();
        if next_pos < self.input.len() {
            self.input[next_pos..].chars().next().unwrap_or('\0')
        } else {
            '\0'
        }
    }

    fn advance(&mut self) {
        if self.position < self.input.len() {
            let ch = self.current_char();
            self.position += ch.len_utf8();
            self.column += 1;
        }
    }

    fn scan_string(&mut self) -> TokenKind {
        let quote = self.current_char();
        self.advance();

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch == quote {
                // A doubled quote is the quote itself (ISO/IEC 39075:2024
                // 21.2, <character string literal>): `'O''Brien'`.
                if self.peek_char() == quote {
                    self.advance();
                    self.advance();
                    continue;
                }
                self.advance();
                return TokenKind::String;
            }
            if ch == '\\' {
                self.advance();
            }
            self.advance();
        }

        TokenKind::Error // Unterminated string
    }

    fn scan_quoted_identifier(&mut self) -> TokenKind {
        // Consume opening backtick
        self.advance();

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch == '`' {
                // Check for escaped backtick ``
                if self.peek_char() == '`' {
                    // Skip both backticks (treat as escaped literal backtick)
                    self.advance();
                    self.advance();
                } else {
                    // Single backtick - end of identifier
                    self.advance();
                    return TokenKind::QuotedIdentifier;
                }
            } else {
                self.advance();
            }
        }

        TokenKind::Error // Unterminated quoted identifier
    }

    /// The digits `is_digit` accepts, with one underscore between two of
    /// them (ISO GQL `[ <underscore> ] <digit>`); `after_prefix` allows the
    /// underscore before the first digit too, as after `0x`.
    fn scan_digits(&mut self, is_digit: fn(char) -> bool, after_prefix: bool) {
        let mut separator_allowed = after_prefix;
        loop {
            let ch = self.current_char();
            if is_digit(ch) {
                self.advance();
                separator_allowed = true;
            } else if ch == '_' && separator_allowed && is_digit(self.peek_char()) {
                self.advance();
                separator_allowed = false;
            } else {
                return;
            }
        }
    }

    /// The character `offset` characters after the current one.
    fn char_at(&self, offset: usize) -> char {
        self.input[self.position..]
            .chars()
            .nth(offset)
            .unwrap_or('\0')
    }

    /// Whether the `.` at the current position belongs to the number before
    /// it: followed by a digit, an exponent or a suffix (`5.5`, `5.e3`,
    /// `5.M`), or by anything that starts no name and no second point (`5.`
    /// before a comma). `1..5` is a range and `5.MB` a property access.
    fn point_continues_number(&self) -> bool {
        let next = self.peek_char();
        if next.is_ascii_digit() {
            return true;
        }
        let after = self.char_at(2);
        match next.to_ascii_uppercase() {
            'E' => {
                after.is_ascii_digit()
                    || (matches!(after, '+' | '-') && self.char_at(3).is_ascii_digit())
            }
            'M' | 'F' | 'D' => !(after.is_alphanumeric() || after == '_'),
            _ => next != '.' && !(next.is_alphabetic() || next == '_'),
        }
    }

    fn scan_number(&mut self) -> TokenKind {
        // A radix prefix: hex (0x), octal (0o) or binary (0b).
        if self.current_char() == '0' {
            let is_digit: Option<fn(char) -> bool> = match self.peek_char() {
                'x' | 'X' => Some(|ch| ch.is_ascii_hexdigit()),
                'o' | 'O' => Some(|ch| ('0'..='7').contains(&ch)),
                'b' | 'B' => Some(|ch| ch == '0' || ch == '1'),
                _ => None,
            };
            if let Some(is_digit) = is_digit {
                self.advance(); // '0'
                self.advance(); // the radix letter
                self.scan_digits(is_digit, true);
                return TokenKind::Integer;
            }
        }

        self.scan_digits(|ch| ch.is_ascii_digit(), false);

        let mut point = false;
        if self.current_char() == '.' && self.point_continues_number() {
            point = true;
            self.advance();
            self.scan_digits(|ch| ch.is_ascii_digit(), false);
        }
        // Scientific notation, with or without a point: 1.5e10, 2E-3
        let mut exponent = false;
        if self.current_char() == 'e' || self.current_char() == 'E' {
            exponent = true;
            self.advance();
            if self.current_char() == '+' || self.current_char() == '-' {
                self.advance();
            }
            self.scan_digits(|ch| ch.is_ascii_digit(), false);
        }
        // ISO GQL number suffixes: `M` exact, `F` and `D` approximate; a
        // letter, digit or `_` after it makes it the start of a name instead.
        let suffix = self.current_char().to_ascii_uppercase();
        let next = self.peek_char();
        if matches!(suffix, 'M' | 'F' | 'D') && !(next.is_alphanumeric() || next == '_') {
            self.advance();
            return match (suffix, exponent) {
                // An exponent is approximate notation: `1e3M` is no number.
                ('M', true) => TokenKind::Error,
                ('M', false) => TokenKind::Decimal,
                _ => TokenKind::Float,
            };
        }
        if exponent {
            TokenKind::Float
        } else if point {
            TokenKind::Decimal
        } else {
            TokenKind::Integer
        }
    }

    fn scan_parameter(&mut self) -> TokenKind {
        // Skip the '$'
        self.advance();

        // Parameter name must start with a letter or underscore
        if self.position >= self.input.len() {
            return TokenKind::Error;
        }

        let ch = self.current_char();
        if !ch.is_alphabetic() && ch != '_' {
            return TokenKind::Error;
        }

        // Scan the rest of the identifier
        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }

        TokenKind::Parameter
    }

    fn scan_identifier(&mut self) -> TokenKind {
        let start = self.position;
        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let text = &self.input[start..self.position];
        Self::keyword_kind(text)
    }

    fn keyword_kind(text: &str) -> TokenKind {
        use crate::query::keywords::CommonKeyword;

        let upper = text.to_uppercase();
        let upper = upper.as_str();

        // Try common keywords first (shared across parsers)
        if let Some(common) = CommonKeyword::from_uppercase(upper) {
            return Self::map_common_keyword(common);
        }

        // GQL-specific keywords
        match upper {
            "INSERT" => TokenKind::Insert,
            "TYPE" => TokenKind::Type,
            "FOR" => TokenKind::For,
            "ORDINALITY" => TokenKind::Ordinality,
            "OFFSET" => TokenKind::Offset,
            "XOR" => TokenKind::Xor,
            "CAST" => TokenKind::Cast,
            "UNION" => TokenKind::Union,
            "EXCEPT" => TokenKind::Except,
            "INTERSECT" => TokenKind::Intersect,
            "ALL" => TokenKind::All,
            "OTHERWISE" => TokenKind::Otherwise,
            "FILTER" => TokenKind::Filter,
            "GROUP" => TokenKind::Group,
            "WALK" => TokenKind::Walk,
            "TRAIL" => TokenKind::Trail,
            "SIMPLE" => TokenKind::Simple,
            "ACYCLIC" => TokenKind::Acyclic,
            "SHORTEST" => TokenKind::Shortest,
            "GROUPS" => TokenKind::Groups,
            "VECTOR" => TokenKind::Vector,
            "INDEX" => TokenKind::Index,
            "DIMENSION" => TokenKind::Dimension,
            "METRIC" => TokenKind::Metric,
            "NODETACH" => TokenKind::Nodetach,
            "FETCH" => TokenKind::Fetch,
            "FIRST" => TokenKind::First,
            "NEXT" => TokenKind::Next,
            "ROWS" => TokenKind::Rows,
            "ROW" => TokenKind::Row,
            "ONLY" => TokenKind::Only,
            _ => TokenKind::Identifier,
        }
    }

    crate::map_common_keywords! {
        Match, Return, Where, As, Distinct, With, Optional,
        Order, By, Asc, Desc, Limit, Skip,
        And, Or, Not, In, Is, Like,
        Null, True, False,
        Create, Delete, Set, Remove, Merge, Detach, On,
        Call, Yield, Exists, Unwind,
        Node, Edge, Having,
        Case, When, Then, Else, End,
        Starts, Ends, Contains,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokens() {
        let mut lexer = Lexer::new("MATCH (n) RETURN n");

        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier);
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Return);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_arrow_tokens() {
        let mut lexer = Lexer::new("->  <-  --");

        assert_eq!(lexer.next_token().kind, TokenKind::Arrow);
        assert_eq!(lexer.next_token().kind, TokenKind::LeftArrow);
        assert_eq!(lexer.next_token().kind, TokenKind::DoubleDash);
    }

    #[test]
    fn test_number_tokens() {
        let mut lexer = Lexer::new("42 3.14");

        let int_token = lexer.next_token();
        assert_eq!(int_token.kind, TokenKind::Integer);
        assert_eq!(int_token.text, "42");

        // Common notation with a point is exact in ISO GQL (#610).
        let decimal_token = lexer.next_token();
        assert_eq!(decimal_token.kind, TokenKind::Decimal);
        assert_eq!(decimal_token.text, "3.14");
    }

    #[test]
    fn test_string_tokens() {
        let mut lexer = Lexer::new("'hello' \"world\"");

        let s1 = lexer.next_token();
        assert_eq!(s1.kind, TokenKind::String);
        assert_eq!(s1.text, "'hello'");

        let s2 = lexer.next_token();
        assert_eq!(s2.kind, TokenKind::String);
        assert_eq!(s2.text, "\"world\"");
    }

    /// A doubled quote of the literal's own kind continues it; the other
    /// kind does not end it either way.
    #[test]
    fn test_doubled_quote_in_string() {
        let mut lexer = Lexer::new("'O''Brien' \"say \"\"hi\"\"\" '''' 'a\"\"b' x");
        for text in ["'O''Brien'", "\"say \"\"hi\"\"\"", "''''", "'a\"\"b'"] {
            let token = lexer.next_token();
            assert_eq!(token.kind, TokenKind::String, "{text}");
            assert_eq!(token.text, text);
        }
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier);
        assert_eq!(Lexer::new("'a''").next_token().kind, TokenKind::Error);
    }

    #[test]
    fn test_parameter_tokens() {
        let mut lexer = Lexer::new("$param1 $another_param");

        let p1 = lexer.next_token();
        assert_eq!(p1.kind, TokenKind::Parameter);
        assert_eq!(p1.text, "$param1");

        let p2 = lexer.next_token();
        assert_eq!(p2.kind, TokenKind::Parameter);
        assert_eq!(p2.text, "$another_param");
    }

    #[test]
    fn test_parameter_in_query() {
        let mut lexer = Lexer::new("n.age > $min_age");

        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // n
        assert_eq!(lexer.next_token().kind, TokenKind::Dot);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // age
        assert_eq!(lexer.next_token().kind, TokenKind::Gt);

        let param = lexer.next_token();
        assert_eq!(param.kind, TokenKind::Parameter);
        assert_eq!(param.text, "$min_age");
    }

    #[test]
    fn test_quoted_identifier() {
        let mut lexer = Lexer::new("`rdf:type` `special-name`");

        let q1 = lexer.next_token();
        assert_eq!(q1.kind, TokenKind::QuotedIdentifier);
        assert_eq!(q1.text, "`rdf:type`");

        let q2 = lexer.next_token();
        assert_eq!(q2.kind, TokenKind::QuotedIdentifier);
        assert_eq!(q2.text, "`special-name`");
    }

    #[test]
    fn test_quoted_identifier_in_pattern() {
        let mut lexer = Lexer::new("(n:`rdf:type`)");

        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // n
        assert_eq!(lexer.next_token().kind, TokenKind::Colon);

        let label = lexer.next_token();
        assert_eq!(label.kind, TokenKind::QuotedIdentifier);
        assert_eq!(label.text, "`rdf:type`");

        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
    }

    #[test]
    fn test_multibyte_utf8_in_string() {
        // Regression: lexer panicked with "byte index is not a char boundary"
        // on multi-byte UTF-8 chars like ç (2 bytes), ã (2 bytes), é (2 bytes)
        let mut lexer = Lexer::new("'François'");

        let tok = lexer.next_token();
        assert_eq!(tok.kind, TokenKind::String);
        assert_eq!(tok.text, "'François'");

        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_multibyte_utf8_multi_pattern_create() {
        // The original bug report: multi-pattern CREATE with UTF-8 names
        let query = "CREATE (:Person {id: 'p1', name: 'François'}), \
                     (:Person {id: 'p2', name: 'São Paulo'})";
        let mut lexer = Lexer::new(query);

        // Should tokenize to completion without panicking
        loop {
            let tok = lexer.next_token();
            if tok.kind == TokenKind::Eof {
                break;
            }
            assert_ne!(
                tok.kind,
                TokenKind::Error,
                "Unexpected error token: {:?}",
                tok.text
            );
        }
    }

    #[test]
    fn test_multibyte_utf8_various_scripts() {
        // Test with CJK, emoji, and accented characters
        let query = "'日本語' '한국어' '中文' 'café' 'naïve'";
        let mut lexer = Lexer::new(query);

        for expected in ["'日本語'", "'한국어'", "'中文'", "'café'", "'naïve'"] {
            let tok = lexer.next_token();
            assert_eq!(tok.kind, TokenKind::String, "Failed on {expected}");
            assert_eq!(tok.text, expected);
        }

        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_line_comment_skipped() {
        let mut lexer = Lexer::new("MATCH -- this is a comment\n(n) RETURN n");

        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // n
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Return);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // n
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_block_comment_skipped() {
        let mut lexer = Lexer::new("MATCH /* multi\nline */ (n) RETURN n");

        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // n
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Return);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // n
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_double_dash_without_space_is_edge() {
        // `--` without trailing space is a double-dash (undirected edge), not a comment
        let mut lexer = Lexer::new("(a)--(b)");

        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // a
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::DoubleDash);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // b
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_xor_token() {
        let mut lexer = Lexer::new("a XOR b");

        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // a
        assert_eq!(lexer.next_token().kind, TokenKind::Xor);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // b
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_path_mode_tokens() {
        let mut lexer = Lexer::new("WALK TRAIL SIMPLE ACYCLIC");

        assert_eq!(lexer.next_token().kind, TokenKind::Walk);
        assert_eq!(lexer.next_token().kind, TokenKind::Trail);
        assert_eq!(lexer.next_token().kind, TokenKind::Simple);
        assert_eq!(lexer.next_token().kind, TokenKind::Acyclic);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_composite_query_tokens() {
        let mut lexer = Lexer::new("UNION EXCEPT INTERSECT OTHERWISE ALL");

        assert_eq!(lexer.next_token().kind, TokenKind::Union);
        assert_eq!(lexer.next_token().kind, TokenKind::Except);
        assert_eq!(lexer.next_token().kind, TokenKind::Intersect);
        assert_eq!(lexer.next_token().kind, TokenKind::Otherwise);
        assert_eq!(lexer.next_token().kind, TokenKind::All);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_cast_token() {
        let mut lexer = Lexer::new("CAST(x AS INTEGER)");

        assert_eq!(lexer.next_token().kind, TokenKind::Cast);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // x
        assert_eq!(lexer.next_token().kind, TokenKind::As);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // INTEGER
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_filter_and_group_tokens() {
        let mut lexer = Lexer::new("FILTER GROUP BY");

        assert_eq!(lexer.next_token().kind, TokenKind::Filter);
        assert_eq!(lexer.next_token().kind, TokenKind::Group);
        assert_eq!(lexer.next_token().kind, TokenKind::By);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_pipe_and_ampersand_tokens() {
        let mut lexer = Lexer::new("| &");

        assert_eq!(lexer.next_token().kind, TokenKind::Pipe);
        assert_eq!(lexer.next_token().kind, TokenKind::Ampersand);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_pipe_plus_pipe_token() {
        let mut lexer = Lexer::new("|+| | || |+|");

        assert_eq!(lexer.next_token().kind, TokenKind::PipePlusPipe);
        assert_eq!(lexer.next_token().kind, TokenKind::Pipe);
        assert_eq!(lexer.next_token().kind, TokenKind::Concat);
        assert_eq!(lexer.next_token().kind, TokenKind::PipePlusPipe);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_is_token_in_pattern() {
        let mut lexer = Lexer::new("(n IS Person)");

        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // n
        assert_eq!(lexer.next_token().kind, TokenKind::Is);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // Person
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_lbrace_rbrace_tokens() {
        // Used by ISO path quantifiers {m,n}
        let mut lexer = Lexer::new("{2,5}");

        assert_eq!(lexer.next_token().kind, TokenKind::LBrace);
        assert_eq!(lexer.next_token().kind, TokenKind::Integer);
        assert_eq!(lexer.next_token().kind, TokenKind::Comma);
        assert_eq!(lexer.next_token().kind, TokenKind::Integer);
        assert_eq!(lexer.next_token().kind, TokenKind::RBrace);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_ordinality_and_offset_tokens() {
        let mut lexer = Lexer::new("FOR x IN list WITH ORDINALITY i");

        assert_eq!(lexer.next_token().kind, TokenKind::For);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // x
        assert_eq!(lexer.next_token().kind, TokenKind::In);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // list
        assert_eq!(lexer.next_token().kind, TokenKind::With);
        assert_eq!(lexer.next_token().kind, TokenKind::Ordinality);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // i
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);

        let mut lexer = Lexer::new("FOR x IN list WITH OFFSET idx");

        assert_eq!(lexer.next_token().kind, TokenKind::For);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // x
        assert_eq!(lexer.next_token().kind, TokenKind::In);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // list
        assert_eq!(lexer.next_token().kind, TokenKind::With);
        assert_eq!(lexer.next_token().kind, TokenKind::Offset);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // idx
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_hex_integer_literals() {
        let mut lexer = Lexer::new("0xFF 0X1A 0x0 0xDEAD");

        let t1 = lexer.next_token();
        assert_eq!(t1.kind, TokenKind::Integer);
        assert_eq!(t1.text, "0xFF");

        let t2 = lexer.next_token();
        assert_eq!(t2.kind, TokenKind::Integer);
        assert_eq!(t2.text, "0X1A");

        let t3 = lexer.next_token();
        assert_eq!(t3.kind, TokenKind::Integer);
        assert_eq!(t3.text, "0x0");

        let t4 = lexer.next_token();
        assert_eq!(t4.kind, TokenKind::Integer);
        assert_eq!(t4.text, "0xDEAD");

        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_octal_integer_literals() {
        let mut lexer = Lexer::new("0o77 0O10 0o0 0o755");

        let t1 = lexer.next_token();
        assert_eq!(t1.kind, TokenKind::Integer);
        assert_eq!(t1.text, "0o77");

        let t2 = lexer.next_token();
        assert_eq!(t2.kind, TokenKind::Integer);
        assert_eq!(t2.text, "0O10");

        let t3 = lexer.next_token();
        assert_eq!(t3.kind, TokenKind::Integer);
        assert_eq!(t3.text, "0o0");

        let t4 = lexer.next_token();
        assert_eq!(t4.kind, TokenKind::Integer);
        assert_eq!(t4.text, "0o755");

        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_binary_integer_literals() {
        let mut lexer = Lexer::new("0b1010 0B1111 0b0 0b11001100");

        let t1 = lexer.next_token();
        assert_eq!(t1.kind, TokenKind::Integer);
        assert_eq!(t1.text, "0b1010");

        let t2 = lexer.next_token();
        assert_eq!(t2.kind, TokenKind::Integer);
        assert_eq!(t2.text, "0B1111");

        let t3 = lexer.next_token();
        assert_eq!(t3.kind, TokenKind::Integer);
        assert_eq!(t3.text, "0b0");

        let t4 = lexer.next_token();
        assert_eq!(t4.kind, TokenKind::Integer);
        assert_eq!(t4.text, "0b11001100");

        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    /// ISO GQL: a number in common notation with a point is exact, and so is
    /// any number with the suffix `M`; a number with an exponent or the
    /// suffix `F` or `D` is approximate (#610). A suffix is part of the number
    /// only when no identifier character follows it.
    #[test]
    fn test_exact_and_approximate_number_literals() {
        for (text, kind, expected) in [
            ("1.5", TokenKind::Decimal, "1.5"),
            ("0.000", TokenKind::Decimal, "0.000"),
            ("1.5M", TokenKind::Decimal, "1.5M"),
            ("2m", TokenKind::Decimal, "2m"),
            ("1.5F", TokenKind::Float, "1.5F"),
            ("1.5d", TokenKind::Float, "1.5d"),
            ("3D", TokenKind::Float, "3D"),
            ("1.5e3", TokenKind::Float, "1.5e3"),
            ("1e5", TokenKind::Float, "1e5"),
            ("1.5e3D", TokenKind::Float, "1.5e3D"),
            ("7", TokenKind::Integer, "7"),
        ] {
            let mut lexer = Lexer::new(text);
            let token = lexer.next_token();
            assert_eq!(
                (token.kind, token.text.as_str()),
                (kind, expected),
                "{text}"
            );
            assert_eq!(lexer.next_token().kind, TokenKind::Eof, "{text}");
        }
        // A letter after the suffix makes it an identifier, not a suffix.
        let mut lexer = Lexer::new("1.5MB");
        assert_eq!(lexer.next_token().kind, TokenKind::Decimal);
        assert_eq!(lexer.next_token().text, "MB");
    }

    #[test]
    fn test_scientific_notation_literals() {
        let mut lexer = Lexer::new("1.5e10 2.0E-3 3.14e+2 1e5");

        let t1 = lexer.next_token();
        assert_eq!(t1.kind, TokenKind::Float);
        assert_eq!(t1.text, "1.5e10");

        let t2 = lexer.next_token();
        assert_eq!(t2.kind, TokenKind::Float);
        assert_eq!(t2.text, "2.0E-3");

        let t3 = lexer.next_token();
        assert_eq!(t3.kind, TokenKind::Float);
        assert_eq!(t3.text, "3.14e+2");

        let t4 = lexer.next_token();
        assert_eq!(t4.kind, TokenKind::Float);
        assert_eq!(t4.text, "1e5");

        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_tilde_token() {
        let mut lexer = Lexer::new("~ ~");
        assert_eq!(lexer.next_token().kind, TokenKind::Tilde);
        assert_eq!(lexer.next_token().kind, TokenKind::Tilde);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    fn kinds_and_texts(text: &str) -> Vec<(TokenKind, String)> {
        let mut lexer = Lexer::new(text);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token();
            if token.kind == TokenKind::Eof {
                return tokens;
            }
            tokens.push((token.kind, token.text));
        }
    }

    /// ISO GQL writes an exact number with digits on one side of the point
    /// only (`.5`, `5.`), and such a number takes an exponent or a suffix.
    #[test]
    fn test_a_point_with_digits_on_one_side() {
        for (text, kind) in [
            (".5", TokenKind::Decimal),
            ("5.", TokenKind::Decimal),
            (".5e3", TokenKind::Float),
            ("5.e3", TokenKind::Float),
            ("5.E-3", TokenKind::Float),
            (".5M", TokenKind::Decimal),
            ("5.D", TokenKind::Float),
        ] {
            assert_eq!(kinds_and_texts(text), [(kind, text.to_string())], "{text}");
        }
        // A number before a closing bracket or a comma keeps its point.
        assert_eq!(
            kinds_and_texts("[5.,.5]"),
            [
                (TokenKind::LBracket, "[".to_string()),
                (TokenKind::Decimal, "5.".to_string()),
                (TokenKind::Comma, ",".to_string()),
                (TokenKind::Decimal, ".5".to_string()),
                (TokenKind::RBracket, "]".to_string()),
            ]
        );
    }

    /// A range, a property and a sum lex as they did before `.5` and `5.`
    /// were numbers: the point of `1..5` is the range, and a name after a
    /// point is a property.
    #[test]
    fn test_points_that_are_no_number() {
        let integer = |text: &str| (TokenKind::Integer, text.to_string());
        let token = |kind, text: &str| (kind, text.to_string());
        assert_eq!(
            kinds_and_texts("1..5"),
            [integer("1"), token(TokenKind::DotDot, ".."), integer("5")]
        );
        assert_eq!(
            kinds_and_texts("*..5"),
            [
                token(TokenKind::Star, "*"),
                token(TokenKind::DotDot, ".."),
                integer("5")
            ]
        );
        assert_eq!(
            kinds_and_texts("n.name"),
            [
                token(TokenKind::Identifier, "n"),
                token(TokenKind::Dot, "."),
                token(TokenKind::Identifier, "name"),
            ]
        );
        assert_eq!(
            kinds_and_texts("5.MB"),
            [
                integer("5"),
                token(TokenKind::Dot, "."),
                token(TokenKind::Identifier, "MB"),
            ]
        );
        assert_eq!(
            kinds_and_texts("x+1"),
            [
                token(TokenKind::Identifier, "x"),
                token(TokenKind::Plus, "+"),
                integer("1"),
            ]
        );
    }

    /// ISO GQL allows one underscore between two digits, in every radix and
    /// in the fraction and the exponent; the token's text is the number
    /// without them.
    #[test]
    fn test_digit_separators() {
        for (text, kind, digits) in [
            ("1_000", TokenKind::Integer, "1000"),
            ("1_000_000", TokenKind::Integer, "1000000"),
            ("0x7F_FF", TokenKind::Integer, "0x7FFF"),
            ("0x_7F", TokenKind::Integer, "0x7F"),
            ("0o7_7", TokenKind::Integer, "0o77"),
            ("0b1010_0101", TokenKind::Integer, "0b10100101"),
            ("1_000.000_5", TokenKind::Decimal, "1000.0005"),
            ("1e1_0", TokenKind::Float, "1e10"),
            ("1_5M", TokenKind::Decimal, "15M"),
        ] {
            assert_eq!(
                kinds_and_texts(text),
                [(kind, digits.to_string())],
                "{text}"
            );
        }
        // Not between two digits: the number ends before the underscore.
        for (text, number) in [("1__0", "1"), ("1_", "1"), ("1_e3", "1"), ("1._5", "1")] {
            let tokens = kinds_and_texts(text);
            assert_eq!(
                tokens[0],
                (TokenKind::Integer, number.to_string()),
                "{text}"
            );
            assert!(tokens.len() > 1, "{text}");
        }
    }
}
