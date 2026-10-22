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
    /// Float literal.
    Float,
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

        let text = self.input[start..self.position].to_string();
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

    fn scan_number(&mut self) -> TokenKind {
        // Check for hex (0x/0X) or octal (0o/0O) prefix
        if self.current_char() == '0' {
            let next = self.peek_char();
            if next == 'x' || next == 'X' {
                self.advance(); // '0'
                self.advance(); // 'x'
                while self.position < self.input.len() && self.current_char().is_ascii_hexdigit() {
                    self.advance();
                }
                return TokenKind::Integer;
            }
            if next == 'o' || next == 'O' {
                self.advance(); // '0'
                self.advance(); // 'o'
                while self.position < self.input.len() && ('0'..='7').contains(&self.current_char())
                {
                    self.advance();
                }
                return TokenKind::Integer;
            }
            if next == 'b' || next == 'B' {
                self.advance(); // '0'
                self.advance(); // 'b'
                while self.position < self.input.len()
                    && (self.current_char() == '0' || self.current_char() == '1')
                {
                    self.advance();
                }
                return TokenKind::Integer;
            }
        }

        while self.position < self.input.len() && self.current_char().is_ascii_digit() {
            self.advance();
        }

        // Only consume '.' if followed by a digit (to avoid consuming '..' as part of a number)
        if self.current_char() == '.' && self.peek_char().is_ascii_digit() {
            self.advance();
            while self.position < self.input.len() && self.current_char().is_ascii_digit() {
                self.advance();
            }
            // Check for exponent after decimal
            if self.current_char() == 'e' || self.current_char() == 'E' {
                self.advance();
                if self.current_char() == '+' || self.current_char() == '-' {
                    self.advance();
                }
                while self.position < self.input.len() && self.current_char().is_ascii_digit() {
                    self.advance();
                }
            }
            TokenKind::Float
        } else if self.current_char() == 'e' || self.current_char() == 'E' {
            // Scientific notation without decimal: 1e10, 2E-3
            self.advance();
            if self.current_char() == '+' || self.current_char() == '-' {
                self.advance();
            }
            while self.position < self.input.len() && self.current_char().is_ascii_digit() {
                self.advance();
            }
            TokenKind::Float
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

        let float_token = lexer.next_token();
        assert_eq!(float_token.kind, TokenKind::Float);
        assert_eq!(float_token.text, "3.14");
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
}
