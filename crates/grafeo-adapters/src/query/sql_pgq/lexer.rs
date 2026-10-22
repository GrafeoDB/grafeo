//! SQL/PGQ Lexer.
//!
//! Tokenizes SQL/PGQ query strings (SQL:2023 GRAPH_TABLE) into a
//! stream of tokens. Handles both SQL-level syntax and GQL-style
//! graph pattern tokens inside the MATCH clause.

use grafeo_common::utils::error::SourceSpan;

/// A token in the SQL/PGQ language.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The token kind.
    pub kind: TokenKind,
    /// The source text.
    pub text: String,
    /// Source span.
    pub span: SourceSpan,
}

/// Token kinds in SQL/PGQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    // SQL keywords
    /// SELECT
    Select,
    /// FROM
    From,
    /// WHERE
    Where,
    /// AS
    As,
    /// ORDER
    Order,
    /// BY
    By,
    /// ASC
    Asc,
    /// DESC
    Desc,
    /// LIMIT
    Limit,
    /// OFFSET
    Offset,
    /// DISTINCT
    Distinct,

    // Logical operators (keywords)
    /// AND
    And,
    /// OR
    Or,
    /// NOT
    Not,
    /// IN
    In,
    /// IS
    Is,
    /// LIKE
    Like,
    /// BETWEEN
    Between,

    // Literal keywords
    /// NULL
    Null,
    /// TRUE
    True,
    /// FALSE
    False,

    // SQL/PGQ specific keywords
    /// GRAPH_TABLE
    GraphTable,
    /// MATCH
    Match,
    /// COLUMNS
    Columns,

    // DDL keywords
    /// CREATE
    Create,
    /// PROPERTY
    Property,
    /// GRAPH
    Graph,
    /// NODE
    Node,
    /// EDGE
    Edge,
    /// TABLES
    Tables,
    /// PRIMARY
    Primary,
    /// KEY
    Key,
    /// REFERENCES
    References,

    // Join keywords
    /// LEFT
    Left,
    /// OUTER
    Outer,
    /// JOIN
    Join,
    /// OPTIONAL
    Optional,

    // Grouping keywords
    /// GROUP
    Group,
    /// HAVING
    Having,

    // CASE expression keywords
    /// CASE
    Case,
    /// WHEN
    When,
    /// THEN
    Then,
    /// ELSE
    Else,
    /// END
    End,

    // Set operation keywords
    /// UNION
    Union,
    /// INTERSECT
    Intersect,
    /// EXCEPT
    Except,
    /// ALL
    All,

    // Procedure keywords
    /// CALL
    Call,
    /// YIELD
    Yield,

    // Literals
    /// Integer literal
    Integer,
    /// Float literal
    Float,
    /// String literal (single or double quoted)
    String,

    // Identifiers
    /// Identifier (unquoted)
    Identifier,
    /// Double-quoted identifier
    QuotedIdentifier,

    // Symbols
    /// (
    LParen,
    /// )
    RParen,
    /// [
    LBracket,
    /// ]
    RBracket,
    /// {
    LBrace,
    /// }
    RBrace,
    /// :
    Colon,
    /// ;
    Semicolon,
    /// ,
    Comma,
    /// .
    Dot,
    /// |
    Pipe,
    /// $
    Dollar,

    // Operators
    /// =
    Eq,
    /// <> or !=
    Ne,
    /// <
    Lt,
    /// <=
    Le,
    /// >
    Gt,
    /// >=
    Ge,
    /// +
    Plus,
    /// -
    Minus,
    /// *
    Star,
    /// /
    Slash,
    /// %
    Percent,

    // Arrows (for GQL pattern syntax inside MATCH)
    /// ->
    Arrow,
    /// <-
    LeftArrow,
    /// --
    DoubleDash,

    // Special
    /// End of input
    Eof,
    /// Lexical error
    Error,
}

/// SQL/PGQ lexer.
#[derive(Clone)]
pub struct Lexer<'a> {
    /// Source text.
    source: &'a str,
    /// Current byte position.
    pos: usize,
    /// Start of current token.
    start: usize,
    /// Current line number.
    line: usize,
    /// Column of current line.
    column: usize,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given source.
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            pos: 0,
            start: 0,
            line: 1,
            column: 1,
        }
    }

    /// Returns the next token.
    pub fn next_token(&mut self) -> Token {
        // A loop, not a recursion: a query may hold any number of comments.
        loop {
            if let Some(token) = self.scan_token() {
                return token;
            }
        }
    }

    /// Scans the next token, or skips a `-- ` line comment and returns `None`.
    fn scan_token(&mut self) -> Option<Token> {
        self.skip_whitespace_and_comments();

        self.start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        if self.is_at_end() {
            return Some(self.make_token(TokenKind::Eof, start_line, start_col));
        }

        let ch = self.advance();

        let kind = match ch {
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            ':' => TokenKind::Colon,
            ';' => TokenKind::Semicolon,
            ',' => TokenKind::Comma,
            '|' => TokenKind::Pipe,
            '$' => TokenKind::Dollar,
            '%' => TokenKind::Percent,
            '*' => TokenKind::Star,
            '/' => TokenKind::Slash,
            '+' => TokenKind::Plus,
            '.' => TokenKind::Dot,
            '=' => TokenKind::Eq,
            '<' => {
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
                if self.current_char() == '=' {
                    self.advance();
                    TokenKind::Ge
                } else {
                    TokenKind::Gt
                }
            }
            '!' => {
                if self.current_char() == '=' {
                    self.advance();
                    TokenKind::Ne
                } else {
                    TokenKind::Error
                }
            }
            '-' => {
                if self.current_char() == '>' {
                    self.advance();
                    TokenKind::Arrow
                } else if self.current_char() == '-' {
                    // Peek further: if this is `-- ` (space or EOF after), it's a SQL line comment
                    let saved_pos = self.pos;
                    let saved_col = self.column;
                    self.advance(); // consume second '-'
                    let next = self.current_char();
                    if next == ' '
                        || next == '\t'
                        || next == '\n'
                        || next == '\r'
                        || self.is_at_end()
                    {
                        // SQL line comment: skip to end of line, then
                        // scan the next real token
                        while !self.is_at_end() && self.current_char() != '\n' {
                            self.advance();
                        }
                        return None;
                    }
                    // Not a comment - restore and return DoubleDash
                    self.pos = saved_pos;
                    self.column = saved_col;
                    self.advance(); // re-consume second '-'
                    TokenKind::DoubleDash
                } else {
                    TokenKind::Minus
                }
            }
            '\'' => self.scan_string(ch),
            '"' => self.scan_quoted_identifier_or_string(ch),
            _ if ch.is_ascii_digit() => self.scan_number(),
            _ if ch.is_alphabetic() || ch == '_' => self.scan_identifier(),
            _ => TokenKind::Error,
        };

        Some(self.make_token(kind, start_line, start_col))
    }

    fn make_token(&self, kind: TokenKind, start_line: usize, start_col: usize) -> Token {
        Token {
            kind,
            text: self.source[self.start..self.pos].to_string(),
            // Source files never exceed 4 billion lines or columns
            // reason: value is bounded by format constraints
            #[allow(clippy::cast_possible_truncation)]
            span: SourceSpan::new(
                self.start,
                self.pos - self.start,
                start_line as u32,
                start_col as u32,
            ),
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.current_char() {
                ' ' | '\t' | '\r' => {
                    self.advance();
                }
                '\n' => {
                    self.advance();
                    self.line += 1;
                    self.column = 1;
                }
                '/' if self.peek_char() == '*' => {
                    // Block comment
                    self.advance(); // /
                    self.advance(); // *
                    while !self.is_at_end() {
                        if self.current_char() == '*' && self.peek_char() == '/' {
                            self.advance(); // *
                            self.advance(); // /
                            break;
                        }
                        if self.current_char() == '\n' {
                            self.line += 1;
                            self.column = 1;
                        }
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    fn scan_string(&mut self, quote: char) -> TokenKind {
        while !self.is_at_end() {
            let ch = self.current_char();
            if ch == quote {
                self.advance();
                return TokenKind::String;
            }
            if ch == '\\' {
                self.advance(); // Skip escape
                if !self.is_at_end() {
                    self.advance(); // Skip escaped char
                }
            } else if ch == '\n' {
                self.line += 1;
                self.column = 1;
                self.advance();
            } else {
                self.advance();
            }
        }
        TokenKind::Error // Unterminated string
    }

    fn scan_quoted_identifier_or_string(&mut self, _quote: char) -> TokenKind {
        // In SQL, double-quoted tokens are identifiers; treat as QuotedIdentifier
        while !self.is_at_end() {
            let ch = self.current_char();
            if ch == '"' {
                self.advance();
                return TokenKind::QuotedIdentifier;
            }
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            }
            self.advance();
        }
        TokenKind::Error // Unterminated
    }

    fn scan_number(&mut self) -> TokenKind {
        while self.current_char().is_ascii_digit() {
            self.advance();
        }

        // Check for decimal point
        if self.current_char() == '.' && self.peek_char().is_ascii_digit() {
            self.advance(); // .
            while self.current_char().is_ascii_digit() {
                self.advance();
            }
            // Check for exponent
            if self.current_char() == 'e' || self.current_char() == 'E' {
                self.advance();
                if self.current_char() == '+' || self.current_char() == '-' {
                    self.advance();
                }
                while self.current_char().is_ascii_digit() {
                    self.advance();
                }
            }
            return TokenKind::Float;
        }

        // Check for exponent without decimal
        if self.current_char() == 'e' || self.current_char() == 'E' {
            self.advance();
            if self.current_char() == '+' || self.current_char() == '-' {
                self.advance();
            }
            while self.current_char().is_ascii_digit() {
                self.advance();
            }
            return TokenKind::Float;
        }

        TokenKind::Integer
    }

    fn scan_identifier(&mut self) -> TokenKind {
        while self.current_char().is_alphanumeric() || self.current_char() == '_' {
            self.advance();
        }

        let text = &self.source[self.start..self.pos];
        Self::keyword_kind(text).unwrap_or(TokenKind::Identifier)
    }

    fn keyword_kind(text: &str) -> Option<TokenKind> {
        use crate::query::keywords::CommonKeyword;

        let upper = text.to_uppercase();
        let upper = upper.as_str();

        // Try common keywords first (shared across parsers)
        if let Some(common) = CommonKeyword::from_uppercase(upper) {
            return Some(Self::map_common_keyword(common));
        }

        // SQL/PGQ-specific keywords
        match upper {
            "SELECT" => Some(TokenKind::Select),
            "FROM" => Some(TokenKind::From),
            "OFFSET" => Some(TokenKind::Offset),
            "BETWEEN" => Some(TokenKind::Between),
            "GRAPH_TABLE" => Some(TokenKind::GraphTable),
            "COLUMNS" => Some(TokenKind::Columns),
            "LEFT" => Some(TokenKind::Left),
            "OUTER" => Some(TokenKind::Outer),
            "JOIN" => Some(TokenKind::Join),
            "OPTIONAL" => Some(TokenKind::Optional),
            "PROPERTY" => Some(TokenKind::Property),
            "GRAPH" => Some(TokenKind::Graph),
            "TABLES" => Some(TokenKind::Tables),
            "PRIMARY" => Some(TokenKind::Primary),
            "KEY" => Some(TokenKind::Key),
            "REFERENCES" => Some(TokenKind::References),
            "GROUP" => Some(TokenKind::Group),
            "UNION" => Some(TokenKind::Union),
            "INTERSECT" => Some(TokenKind::Intersect),
            "EXCEPT" => Some(TokenKind::Except),
            "ALL" => Some(TokenKind::All),
            _ => None,
        }
    }

    crate::map_common_keywords! {
        Match, Where, As, Distinct,
        Order, By, Asc, Desc, Limit,
        And, Or, Not, In, Is, Like,
        Null, True, False,
        Create, Node, Edge,
        Call, Yield, Optional, Having,
        Case, When, Then, Else, End,
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.source.len()
    }

    fn current_char(&self) -> char {
        self.source[self.pos..].chars().next().unwrap_or('\0')
    }

    fn peek_char(&self) -> char {
        let mut chars = self.source[self.pos..].chars();
        chars.next();
        chars.next().unwrap_or('\0')
    }

    fn advance(&mut self) -> char {
        let ch = self.current_char();
        self.pos += ch.len_utf8();
        self.column += 1;
        ch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokens() {
        let mut lexer = Lexer::new("()[]{}:;,.|$");
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::LBracket);
        assert_eq!(lexer.next_token().kind, TokenKind::RBracket);
        assert_eq!(lexer.next_token().kind, TokenKind::LBrace);
        assert_eq!(lexer.next_token().kind, TokenKind::RBrace);
        assert_eq!(lexer.next_token().kind, TokenKind::Colon);
        assert_eq!(lexer.next_token().kind, TokenKind::Semicolon);
        assert_eq!(lexer.next_token().kind, TokenKind::Comma);
        assert_eq!(lexer.next_token().kind, TokenKind::Dot);
        assert_eq!(lexer.next_token().kind, TokenKind::Pipe);
        assert_eq!(lexer.next_token().kind, TokenKind::Dollar);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_sql_keywords() {
        let mut lexer = Lexer::new("SELECT FROM WHERE AS ORDER BY ASC DESC LIMIT OFFSET DISTINCT");
        assert_eq!(lexer.next_token().kind, TokenKind::Select);
        assert_eq!(lexer.next_token().kind, TokenKind::From);
        assert_eq!(lexer.next_token().kind, TokenKind::Where);
        assert_eq!(lexer.next_token().kind, TokenKind::As);
        assert_eq!(lexer.next_token().kind, TokenKind::Order);
        assert_eq!(lexer.next_token().kind, TokenKind::By);
        assert_eq!(lexer.next_token().kind, TokenKind::Asc);
        assert_eq!(lexer.next_token().kind, TokenKind::Desc);
        assert_eq!(lexer.next_token().kind, TokenKind::Limit);
        assert_eq!(lexer.next_token().kind, TokenKind::Offset);
        assert_eq!(lexer.next_token().kind, TokenKind::Distinct);
    }

    #[test]
    fn test_sql_pgq_keywords() {
        let mut lexer = Lexer::new("GRAPH_TABLE MATCH COLUMNS");
        assert_eq!(lexer.next_token().kind, TokenKind::GraphTable);
        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::Columns);
    }

    #[test]
    fn test_logical_keywords() {
        let mut lexer = Lexer::new("AND OR NOT IN IS LIKE BETWEEN");
        assert_eq!(lexer.next_token().kind, TokenKind::And);
        assert_eq!(lexer.next_token().kind, TokenKind::Or);
        assert_eq!(lexer.next_token().kind, TokenKind::Not);
        assert_eq!(lexer.next_token().kind, TokenKind::In);
        assert_eq!(lexer.next_token().kind, TokenKind::Is);
        assert_eq!(lexer.next_token().kind, TokenKind::Like);
        assert_eq!(lexer.next_token().kind, TokenKind::Between);
    }

    #[test]
    fn test_literal_keywords() {
        let mut lexer = Lexer::new("NULL TRUE FALSE");
        assert_eq!(lexer.next_token().kind, TokenKind::Null);
        assert_eq!(lexer.next_token().kind, TokenKind::True);
        assert_eq!(lexer.next_token().kind, TokenKind::False);
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let mut lexer = Lexer::new("select Select SELECT graph_table Graph_Table GRAPH_TABLE");
        assert_eq!(lexer.next_token().kind, TokenKind::Select);
        assert_eq!(lexer.next_token().kind, TokenKind::Select);
        assert_eq!(lexer.next_token().kind, TokenKind::Select);
        assert_eq!(lexer.next_token().kind, TokenKind::GraphTable);
        assert_eq!(lexer.next_token().kind, TokenKind::GraphTable);
        assert_eq!(lexer.next_token().kind, TokenKind::GraphTable);
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 3.14 1e10 2.5e-3");
        assert_eq!(lexer.next_token().kind, TokenKind::Integer);
        assert_eq!(lexer.next_token().kind, TokenKind::Float);
        assert_eq!(lexer.next_token().kind, TokenKind::Float);
        assert_eq!(lexer.next_token().kind, TokenKind::Float);
    }

    #[test]
    fn test_strings() {
        let mut lexer = Lexer::new("'hello' 'world'");
        let t1 = lexer.next_token();
        assert_eq!(t1.kind, TokenKind::String);
        assert_eq!(t1.text, "'hello'");

        let t2 = lexer.next_token();
        assert_eq!(t2.kind, TokenKind::String);
        assert_eq!(t2.text, "'world'");
    }

    #[test]
    fn test_quoted_identifier() {
        let mut lexer = Lexer::new("\"my column\"");
        let token = lexer.next_token();
        assert_eq!(token.kind, TokenKind::QuotedIdentifier);
        assert_eq!(token.text, "\"my column\"");
    }

    #[test]
    fn test_arrows() {
        let mut lexer = Lexer::new("-> <-");
        assert_eq!(lexer.next_token().kind, TokenKind::Arrow);
        assert_eq!(lexer.next_token().kind, TokenKind::LeftArrow);
    }

    #[test]
    fn test_comparison_operators() {
        let mut lexer = Lexer::new("= <> != < <= > >=");
        assert_eq!(lexer.next_token().kind, TokenKind::Eq);
        assert_eq!(lexer.next_token().kind, TokenKind::Ne);
        assert_eq!(lexer.next_token().kind, TokenKind::Ne);
        assert_eq!(lexer.next_token().kind, TokenKind::Lt);
        assert_eq!(lexer.next_token().kind, TokenKind::Le);
        assert_eq!(lexer.next_token().kind, TokenKind::Gt);
        assert_eq!(lexer.next_token().kind, TokenKind::Ge);
    }

    #[test]
    fn test_arithmetic_operators() {
        let mut lexer = Lexer::new("+ - * / %");
        assert_eq!(lexer.next_token().kind, TokenKind::Plus);
        assert_eq!(lexer.next_token().kind, TokenKind::Minus);
        assert_eq!(lexer.next_token().kind, TokenKind::Star);
        assert_eq!(lexer.next_token().kind, TokenKind::Slash);
        assert_eq!(lexer.next_token().kind, TokenKind::Percent);
    }

    #[test]
    fn test_block_comments() {
        let mut lexer = Lexer::new("SELECT /* block\ncomment */ FROM");
        assert_eq!(lexer.next_token().kind, TokenKind::Select);
        assert_eq!(lexer.next_token().kind, TokenKind::From);
    }

    #[test]
    fn test_line_comments() {
        let mut lexer = Lexer::new("SELECT -- line comment\nFROM");
        assert_eq!(lexer.next_token().kind, TokenKind::Select);
        assert_eq!(lexer.next_token().kind, TokenKind::From);
    }

    #[test]
    fn test_double_dash_in_pattern() {
        // `--` without trailing space is a pattern connector, not a comment
        let mut lexer = Lexer::new("(a)--(b)");
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier);
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::DoubleDash);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier);
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
    }

    #[test]
    fn test_identifiers() {
        let mut lexer = Lexer::new("foo bar_baz x123 _private");
        let t1 = lexer.next_token();
        assert_eq!(t1.kind, TokenKind::Identifier);
        assert_eq!(t1.text, "foo");

        let t2 = lexer.next_token();
        assert_eq!(t2.kind, TokenKind::Identifier);
        assert_eq!(t2.text, "bar_baz");

        let t3 = lexer.next_token();
        assert_eq!(t3.kind, TokenKind::Identifier);
        assert_eq!(t3.text, "x123");

        let t4 = lexer.next_token();
        assert_eq!(t4.kind, TokenKind::Identifier);
        assert_eq!(t4.text, "_private");
    }

    #[test]
    fn test_whitespace_handling() {
        let mut lexer = Lexer::new("  SELECT\t\t*\n\nFROM  ");
        assert_eq!(lexer.next_token().kind, TokenKind::Select);
        assert_eq!(lexer.next_token().kind, TokenKind::Star);
        assert_eq!(lexer.next_token().kind, TokenKind::From);
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_full_graph_table_query() {
        let mut lexer = Lexer::new(
            "SELECT g.person FROM GRAPH_TABLE ( \
             MATCH (a:Person)-[e:KNOWS]->(b:Person) \
             COLUMNS (a.name AS person, b.name AS friend) \
             ) AS g WHERE g.person = 'Alix' ORDER BY g.person LIMIT 10",
        );
        assert_eq!(lexer.next_token().kind, TokenKind::Select);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // g
        assert_eq!(lexer.next_token().kind, TokenKind::Dot);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // person
        assert_eq!(lexer.next_token().kind, TokenKind::From);
        assert_eq!(lexer.next_token().kind, TokenKind::GraphTable);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen); // (
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // a
        assert_eq!(lexer.next_token().kind, TokenKind::Colon);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // Person
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Minus);
        assert_eq!(lexer.next_token().kind, TokenKind::LBracket);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // e
        assert_eq!(lexer.next_token().kind, TokenKind::Colon);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // KNOWS
        assert_eq!(lexer.next_token().kind, TokenKind::RBracket);
        assert_eq!(lexer.next_token().kind, TokenKind::Arrow); // ->
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // b
        assert_eq!(lexer.next_token().kind, TokenKind::Colon);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // Person
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Columns);
        assert_eq!(lexer.next_token().kind, TokenKind::LParen);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // a
        assert_eq!(lexer.next_token().kind, TokenKind::Dot);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // name
        assert_eq!(lexer.next_token().kind, TokenKind::As);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // person
        assert_eq!(lexer.next_token().kind, TokenKind::Comma);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // b
        assert_eq!(lexer.next_token().kind, TokenKind::Dot);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // name
        assert_eq!(lexer.next_token().kind, TokenKind::As);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // friend
        assert_eq!(lexer.next_token().kind, TokenKind::RParen);
        assert_eq!(lexer.next_token().kind, TokenKind::RParen); // closing GRAPH_TABLE
        assert_eq!(lexer.next_token().kind, TokenKind::As);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // g
        assert_eq!(lexer.next_token().kind, TokenKind::Where);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // g
        assert_eq!(lexer.next_token().kind, TokenKind::Dot);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // person
        assert_eq!(lexer.next_token().kind, TokenKind::Eq);
        assert_eq!(lexer.next_token().kind, TokenKind::String); // 'Alix'
        assert_eq!(lexer.next_token().kind, TokenKind::Order);
        assert_eq!(lexer.next_token().kind, TokenKind::By);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // g
        assert_eq!(lexer.next_token().kind, TokenKind::Dot);
        assert_eq!(lexer.next_token().kind, TokenKind::Identifier); // person
        assert_eq!(lexer.next_token().kind, TokenKind::Limit);
        assert_eq!(lexer.next_token().kind, TokenKind::Integer); // 10
        assert_eq!(lexer.next_token().kind, TokenKind::Eof);
    }

    #[test]
    fn test_span_tracking() {
        let mut lexer = Lexer::new("SELECT *");
        let t1 = lexer.next_token();
        assert_eq!(t1.kind, TokenKind::Select);
        assert_eq!(t1.span.line, 1);
        assert_eq!(t1.span.column, 1);

        let t2 = lexer.next_token();
        assert_eq!(t2.kind, TokenKind::Star);
        assert!(t2.span.column > t1.span.column);
    }

    #[test]
    fn test_multiline_span() {
        let mut lexer = Lexer::new("SELECT\n*");
        let _ = lexer.next_token(); // SELECT
        let t2 = lexer.next_token(); // *
        assert_eq!(t2.span.line, 2);
        assert_eq!(t2.span.column, 1);
    }

    #[test]
    fn test_error_invalid_char() {
        let mut lexer = Lexer::new("@");
        assert_eq!(lexer.next_token().kind, TokenKind::Error);
    }

    #[test]
    fn test_unterminated_string() {
        let mut lexer = Lexer::new("'hello");
        assert_eq!(lexer.next_token().kind, TokenKind::Error);
    }

    #[test]
    fn test_unterminated_quoted_identifier() {
        let mut lexer = Lexer::new("\"column");
        assert_eq!(lexer.next_token().kind, TokenKind::Error);
    }

    #[test]
    fn test_string_with_escape() {
        let mut lexer = Lexer::new(r"'hello\'world'");
        let token = lexer.next_token();
        assert_eq!(token.kind, TokenKind::String);
    }
}
