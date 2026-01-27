//! Cypher Lexer.
//!
//! Tokenizes Cypher query strings into a stream of tokens.

use graphos_common::utils::error::SourceSpan;

/// A token in the Cypher language.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The token kind.
    pub kind: TokenKind,
    /// The source text.
    pub text: String,
    /// Source span.
    pub span: SourceSpan,
}

/// Token kinds in Cypher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    // Keywords
    /// MATCH
    Match,
    /// OPTIONAL
    Optional,
    /// WHERE
    Where,
    /// RETURN
    Return,
    /// WITH
    With,
    /// UNWIND
    Unwind,
    /// AS
    As,
    /// ORDER
    Order,
    /// BY
    By,
    /// ASC
    Asc,
    /// ASCENDING
    Ascending,
    /// DESC
    Desc,
    /// DESCENDING
    Descending,
    /// SKIP
    Skip,
    /// LIMIT
    Limit,
    /// CREATE
    Create,
    /// MERGE
    Merge,
    /// DELETE
    Delete,
    /// DETACH
    Detach,
    /// SET
    Set,
    /// REMOVE
    Remove,
    /// ON
    On,
    /// AND
    And,
    /// OR
    Or,
    /// XOR
    Xor,
    /// NOT
    Not,
    /// IN
    In,
    /// IS
    Is,
    /// NULL
    Null,
    /// TRUE
    True,
    /// FALSE
    False,
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
    /// DISTINCT
    Distinct,
    /// EXISTS
    Exists,
    /// COUNT
    Count,
    /// STARTS
    Starts,
    /// ENDS
    Ends,
    /// CONTAINS
    Contains,
    /// CALL
    Call,
    /// YIELD
    Yield,
    /// UNION
    Union,
    /// ALL
    All,

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
    /// Backtick-quoted identifier
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
    /// ..
    DotDot,
    /// |
    Pipe,
    /// $
    Dollar,

    // Operators
    /// =
    Eq,
    /// <>
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
    /// ^
    Caret,
    /// +=
    PlusEq,
    /// =~
    RegexMatch,

    // Arrows
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

/// Cypher lexer.
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
        self.skip_whitespace_and_comments();

        self.start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        if self.is_at_end() {
            return self.make_token(TokenKind::Eof, start_line, start_col);
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
            '^' => TokenKind::Caret,
            '%' => TokenKind::Percent,
            '*' => TokenKind::Star,
            '/' => TokenKind::Slash,
            '.' => {
                if self.current_char() == '.' {
                    self.advance();
                    TokenKind::DotDot
                } else {
                    TokenKind::Dot
                }
            }
            '+' => {
                if self.current_char() == '=' {
                    self.advance();
                    TokenKind::PlusEq
                } else {
                    TokenKind::Plus
                }
            }
            '=' => {
                if self.current_char() == '~' {
                    self.advance();
                    TokenKind::RegexMatch
                } else {
                    TokenKind::Eq
                }
            }
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
            '-' => {
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
            '\'' | '"' => self.scan_string(ch),
            '`' => self.scan_quoted_identifier(),
            _ if ch.is_ascii_digit() => self.scan_number(),
            _ if ch.is_ascii_alphabetic() || ch == '_' => self.scan_identifier(),
            _ => TokenKind::Error,
        };

        self.make_token(kind, start_line, start_col)
    }

    fn make_token(&self, kind: TokenKind, start_line: usize, start_col: usize) -> Token {
        Token {
            kind,
            text: self.source[self.start..self.pos].to_string(),
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
                '/' if self.peek_char() == '/' => {
                    // Line comment
                    while !self.is_at_end() && self.current_char() != '\n' {
                        self.advance();
                    }
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

    fn scan_quoted_identifier(&mut self) -> TokenKind {
        while !self.is_at_end() && self.current_char() != '`' {
            if self.current_char() == '\n' {
                self.line += 1;
                self.column = 1;
            }
            self.advance();
        }
        if self.is_at_end() {
            return TokenKind::Error;
        }
        self.advance(); // Closing backtick
        TokenKind::QuotedIdentifier
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
        while self.current_char().is_ascii_alphanumeric() || self.current_char() == '_' {
            self.advance();
        }

        let text = &self.source[self.start..self.pos];
        Self::keyword_kind(text).unwrap_or(TokenKind::Identifier)
    }

    fn keyword_kind(text: &str) -> Option<TokenKind> {
        // Case-insensitive keyword matching
        match text.to_uppercase().as_str() {
            "MATCH" => Some(TokenKind::Match),
            "OPTIONAL" => Some(TokenKind::Optional),
            "WHERE" => Some(TokenKind::Where),
            "RETURN" => Some(TokenKind::Return),
            "WITH" => Some(TokenKind::With),
            "UNWIND" => Some(TokenKind::Unwind),
            "AS" => Some(TokenKind::As),
            "ORDER" => Some(TokenKind::Order),
            "BY" => Some(TokenKind::By),
            "ASC" => Some(TokenKind::Asc),
            "ASCENDING" => Some(TokenKind::Ascending),
            "DESC" => Some(TokenKind::Desc),
            "DESCENDING" => Some(TokenKind::Descending),
            "SKIP" => Some(TokenKind::Skip),
            "LIMIT" => Some(TokenKind::Limit),
            "CREATE" => Some(TokenKind::Create),
            "MERGE" => Some(TokenKind::Merge),
            "DELETE" => Some(TokenKind::Delete),
            "DETACH" => Some(TokenKind::Detach),
            "SET" => Some(TokenKind::Set),
            "REMOVE" => Some(TokenKind::Remove),
            "ON" => Some(TokenKind::On),
            "AND" => Some(TokenKind::And),
            "OR" => Some(TokenKind::Or),
            "XOR" => Some(TokenKind::Xor),
            "NOT" => Some(TokenKind::Not),
            "IN" => Some(TokenKind::In),
            "IS" => Some(TokenKind::Is),
            "NULL" => Some(TokenKind::Null),
            "TRUE" => Some(TokenKind::True),
            "FALSE" => Some(TokenKind::False),
            "CASE" => Some(TokenKind::Case),
            "WHEN" => Some(TokenKind::When),
            "THEN" => Some(TokenKind::Then),
            "ELSE" => Some(TokenKind::Else),
            "END" => Some(TokenKind::End),
            "DISTINCT" => Some(TokenKind::Distinct),
            "EXISTS" => Some(TokenKind::Exists),
            "COUNT" => Some(TokenKind::Count),
            "STARTS" => Some(TokenKind::Starts),
            "ENDS" => Some(TokenKind::Ends),
            "CONTAINS" => Some(TokenKind::Contains),
            "CALL" => Some(TokenKind::Call),
            "YIELD" => Some(TokenKind::Yield),
            "UNION" => Some(TokenKind::Union),
            "ALL" => Some(TokenKind::All),
            _ => None,
        }
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
    fn test_keywords() {
        let mut lexer = Lexer::new("MATCH WHERE RETURN CREATE");
        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::Where);
        assert_eq!(lexer.next_token().kind, TokenKind::Return);
        assert_eq!(lexer.next_token().kind, TokenKind::Create);
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let mut lexer = Lexer::new("match Match MATCH");
        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::Match);
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
        let mut lexer = Lexer::new("'hello' \"world\"");
        let t1 = lexer.next_token();
        assert_eq!(t1.kind, TokenKind::String);
        assert_eq!(t1.text, "'hello'");

        let t2 = lexer.next_token();
        assert_eq!(t2.kind, TokenKind::String);
        assert_eq!(t2.text, "\"world\"");
    }

    #[test]
    fn test_arrows() {
        let mut lexer = Lexer::new("-> <- --");
        assert_eq!(lexer.next_token().kind, TokenKind::Arrow);
        assert_eq!(lexer.next_token().kind, TokenKind::LeftArrow);
        assert_eq!(lexer.next_token().kind, TokenKind::DoubleDash);
    }

    #[test]
    fn test_operators() {
        let mut lexer = Lexer::new("= <> < <= > >= + - * / % ^");
        assert_eq!(lexer.next_token().kind, TokenKind::Eq);
        assert_eq!(lexer.next_token().kind, TokenKind::Ne);
        assert_eq!(lexer.next_token().kind, TokenKind::Lt);
        assert_eq!(lexer.next_token().kind, TokenKind::Le);
        assert_eq!(lexer.next_token().kind, TokenKind::Gt);
        assert_eq!(lexer.next_token().kind, TokenKind::Ge);
        assert_eq!(lexer.next_token().kind, TokenKind::Plus);
        assert_eq!(lexer.next_token().kind, TokenKind::Minus);
        assert_eq!(lexer.next_token().kind, TokenKind::Star);
        assert_eq!(lexer.next_token().kind, TokenKind::Slash);
        assert_eq!(lexer.next_token().kind, TokenKind::Percent);
        assert_eq!(lexer.next_token().kind, TokenKind::Caret);
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("MATCH // comment\nRETURN");
        assert_eq!(lexer.next_token().kind, TokenKind::Match);
        assert_eq!(lexer.next_token().kind, TokenKind::Return);
    }
}
