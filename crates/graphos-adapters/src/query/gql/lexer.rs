//! GQL Lexer.

use graphos_common::utils::error::SourceSpan;

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
    /// INSERT keyword.
    Insert,
    /// DELETE keyword.
    Delete,
    /// SET keyword.
    Set,
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
    /// || operator.
    Concat,

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
    /// -> arrow.
    Arrow,
    /// <- arrow.
    LeftArrow,
    /// -- double dash.
    DoubleDash,

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
                TokenKind::Dot
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
                } else {
                    TokenKind::Error
                }
            }
            '\'' | '"' => self.scan_string(),
            _ if ch.is_ascii_digit() => self.scan_number(),
            _ if ch.is_ascii_alphabetic() || ch == '_' => self.scan_identifier(),
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
                self.position += 1;
            } else {
                break;
            }
        }
    }

    fn current_char(&self) -> char {
        self.input[self.position..].chars().next().unwrap_or('\0')
    }

    fn advance(&mut self) {
        if self.position < self.input.len() {
            self.position += 1;
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

    fn scan_number(&mut self) -> TokenKind {
        while self.position < self.input.len() && self.current_char().is_ascii_digit() {
            self.advance();
        }

        if self.current_char() == '.' {
            self.advance();
            while self.position < self.input.len() && self.current_char().is_ascii_digit() {
                self.advance();
            }
            TokenKind::Float
        } else {
            TokenKind::Integer
        }
    }

    fn scan_identifier(&mut self) -> TokenKind {
        let start = self.position;
        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let text = &self.input[start..self.position];
        match text.to_uppercase().as_str() {
            "MATCH" => TokenKind::Match,
            "RETURN" => TokenKind::Return,
            "WHERE" => TokenKind::Where,
            "AND" => TokenKind::And,
            "OR" => TokenKind::Or,
            "NOT" => TokenKind::Not,
            "INSERT" => TokenKind::Insert,
            "DELETE" => TokenKind::Delete,
            "SET" => TokenKind::Set,
            "CREATE" => TokenKind::Create,
            "NODE" => TokenKind::Node,
            "EDGE" => TokenKind::Edge,
            "TYPE" => TokenKind::Type,
            "AS" => TokenKind::As,
            "DISTINCT" => TokenKind::Distinct,
            "ORDER" => TokenKind::Order,
            "BY" => TokenKind::By,
            "ASC" => TokenKind::Asc,
            "DESC" => TokenKind::Desc,
            "SKIP" => TokenKind::Skip,
            "LIMIT" => TokenKind::Limit,
            "NULL" => TokenKind::Null,
            "TRUE" => TokenKind::True,
            "FALSE" => TokenKind::False,
            "DETACH" => TokenKind::Detach,
            "CALL" => TokenKind::Call,
            "YIELD" => TokenKind::Yield,
            "IN" => TokenKind::In,
            "LIKE" => TokenKind::Like,
            "IS" => TokenKind::Is,
            "CASE" => TokenKind::Case,
            "WHEN" => TokenKind::When,
            "THEN" => TokenKind::Then,
            "ELSE" => TokenKind::Else,
            "END" => TokenKind::End,
            "OPTIONAL" => TokenKind::Optional,
            "WITH" => TokenKind::With,
            _ => TokenKind::Identifier,
        }
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
}
