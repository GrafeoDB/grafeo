//! GraphQL Lexer.
//!
//! Tokenizes GraphQL query strings according to the specification.

use std::iter::Peekable;
use std::str::Chars;

/// Token types for GraphQL.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Punctuation
    Bang,     // !
    Dollar,   // $
    Amp,      // &
    LParen,   // (
    RParen,   // )
    Spread,   // ...
    Colon,    // :
    Eq,       // =
    At,       // @
    LBracket, // [
    RBracket, // ]
    LBrace,   // {
    RBrace,   // }
    Pipe,     // |

    // Literals
    Name(String),
    Int(i64),
    Float(f64),
    String(String),
    BlockString(String),

    // Keywords (these are also valid names in GraphQL)
    Query,
    Mutation,
    Subscription,
    Fragment,
    On,
    True,
    False,
    Null,

    // Type system keywords (for completeness)
    Schema,
    Extend,
    Scalar,
    Type,
    Interface,
    Union,
    Enum,
    Input,
    Directive,
    Implements,

    // End of input
    Eof,
}

/// A token with its position.
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Source code span.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// GraphQL lexer.
pub struct Lexer<'a> {
    source: &'a str,
    chars: Peekable<Chars<'a>>,
    position: usize,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given source.
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            chars: source.chars().peekable(),
            position: 0,
        }
    }

    /// Returns the next token.
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        let start = self.position;

        let kind = match self.advance() {
            Some('!') => TokenKind::Bang,
            Some('$') => TokenKind::Dollar,
            Some('&') => TokenKind::Amp,
            Some('(') => TokenKind::LParen,
            Some(')') => TokenKind::RParen,
            Some(':') => TokenKind::Colon,
            Some('=') => TokenKind::Eq,
            Some('@') => TokenKind::At,
            Some('[') => TokenKind::LBracket,
            Some(']') => TokenKind::RBracket,
            Some('{') => TokenKind::LBrace,
            Some('}') => TokenKind::RBrace,
            Some('|') => TokenKind::Pipe,

            Some('.') => {
                // Check for spread operator
                if self.peek() == Some('.') && self.peek_next() == Some('.') {
                    self.advance();
                    self.advance();
                    TokenKind::Spread
                } else {
                    TokenKind::Eof // Invalid single dot
                }
            }

            Some('"') => {
                // Check for block string
                if self.peek() == Some('"') && self.peek_next() == Some('"') {
                    self.advance();
                    self.advance();
                    self.read_block_string()
                } else {
                    self.read_string()
                }
            }

            Some(c) if c.is_ascii_digit() || c == '-' => self.read_number(c),

            Some(c) if is_name_start(c) => self.read_name(c),

            None => TokenKind::Eof,
            _ => TokenKind::Eof,
        };

        Token {
            kind,
            span: Span {
                start,
                end: self.position,
            },
        }
    }

    /// Returns all tokens.
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            let is_eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        tokens
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next();
        if c.is_some() {
            self.position += 1;
        }
        c
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn peek_next(&self) -> Option<char> {
        let mut iter = self.source[self.position..].chars();
        iter.next();
        iter.next()
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                // Whitespace
                Some(c) if c.is_whitespace() || c == ',' => {
                    self.advance();
                }
                // Comments
                Some('#') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' || c == '\r' {
                            break;
                        }
                        self.advance();
                    }
                }
                // BOM (optional)
                Some('\u{FEFF}') => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    fn read_string(&mut self) -> TokenKind {
        let mut value = String::new();
        loop {
            match self.advance() {
                Some('\\') => {
                    if let Some(escaped) = self.advance() {
                        match escaped {
                            '"' => value.push('"'),
                            '\\' => value.push('\\'),
                            '/' => value.push('/'),
                            'b' => value.push('\x08'),
                            'f' => value.push('\x0C'),
                            'n' => value.push('\n'),
                            'r' => value.push('\r'),
                            't' => value.push('\t'),
                            'u' => {
                                // Unicode escape
                                let mut hex = String::new();
                                for _ in 0..4 {
                                    if let Some(h) = self.advance() {
                                        hex.push(h);
                                    }
                                }
                                if let Ok(code) = u32::from_str_radix(&hex, 16)
                                    && let Some(c) = char::from_u32(code)
                                {
                                    value.push(c);
                                }
                            }
                            _ => value.push(escaped),
                        }
                    }
                }
                Some('"') => break,
                Some(c) => value.push(c),
                None => break,
            }
        }
        TokenKind::String(value)
    }

    fn read_block_string(&mut self) -> TokenKind {
        let mut value = String::new();
        loop {
            match self.peek() {
                Some('"') => {
                    if self.peek_next() == Some('"') {
                        // Check for closing """
                        let mut chars_copy = self.source[self.position..].chars();
                        if chars_copy.next() == Some('"')
                            && chars_copy.next() == Some('"')
                            && chars_copy.next() == Some('"')
                        {
                            self.advance();
                            self.advance();
                            self.advance();
                            break;
                        }
                    }
                    if let Some(c) = self.advance() {
                        value.push(c);
                    }
                }
                Some('\\') => {
                    self.advance();
                    // Block strings only escape """
                    if self.peek() == Some('"') && self.peek_next() == Some('"') {
                        // Check for escaped """
                        value.push_str("\"\"\"");
                        self.advance();
                        self.advance();
                        self.advance();
                    } else {
                        value.push('\\');
                    }
                }
                Some(c) => {
                    value.push(c);
                    self.advance();
                }
                None => break,
            }
        }
        TokenKind::BlockString(dedent_block_string(&value))
    }

    fn read_number(&mut self, first: char) -> TokenKind {
        let mut value = String::from(first);
        let mut is_float = false;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                value.push(c);
                self.advance();
            } else if c == '.' && !is_float {
                is_float = true;
                value.push(c);
                self.advance();
            } else if (c == 'e' || c == 'E') && !value.contains('e') && !value.contains('E') {
                is_float = true;
                value.push(c);
                self.advance();
                if matches!(self.peek(), Some('+') | Some('-'))
                    && let Some(sign) = self.advance()
                {
                    value.push(sign);
                }
            } else {
                break;
            }
        }

        if is_float {
            TokenKind::Float(value.parse().unwrap_or(0.0))
        } else {
            TokenKind::Int(value.parse().unwrap_or(0))
        }
    }

    fn read_name(&mut self, first: char) -> TokenKind {
        let mut value = String::from(first);

        while let Some(c) = self.peek() {
            if is_name_continue(c) {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Match keywords
        match value.as_str() {
            "query" => TokenKind::Query,
            "mutation" => TokenKind::Mutation,
            "subscription" => TokenKind::Subscription,
            "fragment" => TokenKind::Fragment,
            "on" => TokenKind::On,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "null" => TokenKind::Null,
            "schema" => TokenKind::Schema,
            "extend" => TokenKind::Extend,
            "scalar" => TokenKind::Scalar,
            "type" => TokenKind::Type,
            "interface" => TokenKind::Interface,
            "union" => TokenKind::Union,
            "enum" => TokenKind::Enum,
            "input" => TokenKind::Input,
            "directive" => TokenKind::Directive,
            "implements" => TokenKind::Implements,
            _ => TokenKind::Name(value),
        }
    }
}

/// Check if character can start a name.
fn is_name_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

/// Check if character can continue a name.
fn is_name_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Dedent a block string according to GraphQL spec.
fn dedent_block_string(value: &str) -> String {
    let lines: Vec<&str> = value.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    // Find common indent
    let mut common_indent: Option<usize> = None;
    for line in &lines[1..] {
        let indent = line.len() - line.trim_start().len();
        if !line.trim().is_empty() {
            common_indent = Some(match common_indent {
                Some(ci) => ci.min(indent),
                None => indent,
            });
        }
    }

    // Build result
    let mut result = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            result.push('\n');
        }
        if i == 0 {
            result.push_str(line);
        } else if let Some(indent) = common_indent {
            if line.len() > indent {
                result.push_str(&line[indent..]);
            }
        } else {
            result.push_str(line);
        }
    }

    // Trim leading and trailing blank lines
    result.trim_matches('\n').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_query() {
        let mut lexer = Lexer::new("{ user { name } }");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[0].kind, TokenKind::LBrace);
        assert_eq!(tokens[1].kind, TokenKind::Name("user".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::LBrace);
        assert_eq!(tokens[3].kind, TokenKind::Name("name".to_string()));
        assert_eq!(tokens[4].kind, TokenKind::RBrace);
        assert_eq!(tokens[5].kind, TokenKind::RBrace);
        assert_eq!(tokens[6].kind, TokenKind::Eof);
    }

    #[test]
    fn test_query_with_arguments() {
        let mut lexer = Lexer::new("{ user(id: 123) { name } }");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[2].kind, TokenKind::LParen);
        assert_eq!(tokens[3].kind, TokenKind::Name("id".to_string()));
        assert_eq!(tokens[4].kind, TokenKind::Colon);
        assert_eq!(tokens[5].kind, TokenKind::Int(123));
        assert_eq!(tokens[6].kind, TokenKind::RParen);
    }

    #[test]
    fn test_strings() {
        let mut lexer = Lexer::new("\"hello world\"");
        let token = lexer.next_token();
        assert_eq!(token.kind, TokenKind::String("hello world".to_string()));
    }

    #[test]
    fn test_spread_operator() {
        let mut lexer = Lexer::new("...userFields");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[0].kind, TokenKind::Spread);
        assert_eq!(tokens[1].kind, TokenKind::Name("userFields".to_string()));
    }

    #[test]
    fn test_operation_keywords() {
        let mut lexer = Lexer::new("query mutation subscription");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[0].kind, TokenKind::Query);
        assert_eq!(tokens[1].kind, TokenKind::Mutation);
        assert_eq!(tokens[2].kind, TokenKind::Subscription);
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("{ # comment\n user }");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[0].kind, TokenKind::LBrace);
        assert_eq!(tokens[1].kind, TokenKind::Name("user".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::RBrace);
    }
}
