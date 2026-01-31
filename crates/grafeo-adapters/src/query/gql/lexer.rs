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
            '`' => self.scan_quoted_identifier(),
            '$' => self.scan_parameter(),
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

    fn peek_char(&self) -> char {
        if self.position + 1 < self.input.len() {
            self.input[self.position + 1..]
                .chars()
                .next()
                .unwrap_or('\0')
        } else {
            '\0'
        }
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
        while self.position < self.input.len() && self.current_char().is_ascii_digit() {
            self.advance();
        }

        // Only consume '.' if followed by a digit (to avoid consuming '..' as part of a number)
        if self.current_char() == '.' && self.peek_char().is_ascii_digit() {
            self.advance();
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
        if !ch.is_ascii_alphabetic() && ch != '_' {
            return TokenKind::Error;
        }

        // Scan the rest of the identifier
        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_ascii_alphanumeric() || ch == '_' {
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
            "REMOVE" => TokenKind::Remove,
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
            "EXISTS" => TokenKind::Exists,
            "OPTIONAL" => TokenKind::Optional,
            "WITH" => TokenKind::With,
            "UNWIND" => TokenKind::Unwind,
            "MERGE" => TokenKind::Merge,
            "HAVING" => TokenKind::Having,
            "ON" => TokenKind::On,
            "STARTS" => TokenKind::Starts,
            "ENDS" => TokenKind::Ends,
            "CONTAINS" => TokenKind::Contains,
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
}
