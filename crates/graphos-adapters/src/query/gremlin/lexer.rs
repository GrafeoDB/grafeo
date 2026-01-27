//! Gremlin Lexer.
//!
//! Tokenizes Gremlin query strings.

use std::iter::Peekable;
use std::str::Chars;

/// Token types for Gremlin.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    True,
    False,

    // Identifiers and keywords
    Identifier(String),

    // Graph source
    G,
    V,
    E,
    AddV,
    AddE,

    // Navigation steps
    Out,
    In,
    Both,
    OutE,
    InE,
    BothE,
    OutV,
    InV,
    BothV,
    OtherV,

    // Filter steps
    Has,
    HasLabel,
    HasId,
    HasNot,
    Filter,
    Where,
    And,
    Or,
    Not,
    Dedup,
    Limit,
    Skip,
    Range,

    // Map steps
    Values,
    ValueMap,
    ElementMap,
    Id,
    Label,
    Properties,
    Constant,
    Count,
    Sum,
    Mean,
    Min,
    Max,
    Fold,
    Unfold,
    Group,
    GroupCount,
    Path,
    Select,
    Project,
    By,
    Order,
    Coalesce,
    Optional,
    Union,
    Choose,

    // Side effect steps
    As,
    SideEffect,
    Aggregate,
    Store,
    Property,
    Drop,

    // Edge creation
    From,
    To,

    // Predicates (P.*)
    P,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    Within,
    Without,
    Between,
    Inside,
    Outside,
    Containing,
    StartingWith,
    EndingWith,
    Regex,

    // Tokens (T.*)
    T,

    // Order
    Asc,
    Desc,
    Shuffle,

    // Cardinality
    Single,
    List,
    Set,

    // Punctuation
    Dot,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Underscore,

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

/// Gremlin lexer.
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
        self.skip_whitespace();

        let start = self.position;

        let kind = match self.advance() {
            Some('.') => TokenKind::Dot,
            Some(',') => TokenKind::Comma,
            Some('(') => TokenKind::LParen,
            Some(')') => TokenKind::RParen,
            Some('[') => TokenKind::LBracket,
            Some(']') => TokenKind::RBracket,
            Some('_') if self.peek_is(|c| !c.is_alphanumeric()) => TokenKind::Underscore,

            Some('"') => self.read_string('"'),
            Some('\'') => self.read_string('\''),

            Some(c) if c.is_ascii_digit() || (c == '-' && self.peek_is(|c| c.is_ascii_digit())) => {
                self.read_number(c)
            }

            Some(c) if c.is_alphabetic() || c == '_' => self.read_identifier(c),

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

    fn peek_is(&mut self, f: impl FnOnce(char) -> bool) -> bool {
        self.peek().is_some_and(f)
    }

    fn skip_whitespace(&mut self) {
        while self.peek_is(|c| c.is_whitespace()) {
            self.advance();
        }
    }

    fn read_string(&mut self, quote: char) -> TokenKind {
        let mut value = String::new();
        loop {
            match self.advance() {
                Some('\\') => {
                    if let Some(escaped) = self.advance() {
                        match escaped {
                            'n' => value.push('\n'),
                            't' => value.push('\t'),
                            'r' => value.push('\r'),
                            '\\' => value.push('\\'),
                            '"' => value.push('"'),
                            '\'' => value.push('\''),
                            _ => value.push(escaped),
                        }
                    }
                }
                Some(c) if c == quote => break,
                Some(c) => value.push(c),
                None => break,
            }
        }
        TokenKind::String(value)
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
                if self.peek_is(|c| c == '+' || c == '-') {
                    if let Some(sign) = self.advance() {
                        value.push(sign);
                    }
                }
            } else {
                break;
            }
        }

        if is_float {
            TokenKind::Float(value.parse().unwrap_or(0.0))
        } else {
            TokenKind::Integer(value.parse().unwrap_or(0))
        }
    }

    fn read_identifier(&mut self, first: char) -> TokenKind {
        let mut value = String::from(first);

        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Match keywords
        match value.as_str() {
            "g" => TokenKind::G,
            "V" => TokenKind::V,
            "E" => TokenKind::E,
            "addV" => TokenKind::AddV,
            "addE" => TokenKind::AddE,
            "out" => TokenKind::Out,
            "in" | "in_" => TokenKind::In,
            "both" => TokenKind::Both,
            "outE" => TokenKind::OutE,
            "inE" => TokenKind::InE,
            "bothE" => TokenKind::BothE,
            "outV" => TokenKind::OutV,
            "inV" => TokenKind::InV,
            "bothV" => TokenKind::BothV,
            "otherV" => TokenKind::OtherV,
            "has" => TokenKind::Has,
            "hasLabel" => TokenKind::HasLabel,
            "hasId" => TokenKind::HasId,
            "hasNot" => TokenKind::HasNot,
            "filter" => TokenKind::Filter,
            "where" => TokenKind::Where,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "dedup" => TokenKind::Dedup,
            "limit" => TokenKind::Limit,
            "skip" => TokenKind::Skip,
            "range" => TokenKind::Range,
            "values" => TokenKind::Values,
            "valueMap" => TokenKind::ValueMap,
            "elementMap" => TokenKind::ElementMap,
            "id" => TokenKind::Id,
            "label" => TokenKind::Label,
            "properties" => TokenKind::Properties,
            "constant" => TokenKind::Constant,
            "count" => TokenKind::Count,
            "sum" => TokenKind::Sum,
            "mean" | "avg" => TokenKind::Mean,
            "min" => TokenKind::Min,
            "max" => TokenKind::Max,
            "fold" => TokenKind::Fold,
            "unfold" => TokenKind::Unfold,
            "group" => TokenKind::Group,
            "groupCount" => TokenKind::GroupCount,
            "path" => TokenKind::Path,
            "select" => TokenKind::Select,
            "project" => TokenKind::Project,
            "by" => TokenKind::By,
            "order" => TokenKind::Order,
            "coalesce" => TokenKind::Coalesce,
            "optional" => TokenKind::Optional,
            "union" => TokenKind::Union,
            "choose" => TokenKind::Choose,
            "as" | "as_" => TokenKind::As,
            "sideEffect" => TokenKind::SideEffect,
            "aggregate" => TokenKind::Aggregate,
            "store" => TokenKind::Store,
            "property" => TokenKind::Property,
            "drop" => TokenKind::Drop,
            "from" | "from_" => TokenKind::From,
            "to" => TokenKind::To,
            "P" => TokenKind::P,
            "eq" => TokenKind::Eq,
            "neq" => TokenKind::Neq,
            "lt" => TokenKind::Lt,
            "lte" => TokenKind::Lte,
            "gt" => TokenKind::Gt,
            "gte" => TokenKind::Gte,
            "within" => TokenKind::Within,
            "without" => TokenKind::Without,
            "between" => TokenKind::Between,
            "inside" => TokenKind::Inside,
            "outside" => TokenKind::Outside,
            "containing" => TokenKind::Containing,
            "startingWith" => TokenKind::StartingWith,
            "endingWith" => TokenKind::EndingWith,
            "regex" => TokenKind::Regex,
            "T" => TokenKind::T,
            "asc" => TokenKind::Asc,
            "desc" => TokenKind::Desc,
            "shuffle" => TokenKind::Shuffle,
            "single" => TokenKind::Single,
            "list" => TokenKind::List,
            "set" => TokenKind::Set,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Identifier(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_traversal() {
        let mut lexer = Lexer::new("g.V()");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[0].kind, TokenKind::G);
        assert_eq!(tokens[1].kind, TokenKind::Dot);
        assert_eq!(tokens[2].kind, TokenKind::V);
        assert_eq!(tokens[3].kind, TokenKind::LParen);
        assert_eq!(tokens[4].kind, TokenKind::RParen);
        assert_eq!(tokens[5].kind, TokenKind::Eof);
    }

    #[test]
    fn test_has_step() {
        let mut lexer = Lexer::new("g.V().has('name', 'Alice')");
        let tokens = lexer.tokenize();

        // g.V().has('name', 'Alice')
        // 0=G 1=Dot 2=V 3=LParen 4=RParen 5=Dot 6=Has 7=LParen 8=String 9=Comma 10=String 11=RParen
        assert_eq!(tokens[0].kind, TokenKind::G);
        assert_eq!(tokens[6].kind, TokenKind::Has);
        assert_eq!(tokens[8].kind, TokenKind::String("name".to_string()));
        assert_eq!(tokens[10].kind, TokenKind::String("Alice".to_string()));
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 3.14 -7");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
        assert_eq!(tokens[1].kind, TokenKind::Float(3.14));
        assert_eq!(tokens[2].kind, TokenKind::Integer(-7));
    }

    #[test]
    fn test_predicate() {
        let mut lexer = Lexer::new("P.gt(30)");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[0].kind, TokenKind::P);
        assert_eq!(tokens[1].kind, TokenKind::Dot);
        assert_eq!(tokens[2].kind, TokenKind::Gt);
    }
}
