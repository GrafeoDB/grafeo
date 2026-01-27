//! SPARQL Lexer.

use graphos_common::utils::error::SourceSpan;

/// A token in the SPARQL language.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The token kind.
    pub kind: TokenKind,
    /// The source text.
    pub text: String,
    /// Source span.
    pub span: SourceSpan,
}

/// Token kinds in SPARQL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Query form keywords
    /// SELECT keyword.
    Select,
    /// CONSTRUCT keyword.
    Construct,
    /// ASK keyword.
    Ask,
    /// DESCRIBE keyword.
    Describe,

    // Prologue keywords
    /// PREFIX keyword.
    Prefix,
    /// BASE keyword.
    Base,

    // Clause keywords
    /// WHERE keyword.
    Where,
    /// FROM keyword.
    From,
    /// NAMED keyword.
    Named,
    /// OPTIONAL keyword.
    Optional,
    /// UNION keyword.
    Union,
    /// FILTER keyword.
    Filter,
    /// GRAPH keyword.
    Graph,
    /// BIND keyword.
    Bind,
    /// VALUES keyword.
    Values,
    /// SERVICE keyword.
    Service,
    /// SILENT keyword.
    Silent,
    /// MINUS keyword.
    Minus,

    // Solution modifier keywords
    /// ORDER keyword.
    Order,
    /// BY keyword.
    By,
    /// ASC keyword.
    Asc,
    /// DESC keyword.
    Desc,
    /// LIMIT keyword.
    Limit,
    /// OFFSET keyword.
    Offset,
    /// GROUP keyword.
    Group,
    /// HAVING keyword.
    Having,
    /// DISTINCT keyword.
    Distinct,
    /// REDUCED keyword.
    Reduced,

    // Expression keywords
    /// AS keyword.
    As,
    /// EXISTS keyword.
    Exists,
    /// NOT keyword.
    Not,
    /// IN keyword.
    In,
    /// AND keyword (&&).
    And,
    /// OR keyword (||).
    Or,

    // Aggregate keywords
    /// COUNT keyword.
    Count,
    /// SUM keyword.
    Sum,
    /// AVG keyword.
    Avg,
    /// MIN keyword.
    Min,
    /// MAX keyword.
    Max,
    /// SAMPLE keyword.
    Sample,
    /// GROUP_CONCAT keyword.
    GroupConcat,
    /// SEPARATOR keyword.
    Separator,

    // Boolean literals
    /// true literal.
    True,
    /// false literal.
    False,

    // Special keyword
    /// 'a' shorthand for rdf:type.
    A,
    /// UNDEF keyword.
    Undef,

    // Literals
    /// Integer literal (value stored in text).
    Integer,
    /// Decimal literal (value stored in text).
    Decimal,
    /// Double literal (value stored in text).
    Double,
    /// String literal (value stored in text, includes quotes).
    String,
    /// Long string literal (triple-quoted).
    LongString,

    // Identifiers and IRIs
    /// Variable (?name or $name, includes prefix).
    Variable,
    /// Full IRI (<http://...>).
    Iri,
    /// Prefixed name (prefix:local).
    PrefixedName,
    /// Blank node label (_:label).
    BlankNodeLabel,
    /// Anonymous blank node [].
    AnonymousBlank,

    // Operators
    /// = operator.
    Equals,
    /// != operator.
    NotEquals,
    /// < operator.
    LessThan,
    /// <= operator.
    LessOrEqual,
    /// > operator.
    GreaterThan,
    /// >= operator.
    GreaterOrEqual,
    /// + operator.
    Plus,
    /// - operator (also used for Minus keyword context).
    MinusOp,
    /// * operator.
    Star,
    /// / operator.
    Slash,
    /// ! operator.
    Bang,
    /// && operator.
    AndOp,
    /// || operator.
    OrOp,
    /// ^ operator (property path inverse).
    Caret,
    /// ^^ operator (datatype).
    DoubleCaret,
    /// @ symbol (language tag).
    At,
    /// | operator (property path alternative).
    Pipe,
    /// ? operator (property path zero-or-one).
    QuestionMark,

    // Punctuation
    /// ( punctuation.
    LeftParen,
    /// ) punctuation.
    RightParen,
    /// [ punctuation.
    LeftBracket,
    /// ] punctuation.
    RightBracket,
    /// { punctuation.
    LeftBrace,
    /// } punctuation.
    RightBrace,
    /// . punctuation.
    Dot,
    /// , punctuation.
    Comma,
    /// ; punctuation.
    Semicolon,
    /// : punctuation (for prefixed names).
    Colon,

    /// End of input.
    Eof,

    /// Error token.
    Error,
}

/// SPARQL Lexer.
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
        self.skip_whitespace_and_comments();

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
                TokenKind::LeftParen
            }
            ')' => {
                self.advance();
                TokenKind::RightParen
            }
            '[' => {
                self.advance();
                if self.current_char() == ']' {
                    self.advance();
                    TokenKind::AnonymousBlank
                } else {
                    TokenKind::LeftBracket
                }
            }
            ']' => {
                self.advance();
                TokenKind::RightBracket
            }
            '{' => {
                self.advance();
                TokenKind::LeftBrace
            }
            '}' => {
                self.advance();
                TokenKind::RightBrace
            }
            '.' => {
                self.advance();
                TokenKind::Dot
            }
            ',' => {
                self.advance();
                TokenKind::Comma
            }
            ';' => {
                self.advance();
                TokenKind::Semicolon
            }
            ':' => {
                self.advance();
                // Check if this is a prefixed name starting with just ':'
                if self.is_pname_char(self.current_char()) {
                    self.scan_prefixed_name_local()
                } else {
                    TokenKind::Colon
                }
            }
            '+' => {
                self.advance();
                TokenKind::Plus
            }
            '-' => {
                self.advance();
                TokenKind::MinusOp
            }
            '*' => {
                self.advance();
                TokenKind::Star
            }
            '/' => {
                self.advance();
                TokenKind::Slash
            }
            '!' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    TokenKind::NotEquals
                } else {
                    TokenKind::Bang
                }
            }
            '=' => {
                self.advance();
                TokenKind::Equals
            }
            '<' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    TokenKind::LessOrEqual
                } else if self.is_iri_start() {
                    // Go back and scan full IRI
                    self.position = start;
                    self.column = start_column;
                    self.scan_iri()
                } else {
                    TokenKind::LessThan
                }
            }
            '>' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    TokenKind::GreaterOrEqual
                } else {
                    TokenKind::GreaterThan
                }
            }
            '&' => {
                self.advance();
                if self.current_char() == '&' {
                    self.advance();
                    TokenKind::AndOp
                } else {
                    TokenKind::Error
                }
            }
            '|' => {
                self.advance();
                if self.current_char() == '|' {
                    self.advance();
                    TokenKind::OrOp
                } else {
                    TokenKind::Pipe
                }
            }
            '^' => {
                self.advance();
                if self.current_char() == '^' {
                    self.advance();
                    TokenKind::DoubleCaret
                } else {
                    TokenKind::Caret
                }
            }
            '@' => {
                self.advance();
                TokenKind::At
            }
            '?' => {
                self.advance();
                // Check if it's a variable or the ? path operator
                if self.current_char().is_ascii_alphanumeric() || self.current_char() == '_' {
                    self.scan_variable_rest()
                } else {
                    TokenKind::QuestionMark
                }
            }
            '$' => {
                self.advance();
                self.scan_variable_rest()
            }
            '_' => {
                // Could be blank node label _:xxx or identifier
                if self.peek_char() == ':' {
                    self.scan_blank_node_label()
                } else {
                    self.scan_identifier_or_keyword()
                }
            }
            '\'' | '"' => self.scan_string(),
            _ if ch.is_ascii_digit() => self.scan_number(),
            _ if self.is_pname_start_char(ch) => self.scan_identifier_or_keyword(),
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

    fn skip_whitespace_and_comments(&mut self) {
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
            } else if ch == '#' {
                // Line comment
                while self.position < self.input.len() && self.current_char() != '\n' {
                    self.position += self.current_char().len_utf8();
                }
            } else {
                break;
            }
        }
    }

    fn current_char(&self) -> char {
        self.input[self.position..].chars().next().unwrap_or('\0')
    }

    fn peek_char(&self) -> char {
        let mut chars = self.input[self.position..].chars();
        chars.next();
        chars.next().unwrap_or('\0')
    }

    fn advance(&mut self) {
        if self.position < self.input.len() {
            let ch = self.current_char();
            self.position += ch.len_utf8();
            self.column += 1;
        }
    }

    fn is_iri_start(&self) -> bool {
        // After seeing '<', check if this looks like an IRI
        let remaining = &self.input[self.position..];
        // Simple heuristic: IRIs don't start with whitespace or common operators
        !remaining.is_empty()
            && !remaining.starts_with(' ')
            && !remaining.starts_with('\n')
            && !remaining.starts_with('=')
    }

    fn scan_iri(&mut self) -> TokenKind {
        // Consume opening '<'
        self.advance();

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch == '>' {
                self.advance();
                return TokenKind::Iri;
            }
            if ch == '\\' {
                // Escape sequence
                self.advance();
                if self.position < self.input.len() {
                    self.advance();
                }
            } else if ch.is_whitespace() && ch != ' ' {
                // Invalid character in IRI (newlines not allowed)
                return TokenKind::Error;
            } else {
                self.advance();
            }
        }

        TokenKind::Error // Unterminated IRI
    }

    fn scan_variable_rest(&mut self) -> TokenKind {
        // Already consumed ? or $
        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        TokenKind::Variable
    }

    fn scan_blank_node_label(&mut self) -> TokenKind {
        // Consume '_'
        self.advance();
        // Consume ':'
        self.advance();

        // Label
        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                self.advance();
            } else {
                break;
            }
        }

        TokenKind::BlankNodeLabel
    }

    fn scan_string(&mut self) -> TokenKind {
        let quote = self.current_char();
        self.advance();

        // Check for long string (triple quotes)
        if self.current_char() == quote && self.peek_char() == quote {
            self.advance();
            self.advance();
            return self.scan_long_string(quote);
        }

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch == quote {
                self.advance();
                return TokenKind::String;
            }
            if ch == '\\' {
                self.advance();
                if self.position < self.input.len() {
                    self.advance();
                }
            } else if ch == '\n' {
                // Newline not allowed in short string
                return TokenKind::Error;
            } else {
                self.advance();
            }
        }

        TokenKind::Error // Unterminated string
    }

    fn scan_long_string(&mut self, quote: char) -> TokenKind {
        let mut consecutive_quotes = 0;

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch == quote {
                consecutive_quotes += 1;
                self.advance();
                if consecutive_quotes >= 3 {
                    return TokenKind::LongString;
                }
            } else {
                consecutive_quotes = 0;
                if ch == '\\' {
                    self.advance();
                    if self.position < self.input.len() {
                        self.advance();
                    }
                } else {
                    if ch == '\n' {
                        self.line += 1;
                        self.column = 0;
                    }
                    self.advance();
                }
            }
        }

        TokenKind::Error // Unterminated long string
    }

    fn scan_number(&mut self) -> TokenKind {
        let mut has_dot = false;
        let mut has_exponent = false;

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_ascii_digit() {
                self.advance();
            } else if ch == '.' && !has_dot && !has_exponent {
                // Check if next char is a digit (otherwise it might be end of triple)
                if self.peek_char().is_ascii_digit() {
                    has_dot = true;
                    self.advance();
                } else {
                    break;
                }
            } else if (ch == 'e' || ch == 'E') && !has_exponent {
                has_exponent = true;
                self.advance();
                if self.current_char() == '+' || self.current_char() == '-' {
                    self.advance();
                }
            } else {
                break;
            }
        }

        if has_exponent {
            TokenKind::Double
        } else if has_dot {
            TokenKind::Decimal
        } else {
            TokenKind::Integer
        }
    }

    fn scan_identifier_or_keyword(&mut self) -> TokenKind {
        let start = self.position;

        while self.position < self.input.len() {
            let ch = self.current_char();
            if self.is_pname_char(ch) {
                self.advance();
            } else {
                break;
            }
        }

        // Check for prefixed name (contains ':')
        if self.current_char() == ':' {
            self.advance();
            // Scan local part
            return self.scan_prefixed_name_local();
        }

        let text = &self.input[start..self.position];
        self.keyword_or_identifier(text)
    }

    fn scan_prefixed_name_local(&mut self) -> TokenKind {
        // Already consumed the ':'
        while self.position < self.input.len() {
            let ch = self.current_char();
            if self.is_pname_char(ch) || ch == '.' || ch == '-' {
                // Check that '.' isn't followed by whitespace (end of pattern)
                if ch == '.' {
                    let next = self.peek_char();
                    if next.is_whitespace() || next == '\0' {
                        break;
                    }
                }
                self.advance();
            } else {
                break;
            }
        }
        TokenKind::PrefixedName
    }

    fn is_pname_start_char(&self, ch: char) -> bool {
        ch.is_ascii_alphabetic() || ch == '_' || ch > '\u{7F}'
    }

    fn is_pname_char(&self, ch: char) -> bool {
        ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch > '\u{7F}'
    }

    fn keyword_or_identifier(&self, text: &str) -> TokenKind {
        match text.to_uppercase().as_str() {
            "SELECT" => TokenKind::Select,
            "CONSTRUCT" => TokenKind::Construct,
            "ASK" => TokenKind::Ask,
            "DESCRIBE" => TokenKind::Describe,
            "PREFIX" => TokenKind::Prefix,
            "BASE" => TokenKind::Base,
            "WHERE" => TokenKind::Where,
            "FROM" => TokenKind::From,
            "NAMED" => TokenKind::Named,
            "OPTIONAL" => TokenKind::Optional,
            "UNION" => TokenKind::Union,
            "FILTER" => TokenKind::Filter,
            "GRAPH" => TokenKind::Graph,
            "BIND" => TokenKind::Bind,
            "VALUES" => TokenKind::Values,
            "SERVICE" => TokenKind::Service,
            "SILENT" => TokenKind::Silent,
            "MINUS" => TokenKind::Minus,
            "ORDER" => TokenKind::Order,
            "BY" => TokenKind::By,
            "ASC" => TokenKind::Asc,
            "DESC" => TokenKind::Desc,
            "LIMIT" => TokenKind::Limit,
            "OFFSET" => TokenKind::Offset,
            "GROUP" => TokenKind::Group,
            "HAVING" => TokenKind::Having,
            "DISTINCT" => TokenKind::Distinct,
            "REDUCED" => TokenKind::Reduced,
            "AS" => TokenKind::As,
            "EXISTS" => TokenKind::Exists,
            "NOT" => TokenKind::Not,
            "IN" => TokenKind::In,
            "COUNT" => TokenKind::Count,
            "SUM" => TokenKind::Sum,
            "AVG" => TokenKind::Avg,
            "MIN" => TokenKind::Min,
            "MAX" => TokenKind::Max,
            "SAMPLE" => TokenKind::Sample,
            "GROUP_CONCAT" => TokenKind::GroupConcat,
            "SEPARATOR" => TokenKind::Separator,
            "TRUE" => TokenKind::True,
            "FALSE" => TokenKind::False,
            "UNDEF" => TokenKind::Undef,
            // Special: 'a' is shorthand for rdf:type (case-sensitive!)
            "a" => TokenKind::A,
            _ => TokenKind::PrefixedName, // Treat as prefixed name without colon (just prefix part)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_all(input: &str) -> Vec<TokenKind> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token();
            let kind = token.kind.clone();
            tokens.push(kind.clone());
            if kind == TokenKind::Eof {
                break;
            }
        }
        tokens
    }

    #[test]
    fn test_simple_select() {
        let tokens = lex_all("SELECT ?x WHERE { }");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Select,
                TokenKind::Variable,
                TokenKind::Where,
                TokenKind::LeftBrace,
                TokenKind::RightBrace,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_prefix_declaration() {
        let tokens = lex_all("PREFIX foaf: <http://xmlns.com/foaf/0.1/>");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Prefix,
                TokenKind::PrefixedName, // foaf:
                TokenKind::Iri,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_triple_pattern() {
        let tokens = lex_all("?s ?p ?o .");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Variable,
                TokenKind::Variable,
                TokenKind::Variable,
                TokenKind::Dot,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_prefixed_name() {
        let tokens = lex_all("foaf:name");
        assert_eq!(tokens, vec![TokenKind::PrefixedName, TokenKind::Eof,]);
    }

    #[test]
    fn test_blank_node() {
        let tokens = lex_all("_:b0 []");
        assert_eq!(
            tokens,
            vec![
                TokenKind::BlankNodeLabel,
                TokenKind::AnonymousBlank,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_literals() {
        let tokens = lex_all("42 3.14 1.5e10 \"hello\" 'world'");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Integer,
                TokenKind::Decimal,
                TokenKind::Double,
                TokenKind::String,
                TokenKind::String,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_operators() {
        let tokens = lex_all("= != < <= > >= && || + - * /");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Equals,
                TokenKind::NotEquals,
                TokenKind::LessThan,
                TokenKind::LessOrEqual,
                TokenKind::GreaterThan,
                TokenKind::GreaterOrEqual,
                TokenKind::AndOp,
                TokenKind::OrOp,
                TokenKind::Plus,
                TokenKind::MinusOp,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_property_path_operators() {
        let tokens = lex_all("^ ^^ | ? !");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Caret,
                TokenKind::DoubleCaret,
                TokenKind::Pipe,
                TokenKind::QuestionMark,
                TokenKind::Bang,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_comment() {
        let tokens = lex_all("SELECT # comment\n?x");
        assert_eq!(
            tokens,
            vec![TokenKind::Select, TokenKind::Variable, TokenKind::Eof,]
        );
    }

    #[test]
    fn test_a_keyword() {
        let tokens = lex_all("?x a foaf:Person");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Variable,
                TokenKind::A,
                TokenKind::PrefixedName,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_aggregates() {
        let tokens = lex_all("COUNT SUM AVG MIN MAX");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Count,
                TokenKind::Sum,
                TokenKind::Avg,
                TokenKind::Min,
                TokenKind::Max,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_datatype_and_lang() {
        let tokens = lex_all("\"hello\"^^xsd:string \"bonjour\"@fr");
        assert_eq!(
            tokens,
            vec![
                TokenKind::String,
                TokenKind::DoubleCaret,
                TokenKind::PrefixedName,
                TokenKind::String,
                TokenKind::At,
                TokenKind::PrefixedName, // fr treated as prefixed name
                TokenKind::Eof,
            ]
        );
    }
}
