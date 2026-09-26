//! Gremlin Lexer.
//!
//! Tokenizes Gremlin query strings.

use std::iter::Peekable;
use std::str::Chars;

use grafeo_common::utils::error::SourceSpan;

/// Token types for Gremlin.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    /// Integer literal.
    Integer(i64),
    /// Floating-point literal.
    Float(f64),
    /// String literal.
    String(String),
    /// Boolean `true` literal.
    True,
    /// Boolean `false` literal.
    False,

    // Identifiers and keywords
    /// An unrecognized identifier.
    Identifier(String),

    // Graph source
    /// The `g` graph traversal source.
    G,
    /// The `V()` vertex step.
    V,
    /// The `E()` edge step.
    E,
    /// The `addV()` add-vertex step.
    AddV,
    /// The `addE()` add-edge step.
    AddE,

    // Navigation steps
    /// The `out()` outgoing-adjacency step.
    Out,
    /// The `in()` incoming-adjacency step.
    In,
    /// The `both()` bidirectional-adjacency step.
    Both,
    /// The `outE()` outgoing-edge step.
    OutE,
    /// The `inE()` incoming-edge step.
    InE,
    /// The `bothE()` bidirectional-edge step.
    BothE,
    /// The `outV()` outgoing-vertex step.
    OutV,
    /// The `inV()` incoming-vertex step.
    InV,
    /// The `bothV()` both-vertices step.
    BothV,
    /// The `otherV()` opposite-vertex step.
    OtherV,

    // Filter steps
    /// The `has()` property-filter step.
    Has,
    /// The `hasLabel()` label-filter step.
    HasLabel,
    /// The `hasId()` id-filter step.
    HasId,
    /// The `hasNot()` property-absence filter step.
    HasNot,
    /// The `filter()` general-purpose filter step.
    Filter,
    /// The `where()` traversal-filter step.
    Where,
    /// The `and()` logical conjunction step.
    And,
    /// The `or()` logical disjunction step.
    Or,
    /// The `not()` logical negation step.
    Not,
    /// The `dedup()` deduplication step.
    Dedup,
    /// The `limit()` result-limiting step.
    Limit,
    /// The `skip()` result-skipping step.
    Skip,
    /// The `range()` result-range step.
    Range,

    // Map steps
    /// The `values()` property-value projection step.
    Values,
    /// The `valueMap()` property-map projection step.
    ValueMap,
    /// The `elementMap()` full-element projection step.
    ElementMap,
    /// The `id()` element-id projection step.
    Id,
    /// The `label()` element-label projection step.
    Label,
    /// The `properties()` property projection step.
    Properties,
    /// The `constant()` constant-value step.
    Constant,
    /// The `count()` counting step.
    Count,
    /// The `sum()` summation step.
    Sum,
    /// The `mean()` averaging step.
    Mean,
    /// The `min()` minimum step.
    Min,
    /// The `max()` maximum step.
    Max,
    /// The `fold()` list-aggregation step.
    Fold,
    /// The `unfold()` list-expansion step.
    Unfold,
    /// The `group()` grouping step.
    Group,
    /// The `groupCount()` group-and-count step.
    GroupCount,
    /// The `path()` traversal-path step.
    Path,
    /// The `select()` label-selection step.
    Select,
    /// The `project()` named-projection step.
    Project,
    /// The `by()` modulator step.
    By,
    /// The `order()` ordering step.
    Order,
    /// The `coalesce()` first-available step.
    Coalesce,
    /// The `optional()` optional-traversal step.
    Optional,
    /// The `union()` branch-merging step.
    Union,
    /// The `choose()` conditional branching step.
    Choose,

    // Looping steps
    /// The `repeat()` loop step.
    Repeat,
    /// The `until()` loop termination predicate.
    Until,
    /// The `times()` fixed loop count.
    Times,
    /// The `emit()` intermediate result emission.
    Emit,
    /// The `loops()` loop counter access.
    Loops,

    // Side effect steps
    /// The `as()` step-label alias.
    As,
    /// The `sideEffect()` side-effect step.
    SideEffect,
    /// The `aggregate()` eager collection step.
    Aggregate,
    /// The `store()` lazy collection step.
    Store,
    /// The `property()` property-mutation step.
    Property,
    /// The `drop()` element-removal step.
    Drop,

    // Edge creation
    /// The `from()` edge-source modulator.
    From,
    /// The `to()` edge-target modulator.
    To,

    // Predicates (P.*)
    /// The `P` predicate namespace.
    P,
    /// The `eq()` equality predicate.
    Eq,
    /// The `neq()` inequality predicate.
    Neq,
    /// The `lt()` less-than predicate.
    Lt,
    /// The `lte()` less-than-or-equal predicate.
    Lte,
    /// The `gt()` greater-than predicate.
    Gt,
    /// The `gte()` greater-than-or-equal predicate.
    Gte,
    /// The `within()` collection-membership predicate.
    Within,
    /// The `without()` collection-exclusion predicate.
    Without,
    /// The `between()` range-inclusive predicate.
    Between,
    /// The `inside()` range-exclusive predicate.
    Inside,
    /// The `outside()` range-complement predicate.
    Outside,
    /// The `containing()` substring predicate.
    Containing,
    /// The `startingWith()` prefix predicate.
    StartingWith,
    /// The `endingWith()` suffix predicate.
    EndingWith,
    /// The `regex()` regular-expression predicate.
    Regex,
    /// The `notRegex()` negated regular-expression predicate.
    NotRegex,
    /// The `notContaining()` negated substring predicate.
    NotContaining,
    /// The `notStartingWith()` negated prefix predicate.
    NotStartingWith,
    /// The `notEndingWith()` negated suffix predicate.
    NotEndingWith,

    // Tokens (T.*)
    /// The `T` token namespace (e.g., `T.id`, `T.label`).
    T,

    // Order
    /// The `asc` ascending sort order.
    Asc,
    /// The `desc` descending sort order.
    Desc,
    /// The `shuffle` random sort order.
    Shuffle,

    // Cardinality
    /// The `single` cardinality (one value per key).
    Single,
    /// The `list` cardinality (ordered multi-value).
    List,
    /// The `set` cardinality (unique multi-value).
    Set,

    // Punctuation
    /// Dot (`.`) separator.
    Dot,
    /// Comma (`,`) delimiter.
    Comma,
    /// Left parenthesis (`(`).
    LParen,
    /// Right parenthesis (`)`).
    RParen,
    /// Left bracket (`[`).
    LBracket,
    /// Right bracket (`]`).
    RBracket,
    /// Underscore (`_`) token.
    Underscore,

    /// A `$name` parameter reference.
    Parameter(String),

    // End of input
    /// End of input.
    Eof,
    /// Lexer error (e.g. integer overflow).
    Error(String),
}

/// A token with its position.
#[derive(Debug, Clone)]
pub struct Token {
    /// The token type.
    pub kind: TokenKind,
    /// Source location of this token.
    pub span: SourceSpan,
}

/// Gremlin lexer.
pub struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
    position: usize,
    line: u32,
    column: u32,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given source.
    pub fn new(source: &'a str) -> Self {
        Self {
            chars: source.chars().peekable(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Returns the next token.
    ///
    /// # Panics
    ///
    /// Panics if the internal character iterator is inconsistent after a peek-confirmed advance.
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let start = self.position;
        let start_line = self.line;
        let start_column = self.column;

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

            Some('$') => {
                // Parameter: $name
                let mut name = String::new();
                while self.peek_is(|c| c.is_alphanumeric() || c == '_') {
                    name.push(self.advance().unwrap());
                }
                TokenKind::Parameter(name)
            }

            Some(c) if c.is_alphabetic() || c == '_' => self.read_identifier(c),

            None => TokenKind::Eof,
            _ => TokenKind::Eof,
        };

        Token {
            kind,
            span: SourceSpan::new(start, self.position, start_line, start_column),
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
        if let Some(ch) = c {
            self.position += ch.len_utf8();
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
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
                if self.peek_is(|c| c == '+' || c == '-')
                    && let Some(sign) = self.advance()
                {
                    value.push(sign);
                }
            } else {
                break;
            }
        }

        if is_float {
            match value.parse() {
                Ok(f) => TokenKind::Float(f),
                Err(_) => TokenKind::Error(format!("Invalid float literal: '{value}'")),
            }
        } else {
            match value.parse::<i64>() {
                Ok(n) => TokenKind::Integer(n),
                Err(e)
                    if *e.kind() == std::num::IntErrorKind::PosOverflow
                        || *e.kind() == std::num::IntErrorKind::NegOverflow =>
                {
                    TokenKind::Error(format!(
                        "Integer literal '{value}' overflows the valid range ({} to {})",
                        i64::MIN,
                        i64::MAX
                    ))
                }
                Err(_) => TokenKind::Error(format!("Invalid integer literal: '{value}'")),
            }
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
            "repeat" => TokenKind::Repeat,
            "until" => TokenKind::Until,
            "times" => TokenKind::Times,
            "emit" => TokenKind::Emit,
            "loops" => TokenKind::Loops,
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
            "notRegex" => TokenKind::NotRegex,
            "notContaining" => TokenKind::NotContaining,
            "notStartingWith" => TokenKind::NotStartingWith,
            "notEndingWith" => TokenKind::NotEndingWith,
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
        let mut lexer = Lexer::new("g.V().has('name', 'Alix')");
        let tokens = lexer.tokenize();

        // g.V().has('name', 'Alix')
        // 0=G 1=Dot 2=V 3=LParen 4=RParen 5=Dot 6=Has 7=LParen 8=String 9=Comma 10=String 11=RParen
        assert_eq!(tokens[0].kind, TokenKind::G);
        assert_eq!(tokens[6].kind, TokenKind::Has);
        assert_eq!(tokens[8].kind, TokenKind::String("name".to_string()));
        assert_eq!(tokens[10].kind, TokenKind::String("Alix".to_string()));
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 2.78 -7");
        let tokens = lexer.tokenize();

        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
        assert_eq!(tokens[1].kind, TokenKind::Float(2.78));
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
