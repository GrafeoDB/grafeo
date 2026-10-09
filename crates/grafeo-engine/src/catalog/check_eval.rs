//! CHECK constraint expressions: parsed once, evaluated as a query's WHERE.
//!
//! A CHECK expression is a boolean expression over an entity's properties,
//! each named by itself (`begins < ends`): comparison operators (`=`, `<>`,
//! `!=`, `<`, `<=`, `>`, `>=`), `AND`, `OR`, `NOT`, `IS [NOT] NULL`,
//! `[NOT] IN (...)`, `[NOT] BETWEEN ... AND ...`, arithmetic, parentheses
//! and literals (integers, floats, strings, booleans, `NULL`).
//!
//! The expression is parsed into the query engine's filter expression and
//! evaluated by the query's evaluator, with each property a variable bound to
//! its value: a constraint means what the same predicate means in a WHERE
//! (dates order, `1 = 1.0`, a comparison with null is unknown), and holds
//! only where that predicate is true, as a WHERE keeps only those rows.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use grafeo_common::types::Value;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    BinaryFilterOp, ExpressionPredicate, FilterExpression, UnaryFilterOp,
};
use grafeo_core::execution::vector::ValueVector;
use grafeo_core::graph::lpg::LpgStore;

/// A parsed CHECK constraint expression, ready to evaluate.
pub(crate) struct CheckExpression {
    /// The expression as written, for messages.
    text: String,
    /// The properties it reads, in the order of the evaluation row's
    /// columns.
    properties: Vec<String>,
    /// The query evaluator of the expression, each property a variable.
    predicate: ExpressionPredicate,
}

impl CheckExpression {
    /// Parses `text`.
    ///
    /// # Errors
    ///
    /// What is wrong with `text` when it is not an expression.
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let tokens = tokenize(text)?;
        let mut pos = 0;
        let expression = parse_or(&tokens, &mut pos)?;
        if pos < tokens.len() {
            return Err(format!(
                "unexpected token after expression: {:?}",
                tokens[pos]
            ));
        }
        let mut properties = Vec::new();
        collect_properties(&expression, &mut properties);
        let columns = properties
            .iter()
            .enumerate()
            .map(|(column, name)| (name.clone(), column))
            .collect::<HashMap<_, _>>();
        let store = no_graph().ok_or("no store to evaluate the expression with")?;
        Ok(Self {
            text: text.to_string(),
            properties,
            predicate: ExpressionPredicate::new(expression, columns, store),
        })
    }

    /// Whether `properties` (an entity's property map; a property it does not
    /// have is null) satisfy the expression: `Ok(true)` when it is true,
    /// `Ok(false)` when it is false or unknown. Unknown is null, or no value
    /// at all, which the query evaluator gives for a comparison with null
    /// and for an operator that does not apply (`'a' < 3`, a division by
    /// zero, an integer overflow): a WHERE drops the row then too.
    ///
    /// # Errors
    ///
    /// When the expression gives a value other than a boolean (`x + 1`).
    pub(crate) fn evaluate(&self, properties: &[(String, Value)]) -> Result<bool, String> {
        let columns = self
            .properties
            .iter()
            .map(|name| {
                let value = properties
                    .iter()
                    .find(|(key, _)| key == name)
                    .map_or(Value::Null, |(_, value)| value.clone());
                ValueVector::from_values(&[value])
            })
            .collect();
        match self.predicate.eval_at(&DataChunk::new(columns), 0) {
            Some(Value::Bool(holds)) => Ok(holds),
            Some(Value::Null) | None => Ok(false),
            Some(other) => Err(format!("({}) gives {other:?}, not a boolean", self.text)),
        }
    }
}

/// The store an expression evaluator reads graph elements from: an empty
/// one, since a CHECK expression reads property values only. `None` when it
/// cannot be made (an arena that does not allocate).
fn no_graph() -> Option<Arc<LpgStore>> {
    static NO_GRAPH: OnceLock<Option<Arc<LpgStore>>> = OnceLock::new();
    NO_GRAPH
        .get_or_init(|| LpgStore::new().ok().map(Arc::new))
        .clone()
}

/// Adds the properties `expression` reads to `properties`, each once.
fn collect_properties(expression: &FilterExpression, properties: &mut Vec<String>) {
    match expression {
        FilterExpression::Variable(name) => {
            if !properties.contains(name) {
                properties.push(name.clone());
            }
        }
        FilterExpression::Binary { left, right, .. } => {
            collect_properties(left, properties);
            collect_properties(right, properties);
        }
        FilterExpression::Unary { operand, .. } => collect_properties(operand, properties),
        FilterExpression::List(items) => {
            for item in items {
                collect_properties(item, properties);
            }
        }
        // The parser below builds no other expression.
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Integer(i64),
    Float(f64),
    StringLit(String),
    True,
    False,
    Null,
    And,
    Or,
    Not,
    Is,
    In,
    Between,
    LParen,
    RParen,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        // Skip whitespace
        if ch.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Operators and punctuation
        match ch {
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
                continue;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
                continue;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
                continue;
            }
            '=' => {
                tokens.push(Token::Eq);
                i += 1;
                continue;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
                continue;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
                continue;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
                continue;
            }
            '%' => {
                tokens.push(Token::Percent);
                i += 1;
                continue;
            }
            '<' => {
                if i + 1 < len && chars[i + 1] == '>' {
                    tokens.push(Token::Neq);
                    i += 2;
                } else if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(Token::Le);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
                continue;
            }
            '>' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(Token::Ge);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
                continue;
            }
            '!' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(Token::Neq);
                    i += 2;
                    continue;
                }
                return Err(format!("unexpected character '!' at position {i}"));
            }
            '-' => {
                // Unary minus before a number: only when at start or after an
                // operator / open-paren (i.e., not after a value token).
                let is_unary = tokens.is_empty()
                    || matches!(
                        tokens.last(),
                        Some(
                            Token::LParen
                                | Token::Comma
                                | Token::And
                                | Token::Or
                                | Token::Not
                                | Token::Eq
                                | Token::Neq
                                | Token::Lt
                                | Token::Le
                                | Token::Gt
                                | Token::Ge
                                | Token::Plus
                                | Token::Minus
                                | Token::Star
                                | Token::Slash
                                | Token::Percent
                                | Token::Is
                                | Token::Between
                        )
                    );
                if is_unary && i + 1 < len && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '.')
                {
                    // Consume the number with the minus sign
                    let start = i;
                    i += 1; // skip '-'
                    while i < len
                        && (chars[i].is_ascii_digit()
                            || chars[i] == '.'
                            || chars[i] == 'e'
                            || chars[i] == 'E')
                    {
                        i += 1;
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
                        let val: f64 = num_str
                            .parse()
                            .map_err(|e| format!("invalid float '{num_str}': {e}"))?;
                        tokens.push(Token::Float(val));
                    } else {
                        let val: i64 = num_str
                            .parse()
                            .map_err(|e| format!("invalid integer '{num_str}': {e}"))?;
                        tokens.push(Token::Integer(val));
                    }
                } else {
                    tokens.push(Token::Minus);
                    i += 1;
                }
                continue;
            }
            _ => {}
        }

        // String literals (single-quoted)
        if ch == '\'' {
            i += 1;
            let mut s = String::new();
            while i < len {
                if chars[i] == '\'' {
                    if i + 1 < len && chars[i + 1] == '\'' {
                        // Escaped single quote
                        s.push('\'');
                        i += 2;
                    } else {
                        break;
                    }
                } else {
                    s.push(chars[i]);
                    i += 1;
                }
            }
            if i >= len {
                return Err("unterminated string literal".to_string());
            }
            i += 1; // skip closing quote
            tokens.push(Token::StringLit(s));
            continue;
        }

        // Numbers
        if ch.is_ascii_digit() || ch == '.' {
            let start = i;
            while i < len
                && (chars[i].is_ascii_digit()
                    || chars[i] == '.'
                    || chars[i] == 'e'
                    || chars[i] == 'E')
            {
                i += 1;
            }
            let num_str: String = chars[start..i].iter().collect();
            if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
                let val: f64 = num_str
                    .parse()
                    .map_err(|e| format!("invalid float '{num_str}': {e}"))?;
                tokens.push(Token::Float(val));
            } else {
                let val: i64 = num_str
                    .parse()
                    .map_err(|e| format!("invalid integer '{num_str}': {e}"))?;
                tokens.push(Token::Integer(val));
            }
            continue;
        }

        // Identifiers and keywords
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let upper = word.to_ascii_uppercase();
            match upper.as_str() {
                "TRUE" => tokens.push(Token::True),
                "FALSE" => tokens.push(Token::False),
                "NULL" => tokens.push(Token::Null),
                "AND" => tokens.push(Token::And),
                "OR" => tokens.push(Token::Or),
                "NOT" => tokens.push(Token::Not),
                "IS" => tokens.push(Token::Is),
                "IN" => tokens.push(Token::In),
                "BETWEEN" => tokens.push(Token::Between),
                _ => tokens.push(Token::Ident(word)),
            }
            continue;
        }

        return Err(format!("unexpected character '{ch}' at position {i}"));
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn binary(left: FilterExpression, op: BinaryFilterOp, right: FilterExpression) -> FilterExpression {
    FilterExpression::Binary {
        left: Box::new(left),
        op,
        right: Box::new(right),
    }
}

fn unary(op: UnaryFilterOp, operand: FilterExpression) -> FilterExpression {
    FilterExpression::Unary {
        op,
        operand: Box::new(operand),
    }
}

// ---------------------------------------------------------------------------
// Recursive-descent parser, into the query engine's filter expressions:
// `x NOT IN (...)` is `NOT (x IN [...])`, and `x BETWEEN a AND b` is
// `x >= a AND x <= b` (SQL's definition of BETWEEN).
//
// Grammar:
//   expr        -> or_expr
//   or_expr     -> and_expr (OR and_expr)*
//   and_expr    -> not_expr (AND not_expr)*
//   not_expr    -> NOT not_expr | comparison
//   comparison  -> addition ((= | <> | < | <= | > | >=) addition)?
//                | addition IS [NOT] NULL
//                | addition [NOT] IN (list)
//                | addition [NOT] BETWEEN addition AND addition
//   addition    -> multiply ((+ | -) multiply)*
//   multiply    -> unary ((* | / | %) unary)*
//   unary       -> - unary | primary
//   primary     -> Ident | Number | StringLit | True | False | Null
//                | ( expr )
// ---------------------------------------------------------------------------

fn parse_or(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    let mut left = parse_and(tokens, pos)?;
    while *pos < tokens.len() && tokens[*pos] == Token::Or {
        *pos += 1;
        let right = parse_and(tokens, pos)?;
        left = binary(left, BinaryFilterOp::Or, right);
    }
    Ok(left)
}

fn parse_and(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    let mut left = parse_not(tokens, pos)?;
    while *pos < tokens.len() && tokens[*pos] == Token::And {
        *pos += 1;
        let right = parse_not(tokens, pos)?;
        left = binary(left, BinaryFilterOp::And, right);
    }
    Ok(left)
}

fn parse_not(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    if *pos < tokens.len() && tokens[*pos] == Token::Not {
        *pos += 1;
        let inner = parse_not(tokens, pos)?;
        return Ok(unary(UnaryFilterOp::Not, inner));
    }
    parse_comparison(tokens, pos)
}

fn parse_comparison(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    let left = parse_addition(tokens, pos)?;

    if *pos < tokens.len() {
        // IS [NOT] NULL
        if tokens[*pos] == Token::Is {
            *pos += 1;
            if *pos < tokens.len() && tokens[*pos] == Token::Not {
                *pos += 1;
                expect_token(tokens, pos, &Token::Null, "NULL")?;
                return Ok(unary(UnaryFilterOp::IsNotNull, left));
            }
            expect_token(tokens, pos, &Token::Null, "NULL")?;
            return Ok(unary(UnaryFilterOp::IsNull, left));
        }

        // [NOT] IN (list)
        if tokens[*pos] == Token::In {
            *pos += 1;
            let list = parse_in_list(tokens, pos)?;
            return Ok(binary(left, BinaryFilterOp::In, list));
        }
        if tokens[*pos] == Token::Not && *pos + 1 < tokens.len() && tokens[*pos + 1] == Token::In {
            *pos += 2;
            let list = parse_in_list(tokens, pos)?;
            return Ok(unary(
                UnaryFilterOp::Not,
                binary(left, BinaryFilterOp::In, list),
            ));
        }

        // [NOT] BETWEEN low AND high
        if tokens[*pos] == Token::Between {
            *pos += 1;
            return parse_between_rest(left, false, tokens, pos);
        }
        if tokens[*pos] == Token::Not
            && *pos + 1 < tokens.len()
            && tokens[*pos + 1] == Token::Between
        {
            *pos += 2;
            return parse_between_rest(left, true, tokens, pos);
        }

        // Comparison operators
        let op = match tokens[*pos] {
            Token::Eq => Some(BinaryFilterOp::Eq),
            Token::Neq => Some(BinaryFilterOp::Ne),
            Token::Lt => Some(BinaryFilterOp::Lt),
            Token::Le => Some(BinaryFilterOp::Le),
            Token::Gt => Some(BinaryFilterOp::Gt),
            Token::Ge => Some(BinaryFilterOp::Ge),
            _ => None,
        };
        if let Some(op) = op {
            *pos += 1;
            let right = parse_addition(tokens, pos)?;
            return Ok(binary(left, op, right));
        }
    }

    Ok(left)
}

/// Reads `( item, ... )` as a list expression.
fn parse_in_list(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    expect_token(tokens, pos, &Token::LParen, "(")?;
    let mut items = Vec::new();
    if *pos < tokens.len() && tokens[*pos] != Token::RParen {
        items.push(parse_addition(tokens, pos)?);
        while *pos < tokens.len() && tokens[*pos] == Token::Comma {
            *pos += 1;
            items.push(parse_addition(tokens, pos)?);
        }
    }
    expect_token(tokens, pos, &Token::RParen, ")")?;
    Ok(FilterExpression::List(items))
}

fn parse_between_rest(
    value: FilterExpression,
    negated: bool,
    tokens: &[Token],
    pos: &mut usize,
) -> Result<FilterExpression, String> {
    let low = parse_addition(tokens, pos)?;
    expect_token(tokens, pos, &Token::And, "AND")?;
    let high = parse_addition(tokens, pos)?;
    let between = binary(
        binary(value.clone(), BinaryFilterOp::Ge, low),
        BinaryFilterOp::And,
        binary(value, BinaryFilterOp::Le, high),
    );
    Ok(if negated {
        unary(UnaryFilterOp::Not, between)
    } else {
        between
    })
}

fn parse_addition(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    let mut left = parse_multiply(tokens, pos)?;
    while *pos < tokens.len() {
        let op = match tokens[*pos] {
            Token::Plus => BinaryFilterOp::Add,
            Token::Minus => BinaryFilterOp::Sub,
            _ => break,
        };
        *pos += 1;
        let right = parse_multiply(tokens, pos)?;
        left = binary(left, op, right);
    }
    Ok(left)
}

fn parse_multiply(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    let mut left = parse_unary(tokens, pos)?;
    while *pos < tokens.len() {
        let op = match tokens[*pos] {
            Token::Star => BinaryFilterOp::Mul,
            Token::Slash => BinaryFilterOp::Div,
            Token::Percent => BinaryFilterOp::Mod,
            _ => break,
        };
        *pos += 1;
        let right = parse_unary(tokens, pos)?;
        left = binary(left, op, right);
    }
    Ok(left)
}

fn parse_unary(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    if *pos < tokens.len() && tokens[*pos] == Token::Minus {
        *pos += 1;
        let inner = parse_unary(tokens, pos)?;
        return Ok(unary(UnaryFilterOp::Neg, inner));
    }
    parse_primary(tokens, pos)
}

fn parse_primary(tokens: &[Token], pos: &mut usize) -> Result<FilterExpression, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".to_string());
    }
    let literal = FilterExpression::Literal;
    let parsed = match &tokens[*pos] {
        Token::LParen => {
            *pos += 1;
            let inner = parse_or(tokens, pos)?;
            expect_token(tokens, pos, &Token::RParen, ")")?;
            return Ok(inner);
        }
        Token::Ident(name) => FilterExpression::Variable(name.clone()),
        Token::Integer(n) => literal(Value::Int64(*n)),
        Token::Float(f) => literal(Value::Float64(*f)),
        Token::StringLit(s) => literal(Value::String(s.as_str().into())),
        Token::True => literal(Value::Bool(true)),
        Token::False => literal(Value::Bool(false)),
        Token::Null => literal(Value::Null),
        other => return Err(format!("unexpected token: {other:?}")),
    };
    *pos += 1;
    Ok(parsed)
}

fn expect_token(
    tokens: &[Token],
    pos: &mut usize,
    expected: &Token,
    label: &str,
) -> Result<(), String> {
    if *pos >= tokens.len() {
        return Err(format!("expected {label}, found end of expression"));
    }
    if &tokens[*pos] != expected {
        return Err(format!("expected {label}, found {:?}", tokens[*pos]));
    }
    *pos += 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses `expression` and evaluates it for `properties`.
    fn evaluate_check(expression: &str, properties: &[(String, Value)]) -> Result<bool, String> {
        CheckExpression::parse(expression)?.evaluate(properties)
    }

    fn props(pairs: &[(&str, Value)]) -> Vec<(String, Value)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    // -- Basic comparisons --

    #[test]
    fn test_integer_equality() {
        let p = props(&[("age", Value::Int64(30))]);
        assert!(evaluate_check("age = 30", &p).unwrap());
        assert!(!evaluate_check("age = 25", &p).unwrap());
    }

    #[test]
    fn test_integer_inequality() {
        let p = props(&[("age", Value::Int64(30))]);
        assert!(evaluate_check("age <> 25", &p).unwrap());
        assert!(!evaluate_check("age <> 30", &p).unwrap());
    }

    #[test]
    fn test_integer_ordering() {
        let p = props(&[("age", Value::Int64(30))]);
        assert!(evaluate_check("age > 18", &p).unwrap());
        assert!(evaluate_check("age >= 30", &p).unwrap());
        assert!(evaluate_check("age < 100", &p).unwrap());
        assert!(evaluate_check("age <= 30", &p).unwrap());
        assert!(!evaluate_check("age < 30", &p).unwrap());
        assert!(!evaluate_check("age > 30", &p).unwrap());
    }

    #[test]
    fn test_float_comparison() {
        let p = props(&[("score", Value::Float64(3.15))]);
        assert!(evaluate_check("score > 3.0", &p).unwrap());
        assert!(evaluate_check("score < 4.0", &p).unwrap());
    }

    #[test]
    fn test_string_comparison() {
        let p = props(&[("name", Value::String("Gus".into()))]);
        assert!(evaluate_check("name = 'Gus'", &p).unwrap());
        assert!(!evaluate_check("name = 'Alix'", &p).unwrap());
    }

    #[test]
    fn test_cross_type_numeric() {
        let p = props(&[("score", Value::Int64(10))]);
        assert!(evaluate_check("score > 9.5", &p).unwrap());
        assert!(!evaluate_check("score > 10.5", &p).unwrap());
    }

    // -- Boolean operators --

    #[test]
    fn test_and() {
        let p = props(&[("age", Value::Int64(25)), ("score", Value::Int64(90))]);
        assert!(evaluate_check("age >= 18 AND score >= 80", &p).unwrap());
        assert!(!evaluate_check("age >= 30 AND score >= 80", &p).unwrap());
    }

    #[test]
    fn test_or() {
        let p = props(&[("age", Value::Int64(15))]);
        assert!(evaluate_check("age < 18 OR age > 65", &p).unwrap());
        assert!(!evaluate_check("age > 18 OR age < 10", &p).unwrap());
    }

    #[test]
    fn test_not() {
        let p = props(&[("active", Value::Bool(false))]);
        assert!(evaluate_check("NOT active", &p).unwrap());
        assert!(!evaluate_check("active", &p).unwrap());
    }

    #[test]
    fn test_combined_boolean() {
        let p = props(&[("age", Value::Int64(25)), ("vip", Value::Bool(true))]);
        assert!(evaluate_check("age >= 18 AND (vip OR age > 30)", &p).unwrap());
    }

    // -- NULL handling --

    #[test]
    fn test_is_null() {
        let p = props(&[("email", Value::Null)]);
        assert!(evaluate_check("email IS NULL", &p).unwrap());
        assert!(!evaluate_check("email IS NOT NULL", &p).unwrap());
    }

    #[test]
    fn test_is_not_null() {
        let p = props(&[("email", Value::String("a@b.com".into()))]);
        assert!(evaluate_check("email IS NOT NULL", &p).unwrap());
        assert!(!evaluate_check("email IS NULL", &p).unwrap());
    }

    #[test]
    fn test_missing_property_is_null() {
        let p = props(&[]);
        assert!(evaluate_check("phantom IS NULL", &p).unwrap());
    }

    #[test]
    fn test_null_comparison_is_false() {
        // SQL/GQL: NULL = NULL -> false, NULL <> NULL -> false
        let p = props(&[("x", Value::Null)]);
        assert!(!evaluate_check("x = 1", &p).unwrap());
        assert!(!evaluate_check("x <> 1", &p).unwrap());
        assert!(!evaluate_check("x > 0", &p).unwrap());
    }

    // -- Arithmetic in comparisons --

    #[test]
    fn test_arithmetic_addition() {
        let p = props(&[("price", Value::Int64(100)), ("tax", Value::Int64(20))]);
        assert!(evaluate_check("price + tax = 120", &p).unwrap());
    }

    #[test]
    fn test_arithmetic_subtraction() {
        let p = props(&[("a", Value::Int64(50))]);
        assert!(evaluate_check("a - 10 > 30", &p).unwrap());
    }

    #[test]
    fn test_arithmetic_multiplication() {
        let p = props(&[("qty", Value::Int64(5)), ("price", Value::Int64(10))]);
        assert!(evaluate_check("qty * price = 50", &p).unwrap());
    }

    #[test]
    fn test_arithmetic_modulo() {
        let p = props(&[("x", Value::Int64(10))]);
        assert!(evaluate_check("x % 3 = 1", &p).unwrap());
    }

    // -- IN list --

    #[test]
    fn test_in_list() {
        let p = props(&[("status", Value::String("active".into()))]);
        assert!(evaluate_check("status IN ('active', 'pending')", &p).unwrap());
        assert!(!evaluate_check("status IN ('closed', 'archived')", &p).unwrap());
    }

    #[test]
    fn test_not_in_list() {
        let p = props(&[("status", Value::String("active".into()))]);
        assert!(evaluate_check("status NOT IN ('closed', 'archived')", &p).unwrap());
        assert!(!evaluate_check("status NOT IN ('active', 'pending')", &p).unwrap());
    }

    // -- BETWEEN --

    #[test]
    fn test_between() {
        let p = props(&[("age", Value::Int64(25))]);
        assert!(evaluate_check("age BETWEEN 18 AND 65", &p).unwrap());
        assert!(!evaluate_check("age BETWEEN 30 AND 65", &p).unwrap());
    }

    #[test]
    fn test_not_between() {
        let p = props(&[("age", Value::Int64(10))]);
        assert!(evaluate_check("age NOT BETWEEN 18 AND 65", &p).unwrap());
        assert!(!evaluate_check("age NOT BETWEEN 5 AND 15", &p).unwrap());
    }

    // -- Edge cases --

    #[test]
    fn test_escaped_string() {
        let p = props(&[("name", Value::String("it's".into()))]);
        assert!(evaluate_check("name = 'it''s'", &p).unwrap());
    }

    #[test]
    fn test_nested_parentheses() {
        let p = props(&[("x", Value::Int64(5))]);
        assert!(evaluate_check("((x > 1) AND (x < 10))", &p).unwrap());
    }

    #[test]
    fn test_negative_number() {
        let p = props(&[("temp", Value::Int64(-5))]);
        assert!(evaluate_check("temp < 0", &p).unwrap());
        assert!(evaluate_check("temp = -5", &p).unwrap());
    }

    #[test]
    fn test_bool_literal_true() {
        let p = props(&[("active", Value::Bool(true))]);
        assert!(evaluate_check("active = TRUE", &p).unwrap());
    }

    #[test]
    fn test_bang_equals() {
        let p = props(&[("x", Value::Int64(5))]);
        assert!(evaluate_check("x != 3", &p).unwrap());
        assert!(!evaluate_check("x != 5", &p).unwrap());
    }

    // -- Error cases --

    #[test]
    fn test_empty_expression_error() {
        let p = props(&[]);
        assert!(evaluate_check("", &p).is_err());
    }

    #[test]
    fn test_unterminated_string_error() {
        let p = props(&[]);
        assert!(evaluate_check("name = 'oops", &p).is_err());
    }

    /// An operator that does not apply has no value, as in a WHERE, which
    /// drops the row: the check does not hold.
    #[test]
    fn test_division_by_zero_does_not_hold() {
        let p = props(&[("x", Value::Int64(10))]);
        assert_eq!(evaluate_check("x / 0 = 1", &p), Ok(false));
        assert_eq!(evaluate_check("NOT (x / 0 = 1)", &p), Ok(false));
    }

    #[test]
    fn test_incomparable_types_do_not_hold() {
        let p = props(&[("x", Value::Bool(true))]);
        assert_eq!(evaluate_check("x > 5", &p), Ok(false));
        assert_eq!(evaluate_check("NOT (x > 5)", &p), Ok(false));
    }

    // -- Arithmetic in boolean context --

    #[test]
    fn test_arithmetic_in_boolean_context_errors() {
        let p = props(&[("x", Value::Int64(5))]);
        assert!(evaluate_check("x + 1", &p).is_err());
    }

    // -- Integer overflow / underflow --

    #[test]
    fn test_integer_overflow_add() {
        let p = props(&[("x", Value::Int64(i64::MAX))]);
        assert_eq!(evaluate_check("x + 1 > 0", &p), Ok(false));
    }

    #[test]
    fn test_integer_underflow_sub() {
        let p = props(&[("x", Value::Int64(i64::MIN))]);
        assert_eq!(evaluate_check("x - 1 < 0", &p), Ok(false));
    }

    #[test]
    fn test_integer_overflow_mul() {
        let p = props(&[("x", Value::Int64(i64::MAX))]);
        assert_eq!(evaluate_check("x * 2 > 0", &p), Ok(false));
    }

    #[test]
    fn test_modulo_by_zero() {
        let p = props(&[("x", Value::Int64(10))]);
        assert_eq!(evaluate_check("x % 0 = 0", &p), Ok(false));
    }

    // -- Float arithmetic --

    #[test]
    fn test_float_arithmetic() {
        let p = props(&[("x", Value::Float64(2.5))]);
        assert!(evaluate_check("x * 2.0 = 5.0", &p).unwrap());
        assert!(evaluate_check("x + 1.5 = 4.0", &p).unwrap());
        assert!(evaluate_check("x - 0.5 = 2.0", &p).unwrap());
    }

    #[test]
    fn test_float_division() {
        let p = props(&[("x", Value::Float64(10.0))]);
        assert!(evaluate_check("x / 2.0 = 5.0", &p).unwrap());
    }

    #[test]
    fn test_float_modulo() {
        let p = props(&[("x", Value::Float64(10.0))]);
        assert!(evaluate_check("x % 3.0 = 1.0", &p).unwrap());
    }

    // -- Cross-type numeric promotion --

    #[test]
    fn test_int_float_cross_promotion() {
        let p = props(&[("x", Value::Int64(5)), ("y", Value::Float64(2.5))]);
        assert!(evaluate_check("x + y = 7.5", &p).unwrap());
        assert!(evaluate_check("y * x = 12.5", &p).unwrap());
    }

    // -- Unsupported arithmetic types --

    /// A string plus a number is the concatenation, as in a query, so the
    /// comparison is false; arithmetic on a boolean has no value, so the check does not hold.
    #[test]
    fn test_arithmetic_on_strings_concatenates() {
        let p = props(&[
            ("x", Value::String("hello".into())),
            ("b", Value::Bool(true)),
        ]);
        assert_eq!(evaluate_check("x + 1 = 2", &p), Ok(false));
        assert_eq!(evaluate_check("x + 1 = 'hello1'", &p), Ok(true));
        assert_eq!(evaluate_check("b + 1 = 2", &p), Ok(false));
        // A value that is not a boolean is not a condition.
        assert!(evaluate_check("b", &p).is_ok());
        assert!(evaluate_check("x + 1", &p).is_err());
    }

    // -- Negated IN and BETWEEN --

    #[test]
    fn test_not_in_list_generic() {
        let p = props(&[("x", Value::Int64(5))]);
        assert!(evaluate_check("x NOT IN (1, 2, 3)", &p).unwrap());
        assert!(!evaluate_check("x NOT IN (5, 6, 7)", &p).unwrap());
    }

    #[test]
    fn test_not_between_generic() {
        let p = props(&[("x", Value::Int64(5))]);
        assert!(evaluate_check("x NOT BETWEEN 10 AND 20", &p).unwrap());
        assert!(!evaluate_check("x NOT BETWEEN 1 AND 10", &p).unwrap());
    }

    // -- Null propagation in arithmetic --

    #[test]
    fn test_null_in_arithmetic_comparison() {
        let p = props(&[("x", Value::Null)]);
        // NULL + 1 comparison should yield false (not error)
        assert!(!evaluate_check("x > 5", &p).unwrap());
    }

    // -- What the predicate means in a query --

    /// Dates, times and timestamps order as in a query; every write to a
    /// node with `CHECK (begins < ends)` on two dates used to fail with
    /// "cannot compare".
    #[test]
    fn temporal_values_compare_as_in_a_query() {
        use grafeo_common::types::{Date, Time, Timestamp};
        let p = props(&[
            ("begins", Value::Date(Date::from_ymd(2024, 3, 19).unwrap())),
            ("ends", Value::Date(Date::from_ymd(2024, 3, 22).unwrap())),
            ("opens", Value::Time(Time::from_hms(8, 30, 0).unwrap())),
            ("closes", Value::Time(Time::from_hms(19, 0, 0).unwrap())),
            ("sent", Value::Timestamp(Timestamp::from_secs(3))),
            ("read", Value::Timestamp(Timestamp::from_secs(88))),
        ]);
        assert_eq!(evaluate_check("begins < ends", &p), Ok(true));
        assert_eq!(evaluate_check("ends <= begins", &p), Ok(false));
        assert_eq!(evaluate_check("opens < closes", &p), Ok(true));
        assert_eq!(evaluate_check("sent >= read", &p), Ok(false));
        assert_eq!(
            evaluate_check("begins BETWEEN begins AND ends", &p),
            Ok(true)
        );
    }

    /// An integer equals the float of the same value, as `=` and `<>` say in
    /// a query and as `<=` already said here; `1 = 1.0` used to be false.
    #[test]
    fn an_integer_equals_the_same_float() {
        let p = props(&[("x", Value::Int64(1)), ("y", Value::Float64(19.0))]);
        assert_eq!(evaluate_check("x = 1.0", &p), Ok(true));
        assert_eq!(evaluate_check("x <> 1.0", &p), Ok(false));
        assert_eq!(evaluate_check("y = 19", &p), Ok(true));
        assert_eq!(evaluate_check("x IN (3.0, 1.0)", &p), Ok(true));
    }

    /// A comparison with null is unknown, and so is its negation: the check
    /// holds only for a true predicate, as a WHERE keeps only the rows its
    /// predicate is true for. `NOT (x > 0)` used to hold for a null `x`.
    #[test]
    fn an_unknown_predicate_does_not_hold() {
        let p = props(&[("x", Value::Null), ("y", Value::Int64(3))]);
        assert_eq!(evaluate_check("NOT (x > 0)", &p), Ok(false));
        assert_eq!(evaluate_check("NOT (x = 1)", &p), Ok(false));
        assert_eq!(evaluate_check("x > 0 OR y = 3", &p), Ok(true));
        assert_eq!(evaluate_check("x IS NULL OR x > 0", &p), Ok(true));
        // No item matches and one is null: unknown, also negated.
        assert_eq!(evaluate_check("y NOT IN (1, NULL)", &p), Ok(false));
        assert_eq!(evaluate_check("y IN (1, NULL)", &p), Ok(false));
        assert_eq!(evaluate_check("y IN (3, NULL)", &p), Ok(true));
    }

    // -- Complex nested boolean --

    #[test]
    fn test_nested_and_or_not() {
        let p = props(&[("x", Value::Int64(5)), ("y", Value::Int64(10))]);
        assert!(evaluate_check("(x > 0 AND y > 0) OR x < -100", &p).unwrap());
        assert!(evaluate_check("NOT (x > 100)", &p).unwrap());
        assert!(!evaluate_check("NOT (x > 0 AND y > 0)", &p).unwrap());
    }
}
