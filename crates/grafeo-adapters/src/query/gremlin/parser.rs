//! Gremlin Parser.
//!
//! Parses tokenized Gremlin queries into an AST.

#[allow(clippy::wildcard_imports)]
use super::ast::*;
use super::lexer::{Lexer, Token, TokenKind};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};

/// Maximum nesting depth for recursive parsing constructs (nested traversals
/// such as `.where()`, `.local()`, `.union()`, `.choose()`).
const MAX_NESTING_DEPTH: u32 = 128;

/// Gremlin parser.
pub struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    source: &'a str,
    /// Current nesting depth for recursive parsing constructs.
    nesting_depth: u32,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given source.
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        Self {
            tokens,
            position: 0,
            source,
            nesting_depth: 0,
        }
    }

    /// Increments the nesting depth and returns an error if the limit is exceeded.
    fn enter_nesting(&mut self) -> Result<()> {
        self.nesting_depth += 1;
        if self.nesting_depth > MAX_NESTING_DEPTH {
            return Err(self.error(&format!(
                "Maximum nesting depth of {MAX_NESTING_DEPTH} exceeded"
            )));
        }
        Ok(())
    }

    /// Decrements the nesting depth.
    fn exit_nesting(&mut self) {
        self.nesting_depth = self.nesting_depth.saturating_sub(1);
    }

    /// Parses the query into a statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the input contains invalid or unexpected Gremlin syntax.
    pub fn parse(&mut self) -> Result<Statement> {
        self.parse_statement()
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        // Expect 'g' at the start
        self.expect(TokenKind::G)?;
        self.expect(TokenKind::Dot)?;

        // Parse source (V, E, addV, addE)
        let source = self.parse_source()?;

        // Parse steps
        let mut steps = Vec::new();
        while self.check(TokenKind::Dot) {
            self.advance(); // consume '.'
            let step = self.parse_step()?;
            steps.push(step);
        }

        Ok(Statement { source, steps })
    }

    fn parse_source(&mut self) -> Result<TraversalSource> {
        let token = self.advance_token()?;
        match &token.kind {
            TokenKind::V => {
                self.expect(TokenKind::LParen)?;
                let ids = self.parse_optional_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(TraversalSource::V(if ids.is_empty() {
                    None
                } else {
                    Some(ids)
                }))
            }
            TokenKind::E => {
                self.expect(TokenKind::LParen)?;
                let ids = self.parse_optional_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(TraversalSource::E(if ids.is_empty() {
                    None
                } else {
                    Some(ids)
                }))
            }
            TokenKind::AddV => {
                self.expect(TokenKind::LParen)?;
                let label = if self.check_string() {
                    Some(self.parse_string()?)
                } else {
                    None
                };
                self.expect(TokenKind::RParen)?;
                Ok(TraversalSource::AddV(label))
            }
            TokenKind::AddE => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(TraversalSource::AddE(label))
            }
            _ => Err(self.error("Expected V, E, addV, or addE")),
        }
    }

    fn parse_step(&mut self) -> Result<Step> {
        let token = self.advance_token()?;
        match &token.kind {
            // Mid-traversal V() restarts from all vertices
            TokenKind::V => {
                self.expect(TokenKind::LParen)?;
                let ids = self.parse_optional_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::MidV(if ids.is_empty() { None } else { Some(ids) }))
            }
            // Navigation steps
            TokenKind::Out => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Out(labels))
            }
            TokenKind::In => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::In(labels))
            }
            TokenKind::Both => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Both(labels))
            }
            TokenKind::OutE => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::OutE(labels))
            }
            TokenKind::InE => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::InE(labels))
            }
            TokenKind::BothE => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::BothE(labels))
            }
            TokenKind::OutV => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::OutV)
            }
            TokenKind::InV => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::InV)
            }
            TokenKind::BothV => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::BothV)
            }
            TokenKind::OtherV => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::OtherV)
            }

            // Filter steps
            TokenKind::Has => {
                self.expect(TokenKind::LParen)?;
                let has_step = self.parse_has_args()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Has(has_step))
            }
            TokenKind::HasLabel => {
                self.expect(TokenKind::LParen)?;
                let labels = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::HasLabel(labels))
            }
            TokenKind::HasId => {
                self.expect(TokenKind::LParen)?;
                let ids = self.parse_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::HasId(ids))
            }
            TokenKind::HasNot => {
                self.expect(TokenKind::LParen)?;
                let key = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::HasNot(key))
            }
            TokenKind::Dedup => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Dedup(keys))
            }
            TokenKind::Limit => {
                self.expect(TokenKind::LParen)?;
                let n = self.parse_non_negative_integer("limit")?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Limit(n))
            }
            TokenKind::Skip => {
                self.expect(TokenKind::LParen)?;
                let n = self.parse_non_negative_integer("skip")?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Skip(n))
            }
            TokenKind::Range => {
                self.expect(TokenKind::LParen)?;
                let start = self.parse_non_negative_integer("range")?;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_non_negative_integer("range")?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Range(start, end))
            }

            // Map steps
            TokenKind::Values => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Values(keys))
            }
            TokenKind::ValueMap => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::ValueMap(keys))
            }
            TokenKind::ElementMap => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::ElementMap(keys))
            }
            TokenKind::Id => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Id)
            }
            TokenKind::Label => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Label)
            }
            TokenKind::Properties => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Properties(keys))
            }
            TokenKind::Constant => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Constant(value))
            }
            TokenKind::Count => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Count)
            }
            TokenKind::Sum => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Sum)
            }
            TokenKind::Mean => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Mean)
            }
            TokenKind::Min => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Min)
            }
            TokenKind::Max => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Max)
            }
            TokenKind::Fold => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Fold)
            }
            TokenKind::Unfold => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Unfold)
            }
            TokenKind::Group => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Group(None))
            }
            TokenKind::GroupCount => {
                self.expect(TokenKind::LParen)?;
                let label = if self.check_string() {
                    Some(self.parse_string()?)
                } else {
                    None
                };
                self.expect(TokenKind::RParen)?;
                Ok(Step::GroupCount(label))
            }
            TokenKind::Path => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Path)
            }
            TokenKind::Select => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Select(keys))
            }
            TokenKind::Project => {
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Project(keys))
            }
            TokenKind::By => {
                self.expect(TokenKind::LParen)?;
                let modifier = self.parse_by_modifier()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::By(modifier))
            }
            TokenKind::Order => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Order(Vec::new()))
            }

            // Side effect steps
            TokenKind::As => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::As(label))
            }
            TokenKind::Aggregate => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Aggregate(label))
            }
            TokenKind::Store => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Store(label))
            }
            TokenKind::Property => {
                self.expect(TokenKind::LParen)?;
                let prop_step = self.parse_property_args()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Property(prop_step))
            }
            TokenKind::Drop => {
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Drop)
            }

            // Edge creation
            TokenKind::From => {
                self.expect(TokenKind::LParen)?;
                let from_to = self.parse_from_to()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::From(from_to))
            }
            TokenKind::To => {
                self.expect(TokenKind::LParen)?;
                let from_to = self.parse_from_to()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::To(from_to))
            }
            TokenKind::AddV => {
                self.expect(TokenKind::LParen)?;
                let label = if self.check_string() {
                    Some(self.parse_string()?)
                } else {
                    None
                };
                self.expect(TokenKind::RParen)?;
                Ok(Step::AddV(label))
            }
            TokenKind::AddE => {
                self.expect(TokenKind::LParen)?;
                let label = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::AddE(label))
            }

            // Branch / conditional steps
            TokenKind::And => {
                self.expect(TokenKind::LParen)?;
                let traversals = self.parse_inner_traversal_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::And(traversals))
            }
            TokenKind::Or => {
                self.expect(TokenKind::LParen)?;
                let traversals = self.parse_inner_traversal_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Or(traversals))
            }
            TokenKind::Not => {
                self.expect(TokenKind::LParen)?;
                let steps = self.parse_inner_steps()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Not(steps))
            }
            TokenKind::Where => {
                self.expect(TokenKind::LParen)?;
                // where() can take a traversal or a P.predicate
                if self.check(TokenKind::P) {
                    let pred = self.parse_predicate()?;
                    self.expect(TokenKind::RParen)?;
                    // We need a variable name for the predicate, use empty string
                    Ok(Step::Where(WhereClause::Predicate(String::new(), pred)))
                } else {
                    let steps = self.parse_inner_steps()?;
                    self.expect(TokenKind::RParen)?;
                    Ok(Step::Where(WhereClause::Traversal(steps)))
                }
            }
            TokenKind::Filter => {
                self.expect(TokenKind::LParen)?;
                let steps = self.parse_inner_steps()?;
                self.expect(TokenKind::RParen)?;
                // filter() takes a traversal, convert to Filter step
                // We'll use the Where(Traversal) variant since it has the same semantics
                Ok(Step::Where(WhereClause::Traversal(steps)))
            }
            TokenKind::Choose => {
                self.expect(TokenKind::LParen)?;
                // First argument is the condition (traversal)
                let condition_steps = self.parse_inner_steps()?;
                self.expect(TokenKind::Comma)?;
                // Second argument is the true branch
                let true_steps = self.parse_inner_steps()?;
                // Optional third argument is the false branch
                let false_steps = if self.check(TokenKind::Comma) {
                    self.advance();
                    Some(self.parse_inner_steps()?)
                } else {
                    None
                };
                self.expect(TokenKind::RParen)?;
                Ok(Step::Choose(ChooseClause {
                    condition: ChooseCondition::Traversal(condition_steps),
                    true_branch: true_steps,
                    false_branch: false_steps,
                }))
            }
            TokenKind::Optional => {
                self.expect(TokenKind::LParen)?;
                let steps = self.parse_inner_steps()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Optional(steps))
            }
            TokenKind::Union => {
                self.expect(TokenKind::LParen)?;
                let traversals = self.parse_inner_traversal_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Union(traversals))
            }
            TokenKind::Coalesce => {
                self.expect(TokenKind::LParen)?;
                let traversals = self.parse_inner_traversal_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::Coalesce(traversals))
            }
            TokenKind::SideEffect => {
                self.expect(TokenKind::LParen)?;
                let steps = self.parse_inner_steps()?;
                self.expect(TokenKind::RParen)?;
                Ok(Step::SideEffect(steps))
            }

            // Looping steps
            TokenKind::Repeat => {
                self.expect(TokenKind::LParen)?;
                let body = self.parse_inner_steps()?;
                self.expect(TokenKind::RParen)?;
                let mut repeat = RepeatStep {
                    body,
                    termination: None,
                    emit: false,
                };
                // Consume .times()/.until()/.emit() modifiers at any nesting level
                self.parse_repeat_modifiers(&mut repeat)?;
                Ok(Step::Repeat(repeat))
            }

            _ => Err(self.error(&format!("Unknown step: {:?}", token.kind))),
        }
    }

    /// Consumes `.times()`, `.until()`, and `.emit()` modifiers that follow a `repeat()` step.
    fn parse_repeat_modifiers(&mut self, repeat: &mut RepeatStep) -> Result<()> {
        while self.check(TokenKind::Dot) {
            let saved = self.position;
            self.advance(); // consume '.'
            match self.current_kind().cloned() {
                Some(TokenKind::Times) => {
                    self.advance(); // consume 'times'
                    self.expect(TokenKind::LParen)?;
                    let n = self.parse_non_negative_integer("times")?;
                    self.expect(TokenKind::RParen)?;
                    repeat.termination = Some(RepeatTermination::Times(n));
                }
                Some(TokenKind::Until) => {
                    self.advance(); // consume 'until'
                    self.expect(TokenKind::LParen)?;
                    let until_steps = self.parse_inner_steps()?;
                    self.expect(TokenKind::RParen)?;
                    repeat.termination = Some(RepeatTermination::Until(until_steps));
                }
                Some(TokenKind::Emit) => {
                    self.advance(); // consume 'emit'
                    self.expect(TokenKind::LParen)?;
                    self.expect(TokenKind::RParen)?;
                    repeat.emit = true;
                }
                _ => {
                    // Not a repeat modifier, backtrack
                    self.position = saved;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Parses an integer and validates it is non-negative, returning it as `usize`.
    fn parse_non_negative_integer(&mut self, context: &str) -> Result<usize> {
        let n = self.parse_integer()?;
        if n < 0 {
            return Err(self.error(&format!(
                "{context}() requires a non-negative integer, got {n}"
            )));
        }
        usize::try_from(n).map_err(|_| {
            self.error(&format!(
                "{context}() value {n} exceeds maximum supported size"
            ))
        })
    }

    fn parse_has_args(&mut self) -> Result<HasStep> {
        let first = self.parse_string()?;

        if !self.check(TokenKind::Comma) {
            return Ok(HasStep::Key(first));
        }
        self.advance(); // consume ','

        // Check if next is a predicate (P.*)
        if self.check(TokenKind::P) {
            let pred = self.parse_predicate()?;
            return Ok(HasStep::KeyPredicate(first, pred));
        }

        // Check if next is a direct predicate call (gt, lt, etc. without P. prefix)
        if let Some(pred) = self.try_parse_direct_predicate()? {
            return Ok(HasStep::KeyPredicate(first, pred));
        }

        let second = self.parse_value()?;

        if !self.check(TokenKind::Comma) {
            return Ok(HasStep::KeyValue(first, second));
        }
        self.advance(); // consume ','

        // Three arguments: label, key, value
        let third = self.parse_value()?;
        let key = match second {
            Value::String(s) => s.to_string(),
            _ => return Err(self.error("Expected string for key")),
        };
        Ok(HasStep::LabelKeyValue(first, key, third))
    }

    /// Tries to parse a direct predicate call like `gt(28)` without the `P.` prefix.
    /// Returns `Ok(None)` if the current token is not a predicate keyword.
    fn try_parse_direct_predicate(&mut self) -> Result<Option<Predicate>> {
        let pred_kind = match self.current_kind() {
            Some(TokenKind::Eq) => Some(TokenKind::Eq),
            Some(TokenKind::Neq) => Some(TokenKind::Neq),
            Some(TokenKind::Lt) => Some(TokenKind::Lt),
            Some(TokenKind::Lte) => Some(TokenKind::Lte),
            Some(TokenKind::Gt) => Some(TokenKind::Gt),
            Some(TokenKind::Gte) => Some(TokenKind::Gte),
            Some(TokenKind::Within) => Some(TokenKind::Within),
            Some(TokenKind::Without) => Some(TokenKind::Without),
            Some(TokenKind::Between) => Some(TokenKind::Between),
            Some(TokenKind::Inside) => Some(TokenKind::Inside),
            Some(TokenKind::Outside) => Some(TokenKind::Outside),
            Some(TokenKind::Containing) => Some(TokenKind::Containing),
            Some(TokenKind::StartingWith) => Some(TokenKind::StartingWith),
            Some(TokenKind::EndingWith) => Some(TokenKind::EndingWith),
            Some(TokenKind::Regex) => Some(TokenKind::Regex),
            Some(TokenKind::NotRegex) => Some(TokenKind::NotRegex),
            Some(TokenKind::NotContaining) => Some(TokenKind::NotContaining),
            Some(TokenKind::NotStartingWith) => Some(TokenKind::NotStartingWith),
            Some(TokenKind::NotEndingWith) => Some(TokenKind::NotEndingWith),
            _ => None,
        };

        let Some(kind) = pred_kind else {
            return Ok(None);
        };

        self.advance(); // consume predicate keyword
        self.expect(TokenKind::LParen)?;

        let pred = match kind {
            TokenKind::Eq => {
                let value = self.parse_value()?;
                Predicate::Eq(value)
            }
            TokenKind::Neq => {
                let value = self.parse_value()?;
                Predicate::Neq(value)
            }
            TokenKind::Lt => {
                let value = self.parse_value()?;
                Predicate::Lt(value)
            }
            TokenKind::Lte => {
                let value = self.parse_value()?;
                Predicate::Lte(value)
            }
            TokenKind::Gt => {
                let value = self.parse_value()?;
                Predicate::Gt(value)
            }
            TokenKind::Gte => {
                let value = self.parse_value()?;
                Predicate::Gte(value)
            }
            TokenKind::Within => {
                let values = self.parse_value_list()?;
                Predicate::Within(values)
            }
            TokenKind::Without => {
                let values = self.parse_value_list()?;
                Predicate::Without(values)
            }
            TokenKind::Between => {
                let start = self.parse_value()?;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_value()?;
                Predicate::Between(start, end)
            }
            TokenKind::Inside => {
                let start = self.parse_value()?;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_value()?;
                Predicate::Inside(start, end)
            }
            TokenKind::Outside => {
                let start = self.parse_value()?;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_value()?;
                Predicate::Outside(start, end)
            }
            TokenKind::Containing => {
                let s = self.parse_string()?;
                Predicate::Containing(s)
            }
            TokenKind::StartingWith => {
                let s = self.parse_string()?;
                Predicate::StartingWith(s)
            }
            TokenKind::EndingWith => {
                let s = self.parse_string()?;
                Predicate::EndingWith(s)
            }
            TokenKind::Regex => {
                let s = self.parse_string()?;
                Predicate::Regex(s)
            }
            TokenKind::NotRegex => {
                let s = self.parse_string()?;
                Predicate::NotRegex(s)
            }
            TokenKind::NotContaining => {
                let s = self.parse_string()?;
                Predicate::NotContaining(s)
            }
            TokenKind::NotStartingWith => {
                let s = self.parse_string()?;
                Predicate::NotStartingWith(s)
            }
            TokenKind::NotEndingWith => {
                let s = self.parse_string()?;
                Predicate::NotEndingWith(s)
            }
            _ => return Err(self.error("Unknown predicate")),
        };

        self.expect(TokenKind::RParen)?;
        Ok(Some(pred))
    }

    fn parse_predicate(&mut self) -> Result<Predicate> {
        self.expect(TokenKind::P)?;
        self.expect(TokenKind::Dot)?;

        let token = self.advance_token()?;
        match &token.kind {
            TokenKind::Eq => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Eq(value))
            }
            TokenKind::Neq => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Neq(value))
            }
            TokenKind::Lt => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Lt(value))
            }
            TokenKind::Lte => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Lte(value))
            }
            TokenKind::Gt => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Gt(value))
            }
            TokenKind::Gte => {
                self.expect(TokenKind::LParen)?;
                let value = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Gte(value))
            }
            TokenKind::Within => {
                self.expect(TokenKind::LParen)?;
                let values = self.parse_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Within(values))
            }
            TokenKind::Without => {
                self.expect(TokenKind::LParen)?;
                let values = self.parse_value_list()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Without(values))
            }
            TokenKind::Between => {
                self.expect(TokenKind::LParen)?;
                let start = self.parse_value()?;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Between(start, end))
            }
            TokenKind::Inside => {
                self.expect(TokenKind::LParen)?;
                let start = self.parse_value()?;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Inside(start, end))
            }
            TokenKind::Outside => {
                self.expect(TokenKind::LParen)?;
                let start = self.parse_value()?;
                self.expect(TokenKind::Comma)?;
                let end = self.parse_value()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Outside(start, end))
            }
            TokenKind::Containing => {
                self.expect(TokenKind::LParen)?;
                let s = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::Containing(s))
            }
            TokenKind::StartingWith => {
                self.expect(TokenKind::LParen)?;
                let s = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::StartingWith(s))
            }
            TokenKind::EndingWith => {
                self.expect(TokenKind::LParen)?;
                let s = self.parse_string()?;
                self.expect(TokenKind::RParen)?;
                Ok(Predicate::EndingWith(s))
            }
            _ => Err(self.error("Unknown predicate")),
        }
    }

    fn parse_by_modifier(&mut self) -> Result<ByModifier> {
        if self.check(TokenKind::RParen) {
            return Ok(ByModifier::Identity);
        }

        if self.check(TokenKind::T) {
            self.advance();
            self.expect(TokenKind::Dot)?;
            let token = self.advance_token()?;
            let t = match &token.kind {
                TokenKind::Id => TokenType::Id,
                TokenKind::Label => TokenType::Label,
                _ => return Err(self.error("Expected T.id or T.label")),
            };
            return Ok(ByModifier::Token(t));
        }

        // Bare `label` or `id` tokens are shorthand for T.label / T.id
        if self.check(TokenKind::Label) {
            self.advance();
            return Ok(ByModifier::Token(TokenType::Label));
        }
        if self.check(TokenKind::Id) {
            self.advance();
            return Ok(ByModifier::Token(TokenType::Id));
        }

        // Check for direct order tokens: asc, desc, shuffle
        if let Some(order) = self.try_parse_sort_order() {
            return Ok(ByModifier::Order(order));
        }

        // Check for step-like tokens (count(), sum(), etc.) for traversal-style by()
        if let Some(step) = self.try_parse_by_step()? {
            return Ok(ByModifier::Traversal(vec![step]));
        }

        if self.check_string() {
            let key = self.parse_string()?;

            // Check for optional second argument: order direction
            if self.check(TokenKind::Comma) {
                self.advance(); // consume ','
                if let Some(order) = self.try_parse_sort_order() {
                    return Ok(ByModifier::KeyWithOrder(key, order));
                }
                return Err(self.error("Expected sort order (asc, desc, or shuffle) after comma"));
            }

            return Ok(ByModifier::Key(key));
        }

        Ok(ByModifier::Identity)
    }

    /// Tries to parse a step inside by() like count(), sum(), etc.
    fn try_parse_by_step(&mut self) -> Result<Option<Step>> {
        let step = match self.current_kind() {
            Some(TokenKind::Count) => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Some(Step::Count)
            }
            Some(TokenKind::Sum) => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Some(Step::Sum)
            }
            Some(TokenKind::Mean) => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Some(Step::Mean)
            }
            Some(TokenKind::Min) => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Some(Step::Min)
            }
            Some(TokenKind::Max) => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Some(Step::Max)
            }
            Some(TokenKind::Fold) => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                self.expect(TokenKind::RParen)?;
                Some(Step::Fold)
            }
            Some(TokenKind::Values) => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let keys = self.parse_string_list()?;
                self.expect(TokenKind::RParen)?;
                Some(Step::Values(keys))
            }
            _ => None,
        };
        Ok(step)
    }

    /// Tries to parse a sort order token (asc, desc, shuffle).
    fn try_parse_sort_order(&mut self) -> Option<SortOrder> {
        match self.current_kind() {
            Some(TokenKind::Asc) => {
                self.advance();
                Some(SortOrder::Asc)
            }
            Some(TokenKind::Desc) => {
                self.advance();
                Some(SortOrder::Desc)
            }
            Some(TokenKind::Shuffle) => {
                self.advance();
                Some(SortOrder::Shuffle)
            }
            _ => None,
        }
    }

    fn parse_property_args(&mut self) -> Result<PropertyStep> {
        let mut cardinality = None;

        // Check for cardinality
        match self.current_kind() {
            Some(TokenKind::Single) => {
                cardinality = Some(Cardinality::Single);
                self.advance();
                self.expect(TokenKind::Comma)?;
            }
            Some(TokenKind::List) => {
                cardinality = Some(Cardinality::List);
                self.advance();
                self.expect(TokenKind::Comma)?;
            }
            Some(TokenKind::Set) => {
                cardinality = Some(Cardinality::Set);
                self.advance();
                self.expect(TokenKind::Comma)?;
            }
            _ => {}
        }

        let key = self.parse_string()?;
        self.expect(TokenKind::Comma)?;
        let value = self.parse_value()?;

        Ok(PropertyStep {
            cardinality,
            key,
            value,
        })
    }

    fn parse_from_to(&mut self) -> Result<FromTo> {
        // Check for string label first
        if self.check_string() {
            let label = self.parse_string()?;
            return Ok(FromTo::Label(label));
        }

        // Check for traversal starting with 'g'
        if self.check(TokenKind::G) {
            let steps = self.parse_sub_traversal()?;
            return Ok(FromTo::Traversal(steps));
        }

        // Check for bare V()/E() traversal (without 'g.' prefix)
        if self.check(TokenKind::V) || self.check(TokenKind::E) {
            let steps = self.parse_bare_traversal()?;
            return Ok(FromTo::Traversal(steps));
        }

        Err(self.error("Expected label or traversal for from/to"))
    }

    /// Parse a chain of steps inside parentheses (e.g., `has('age', gt(29))` in
    /// `and(has(...), has(...))`).  Steps are separated by dots: the caller is
    /// responsible for consuming the outer delimiters.
    fn parse_inner_steps(&mut self) -> Result<Vec<Step>> {
        self.enter_nesting()?;
        let mut steps = Vec::new();
        // Parse first step
        let step = self.parse_step()?;
        steps.push(step);
        // Parse chained steps separated by dots
        while self.check(TokenKind::Dot) {
            self.advance(); // consume '.'
            let step = self.parse_step()?;
            steps.push(step);
        }
        self.exit_nesting();
        Ok(steps)
    }

    /// Parse a comma-separated list of inner traversals, e.g. inside
    /// `and(has('age', gt(29)), has('city', 'NYC'))`.
    fn parse_inner_traversal_list(&mut self) -> Result<Vec<Vec<Step>>> {
        let mut traversals = Vec::new();
        traversals.push(self.parse_inner_steps()?);
        while self.check(TokenKind::Comma) {
            self.advance(); // consume ','
            traversals.push(self.parse_inner_steps()?);
        }
        Ok(traversals)
    }

    /// Parse a bare traversal starting with V() or E() (without 'g.' prefix).
    /// Used inside from()/to() arguments, e.g. `from(V().has('name', 'Gus'))`.
    fn parse_bare_traversal(&mut self) -> Result<Vec<Step>> {
        self.enter_nesting()?;

        // Parse source (V, E, etc.) and convert to a step
        let source = self.parse_source()?;

        // Convert source to initial steps
        let mut steps = match source {
            TraversalSource::V(ids) => {
                if let Some(ids) = ids {
                    vec![Step::HasId(ids)]
                } else {
                    Vec::new()
                }
            }
            TraversalSource::E(ids) => {
                if let Some(ids) = ids {
                    vec![Step::HasId(ids)]
                } else {
                    Vec::new()
                }
            }
            TraversalSource::AddV(label) => vec![Step::AddV(label)],
            TraversalSource::AddE(label) => vec![Step::AddE(label)],
        };

        // Parse additional steps until we hit the closing paren of from/to
        while self.check(TokenKind::Dot) {
            self.advance(); // consume '.'
            let step = self.parse_step()?;
            steps.push(step);
        }

        self.exit_nesting();
        Ok(steps)
    }

    /// Parse a sub-traversal (e.g., g.V().has('name', 'Gus'))
    /// Returns the steps as a Vec<Step>
    fn parse_sub_traversal(&mut self) -> Result<Vec<Step>> {
        self.enter_nesting()?;

        // Consume 'g'
        self.expect(TokenKind::G)?;
        self.expect(TokenKind::Dot)?;

        // Parse source (V, E, etc.) and convert to a step
        let source = self.parse_source()?;

        // Convert source to initial steps
        let mut steps = match source {
            TraversalSource::V(ids) => {
                if let Some(ids) = ids {
                    vec![Step::HasId(ids)]
                } else {
                    Vec::new()
                }
            }
            TraversalSource::E(ids) => {
                if let Some(ids) = ids {
                    vec![Step::HasId(ids)]
                } else {
                    Vec::new()
                }
            }
            TraversalSource::AddV(label) => vec![Step::AddV(label)],
            TraversalSource::AddE(label) => vec![Step::AddE(label)],
        };

        // Parse additional steps until we hit the closing paren of from/to
        while self.check(TokenKind::Dot) {
            self.advance(); // consume '.'
            let step = self.parse_step()?;
            steps.push(step);
        }

        self.exit_nesting();
        Ok(steps)
    }

    fn parse_string_list(&mut self) -> Result<Vec<String>> {
        let mut result = Vec::new();
        while self.check_string() {
            result.push(self.parse_string()?);
            if !self.check(TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        Ok(result)
    }

    fn parse_value_list(&mut self) -> Result<Vec<Value>> {
        let mut result = Vec::new();
        loop {
            if self.check(TokenKind::RParen) {
                break;
            }
            result.push(self.parse_value()?);
            if !self.check(TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        Ok(result)
    }

    fn parse_optional_value_list(&mut self) -> Result<Vec<Value>> {
        if self.check(TokenKind::RParen) {
            return Ok(Vec::new());
        }
        self.parse_value_list()
    }

    fn parse_string(&mut self) -> Result<String> {
        let token = self.advance_token()?;
        match token.kind {
            TokenKind::String(s) => Ok(s),
            TokenKind::Identifier(s) => Ok(s),
            _ => Err(self.error("Expected string")),
        }
    }

    fn parse_integer(&mut self) -> Result<i64> {
        let token = self.advance_token()?;
        match token.kind {
            TokenKind::Integer(n) => Ok(n),
            _ => Err(self.error("Expected integer")),
        }
    }

    fn parse_value(&mut self) -> Result<Value> {
        let token = self.advance_token()?;
        match token.kind {
            TokenKind::Integer(n) => Ok(Value::Int64(n)),
            TokenKind::Float(f) => Ok(Value::Float64(f)),
            TokenKind::String(s) => Ok(Value::String(s.into())),
            TokenKind::True => Ok(Value::Bool(true)),
            TokenKind::False => Ok(Value::Bool(false)),
            TokenKind::Parameter(name) => Ok(Value::String(format!("${name}").into())),
            _ => Err(self.error("Expected value")),
        }
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.current_kind() == Some(&kind)
    }

    fn check_string(&self) -> bool {
        matches!(
            self.current_kind(),
            Some(TokenKind::String(_)) | Some(TokenKind::Identifier(_))
        )
    }

    fn current_kind(&self) -> Option<&TokenKind> {
        self.tokens.get(self.position).map(|t| &t.kind)
    }

    fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.position);
        self.position += 1;
        token
    }

    fn advance_token(&mut self) -> Result<Token> {
        let token = self
            .advance()
            .cloned()
            .ok_or_else(|| self.error("Unexpected end of input"))?;
        if let TokenKind::Error(msg) = &token.kind {
            return Err(self.error(msg));
        }
        Ok(token)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token> {
        let token = self.advance_token()?;
        if std::mem::discriminant(&token.kind) == std::mem::discriminant(&kind) {
            Ok(token)
        } else {
            Err(self.error(&format!("Expected {:?}, found {:?}", kind, token.kind)))
        }
    }

    fn current_span(&self) -> grafeo_common::utils::error::SourceSpan {
        self.tokens
            .get(self.position)
            .or_else(|| self.tokens.last())
            .map_or(
                grafeo_common::utils::error::SourceSpan::new(0, 0, 1, 1),
                |t| t.span,
            )
    }

    fn error(&self, message: &str) -> Error {
        Error::Query(
            grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Syntax,
                format!("[Gremlin] {message}"),
            )
            .with_span(self.current_span())
            .with_source(self.source.to_string()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_traversal() {
        let mut parser = Parser::new("g.V()");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(matches!(stmt.source, TraversalSource::V(None)));
        assert!(stmt.steps.is_empty());
    }

    #[test]
    fn test_parse_with_steps() {
        let mut parser = Parser::new("g.V().hasLabel('Person').out('knows')");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert_eq!(stmt.steps.len(), 2);
    }

    #[test]
    fn test_parse_has_with_value() {
        let mut parser = Parser::new("g.V().has('name', 'Alix')");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert_eq!(stmt.steps.len(), 1);
        if let Step::Has(HasStep::KeyValue(key, value)) = &stmt.steps[0] {
            assert_eq!(key, "name");
            assert_eq!(*value, Value::String("Alix".into()));
        } else {
            panic!("Expected Has step with key-value");
        }
    }

    #[test]
    fn test_parse_limit() {
        let mut parser = Parser::new("g.V().limit(10)");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let Step::Limit(n) = &stmt.steps[0] {
            assert_eq!(*n, 10);
        } else {
            panic!("Expected Limit step");
        }
    }

    #[test]
    fn test_parse_values() {
        let mut parser = Parser::new("g.V().values('name', 'age')");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let Step::Values(keys) = &stmt.steps[0] {
            assert_eq!(keys.len(), 2);
            assert_eq!(keys[0], "name");
            assert_eq!(keys[1], "age");
        } else {
            panic!("Expected Values step");
        }
    }

    #[test]
    fn test_parse_has_with_direct_predicate_gt() {
        let mut parser = Parser::new("g.V().has('age', gt(28))");
        let result = parser.parse();
        assert!(result.is_ok(), "Failed to parse: {:?}", result);
        let stmt = result.unwrap();
        assert_eq!(stmt.steps.len(), 1);
        if let Step::Has(HasStep::KeyPredicate(key, Predicate::Gt(value))) = &stmt.steps[0] {
            assert_eq!(key, "age");
            assert_eq!(*value, Value::Int64(28));
        } else {
            panic!(
                "Expected Has step with gt predicate, got: {:?}",
                stmt.steps[0]
            );
        }
    }

    #[test]
    fn test_parse_has_with_direct_predicate_lt() {
        let mut parser = Parser::new("g.V().has('age', lt(50))");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let Step::Has(HasStep::KeyPredicate(key, Predicate::Lt(value))) = &stmt.steps[0] {
            assert_eq!(key, "age");
            assert_eq!(*value, Value::Int64(50));
        } else {
            panic!("Expected Has step with lt predicate");
        }
    }

    #[test]
    fn test_parse_has_with_direct_predicate_within() {
        let mut parser = Parser::new("g.V().has('status', within('active', 'pending'))");
        let result = parser.parse();
        assert!(result.is_ok());
        let stmt = result.unwrap();
        if let Step::Has(HasStep::KeyPredicate(key, Predicate::Within(values))) = &stmt.steps[0] {
            assert_eq!(key, "status");
            assert_eq!(values.len(), 2);
            assert_eq!(values[0], Value::String("active".into()));
            assert_eq!(values[1], Value::String("pending".into()));
        } else {
            panic!("Expected Has step with within predicate");
        }
    }

    // ── Traversal sources ────────────────────────────────────────

    #[test]
    fn test_parse_edge_source() {
        let stmt = Parser::new("g.E()").parse().unwrap();
        assert!(matches!(stmt.source, TraversalSource::E(None)));
    }

    #[test]
    fn test_parse_addv_source() {
        let stmt = Parser::new("g.addV('Person')").parse().unwrap();
        assert!(matches!(&stmt.source, TraversalSource::AddV(Some(l)) if l == "Person"));
    }

    #[test]
    fn test_parse_adde_source() {
        let stmt = Parser::new("g.addE('KNOWS').from('a').to('b')")
            .parse()
            .unwrap();
        assert!(matches!(&stmt.source, TraversalSource::AddE(l) if l == "KNOWS"));
        assert_eq!(stmt.steps.len(), 2);
    }

    #[test]
    fn test_parse_v_with_ids() {
        let stmt = Parser::new("g.V(1, 2, 3)").parse().unwrap();
        if let TraversalSource::V(Some(ids)) = &stmt.source {
            assert_eq!(ids.len(), 3);
        } else {
            panic!("Expected V with IDs");
        }
    }

    // ── Navigation steps ─────────────────────────────────────────

    #[test]
    fn test_parse_in_step() {
        let stmt = Parser::new("g.V().in('knows')").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::In(l) if l == &["knows"]));
    }

    #[test]
    fn test_parse_both_step() {
        let stmt = Parser::new("g.V().both('knows', 'follows')")
            .parse()
            .unwrap();
        assert!(matches!(&stmt.steps[0], Step::Both(l) if l == &["knows", "follows"]));
    }

    #[test]
    fn test_parse_oute_inv() {
        let stmt = Parser::new("g.V().outE('knows').inV()").parse().unwrap();
        assert_eq!(stmt.steps.len(), 2);
        assert!(matches!(&stmt.steps[0], Step::OutE(l) if l == &["knows"]));
        assert!(matches!(&stmt.steps[1], Step::InV));
    }

    #[test]
    fn test_parse_one_both_bothv() {
        let stmt = Parser::new("g.V().inE().bothV()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::InE(l) if l.is_empty()));
        assert!(matches!(&stmt.steps[1], Step::BothV));
    }

    #[test]
    fn test_parse_outv_otherv() {
        let stmt = Parser::new("g.E().outV()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::OutV));

        let stmt = Parser::new("g.V().outE().otherV()").parse().unwrap();
        assert!(matches!(&stmt.steps[1], Step::OtherV));
    }

    // ── Filter steps ─────────────────────────────────────────────

    #[test]
    fn test_parse_has_id() {
        let stmt = Parser::new("g.V().hasId(1, 2, 3)").parse().unwrap();
        if let Step::HasId(ids) = &stmt.steps[0] {
            assert_eq!(ids.len(), 3);
        } else {
            panic!("Expected HasId step");
        }
    }

    #[test]
    fn test_parse_has_not() {
        let stmt = Parser::new("g.V().hasNot('age')").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::HasNot(k) if k == "age"));
    }

    #[test]
    fn test_parse_dedup() {
        let stmt = Parser::new("g.V().dedup()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Dedup(k) if k.is_empty()));
    }

    #[test]
    fn test_parse_skip() {
        let stmt = Parser::new("g.V().skip(5)").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Skip(5)));
    }

    #[test]
    fn test_parse_range() {
        let stmt = Parser::new("g.V().range(2, 5)").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Range(2, 5)));
    }

    // ── Map steps ────────────────────────────────────────────────

    #[test]
    fn test_parse_value_map() {
        let stmt = Parser::new("g.V().valueMap('name', 'age')")
            .parse()
            .unwrap();
        assert!(matches!(&stmt.steps[0], Step::ValueMap(k) if k == &["name", "age"]));
    }

    #[test]
    fn test_parse_element_map() {
        let stmt = Parser::new("g.V().elementMap()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::ElementMap(k) if k.is_empty()));
    }

    #[test]
    fn test_parse_id_and_label() {
        let stmt = Parser::new("g.V().id()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Id));

        let stmt = Parser::new("g.V().label()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Label));
    }

    #[test]
    fn test_parse_count() {
        let stmt = Parser::new("g.V().count()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Count));
    }

    #[test]
    fn test_parse_aggregate_functions() {
        assert!(matches!(
            &Parser::new("g.V().values('x').sum()")
                .parse()
                .unwrap()
                .steps[1],
            Step::Sum
        ));
        assert!(matches!(
            &Parser::new("g.V().values('x').mean()")
                .parse()
                .unwrap()
                .steps[1],
            Step::Mean
        ));
        assert!(matches!(
            &Parser::new("g.V().values('x').min()")
                .parse()
                .unwrap()
                .steps[1],
            Step::Min
        ));
        assert!(matches!(
            &Parser::new("g.V().values('x').max()")
                .parse()
                .unwrap()
                .steps[1],
            Step::Max
        ));
    }

    #[test]
    fn test_parse_fold_unfold() {
        let stmt = Parser::new("g.V().fold().unfold()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Fold));
        assert!(matches!(&stmt.steps[1], Step::Unfold));
    }

    #[test]
    fn test_parse_constant() {
        let stmt = Parser::new("g.V().constant('default')").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Constant(Value::String(_))));
    }

    #[test]
    fn test_parse_path() {
        let stmt = Parser::new("g.V().out().path()").parse().unwrap();
        assert!(matches!(&stmt.steps[1], Step::Path));
    }

    #[test]
    fn test_parse_select() {
        let stmt = Parser::new("g.V().as('a').out().as('b').select('a', 'b')")
            .parse()
            .unwrap();
        assert!(matches!(&stmt.steps[3], Step::Select(k) if k == &["a", "b"]));
    }

    #[test]
    fn test_parse_project() {
        let stmt = Parser::new("g.V().project('name', 'age')").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Project(k) if k == &["name", "age"]));
    }

    #[test]
    fn test_parse_order() {
        let stmt = Parser::new("g.V().order()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Order(_)));
    }

    #[test]
    fn test_parse_group_count() {
        let stmt = Parser::new("g.V().groupCount()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::GroupCount(_)));
    }

    #[test]
    fn test_parse_group() {
        let stmt = Parser::new("g.V().group()").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Group(_)));
    }

    #[test]
    fn test_parse_properties() {
        let stmt = Parser::new("g.V().properties('name')").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Properties(k) if k == &["name"]));
    }

    // ── Side effect steps ────────────────────────────────────────

    #[test]
    fn test_parse_as_step() {
        let stmt = Parser::new("g.V().as('a')").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::As(l) if l == "a"));
    }

    #[test]
    fn test_parse_aggregate() {
        let stmt = Parser::new("g.V().aggregate('x')").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Aggregate(l) if l == "x"));
    }

    #[test]
    fn test_parse_store() {
        let stmt = Parser::new("g.V().store('x')").parse().unwrap();
        assert!(matches!(&stmt.steps[0], Step::Store(l) if l == "x"));
    }

    #[test]
    fn test_parse_property() {
        let stmt = Parser::new("g.addV('Person').property('name', 'Alix')")
            .parse()
            .unwrap();
        if let Step::Property(prop) = &stmt.steps[0] {
            assert_eq!(prop.key, "name");
            assert_eq!(prop.value, Value::String("Alix".into()));
        } else {
            panic!("Expected Property step");
        }
    }

    #[test]
    fn test_parse_drop() {
        let stmt = Parser::new("g.V().hasLabel('Temp').drop()")
            .parse()
            .unwrap();
        assert!(matches!(&stmt.steps[1], Step::Drop));
    }

    // ── Predicate variants ───────────────────────────────────────

    #[test]
    fn test_parse_neq_predicate() {
        let stmt = Parser::new("g.V().has('status', neq('inactive'))")
            .parse()
            .unwrap();
        assert!(matches!(
            &stmt.steps[0],
            Step::Has(HasStep::KeyPredicate(_, Predicate::Neq(_)))
        ));
    }

    #[test]
    fn test_parse_lte_gte_predicates() {
        let stmt = Parser::new("g.V().has('age', lte(50))").parse().unwrap();
        assert!(matches!(
            &stmt.steps[0],
            Step::Has(HasStep::KeyPredicate(_, Predicate::Lte(_)))
        ));

        let stmt = Parser::new("g.V().has('age', gte(18))").parse().unwrap();
        assert!(matches!(
            &stmt.steps[0],
            Step::Has(HasStep::KeyPredicate(_, Predicate::Gte(_)))
        ));
    }

    #[test]
    fn test_parse_between_predicate() {
        let stmt = Parser::new("g.V().has('age', between(18, 65))")
            .parse()
            .unwrap();
        if let Step::Has(HasStep::KeyPredicate(_, Predicate::Between(lo, hi))) = &stmt.steps[0] {
            assert_eq!(*lo, Value::Int64(18));
            assert_eq!(*hi, Value::Int64(65));
        } else {
            panic!("Expected between predicate");
        }
    }

    #[test]
    fn test_parse_without_predicate() {
        let stmt = Parser::new("g.V().has('status', without('deleted', 'banned'))")
            .parse()
            .unwrap();
        assert!(matches!(
            &stmt.steps[0],
            Step::Has(HasStep::KeyPredicate(_, Predicate::Without(v))) if v.len() == 2
        ));
    }

    #[test]
    fn test_parse_string_predicates() {
        let stmt = Parser::new("g.V().has('name', containing('ali'))")
            .parse()
            .unwrap();
        assert!(matches!(
            &stmt.steps[0],
            Step::Has(HasStep::KeyPredicate(_, Predicate::Containing(_)))
        ));

        let stmt = Parser::new("g.V().has('name', startingWith('A'))")
            .parse()
            .unwrap();
        assert!(matches!(
            &stmt.steps[0],
            Step::Has(HasStep::KeyPredicate(_, Predicate::StartingWith(_)))
        ));

        let stmt = Parser::new("g.V().has('name', endingWith('son'))")
            .parse()
            .unwrap();
        assert!(matches!(
            &stmt.steps[0],
            Step::Has(HasStep::KeyPredicate(_, Predicate::EndingWith(_)))
        ));
    }

    // ── Edge creation (from/to) ──────────────────────────────────

    #[test]
    fn test_parse_from_to_labels() {
        let stmt = Parser::new("g.addE('KNOWS').from('a').to('b')")
            .parse()
            .unwrap();
        assert!(matches!(&stmt.steps[0], Step::From(FromTo::Label(l)) if l == "a"));
        assert!(matches!(&stmt.steps[1], Step::To(FromTo::Label(l)) if l == "b"));
    }

    // ── has(label, key, value) ───────────────────────────────────

    #[test]
    fn test_parse_has_label_key_value() {
        let stmt = Parser::new("g.V().has('Person', 'name', 'Alix')")
            .parse()
            .unwrap();
        if let Step::Has(HasStep::LabelKeyValue(label, key, val)) = &stmt.steps[0] {
            assert_eq!(label, "Person");
            assert_eq!(key, "name");
            assert_eq!(*val, Value::String("Alix".into()));
        } else {
            panic!("Expected LabelKeyValue has step");
        }
    }

    // ── Complex multi-step traversals ────────────────────────────

    #[test]
    fn test_parse_complex_traversal() {
        let stmt = Parser::new(
            "g.V().hasLabel('Person').has('age', gt(25)).out('knows').dedup().values('name').limit(10)",
        )
        .parse()
        .unwrap();
        assert_eq!(stmt.steps.len(), 6);
        assert!(matches!(&stmt.steps[0], Step::HasLabel(_)));
        assert!(matches!(&stmt.steps[1], Step::Has(_)));
        assert!(matches!(&stmt.steps[2], Step::Out(_)));
        assert!(matches!(&stmt.steps[3], Step::Dedup(_)));
        assert!(matches!(&stmt.steps[4], Step::Values(_)));
        assert!(matches!(&stmt.steps[5], Step::Limit(10)));
    }

    #[test]
    fn test_parse_vertex_creation_with_properties() {
        let stmt = Parser::new("g.addV('Person').property('name', 'Alix').as('a')")
            .parse()
            .unwrap();
        assert!(matches!(&stmt.source, TraversalSource::AddV(Some(l)) if l == "Person"));
        assert_eq!(stmt.steps.len(), 2);
        assert!(matches!(&stmt.steps[0], Step::Property(_)));
        assert!(matches!(&stmt.steps[1], Step::As(_)));
    }

    // --- Error tests ---

    #[test]
    fn test_parse_empty_input_fails() {
        let result = Parser::new("").parse();
        assert!(result.is_err(), "Empty input should fail");
    }

    #[test]
    fn test_parse_missing_g_prefix_fails() {
        let result = Parser::new("V()").parse();
        assert!(result.is_err(), "Missing 'g.' prefix should fail");
    }

    #[test]
    fn test_parse_unclosed_paren_fails() {
        let result = Parser::new("g.V(").parse();
        assert!(result.is_err(), "Unclosed parenthesis should fail");
    }

    #[test]
    fn test_parse_invalid_step_name_fails() {
        let result = Parser::new("g.V().notAStep()").parse();
        assert!(result.is_err(), "Unknown step name should fail");
    }

    #[test]
    fn test_parse_garbage_input_fails() {
        let result = Parser::new("@#$%^&*").parse();
        assert!(result.is_err(), "Garbage input should fail");
    }

    #[test]
    fn test_parse_repeat_times() {
        let stmt = Parser::new("g.V().repeat(out('KNOWS')).times(3)")
            .parse()
            .unwrap();
        assert_eq!(stmt.steps.len(), 1);
        if let Step::Repeat(ref r) = stmt.steps[0] {
            assert_eq!(r.body.len(), 1);
            assert!(matches!(r.termination, Some(RepeatTermination::Times(3))));
            assert!(!r.emit);
        } else {
            panic!("Expected Repeat step");
        }
    }

    #[test]
    fn test_parse_repeat_until() {
        let stmt = Parser::new("g.V().repeat(out()).until(has('name', 'Gus'))")
            .parse()
            .unwrap();
        if let Step::Repeat(ref r) = stmt.steps[0] {
            assert!(matches!(r.termination, Some(RepeatTermination::Until(_))));
        } else {
            panic!("Expected Repeat step");
        }
    }

    #[test]
    fn test_parse_repeat_emit() {
        let stmt = Parser::new("g.V().repeat(out()).emit().times(2)")
            .parse()
            .unwrap();
        if let Step::Repeat(ref r) = stmt.steps[0] {
            assert!(r.emit);
            assert!(matches!(r.termination, Some(RepeatTermination::Times(2))));
        } else {
            panic!("Expected Repeat step");
        }
    }

    #[test]
    fn test_parse_repeat_no_termination() {
        let stmt = Parser::new("g.V().repeat(out()).emit()").parse().unwrap();
        if let Step::Repeat(ref r) = stmt.steps[0] {
            assert!(r.emit);
            assert!(r.termination.is_none());
        } else {
            panic!("Expected Repeat step");
        }
    }

    #[test]
    fn test_parse_repeat_negative_times_fails() {
        let result = Parser::new("g.V().repeat(out()).times(-1)").parse();
        assert!(result.is_err(), "Negative times should fail");
    }

    #[test]
    fn test_parse_repeat_in_union() {
        let stmt = Parser::new("g.V().union(repeat(out()).times(2), repeat(in()).times(1))")
            .parse()
            .unwrap();
        if let Step::Union(ref branches) = stmt.steps[0] {
            assert_eq!(branches.len(), 2);
            if let Step::Repeat(ref r) = branches[0][0] {
                assert!(matches!(r.termination, Some(RepeatTermination::Times(2))));
            } else {
                panic!("Expected Repeat in first branch");
            }
        } else {
            panic!("Expected Union step");
        }
    }

    // ==================== Recursion depth limit tests ====================

    #[test]
    fn test_deeply_nested_steps_errors_not_stack_overflow() {
        // Gremlin nesting happens via nested traversals in steps like not(), and(), or()
        let prefix = "not(".repeat(300);
        let suffix = ")".repeat(300);
        let query = format!("g.V().{prefix}has('name', 'x'){suffix}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("nesting depth"),
            "Expected nesting depth error, got: {err}"
        );
    }

    #[test]
    fn test_moderate_step_nesting_succeeds() {
        // 10 levels of not() should succeed
        let prefix = "not(".repeat(10);
        let suffix = ")".repeat(10);
        let query = format!("g.V().{prefix}has('name', 'x'){suffix}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(
            result.is_ok(),
            "10 levels of step nesting should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_deeply_nested_and_steps() {
        let prefix = "and(".repeat(300);
        let suffix = ")".repeat(300);
        let query = format!("g.V().{prefix}has('x', 1){suffix}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(
            result.is_err(),
            "300 levels of and() should error, not stack overflow"
        );
    }

    #[test]
    fn test_deeply_nested_or_steps() {
        let prefix = "or(".repeat(300);
        let suffix = ")".repeat(300);
        let query = format!("g.V().{prefix}has('x', 1){suffix}");
        let mut parser = Parser::new(&query);
        let result = parser.parse();
        assert!(
            result.is_err(),
            "300 levels of or() should error, not stack overflow"
        );
    }
}
