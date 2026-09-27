//! Gremlin Abstract Syntax Tree.
//!
//! Represents the traversal-based structure of Gremlin queries.

use grafeo_common::types::Value;

/// A complete Gremlin statement.
#[derive(Debug, Clone)]
pub struct Statement {
    /// The source (g.V(), g.E(), etc.).
    pub source: TraversalSource,
    /// The traversal steps.
    pub steps: Vec<Step>,
}

/// Traversal source (starting point).
#[derive(Debug, Clone)]
pub enum TraversalSource {
    /// g.V() - start from vertices
    V(Option<Vec<Value>>),
    /// g.E() - start from edges
    E(Option<Vec<Value>>),
    /// g.addV() - add a vertex
    AddV(Option<String>),
    /// g.addE() - add an edge
    AddE(String),
}

/// A traversal step.
#[derive(Debug, Clone)]
pub enum Step {
    // === Navigation Steps ===
    /// .out(labels...) - traverse outgoing edges
    Out(Vec<String>),
    /// .in(labels...) - traverse incoming edges
    In(Vec<String>),
    /// .both(labels...) - traverse both directions
    Both(Vec<String>),
    /// .outE(labels...) - traverse to outgoing edges
    OutE(Vec<String>),
    /// .inE(labels...) - traverse to incoming edges
    InE(Vec<String>),
    /// .bothE(labels...) - traverse to edges in both directions
    BothE(Vec<String>),
    /// .outV() - traverse to outgoing vertex of edge
    OutV,
    /// .inV() - traverse to incoming vertex of edge
    InV,
    /// .bothV() - traverse to both vertices of edge
    BothV,
    /// .otherV() - traverse to the other vertex
    OtherV,

    /// .V() - mid-traversal vertex scan (restarts from all vertices)
    MidV(Option<Vec<Value>>),

    // === Filter Steps ===
    /// .has(key, value) or .has(label, key, value)
    Has(HasStep),
    /// .hasLabel(labels...)
    HasLabel(Vec<String>),
    /// .hasId(ids...)
    HasId(Vec<Value>),
    /// .hasNot(key)
    HasNot(String),
    /// .filter(predicate)
    Filter(Box<Predicate>),
    /// .where(traversal or predicate)
    Where(WhereClause),
    /// .and(traversals...)
    And(Vec<Vec<Step>>),
    /// .or(traversals...)
    Or(Vec<Vec<Step>>),
    /// .not(traversal)
    Not(Vec<Step>),
    /// .dedup(keys...)
    Dedup(Vec<String>),
    /// .limit(n)
    Limit(usize),
    /// .skip(n)
    Skip(usize),
    /// .range(start, end)
    Range(usize, usize),

    // === Map Steps ===
    /// .values(keys...)
    Values(Vec<String>),
    /// .valueMap(keys...)
    ValueMap(Vec<String>),
    /// .elementMap(keys...)
    ElementMap(Vec<String>),
    /// .id()
    Id,
    /// .label()
    Label,
    /// .properties(keys...)
    Properties(Vec<String>),
    /// .constant(value)
    Constant(Value),
    /// .count()
    Count,
    /// .sum()
    Sum,
    /// .mean() / .avg()
    Mean,
    /// .min()
    Min,
    /// .max()
    Max,
    /// .fold()
    Fold,
    /// .unfold()
    Unfold,
    /// .group()
    Group(Option<GroupModifiers>),
    /// .groupCount()
    GroupCount(Option<String>),
    /// .path()
    Path,
    /// .select(keys...)
    Select(Vec<String>),
    /// .project(keys...)
    Project(Vec<String>),
    /// .by(key or traversal)
    By(ByModifier),
    /// .order()
    Order(Vec<OrderModifier>),
    /// .coalesce(traversals...)
    Coalesce(Vec<Vec<Step>>),
    /// .optional(traversal)
    Optional(Vec<Step>),
    /// .union(traversals...)
    Union(Vec<Vec<Step>>),
    /// .choose(predicate, true_branch, false_branch)
    Choose(ChooseClause),

    // === Looping Steps ===
    /// .repeat(traversal).times(n) or .repeat(traversal).until(predicate)
    Repeat(RepeatStep),

    // === Side Effect Steps ===
    /// .as(label)
    As(String),
    /// .sideEffect(traversal)
    SideEffect(Vec<Step>),
    /// .aggregate(label)
    Aggregate(String),
    /// .store(label)
    Store(String),
    /// .property(key, value) or .property(cardinality, key, value)
    Property(PropertyStep),
    /// .drop()
    Drop,

    // === Vertex/Edge Creation ===
    /// .from(vertex or label)
    From(FromTo),
    /// .to(vertex or label)
    To(FromTo),
    /// .addV(label)
    AddV(Option<String>),
    /// .addE(label)
    AddE(String),
}

/// Has step variants.
#[derive(Debug, Clone)]
pub enum HasStep {
    /// .has(key)
    Key(String),
    /// .has(key, value)
    KeyValue(String, Value),
    /// .has(key, predicate)
    KeyPredicate(String, Predicate),
    /// .has(label, key, value)
    LabelKeyValue(String, String, Value),
}

/// A predicate for filtering.
#[derive(Debug, Clone)]
pub enum Predicate {
    /// P.eq(value)
    Eq(Value),
    /// P.neq(value)
    Neq(Value),
    /// P.lt(value)
    Lt(Value),
    /// P.lte(value)
    Lte(Value),
    /// P.gt(value)
    Gt(Value),
    /// P.gte(value)
    Gte(Value),
    /// P.within(values...)
    Within(Vec<Value>),
    /// P.without(values...)
    Without(Vec<Value>),
    /// P.between(start, end)
    Between(Value, Value),
    /// P.inside(start, end)
    Inside(Value, Value),
    /// P.outside(start, end)
    Outside(Value, Value),
    /// P.containing(substring)
    Containing(String),
    /// P.startingWith(prefix)
    StartingWith(String),
    /// P.endingWith(suffix)
    EndingWith(String),
    /// P.regex(pattern)
    Regex(String),
    /// P.notRegex(pattern)
    NotRegex(String),
    /// P.notContaining(substring)
    NotContaining(String),
    /// P.notStartingWith(prefix)
    NotStartingWith(String),
    /// P.notEndingWith(suffix)
    NotEndingWith(String),
    /// P.and(predicates...)
    And(Vec<Predicate>),
    /// P.or(predicates...)
    Or(Vec<Predicate>),
    /// P.not(predicate)
    Not(Box<Predicate>),
}

/// Where clause variants.
#[derive(Debug, Clone)]
pub enum WhereClause {
    /// .where(P.eq(label))
    Predicate(String, Predicate),
    /// .where(traversal)
    Traversal(Vec<Step>),
}

/// Group modifiers.
#[derive(Debug, Clone)]
pub struct GroupModifiers {
    /// Key selector
    pub key: Option<ByModifier>,
    /// Value selector
    pub value: Option<ByModifier>,
}

/// By modifier for ordering, grouping, etc.
#[derive(Debug, Clone)]
pub enum ByModifier {
    /// .by() - use element itself
    Identity,
    /// .by(key)
    Key(String),
    /// .by(key, order) - e.g., .by('age', asc)
    KeyWithOrder(String, SortOrder),
    /// .by(traversal)
    Traversal(Vec<Step>),
    /// .by(T.id), .by(T.label)
    Token(TokenType),
    /// .by(Order.asc), .by(Order.desc)
    Order(SortOrder),
}

/// Token type for by() modifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// `T.id`: the element identifier.
    Id,
    /// `T.label`: the element label.
    Label,
    /// `T.key`: the property key.
    Key,
    /// `T.value`: the property value.
    Value,
}

/// Sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Ascending order.
    Asc,
    /// Descending order.
    Desc,
    /// Random order.
    Shuffle,
}

/// Order modifier.
#[derive(Debug, Clone)]
pub struct OrderModifier {
    /// The key or traversal to order by.
    pub by: ByModifier,
    /// The sort direction.
    pub order: SortOrder,
}

/// Choose clause for branching.
#[derive(Debug, Clone)]
pub struct ChooseClause {
    /// The branching condition.
    pub condition: ChooseCondition,
    /// Steps to execute when the condition is true.
    pub true_branch: Vec<Step>,
    /// Steps to execute when the condition is false.
    pub false_branch: Option<Vec<Step>>,
}

/// Choose condition.
#[derive(Debug, Clone)]
pub enum ChooseCondition {
    /// A predicate condition.
    Predicate(Box<Predicate>),
    /// A traversal condition (truthy if non-empty).
    Traversal(Vec<Step>),
    /// A `has(key)` existence check.
    HasKey(String),
}

/// Repeat loop configuration.
///
/// Combines the repeat body with its termination condition (times or until)
/// and optional emit behavior.
#[derive(Debug, Clone)]
pub struct RepeatStep {
    /// The traversal steps to repeat.
    pub body: Vec<Step>,
    /// How to terminate the loop. `None` means traverse until exhausted.
    pub termination: Option<RepeatTermination>,
    /// Whether to emit intermediate results.
    pub emit: bool,
}

/// How a repeat loop terminates.
#[derive(Debug, Clone)]
pub enum RepeatTermination {
    /// Repeat exactly N times.
    Times(usize),
    /// Repeat until a predicate is satisfied on the traverser.
    Until(Vec<Step>),
}

/// Property step for setting properties.
#[derive(Debug, Clone)]
pub struct PropertyStep {
    /// Optional cardinality (`single`, `list`, or `set`).
    pub cardinality: Option<Cardinality>,
    /// The property key.
    pub key: String,
    /// The property value.
    pub value: Value,
}

/// Property cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    /// At most one value per key (default).
    Single,
    /// Multiple values per key (ordered).
    List,
    /// Multiple unique values per key.
    Set,
}

/// From/To target for edge creation.
#[derive(Debug, Clone)]
pub enum FromTo {
    /// Reference to a labeled step
    Label(String),
    /// A sub-traversal
    Traversal(Vec<Step>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statement_creation() {
        let stmt = Statement {
            source: TraversalSource::V(None),
            steps: vec![
                Step::HasLabel(vec!["Person".to_string()]),
                Step::Out(vec!["knows".to_string()]),
                Step::Values(vec!["name".to_string()]),
            ],
        };

        assert!(matches!(stmt.source, TraversalSource::V(None)));
        assert_eq!(stmt.steps.len(), 3);
    }

    #[test]
    fn test_predicate_creation() {
        let pred = Predicate::And(vec![
            Predicate::Gt(Value::Int64(20)),
            Predicate::Lt(Value::Int64(40)),
        ]);

        if let Predicate::And(preds) = pred {
            assert_eq!(preds.len(), 2);
        } else {
            panic!("Expected And predicate");
        }
    }
}
