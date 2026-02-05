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
    Id,
    Label,
    Key,
    Value,
}

/// Sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Asc,
    Desc,
    Shuffle,
}

/// Order modifier.
#[derive(Debug, Clone)]
pub struct OrderModifier {
    pub by: ByModifier,
    pub order: SortOrder,
}

/// Choose clause for branching.
#[derive(Debug, Clone)]
pub struct ChooseClause {
    pub condition: ChooseCondition,
    pub true_branch: Vec<Step>,
    pub false_branch: Option<Vec<Step>>,
}

/// Choose condition.
#[derive(Debug, Clone)]
pub enum ChooseCondition {
    Predicate(Box<Predicate>),
    Traversal(Vec<Step>),
    HasKey(String),
}

/// Property step for setting properties.
#[derive(Debug, Clone)]
pub struct PropertyStep {
    pub cardinality: Option<Cardinality>,
    pub key: String,
    pub value: Value,
}

/// Property cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    Single,
    List,
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
