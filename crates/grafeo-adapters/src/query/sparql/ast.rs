//! SPARQL Abstract Syntax Tree.
//!
//! Defines the AST types for SPARQL 1.1 Query Language.

use std::fmt;

/// A complete SPARQL query.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// Base IRI declaration.
    pub base: Option<Iri>,
    /// Prefix declarations.
    pub prefixes: Vec<PrefixDeclaration>,
    /// The query form.
    pub query_form: QueryForm,
}

/// A PREFIX declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefixDeclaration {
    /// The prefix (empty string for default prefix).
    pub prefix: String,
    /// The namespace IRI.
    pub namespace: Iri,
}

/// An IRI (Internationalized Resource Identifier).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Iri(pub String);

impl Iri {
    /// Creates a new IRI from a string.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the IRI as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Iri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.0)
    }
}

/// The form of a SPARQL query.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryForm {
    /// SELECT query.
    Select(SelectQuery),
    /// CONSTRUCT query.
    Construct(ConstructQuery),
    /// ASK query.
    Ask(AskQuery),
    /// DESCRIBE query.
    Describe(DescribeQuery),
    /// Update operation (SPARQL Update).
    Update(UpdateOperation),
}

/// A SPARQL Update operation.
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateOperation {
    /// INSERT DATA { triples }.
    InsertData {
        /// The triples to insert.
        data: Vec<QuadPattern>,
    },

    /// DELETE DATA { triples }.
    DeleteData {
        /// The triples to delete.
        data: Vec<QuadPattern>,
    },

    /// DELETE WHERE { pattern }.
    DeleteWhere {
        /// The pattern to match and delete.
        pattern: GraphPattern,
    },

    /// DELETE { template } INSERT { template } WHERE { pattern }
    /// (also handles DELETE-only and INSERT-only variants).
    Modify {
        /// Optional WITH graph.
        with_graph: Option<Iri>,
        /// Optional delete template.
        delete_template: Option<Vec<QuadPattern>>,
        /// Optional insert template.
        insert_template: Option<Vec<QuadPattern>>,
        /// Optional USING clauses.
        using_clauses: Vec<UsingClause>,
        /// The WHERE pattern.
        where_clause: GraphPattern,
    },

    /// LOAD url INTO graph.
    Load {
        /// Whether SILENT is specified.
        silent: bool,
        /// Source URL.
        source: Iri,
        /// Optional destination graph.
        destination: Option<Iri>,
    },

    /// CLEAR graph.
    Clear {
        /// Whether SILENT is specified.
        silent: bool,
        /// The graph target.
        target: GraphTarget,
    },

    /// DROP graph.
    Drop {
        /// Whether SILENT is specified.
        silent: bool,
        /// The graph target.
        target: GraphTarget,
    },

    /// CREATE GRAPH uri.
    Create {
        /// Whether SILENT is specified.
        silent: bool,
        /// The graph to create.
        graph: Iri,
    },

    /// COPY source TO destination.
    Copy {
        /// Whether SILENT is specified.
        silent: bool,
        /// Source graph.
        source: GraphTarget,
        /// Destination graph.
        destination: GraphTarget,
    },

    /// MOVE source TO destination.
    Move {
        /// Whether SILENT is specified.
        silent: bool,
        /// Source graph.
        source: GraphTarget,
        /// Destination graph.
        destination: GraphTarget,
    },

    /// ADD source TO destination.
    Add {
        /// Whether SILENT is specified.
        silent: bool,
        /// Source graph.
        source: GraphTarget,
        /// Destination graph.
        destination: GraphTarget,
    },
}

/// A quad pattern (triple with optional graph).
#[derive(Debug, Clone, PartialEq)]
pub struct QuadPattern {
    /// Optional graph.
    pub graph: Option<VariableOrIri>,
    /// The triple pattern.
    pub triple: TriplePattern,
}

/// USING clause for updates.
#[derive(Debug, Clone, PartialEq)]
pub enum UsingClause {
    /// `USING <iri>`.
    Default(Iri),
    /// `USING NAMED <iri>`.
    Named(Iri),
}

/// Target for graph management operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphTarget {
    /// DEFAULT graph.
    Default,
    /// NAMED graph by IRI.
    Named(Iri),
    /// ALL graphs.
    All,
}

/// A SELECT query.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectQuery {
    /// DISTINCT or REDUCED modifier.
    pub modifier: SelectModifier,
    /// The projection (variables to return).
    pub projection: Projection,
    /// Dataset clause (FROM / FROM NAMED).
    pub dataset: Option<DatasetClause>,
    /// The WHERE clause pattern.
    pub where_clause: GraphPattern,
    /// Solution modifiers (GROUP BY, ORDER BY, etc.).
    pub solution_modifiers: SolutionModifiers,
}

/// A CONSTRUCT query.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstructQuery {
    /// The template triples.
    pub template: Vec<TriplePattern>,
    /// Dataset clause.
    pub dataset: Option<DatasetClause>,
    /// The WHERE clause pattern.
    pub where_clause: GraphPattern,
    /// Solution modifiers.
    pub solution_modifiers: SolutionModifiers,
}

/// An ASK query.
#[derive(Debug, Clone, PartialEq)]
pub struct AskQuery {
    /// Dataset clause.
    pub dataset: Option<DatasetClause>,
    /// The WHERE clause pattern.
    pub where_clause: GraphPattern,
}

/// A DESCRIBE query.
#[derive(Debug, Clone, PartialEq)]
pub struct DescribeQuery {
    /// Resources to describe (variables or IRIs).
    pub resources: Vec<VariableOrIri>,
    /// Dataset clause.
    pub dataset: Option<DatasetClause>,
    /// Optional WHERE clause.
    pub where_clause: Option<GraphPattern>,
}

/// SELECT modifier (DISTINCT/REDUCED).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectModifier {
    /// No modifier.
    #[default]
    None,
    /// DISTINCT modifier.
    Distinct,
    /// REDUCED modifier.
    Reduced,
}

/// Projection in a SELECT query.
#[derive(Debug, Clone, PartialEq)]
pub enum Projection {
    /// SELECT * - all variables.
    Wildcard,
    /// SELECT ?x ?y (expr AS ?z) - specific variables/expressions.
    Variables(Vec<ProjectionVariable>),
}

/// A single projected variable or expression.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionVariable {
    /// The expression (can be just a variable).
    pub expression: Expression,
    /// Optional alias (AS ?name).
    pub alias: Option<String>,
}

/// Dataset clause (FROM / FROM NAMED).
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetClause {
    /// Default graph IRIs.
    pub default_graphs: Vec<Iri>,
    /// Named graph IRIs.
    pub named_graphs: Vec<Iri>,
}

/// A graph pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphPattern {
    /// Basic graph pattern (sequence of triple patterns).
    Basic(Vec<TriplePattern>),

    /// Group graph pattern { ... }.
    Group(Vec<GraphPattern>),

    /// OPTIONAL { ... }.
    Optional(Box<GraphPattern>),

    /// UNION of patterns.
    Union(Vec<GraphPattern>),

    /// GRAPH ?g { ... } or GRAPH `<iri>` { ... }.
    NamedGraph {
        /// The graph (variable or IRI).
        graph: VariableOrIri,
        /// The inner pattern.
        pattern: Box<GraphPattern>,
    },

    /// MINUS { ... }.
    Minus(Box<GraphPattern>),

    /// FILTER expression.
    Filter(Expression),

    /// BIND (expr AS ?var).
    Bind {
        /// The expression to bind.
        expression: Expression,
        /// The variable to bind to.
        variable: String,
    },

    /// VALUES inline data.
    InlineData(InlineDataClause),

    /// Subquery (nested SELECT).
    SubSelect(Box<SelectQuery>),

    /// SERVICE endpoint { ... }.
    Service {
        /// Whether SILENT is specified.
        silent: bool,
        /// The service endpoint.
        endpoint: VariableOrIri,
        /// The pattern to execute remotely.
        pattern: Box<GraphPattern>,
    },
}

/// Inline data (VALUES clause).
#[derive(Debug, Clone, PartialEq)]
pub struct InlineDataClause {
    /// Variables to bind.
    pub variables: Vec<String>,
    /// Data rows.
    pub values: Vec<Vec<Option<DataValue>>>,
}

/// A value in inline data.
#[derive(Debug, Clone, PartialEq)]
pub enum DataValue {
    /// IRI value.
    Iri(Iri),
    /// Literal value.
    Literal(Literal),
}

/// A triple pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct TriplePattern {
    /// The subject.
    pub subject: TripleTerm,
    /// The predicate (can be a property path).
    pub predicate: PropertyPath,
    /// The object.
    pub object: TripleTerm,
}

/// A term in a triple pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum TripleTerm {
    /// Variable (?x).
    Variable(String),
    /// IRI.
    Iri(Iri),
    /// Blank node (_:label or []).
    BlankNode(BlankNode),
    /// Literal value.
    Literal(Literal),
}

/// A blank node.
#[derive(Debug, Clone, PartialEq)]
pub enum BlankNode {
    /// Labeled blank node (_:label).
    Labeled(String),
    /// Anonymous blank node with property list.
    Anonymous(Vec<(PropertyPath, TripleTerm)>),
}

/// A property path expression.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyPath {
    /// Simple predicate (IRI).
    Predicate(Iri),

    /// Variable predicate (?p).
    Variable(String),

    /// 'a' shorthand for rdf:type.
    RdfType,

    /// Inverse path (^path).
    Inverse(Box<PropertyPath>),

    /// Sequence path (path1/path2).
    Sequence(Vec<PropertyPath>),

    /// Alternative path (path1|path2).
    Alternative(Vec<PropertyPath>),

    /// Zero or more (*).
    ZeroOrMore(Box<PropertyPath>),

    /// One or more (+).
    OneOrMore(Box<PropertyPath>),

    /// Zero or one (?).
    ZeroOrOne(Box<PropertyPath>),

    /// Negated property set (!(iri1|iri2)).
    Negation(Vec<Iri>),
}

/// A literal value.
#[derive(Debug, Clone, PartialEq)]
pub struct Literal {
    /// The lexical value.
    pub value: String,
    /// Optional datatype IRI.
    pub datatype: Option<Iri>,
    /// Optional language tag.
    pub language: Option<String>,
}

impl Literal {
    /// Creates a simple string literal.
    pub fn string(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            datatype: None,
            language: None,
        }
    }

    /// Creates a typed literal.
    pub fn typed(value: impl Into<String>, datatype: Iri) -> Self {
        Self {
            value: value.into(),
            datatype: Some(datatype),
            language: None,
        }
    }

    /// Creates a language-tagged literal.
    pub fn with_language(value: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            datatype: None,
            language: Some(language.into()),
        }
    }
}

/// Either a variable or an IRI.
#[derive(Debug, Clone, PartialEq)]
pub enum VariableOrIri {
    /// Variable.
    Variable(String),
    /// IRI.
    Iri(Iri),
}

/// Solution modifiers (GROUP BY, HAVING, ORDER BY, LIMIT, OFFSET).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SolutionModifiers {
    /// GROUP BY clause.
    pub group_by: Option<Vec<GroupCondition>>,
    /// HAVING clause.
    pub having: Option<Expression>,
    /// ORDER BY clause.
    pub order_by: Option<Vec<OrderCondition>>,
    /// LIMIT clause.
    pub limit: Option<u64>,
    /// OFFSET clause.
    pub offset: Option<u64>,
}

/// A GROUP BY condition.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupCondition {
    /// Group by variable.
    Variable(String),
    /// Group by expression with optional alias.
    Expression {
        /// The expression.
        expression: Expression,
        /// Optional alias.
        alias: Option<String>,
    },
    /// Group by function call.
    BuiltInCall(Expression),
}

/// An ORDER BY condition.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderCondition {
    /// The expression to order by.
    pub expression: Expression,
    /// The sort direction.
    pub direction: SortDirection,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortDirection {
    /// Ascending order.
    #[default]
    Ascending,
    /// Descending order.
    Descending,
}

/// A SPARQL expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Variable reference.
    Variable(String),

    /// IRI reference.
    Iri(Iri),

    /// Literal value.
    Literal(Literal),

    /// Binary operation.
    Binary {
        /// Left operand.
        left: Box<Expression>,
        /// Operator.
        operator: BinaryOperator,
        /// Right operand.
        right: Box<Expression>,
    },

    /// Unary operation.
    Unary {
        /// Operator.
        operator: UnaryOperator,
        /// Operand.
        operand: Box<Expression>,
    },

    /// Function call.
    FunctionCall {
        /// Function name (IRI or built-in name).
        function: FunctionName,
        /// Arguments.
        arguments: Vec<Expression>,
    },

    /// BOUND(?var).
    Bound(String),

    /// IF(cond, then, else).
    Conditional {
        /// Condition.
        condition: Box<Expression>,
        /// Then expression.
        then_expression: Box<Expression>,
        /// Else expression.
        else_expression: Box<Expression>,
    },

    /// COALESCE(expr, ...).
    Coalesce(Vec<Expression>),

    /// EXISTS { pattern }.
    Exists(Box<GraphPattern>),

    /// NOT EXISTS { pattern }.
    NotExists(Box<GraphPattern>),

    /// expr IN (list).
    In {
        /// Expression to check.
        expression: Box<Expression>,
        /// List of values.
        list: Vec<Expression>,
    },

    /// expr NOT IN (list).
    NotIn {
        /// Expression to check.
        expression: Box<Expression>,
        /// List of values.
        list: Vec<Expression>,
    },

    /// Aggregate expression.
    Aggregate(AggregateExpression),

    /// Bracketed expression (for precedence).
    Bracketed(Box<Expression>),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    /// Logical OR (||).
    Or,
    /// Logical AND (&&).
    And,
    /// Equality (=).
    Equal,
    /// Inequality (!=).
    NotEqual,
    /// Less than (<).
    LessThan,
    /// Less than or equal (<=).
    LessOrEqual,
    /// Greater than (>).
    GreaterThan,
    /// Greater than or equal (>=).
    GreaterOrEqual,
    /// Addition (+).
    Add,
    /// Subtraction (-).
    Subtract,
    /// Multiplication (*).
    Multiply,
    /// Division (/).
    Divide,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    /// Logical NOT (!).
    Not,
    /// Unary plus (+).
    Plus,
    /// Unary minus (-).
    Minus,
}

/// Function name (built-in or IRI).
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionName {
    /// Built-in function.
    BuiltIn(BuiltInFunction),
    /// Custom function (IRI).
    Custom(Iri),
}

/// Built-in SPARQL functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltInFunction {
    // String functions
    /// STR(expr).
    Str,
    /// LANG(expr).
    Lang,
    /// LANGMATCHES(expr, pattern).
    LangMatches,
    /// DATATYPE(expr).
    Datatype,
    /// STRLEN(expr).
    StrLen,
    /// SUBSTR(expr, start, length?).
    Substr,
    /// UCASE(expr).
    Ucase,
    /// LCASE(expr).
    Lcase,
    /// STRSTARTS(expr, prefix).
    StrStarts,
    /// STRENDS(expr, suffix).
    StrEnds,
    /// CONTAINS(expr, pattern).
    Contains,
    /// STRBEFORE(expr, pattern).
    StrBefore,
    /// STRAFTER(expr, pattern).
    StrAfter,
    /// ENCODE_FOR_URI(expr).
    EncodeForUri,
    /// CONCAT(expr, ...).
    Concat,
    /// REPLACE(expr, pattern, replacement, flags?).
    Replace,
    /// REGEX(expr, pattern, flags?).
    Regex,

    // Numeric functions
    /// ABS(expr).
    Abs,
    /// ROUND(expr).
    Round,
    /// CEIL(expr).
    Ceil,
    /// FLOOR(expr).
    Floor,
    /// RAND().
    Rand,

    // Date/time functions
    /// NOW().
    Now,
    /// YEAR(expr).
    Year,
    /// MONTH(expr).
    Month,
    /// DAY(expr).
    Day,
    /// HOURS(expr).
    Hours,
    /// MINUTES(expr).
    Minutes,
    /// SECONDS(expr).
    Seconds,
    /// TIMEZONE(expr).
    Timezone,
    /// TZ(expr).
    Tz,

    // Hash functions
    /// MD5(expr).
    Md5,
    /// SHA1(expr).
    Sha1,
    /// SHA256(expr).
    Sha256,
    /// SHA384(expr).
    Sha384,
    /// SHA512(expr).
    Sha512,

    // RDF term functions
    /// isIRI(expr) / isURI(expr).
    IsIri,
    /// isBLANK(expr).
    IsBlank,
    /// isLITERAL(expr).
    IsLiteral,
    /// isNUMERIC(expr).
    IsNumeric,
    /// IRI(expr) / URI(expr).
    Iri,
    /// BNODE(expr?).
    Bnode,
    /// STRDT(expr, datatype).
    StrDt,
    /// STRLANG(expr, lang).
    StrLang,
    /// UUID().
    Uuid,
    /// STRUUID().
    StrUuid,

    // Comparison
    /// sameTerm(expr1, expr2).
    SameTerm,

    // Vector functions (extension for AI/ML workloads)
    /// COSINE_SIMILARITY(vec1, vec2).
    CosineSimilarity,
    /// EUCLIDEAN_DISTANCE(vec1, vec2).
    EuclideanDistance,
    /// DOT_PRODUCT(vec1, vec2).
    DotProduct,
    /// MANHATTAN_DISTANCE(vec1, vec2).
    ManhattanDistance,
    /// VECTOR([f1, f2, ...]) - creates a vector literal.
    Vector,
}

/// Aggregate expression.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateExpression {
    /// COUNT(*) or COUNT(DISTINCT? expr).
    Count {
        /// Whether DISTINCT is specified.
        distinct: bool,
        /// The expression (None for COUNT(*)).
        expression: Option<Box<Expression>>,
    },

    /// SUM(DISTINCT? expr).
    Sum {
        /// Whether DISTINCT is specified.
        distinct: bool,
        /// The expression.
        expression: Box<Expression>,
    },

    /// AVG(DISTINCT? expr).
    Average {
        /// Whether DISTINCT is specified.
        distinct: bool,
        /// The expression.
        expression: Box<Expression>,
    },

    /// MIN(expr).
    Minimum {
        /// The expression.
        expression: Box<Expression>,
    },

    /// MAX(expr).
    Maximum {
        /// The expression.
        expression: Box<Expression>,
    },

    /// SAMPLE(expr).
    Sample {
        /// The expression.
        expression: Box<Expression>,
    },

    /// GROUP_CONCAT(DISTINCT? expr; SEPARATOR=sep?).
    GroupConcat {
        /// Whether DISTINCT is specified.
        distinct: bool,
        /// The expression.
        expression: Box<Expression>,
        /// Optional separator string.
        separator: Option<String>,
    },
}
