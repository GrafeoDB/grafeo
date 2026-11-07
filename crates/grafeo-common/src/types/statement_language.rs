//! The language a statement is written in, which some value rules follow.

/// The language a statement is written in. The value rules that differ between languages read it:
/// how two values of different kinds compare, what `+` does with a list, and how a zoned datetime
/// compares with a local one (see [`TemporalComparison`](super::TemporalComparison), derived from
/// it and the session time zone). A stored procedure's body is GQL, whatever calls it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum StatementLanguage {
    /// ISO GQL, the default.
    #[default]
    Gql,
    /// openCypher.
    Cypher,
    /// Gremlin.
    Gremlin,
    /// SQL/PGQ (`GRAPH_TABLE`).
    SqlPgq,
    /// GraphQL.
    GraphQl,
    /// SPARQL.
    Sparql,
}

impl StatementLanguage {
    /// Whether values follow openCypher's rules rather than ISO GQL's: Cypher's, and Gremlin's,
    /// which take them (TinkerPop compares across types as false, and a filter drops false as it
    /// drops null).
    #[must_use]
    pub const fn follows_opencypher(self) -> bool {
        matches!(self, Self::Cypher | Self::Gremlin)
    }
}
