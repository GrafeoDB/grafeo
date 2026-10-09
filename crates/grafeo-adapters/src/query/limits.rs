//! Limits every parser enforces on the statements it accepts.
//!
//! Every stage after the parser (translation, binding, optimization,
//! planning, expression evaluation) walks a statement recursively, one stack
//! frame or more per level of nesting. A statement nested deeper than the
//! stack allows used to overflow it: the process aborted, and a WebAssembly
//! module trapped and stayed broken (#573). The parsers bound the nesting
//! instead, so a deeper statement fails with an error that names the limit.

/// How deep a statement may nest.
///
/// An expression nests one level in the expression, list, map or clause it
/// is part of (so do parentheses around it), and so does each operator of a
/// chain of binary operators (`a OR b OR c` nests two levels, as
/// `(a OR b) OR c`), `NOT` and a sign. A function call, `CASE` and a subquery
/// nest one more level, as parsing and planning them take more stack. A long
/// flat list (`[1, 2, 3, ...]`) nests one level however long it is. A deeper
/// statement fails with a syntax error naming this limit.
///
/// The limit keeps the stack a statement can take at every stage well below
/// the smallest stack Grafeo runs on (1 MiB: the main thread on Windows; the
/// WebAssembly builds have 8 MiB): about 470 KiB at the most on x86-64. The
/// engine's tests run every kind of nesting at the limit on a 640 KiB stack.
pub const MAX_NESTING_DEPTH: u32 = 64;

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
/// The message of the error a parser returns for a statement that nests
/// deeper than [`MAX_NESTING_DEPTH`].
#[must_use]
pub(crate) fn nesting_error_message() -> String {
    format!("Maximum nesting depth of {MAX_NESTING_DEPTH} exceeded")
}

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
/// How deep the statement a parser reads nests, as [`MAX_NESTING_DEPTH`]
/// counts it.
///
/// A parser [`enter`](Self::enter)s a level around each construct it parses
/// recursively, and brackets each loop that builds a chain of binary
/// operators with [`begin_chain`](Self::begin_chain) and
/// [`end_chain`](Self::end_chain), calling [`link`](Self::link) for each
/// operator. A chain is built without recursion, but the tree it builds is as
/// deep as the chain is long, and every later stage recurses into it. A
/// parser that backtracks restores a copy taken where it started, as it
/// restores its lexer.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Nesting {
    /// The levels the parser is inside now.
    depth: u32,
    /// The deepest level the syntax parsed since the innermost open chain
    /// began reaches.
    reached: u32,
}

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
/// A chain of binary operators being parsed (see [`Nesting::begin_chain`]).
#[derive(Debug)]
#[must_use = "a chain ends with `Nesting::end_chain`"]
pub(crate) struct Chain {
    /// The deepest level reached before the chain began.
    outer: u32,
}

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
impl Nesting {
    /// Enters one level of nesting. Returns `false` when that is deeper than
    /// [`MAX_NESTING_DEPTH`]: the parser then fails.
    #[must_use]
    pub(crate) fn enter(&mut self) -> bool {
        self.depth += 1;
        self.reached = self.reached.max(self.depth);
        self.depth <= MAX_NESTING_DEPTH
    }

    /// Leaves the level [`enter`](Self::enter) entered.
    pub(crate) fn exit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Begins a chain of binary operators at the current level, before its
    /// first operand is parsed.
    pub(crate) fn begin_chain(&mut self) -> Chain {
        Chain {
            outer: std::mem::replace(&mut self.reached, self.depth),
        }
    }

    /// Counts the operator that joins the chain so far with the operand just
    /// parsed: it is one level above the deeper of the two. Returns `false`
    /// when that is deeper than [`MAX_NESTING_DEPTH`]: the parser then fails.
    #[must_use]
    pub(crate) fn link(&mut self) -> bool {
        self.reached += 1;
        self.reached <= MAX_NESTING_DEPTH
    }

    /// Ends `chain`: what encloses it reaches at least as deep as the chain.
    pub(crate) fn end_chain(&mut self, chain: Chain) {
        self.reached = self.reached.max(chain.outer);
    }
}

#[cfg(all(
    test,
    any(
        feature = "gql",
        feature = "cypher",
        feature = "sparql",
        feature = "gremlin",
        feature = "graphql",
        feature = "sql-pgq"
    )
))]
mod tests {
    use super::*;

    /// The deepest level `parse` reaches, starting at the top.
    fn reached(parse: impl FnOnce(&mut Nesting) -> bool) -> Option<u32> {
        let mut nesting = Nesting::default();
        let chain = nesting.begin_chain();
        let ok = parse(&mut nesting);
        let reached = nesting.reached;
        nesting.end_chain(chain);
        ok.then_some(reached)
    }

    #[test]
    fn a_chain_is_as_deep_as_its_operators() {
        // a OR b OR c: two operators.
        let depth = reached(|n| {
            let chain = n.begin_chain();
            let ok = n.link() && n.link();
            n.end_chain(chain);
            ok
        });
        assert_eq!(depth, Some(2));
    }

    #[test]
    fn a_chain_over_a_parenthesized_operand_counts_both() {
        // (a OR b) AND c: the parentheses, the OR inside, the AND outside.
        let depth = reached(|n| {
            let and = n.begin_chain();
            let ok = n.enter() && {
                let or = n.begin_chain();
                let ok = n.link();
                n.end_chain(or);
                ok
            };
            n.exit();
            let ok = ok && n.link();
            n.end_chain(and);
            ok
        });
        assert_eq!(depth, Some(3));
    }

    #[test]
    fn siblings_take_the_deeper_one_not_their_sum() {
        // f((a OR b), (c OR d)): two arguments of depth 2 each, in a call.
        let depth = reached(|n| {
            let ok = n.enter();
            let mut ok = ok;
            for _ in 0..2 {
                ok &= n.enter();
                let or = n.begin_chain();
                ok &= n.link();
                n.end_chain(or);
                n.exit();
            }
            n.exit();
            ok
        });
        assert_eq!(depth, Some(3));
    }

    #[test]
    fn the_limit_is_inclusive() {
        let at_limit = reached(|n| {
            let chain = n.begin_chain();
            let ok = (0..MAX_NESTING_DEPTH).all(|_| n.link());
            n.end_chain(chain);
            ok
        });
        assert_eq!(at_limit, Some(MAX_NESTING_DEPTH));
        let beyond = reached(|n| {
            let chain = n.begin_chain();
            let ok = (0..=MAX_NESTING_DEPTH).all(|_| n.link());
            n.end_chain(chain);
            ok
        });
        assert_eq!(beyond, None);
        let mut nesting = Nesting::default();
        assert!((0..MAX_NESTING_DEPTH).all(|_| nesting.enter()));
        assert!(!nesting.enter(), "one level beyond the limit");
    }
}
