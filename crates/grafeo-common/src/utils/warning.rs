//! The warnings a statement can raise, in one registry.
//!
//! A warning is a completion condition of class `01` (ISO/IEC 39075, 23.1) with an
//! implementation-defined subclass: `01X..` for a deprecated form, `01Y..` for another notice,
//! `01Z..` for test probes. Each [`Warning`] has a [`WarningSpec`], and one formatter builds its
//! message from it, so a raise site names the variant only. A statement shows each warning it
//! raised once, ordered by code, as a [`QueryWarning`].
//!
//! Two carriers collect them: a [`WarningSet`] for what translation finds (it travels with the
//! plan, and so with a cached plan), and a [`WarningSink`] for what a run finds, shared by every
//! operator and worker of the run.
//!
//! A new warning is one entry in the registry's list (the `warnings!` call below): its code is
//! never reused, and a removed one moves its code to [`RETIRED`].

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use super::gqlstatus::GqlStatus;

/// What a warning tells: a form that is going away, or something else worth knowing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WarningKind {
    /// A form that still runs, deprecated in favour of another.
    Deprecation,
    /// Anything else the statement wants its caller to know.
    Notice,
}

impl WarningKind {
    /// The kind's name, as the bindings show it: `deprecation` or `notice`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Deprecation => "deprecation",
            Self::Notice => "notice",
        }
    }
}

impl fmt::Display for WarningKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// When a deprecated form went and what replaces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeprecationSpec {
    /// The release that deprecated it.
    pub since: &'static str,
    /// Where it is deprecated, when not everywhere (`in GQL and SQL/PGQ`); empty for everywhere.
    pub scope: &'static str,
    /// The release that refuses it.
    pub removed_in: &'static str,
    /// What to write instead.
    pub replacement: &'static str,
}

/// Everything the registry says of a warning.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WarningSpec {
    /// Its GQLSTATUS code, of class `01`.
    pub code: GqlStatus,
    /// Whether it is a deprecation.
    pub kind: WarningKind,
    /// What the statement wrote or did, as the message names it.
    pub form: &'static str,
    /// For a deprecation, its releases and replacement.
    pub deprecation: Option<DeprecationSpec>,
    /// For a notice, what the message says after the form.
    pub notice: &'static str,
}

impl WarningSpec {
    const fn deprecation(
        code: [u8; 5],
        form: &'static str,
        scope: &'static str,
        since: &'static str,
        removed_in: &'static str,
        replacement: &'static str,
    ) -> Self {
        Self {
            code: GqlStatus::from_bytes(code),
            kind: WarningKind::Deprecation,
            form,
            deprecation: Some(DeprecationSpec {
                since,
                scope,
                removed_in,
                replacement,
            }),
            notice: "",
        }
    }

    const fn notice(code: [u8; 5], form: &'static str, notice: &'static str) -> Self {
        Self {
            code: GqlStatus::from_bytes(code),
            kind: WarningKind::Notice,
            form,
            deprecation: None,
            notice,
        }
    }
}

/// Declares every warning once, with its spec; a variant without one does not compile.
macro_rules! warnings {
    ($(
        $(#[doc = $doc:literal])*
        $(#[cfg($cfg:meta)])?
        $variant:ident => $spec:expr;
    )+) => {
        /// A warning a statement can raise (see the [module docs](self)).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Warning {
            $(
                $(#[doc = $doc])*
                $(#[cfg($cfg)])?
                $variant,
            )+
        }

        impl Warning {
            /// Every warning of this build, in the registry's order.
            pub const ALL: &'static [Warning] = &[$($(#[cfg($cfg)])? Warning::$variant,)+];

            /// What the registry says of it.
            #[must_use]
            pub const fn spec(self) -> WarningSpec {
                match self {
                    $($(#[cfg($cfg)])? Self::$variant => $spec,)+
                }
            }
        }
    };
}

warnings! {
    /// A `ZONED DATETIME '...'` literal, which ISO GQL writes as `DATETIME '...'`.
    ZonedDatetimeLiteral => WarningSpec::deprecation(
        *b"01X01",
        "`ZONED DATETIME '...'`",
        "",
        "0.6.2",
        "0.7.0",
        "`DATETIME '...'` with an offset",
    );
    /// `+` on two lists in GQL and SQL/PGQ, which joins them as `||` does.
    ListPlus => WarningSpec::deprecation(
        *b"01X02",
        "`+` on two lists",
        "in GQL and SQL/PGQ",
        "0.6.2",
        "0.7.0",
        "`||`",
    );
    /// A `ZONED TIME '...'` literal, which ISO GQL writes as `TIME '...'`.
    ZonedTimeLiteral => WarningSpec::deprecation(
        *b"01X03",
        "`ZONED TIME '...'`",
        "",
        "0.6.2",
        "0.7.0",
        "`TIME '...'` with an offset",
    );
    /// An iterative procedure that ran out of iterations before its scores settled within its
    /// `tolerance`.
    NotConverged => WarningSpec::notice(
        *b"01Y01",
        "`max_iterations`",
        "ran out before the scores changed by less than `tolerance`; the scores are those of \
         the last iteration",
    );
    /// The test probe (`Session::set_warning_probe`), in test builds only.
    #[cfg(feature = "testing-warning-probe")]
    Probe => WarningSpec::notice(
        *b"01Z99",
        "the warning probe",
        "raised this notice, as the test asked",
    );
}

/// The codes of warnings removed from the registry, which no warning uses again.
pub const RETIRED: &[GqlStatus] = &[];

// A `WarningSet` and a `WarningSink` hold one bit per warning.
const _: () = assert!(Warning::ALL.len() <= 64);

impl Warning {
    /// The message a result shows for it, built from its [`spec`](Self::spec).
    #[must_use]
    pub fn message(self) -> String {
        let spec = self.spec();
        match spec.deprecation {
            Some(deprecation) if deprecation.scope.is_empty() => format!(
                "{} is deprecated since {} and refused from {}; use {}.",
                spec.form, deprecation.since, deprecation.removed_in, deprecation.replacement
            ),
            Some(deprecation) => format!(
                "{} is deprecated since {} {} and refused there from {}; use {}.",
                spec.form,
                deprecation.since,
                deprecation.scope,
                deprecation.removed_in,
                deprecation.replacement
            ),
            None => format!("{} {}.", spec.form, spec.notice),
        }
    }

    /// Its bit in a [`WarningSet`] or [`WarningSink`].
    const fn bit(self) -> u64 {
        1 << self as u32
    }
}

/// The warnings a statement raised, each once: what translation finds, which travels with the
/// plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct WarningSet(u64);

impl WarningSet {
    /// No warning.
    pub const EMPTY: Self = Self(0);

    /// Adds `warning`.
    pub fn insert(&mut self, warning: Warning) {
        self.0 |= warning.bit();
    }

    /// Whether it holds `warning`.
    #[must_use]
    pub const fn contains(self, warning: Warning) -> bool {
        self.0 & warning.bit() != 0
    }

    /// The warnings of both.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether it holds none.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Its warnings, in the registry's order.
    pub fn iter(self) -> impl Iterator<Item = Warning> {
        Warning::ALL
            .iter()
            .copied()
            .filter(move |warning| self.contains(*warning))
    }

    /// Its warnings as a result shows them: each once, ordered by code.
    #[must_use]
    pub fn to_query_warnings(self) -> Vec<QueryWarning> {
        let mut warnings: Vec<QueryWarning> = self.iter().map(QueryWarning::from).collect();
        warnings.sort_by(|a, b| a.code.as_str().cmp(b.code.as_str()));
        warnings
    }
}

impl FromIterator<Warning> for WarningSet {
    fn from_iter<I: IntoIterator<Item = Warning>>(warnings: I) -> Self {
        let mut set = Self::EMPTY;
        for warning in warnings {
            set.insert(warning);
        }
        set
    }
}

/// Where a run raises its warnings: shared by every operator and worker of the run, and by the
/// bodies it runs (subqueries, `EXISTS`, procedure bodies), so a warning reaches the statement
/// that started it.
#[derive(Debug, Default)]
pub struct WarningSink(AtomicU64);

impl WarningSink {
    /// A sink with nothing raised.
    #[must_use]
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Raises `warning`: one load once it is raised, so a form evaluated per row costs a read.
    #[inline]
    pub fn raise(&self, warning: Warning) {
        let bit = warning.bit();
        if self.0.load(Ordering::Relaxed) & bit == 0 {
            self.0.fetch_or(bit, Ordering::Relaxed);
        }
    }

    /// Raises each of `warnings`.
    pub fn raise_all(&self, warnings: WarningSet) {
        if !warnings.is_empty() {
            self.0.fetch_or(warnings.0, Ordering::Relaxed);
        }
    }

    /// What was raised so far.
    #[must_use]
    pub fn raised(&self) -> WarningSet {
        WarningSet(self.0.load(Ordering::Relaxed))
    }
}

/// A warning as a statement's result shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryWarning {
    /// Its GQLSTATUS code, of class `01`.
    pub code: GqlStatus,
    /// Whether it is a deprecation.
    pub kind: WarningKind,
    /// What it says, naming the replacement of a deprecated form.
    pub message: String,
}

impl From<Warning> for QueryWarning {
    fn from(warning: Warning) -> Self {
        let spec = warning.spec();
        Self {
            code: spec.code,
            kind: spec.kind,
            message: warning.message(),
        }
    }
}

impl fmt::Display for QueryWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}
