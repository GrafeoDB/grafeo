//! Role-based access control for Grafeo sessions.
//!
//! This module provides [`Identity`], [`Role`], and [`StatementKind`] types
//! that let callers scope sessions to specific permission levels. The caller
//! is trusted to assign the correct role: there are no credentials or
//! cryptographic verification at this layer.
//!
//! # Roles
//!
//! Roles follow a hierarchy: [`Role::Admin`] implies [`Role::ReadWrite`]
//! implies [`Role::ReadOnly`]. Permission checks use the convenience methods
//! on [`Identity`] (`can_read`, `can_write`, `can_admin`) which respect this
//! hierarchy.
//!
//! # Usage
//!
//! ```
//! use grafeo_engine::auth::{Identity, Role};
//! use grafeo_engine::GrafeoDB;
//!
//! let db = GrafeoDB::new_in_memory();
//!
//! // Anonymous session (full access, backward compatible)
//! let admin_session = db.session();
//!
//! // Scoped session by role
//! let reader = db.session_with_role(Role::ReadOnly);
//!
//! // Scoped session with full identity
//! let identity = Identity::new("app-service", [Role::ReadWrite]);
//! let writer = db.session_with_identity(identity);
//! ```

use std::collections::HashSet;
use std::fmt;

/// A per-graph access grant.
///
/// When an identity has grants, it can only access the listed graphs at the
/// specified role level. An identity with no grants has unrestricted access
/// (governed only by its top-level roles).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Grant {
    /// The graph name this grant applies to.
    pub graph: String,
    /// The maximum role level for this graph.
    pub role: Role,
}

impl Grant {
    /// Creates a new grant for the given graph and role.
    #[must_use]
    pub fn new(graph: impl Into<String>, role: Role) -> Self {
        Self {
            graph: graph.into(),
            role,
        }
    }
}

/// A verified identity bound to a session.
///
/// Created by the caller (typically a server or application layer) and
/// passed to [`GrafeoDB::session_with_identity`](crate::GrafeoDB::session_with_identity).
/// The engine trusts the caller to construct the identity correctly.
#[derive(Debug, Clone)]
pub struct Identity {
    /// Unique user identifier (e.g. "admin", "app-service-1", "anonymous").
    user_id: String,
    /// Roles assigned to this identity.
    roles: HashSet<Role>,
    /// Per-graph access grants. Empty means unrestricted (all graphs accessible
    /// at the identity's role level).
    grants: Vec<Grant>,
}

impl Identity {
    /// Creates a new identity with the given user ID and roles.
    #[must_use]
    pub fn new(user_id: impl Into<String>, roles: impl IntoIterator<Item = Role>) -> Self {
        Self {
            user_id: user_id.into(),
            roles: roles.into_iter().collect(),
            grants: Vec::new(),
        }
    }

    /// Creates an anonymous identity with full access.
    ///
    /// Used internally when no identity is provided (backward-compatible
    /// default). Anonymous sessions have the [`Role::Admin`] role.
    #[must_use]
    pub fn anonymous() -> Self {
        Self {
            user_id: "anonymous".to_owned(),
            roles: [Role::Admin].into_iter().collect(),
            grants: Vec::new(),
        }
    }

    /// Adds per-graph access grants to this identity.
    ///
    /// When grants are set, the identity can only access the listed graphs
    /// at the specified role level. Graphs not in the grant list are
    /// inaccessible regardless of the identity's top-level roles.
    #[must_use]
    pub fn with_grants(mut self, grants: impl IntoIterator<Item = Grant>) -> Self {
        self.grants = grants.into_iter().collect();
        self
    }

    /// Returns the user ID.
    #[must_use]
    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    /// Returns the roles assigned to this identity.
    #[must_use]
    pub fn roles(&self) -> &HashSet<Role> {
        &self.roles
    }

    /// Returns true if this identity has the given role.
    #[must_use]
    pub fn has_role(&self, role: Role) -> bool {
        self.roles.contains(&role)
    }

    /// Returns true if this identity can perform read operations.
    ///
    /// Any assigned role grants read access.
    #[must_use]
    pub fn can_read(&self) -> bool {
        !self.roles.is_empty()
    }

    /// Returns true if this identity can perform write operations
    /// (create/update/delete nodes and edges, graph management).
    #[must_use]
    pub fn can_write(&self) -> bool {
        self.has_role(Role::Admin) || self.has_role(Role::ReadWrite)
    }

    /// Returns true if this identity can perform admin operations
    /// (schema DDL, index management, GC, configuration changes).
    #[must_use]
    pub fn can_admin(&self) -> bool {
        self.has_role(Role::Admin)
    }

    /// Returns the per-graph grants, if any.
    #[must_use]
    pub fn grants(&self) -> &[Grant] {
        &self.grants
    }

    /// Returns true if this identity has per-graph restrictions.
    #[must_use]
    pub fn has_grants(&self) -> bool {
        !self.grants.is_empty()
    }

    /// Checks whether this identity can access the given graph at the
    /// required role level.
    ///
    /// If no grants are configured, access is governed only by the
    /// identity's top-level roles. If grants are configured, the graph
    /// must appear in the grant list with a sufficient role.
    #[must_use]
    pub fn can_access_graph(&self, graph: &str, required: Role) -> bool {
        if self.grants.is_empty() {
            // No per-graph restrictions: use top-level role check
            return match required {
                Role::ReadOnly => self.can_read(),
                Role::ReadWrite => self.can_write(),
                Role::Admin => self.can_admin(),
            };
        }
        // Check if any grant covers this graph at the required level
        self.grants.iter().any(|g| {
            g.graph.eq_ignore_ascii_case(graph)
                && match required {
                    Role::ReadOnly => true, // Any grant implies read access
                    Role::ReadWrite => g.role == Role::ReadWrite || g.role == Role::Admin,
                    Role::Admin => g.role == Role::Admin,
                }
        })
    }
}

impl fmt::Display for Identity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.user_id)
    }
}

/// Database-level roles.
///
/// Roles follow a hierarchy: `Admin` implies `ReadWrite` implies `ReadOnly`.
/// Permission checks use the hierarchy via [`Identity::can_write`] and
/// [`Identity::can_admin`], but roles are stored explicitly (not inherited)
/// to keep the model simple and auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Full administrative access: schema DDL, index management, GC,
    /// plus all read-write operations.
    Admin,
    /// Read and write data: create/update/delete nodes, edges, and
    /// properties. Cannot modify schema or indexes.
    ReadWrite,
    /// Read-only access: MATCH queries, graph traversals, read-only
    /// introspection (database stats, schema info).
    ReadOnly,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admin => write!(f, "Admin"),
            Self::ReadWrite => write!(f, "ReadWrite"),
            Self::ReadOnly => write!(f, "ReadOnly"),
        }
    }
}

/// Classification of a parsed statement for permission checking.
///
/// Determined after parsing but before execution. The session checks the
/// caller's [`Identity`] against the statement kind and rejects operations
/// that exceed the caller's role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementKind {
    /// Read-only: MATCH, RETURN, WITH, UNWIND, CALL (read-only procedures).
    Read,
    /// Data mutation: CREATE, SET, DELETE, REMOVE, MERGE.
    Write,
    /// Schema/admin: CREATE TYPE, DROP TYPE, CREATE INDEX, DROP INDEX,
    /// CREATE CONSTRAINT, DROP CONSTRAINT.
    Admin,
    /// Transaction control: START TRANSACTION, COMMIT, ROLLBACK, SAVEPOINT.
    /// Always allowed regardless of role.
    Transaction,
}

impl StatementKind {
    /// Returns the minimum [`Role`] required for this statement kind.
    ///
    /// Returns `None` for [`StatementKind::Transaction`] (always allowed).
    #[must_use]
    pub fn required_role(self) -> Option<Role> {
        match self {
            Self::Read => Some(Role::ReadOnly),
            Self::Write => Some(Role::ReadWrite),
            Self::Admin => Some(Role::Admin),
            Self::Transaction => None,
        }
    }
}

impl fmt::Display for StatementKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Admin => write!(f, "admin"),
            Self::Transaction => write!(f, "transaction control"),
        }
    }
}

/// Checks whether an identity is permitted to execute a statement of the
/// given kind. Returns `Ok(())` on success, or an error describing the
/// denial.
///
/// Transaction control statements are always permitted.
pub(crate) fn check_permission(
    identity: &Identity,
    kind: StatementKind,
) -> std::result::Result<(), PermissionDenied> {
    match kind {
        StatementKind::Transaction => Ok(()),
        StatementKind::Read => {
            if identity.can_read() {
                Ok(())
            } else {
                Err(PermissionDenied {
                    operation: kind,
                    required: Role::ReadOnly,
                    user_id: identity.user_id.clone(),
                })
            }
        }
        StatementKind::Write => {
            if identity.can_write() {
                Ok(())
            } else {
                Err(PermissionDenied {
                    operation: kind,
                    required: Role::ReadWrite,
                    user_id: identity.user_id.clone(),
                })
            }
        }
        StatementKind::Admin => {
            if identity.can_admin() {
                Ok(())
            } else {
                Err(PermissionDenied {
                    operation: kind,
                    required: Role::Admin,
                    user_id: identity.user_id.clone(),
                })
            }
        }
    }
}

/// Permission denied error with context about what was attempted.
#[derive(Debug, Clone)]
pub struct PermissionDenied {
    /// What kind of statement was attempted.
    pub operation: StatementKind,
    /// The minimum role that would have been required.
    pub required: Role,
    /// The user who was denied.
    pub user_id: String,
}

impl fmt::Display for PermissionDenied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "permission denied: {} operations require {} role (user: {})",
            self.operation, self.required, self.user_id
        )
    }
}

impl std::error::Error for PermissionDenied {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anonymous_has_admin_role() {
        let id = Identity::anonymous();
        assert_eq!(id.user_id(), "anonymous");
        assert!(id.has_role(Role::Admin));
        assert!(id.can_read());
        assert!(id.can_write());
        assert!(id.can_admin());
    }

    #[test]
    fn read_only_identity() {
        let id = Identity::new("reader", [Role::ReadOnly]);
        assert!(id.can_read());
        assert!(!id.can_write());
        assert!(!id.can_admin());
    }

    #[test]
    fn read_write_identity() {
        let id = Identity::new("writer", [Role::ReadWrite]);
        assert!(id.can_read());
        assert!(id.can_write());
        assert!(!id.can_admin());
    }

    #[test]
    fn admin_identity() {
        let id = Identity::new("admin", [Role::Admin]);
        assert!(id.can_read());
        assert!(id.can_write());
        assert!(id.can_admin());
    }

    #[test]
    fn empty_roles_cannot_read() {
        let id = Identity::new("nobody", std::iter::empty::<Role>());
        assert!(!id.can_read());
        assert!(!id.can_write());
        assert!(!id.can_admin());
    }

    #[test]
    fn role_display() {
        assert_eq!(Role::Admin.to_string(), "Admin");
        assert_eq!(Role::ReadWrite.to_string(), "ReadWrite");
        assert_eq!(Role::ReadOnly.to_string(), "ReadOnly");
    }

    #[test]
    fn statement_kind_required_role() {
        assert_eq!(StatementKind::Read.required_role(), Some(Role::ReadOnly));
        assert_eq!(StatementKind::Write.required_role(), Some(Role::ReadWrite));
        assert_eq!(StatementKind::Admin.required_role(), Some(Role::Admin));
        assert_eq!(StatementKind::Transaction.required_role(), None);
    }

    #[test]
    fn check_permission_allows_transaction_for_all() {
        let readonly = Identity::new("r", [Role::ReadOnly]);
        assert!(check_permission(&readonly, StatementKind::Transaction).is_ok());

        let nobody = Identity::new("n", std::iter::empty::<Role>());
        assert!(check_permission(&nobody, StatementKind::Transaction).is_ok());
    }

    #[test]
    fn check_permission_denies_write_for_readonly() {
        let id = Identity::new("reader", [Role::ReadOnly]);
        let err = check_permission(&id, StatementKind::Write).unwrap_err();
        assert_eq!(err.required, Role::ReadWrite);
        assert_eq!(err.operation, StatementKind::Write);
        assert!(err.to_string().contains("permission denied"));
    }

    #[test]
    fn check_permission_denies_admin_for_readwrite() {
        let id = Identity::new("writer", [Role::ReadWrite]);
        let err = check_permission(&id, StatementKind::Admin).unwrap_err();
        assert_eq!(err.required, Role::Admin);
    }

    #[test]
    fn identity_display() {
        let id = Identity::new("app-service", [Role::ReadWrite]);
        assert_eq!(id.to_string(), "app-service");
    }

    #[test]
    fn identity_with_multiple_roles() {
        let id = Identity::new("alix", [Role::ReadOnly, Role::ReadWrite]);
        assert!(id.can_read());
        assert!(id.can_write());
        assert!(!id.can_admin());
        assert!(id.has_role(Role::ReadOnly));
        assert!(id.has_role(Role::ReadWrite));
        assert!(!id.has_role(Role::Admin));
        assert_eq!(id.roles().len(), 2);
    }

    #[test]
    fn identity_with_all_roles() {
        let id = Identity::new("gus", [Role::ReadOnly, Role::ReadWrite, Role::Admin]);
        assert!(id.can_read());
        assert!(id.can_write());
        assert!(id.can_admin());
        assert_eq!(id.roles().len(), 3);
    }

    #[test]
    fn permission_denied_error_message_contains_user_and_role() {
        let id = Identity::new("alix", [Role::ReadOnly]);
        let err = check_permission(&id, StatementKind::Write).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("alix"), "error should contain user id");
        assert!(
            msg.contains("ReadWrite"),
            "error should contain required role"
        );
        assert!(msg.contains("write"), "error should contain operation kind");
    }

    #[test]
    fn permission_denied_admin_error_message() {
        let id = Identity::new("gus", [Role::ReadOnly]);
        let err = check_permission(&id, StatementKind::Admin).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("gus"));
        assert!(msg.contains("Admin"));
        assert!(msg.contains("admin"));
    }

    #[test]
    fn check_permission_denies_read_for_no_roles() {
        let id = Identity::new("nobody", std::iter::empty::<Role>());
        let err = check_permission(&id, StatementKind::Read).unwrap_err();
        assert_eq!(err.required, Role::ReadOnly);
        assert_eq!(err.operation, StatementKind::Read);
        assert!(err.to_string().contains("nobody"));
    }

    #[test]
    fn check_permission_allows_read_for_readonly() {
        let id = Identity::new("alix", [Role::ReadOnly]);
        assert!(check_permission(&id, StatementKind::Read).is_ok());
    }

    #[test]
    fn check_permission_allows_admin_for_admin() {
        let id = Identity::new("gus", [Role::Admin]);
        assert!(check_permission(&id, StatementKind::Admin).is_ok());
    }

    #[test]
    fn check_permission_allows_write_for_readwrite() {
        let id = Identity::new("alix", [Role::ReadWrite]);
        assert!(check_permission(&id, StatementKind::Write).is_ok());
    }

    #[test]
    fn statement_kind_display() {
        assert_eq!(StatementKind::Read.to_string(), "read");
        assert_eq!(StatementKind::Write.to_string(), "write");
        assert_eq!(StatementKind::Admin.to_string(), "admin");
        assert_eq!(
            StatementKind::Transaction.to_string(),
            "transaction control"
        );
    }

    #[test]
    fn permission_denied_is_std_error() {
        let id = Identity::new("alix", [Role::ReadOnly]);
        let err = check_permission(&id, StatementKind::Write).unwrap_err();
        // Verify PermissionDenied implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    // --- Grant tests ---

    #[test]
    fn no_grants_means_unrestricted() {
        let id = Identity::new("alix", [Role::ReadWrite]);
        assert!(id.can_access_graph("any_graph", Role::ReadWrite));
        assert!(id.can_access_graph("other", Role::ReadOnly));
        assert!(!id.has_grants());
    }

    #[test]
    fn grant_restricts_to_listed_graphs() {
        let id = Identity::new("gus", [Role::ReadWrite]).with_grants([
            Grant::new("social", Role::ReadWrite),
            Grant::new("analytics", Role::ReadOnly),
        ]);
        assert!(id.has_grants());
        assert!(id.can_access_graph("social", Role::ReadWrite));
        assert!(id.can_access_graph("social", Role::ReadOnly));
        assert!(id.can_access_graph("analytics", Role::ReadOnly));
        assert!(!id.can_access_graph("analytics", Role::ReadWrite));
        assert!(!id.can_access_graph("secret", Role::ReadOnly));
    }

    #[test]
    fn grant_admin_implies_all() {
        let id =
            Identity::new("admin", [Role::Admin]).with_grants([Grant::new("prod", Role::Admin)]);
        assert!(id.can_access_graph("prod", Role::Admin));
        assert!(id.can_access_graph("prod", Role::ReadWrite));
        assert!(id.can_access_graph("prod", Role::ReadOnly));
    }

    #[test]
    fn grant_case_insensitive() {
        let id = Identity::new("alix", [Role::ReadWrite])
            .with_grants([Grant::new("Social", Role::ReadWrite)]);
        assert!(id.can_access_graph("social", Role::ReadOnly));
        assert!(id.can_access_graph("SOCIAL", Role::ReadWrite));
    }

    #[test]
    fn grant_display() {
        let g = Grant::new("social", Role::ReadWrite);
        assert_eq!(g.graph, "social");
        assert_eq!(g.role, Role::ReadWrite);
    }
}
