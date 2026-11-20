//! The warning probe (feature `testing-warning-probe`): a test sets it on a session, and each
//! statement the session runs after that raises the probe notice (`Warning::Probe`, 01Z99) at
//! the stage the test names, so the warning channel can be tested end to end on any entry path
//! without a deprecated form.

use std::sync::Arc;

use grafeo_common::utils::warning::{Warning, WarningSink};

use super::Session;
use super::SessionInner;
use crate::query::plan::LogicalPlan;

/// Where the warning probe raises its notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbeStage {
    /// Into each statement's translate-time set, which the plan cache keeps with the plan: a
    /// cache hit warns after the probe is cleared, a statement translated after it does not.
    Translate,
    /// Into each run's sink, when its planner is made: every run warns, a cache hit included.
    Run,
}

impl SessionInner {
    pub(crate) fn set_warning_probe(&self, stage: Option<ProbeStage>) {
        *self.warning_probe.lock() = stage;
    }

    /// `plan`, translated for a statement now, with the probe's notice when it probes
    /// translation.
    pub(super) fn probe_translation(&self, mut plan: LogicalPlan) -> LogicalPlan {
        if *self.warning_probe.lock() == Some(ProbeStage::Translate) {
            plan.warnings.insert(Warning::Probe);
        }
        plan
    }

    /// Raises the probe's notice into `sink`, a run's, when it probes runs.
    pub(super) fn probe_run(&self, sink: &Arc<WarningSink>) {
        if *self.warning_probe.lock() == Some(ProbeStage::Run) {
            sink.raise(Warning::Probe);
        }
    }
}

// The session's public methods: each reaches the session's slot for its call (#712).
impl Session {
    /// Sets the warning probe of this session (`None` clears it); other sessions are not
    /// affected. After `close()` it does nothing; it returns a `Result` from 0.7.0.
    pub fn set_warning_probe(&self, stage: Option<ProbeStage>) {
        if let Ok(inner) = self.inner() {
            inner.set_warning_probe(stage);
        }
    }
}
