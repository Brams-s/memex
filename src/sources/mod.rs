//! Source-owned transcript discovery, identity, and projection adapters.
//!
//! Indexing and usage reconstruction deliberately remain independent projections.  The
//! adapters in this module make them share source identity, discovery, hierarchy, and
//! parser-version rules without introducing a persisted normalized transcript store.

pub mod claude;
pub mod codex;
pub mod common;
pub mod copilot;
pub mod cursor;
pub mod opencode;
pub mod pi;

use crate::state::PendingToolCall;
use crate::types::SourceKind;
use crate::usage::UsageEvent;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationKind {
    Main,
    Subagent,
    Sidechain,
    Fork,
    Branch,
    Compaction,
}

impl ConversationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Subagent => "subagent",
            Self::Sidechain => "sidechain",
            Self::Fork => "fork",
            Self::Branch => "branch",
            Self::Compaction => "compaction",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceFile {
    pub source: SourceKind,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionIdentity {
    pub source: SourceKind,
    pub session_id: String,
    pub parent_session_id: Option<String>,
    pub conversation_kind: ConversationKind,
    pub source_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceMetadata {
    pub session: SessionIdentity,
    pub cwd: Option<PathBuf>,
    pub project: Option<String>,
    pub git_branch: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParserVersions {
    /// Rules shared by both projections: discovery, logical session, hierarchy, and project.
    pub identity: u32,
    /// Byte-offset/Tantivy record projection.
    pub index: u32,
    /// Request/token projection.
    pub usage: i64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct IndexParseState {
    pub offset: u64,
    pub turn_id: u32,
    pub pending_tool_calls: std::collections::HashMap<String, PendingToolCall>,
}

#[derive(Clone, Debug)]
pub(crate) struct IndexParseOutput {
    pub offset: u64,
    pub turn_id: u32,
    pub pending_tool_calls: std::collections::HashMap<String, PendingToolCall>,
    pub session_id: Option<String>,
}

/// A file whose content a cached usage projection depends on. Sources that reconstruct
/// cross-file state (currently Codex forks) attach these fingerprints to their parse result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct UsageDependency {
    pub path: String,
    pub size: u64,
    pub mtime_ns: i64,
}

impl UsageDependency {
    pub fn from_path(path: &Path) -> std::io::Result<Self> {
        let metadata = path.metadata()?;
        let mtime_ns = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .min(i64::MAX as u128) as i64;
        Ok(Self {
            path: path.to_string_lossy().to_string(),
            size: metadata.len(),
            mtime_ns,
        })
    }

    pub fn is_current(&self) -> bool {
        Self::from_path(Path::new(&self.path))
            .is_ok_and(|current| current.size == self.size && current.mtime_ns == self.mtime_ns)
    }
}

/// Source-owned usage parsing output. The shared usage pipeline only caches and assembles it.
pub(crate) struct UsageParseOutput {
    pub events: Vec<UsageEvent>,
    pub cacheable: bool,
    pub deps: Vec<UsageDependency>,
}

impl UsageParseOutput {
    pub fn cacheable(events: Vec<UsageEvent>) -> Self {
        Self {
            events,
            cacheable: true,
            deps: Vec::new(),
        }
    }
}

pub fn versions(source: SourceKind) -> ParserVersions {
    match source {
        SourceKind::Claude => claude::VERSIONS,
        SourceKind::CodexSession | SourceKind::CodexHistory => codex::VERSIONS,
        SourceKind::Cursor => cursor::VERSIONS,
        SourceKind::Opencode => opencode::VERSIONS,
        SourceKind::Pi => pi::VERSIONS,
        SourceKind::Copilot => copilot::VERSIONS,
    }
}

pub fn index_state_version(source: SourceKind) -> u32 {
    let versions = versions(source);
    versions.identity.saturating_mul(10_000) + versions.index
}

/// Compatibility classification for persisted records that only carry a source path.
/// Individual path rules stay beside the source discovery code that defines them.
pub fn classify_path(path: &str) -> SourceKind {
    if let Some(source) = codex::classify_path(path) {
        source
    } else if opencode::matches_path(path) {
        SourceKind::Opencode
    } else if cursor::matches_path(path) {
        SourceKind::Cursor
    } else if pi::matches_path(path) {
        SourceKind::Pi
    } else if copilot::matches_path(path) {
        SourceKind::Copilot
    } else {
        SourceKind::Claude
    }
}
