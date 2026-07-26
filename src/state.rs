use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileIdentity {
    /// Stable filesystem identity when the platform exposes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inode: Option<u64>,
    /// Hash of a bounded prefix, used to detect in-place replacement without rescanning a file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_sha256: Option<String>,
    /// Number of leading bytes covered by `prefix_sha256`. Keeping this stable across appends
    /// prevents a short file's fingerprint from changing merely because its prefix grew.
    #[serde(default)]
    pub prefix_bytes: u64,
    /// Nanosecond-resolution modification marker for detecting same-size rewrites.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_ns: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_doc_id: Option<u64>,
    #[serde(default)]
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argument_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argument_bytes: Option<u64>,
    /// Source-native parent event for formats whose result event does not repeat it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_tool_use_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_tool_assistant_uuid: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    pub size: u64,
    pub mtime: i64,
    pub offset: u64,
    pub turn_id: u32,
    #[serde(default)]
    pub parser_version: u32,
    /// Source-specific invalidation generation. Append-only sources leave this at zero;
    /// SQLite-backed sources can advance it when prior rows may have changed.
    #[serde(default)]
    pub source_generation: u64,
    #[serde(default)]
    pub pending_tool_calls: HashMap<String, PendingToolCall>,
    #[serde(default)]
    pub identity: FileIdentity,
}

/// Tracks when we last scanned for changes, allowing us to skip
/// redundant scans if called again within a short TTL.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScanCache {
    /// Unix timestamp (seconds) of last successful scan
    pub last_scan_ts: u64,
    /// Number of files found in last scan
    pub file_count: usize,
    /// Total bytes across all source files
    pub total_bytes: u64,
}

impl ScanCache {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = fs::read_to_string(path)?;
        let cache = serde_json::from_str(&data)?;
        Ok(cache)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Check if the cache is still valid (within TTL seconds)
    pub fn is_fresh(&self, ttl_seconds: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_sub(self.last_scan_ts) < ttl_seconds
    }

    /// Update cache with current scan results
    pub fn update(&mut self, file_count: usize, total_bytes: u64) {
        self.last_scan_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.file_count = file_count;
        self.total_bytes = total_bytes;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestState {
    pub next_doc_id: u64,
    pub files: HashMap<String, FileState>,
}

impl Default for IngestState {
    fn default() -> Self {
        Self {
            next_doc_id: 1,
            files: HashMap::new(),
        }
    }
}

impl IngestState {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = fs::read_to_string(path)?;
        let state = serde_json::from_str(&data)?;
        Ok(state)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }
}
