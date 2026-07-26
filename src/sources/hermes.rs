use super::{
    ConversationKind, IndexParseOutput, IndexParseState, ParseDiagnostics, ParserVersions,
    SourceFile,
};
use crate::state::PendingToolCall;
use crate::types::{Record, RecordLinks, SourceKind};
use crate::usage::{TokenBuckets, UsageEvent};
use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

pub const VERSIONS: ParserVersions = ParserVersions {
    identity: 1,
    index: 2,
    usage: 1,
};

const CONTENT_JSON_PREFIX: &str = "\0json:";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Checkpoint {
    pub max_message_id: u64,
    pub generation: u64,
}

#[derive(Clone, Debug)]
struct MessageRow {
    id: u64,
    session_id: String,
    role: String,
    content: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Option<String>,
    tool_name: Option<String>,
    timestamp_ms: u64,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    cwd: Option<String>,
    parent_session_id: Option<String>,
}

#[derive(Clone, Debug)]
struct ToolCall {
    id: String,
    name: Option<String>,
    arguments: String,
}

pub fn matches_path(path: &str) -> bool {
    Path::new(path) == database_path()
        || ((path.contains(".hermes/") || path.contains(".hermes\\"))
            && (path.ends_with("state.db") || path.ends_with("state.db-wal")))
}

pub fn home() -> PathBuf {
    std::env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| super::common::home().join(".hermes"))
}

pub fn database_path() -> PathBuf {
    home().join("state.db")
}

pub fn discover() -> Vec<SourceFile> {
    let path = database_path();
    if path.is_file() {
        vec![SourceFile {
            source: SourceKind::Hermes,
            path,
        }]
    } else {
        Vec::new()
    }
}

pub fn checkpoint(path: &Path) -> Result<Checkpoint> {
    let connection = open_read_only(path)?;
    let message_columns = table_columns(&connection, "messages")?;
    let columns = table_columns(&connection, "sessions")?;
    let active = if message_columns.contains("active") {
        " WHERE COALESCE(active, 1) != 0"
    } else {
        ""
    };
    let max_message_id = connection
        .query_row(
            &format!("SELECT COALESCE(MAX(id), 0) FROM messages{active}"),
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        .max(0) as u64;
    let schema_version = connection
        .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
            row.get::<_, Option<i64>>(0)
        })
        .optional()
        .ok()
        .flatten()
        .flatten()
        .unwrap_or(0)
        .max(0) as u64;
    let rewind_generation = if columns.contains("rewind_count") {
        connection
            .query_row(
                "SELECT COALESCE(SUM(rewind_count), 0) FROM sessions",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            .max(0) as u64
    } else {
        0
    };
    Ok(Checkpoint {
        max_message_id,
        generation: schema_version.rotate_left(32) ^ rewind_generation,
    })
}

pub fn session_cwd(path: &Path, session_id: &str) -> Option<String> {
    let connection = open_read_only(path).ok()?;
    let columns = table_columns(&connection, "sessions").ok()?;
    if !columns.contains("cwd") {
        return None;
    }
    connection
        .query_row(
            "SELECT cwd FROM sessions WHERE id = ?1",
            params![session_id],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()
        .ok()
        .flatten()
        .flatten()
        .filter(|cwd| !cwd.is_empty())
}

pub(crate) fn parse_index_records(
    path: &Path,
    state: IndexParseState,
    include_reasoning: bool,
    next_doc_id: &AtomicU64,
    mut emit: impl FnMut(Record) -> Result<()>,
) -> Result<IndexParseOutput> {
    let connection = open_read_only(path)?;
    let rows = load_rows(&connection, state.offset)?;
    let source_path = path.to_string_lossy().to_string();
    let mut diagnostics = ParseDiagnostics::default();
    let mut pending_tool_calls = state.pending_tool_calls;
    let mut planned_calls = plan_tool_calls(&rows, &mut diagnostics);
    let mut max_message_id = state.offset;

    for row in &rows {
        max_message_id = max_message_id.max(row.id);
        let project = row
            .cwd
            .as_deref()
            .map(super::common::project_from_path)
            .unwrap_or_else(|| SourceKind::Hermes.label().to_string());
        let links = row_links(row);
        let base_turn = row.id.min((u32::MAX / 8) as u64).saturating_mul(8) as u32;
        let mut component = 0u32;

        match row.role.as_str() {
            "user" => {
                let text = content_text(row.content.as_deref());
                if !text.trim().is_empty() {
                    emit(Record {
                        source: SourceKind::Hermes,
                        doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                        ts: row.timestamp_ms,
                        project,
                        session_id: row.session_id.clone(),
                        turn_id: base_turn + component,
                        role: "user".to_string(),
                        text,
                        tool_name: None,
                        tool_input: None,
                        tool_output: None,
                        links,
                        source_path: source_path.clone(),
                    })?;
                }
            }
            "assistant" => {
                if include_reasoning && let Some(reasoning) = reasoning_text(row, &mut diagnostics)
                {
                    let mut reasoning_links = links.clone();
                    reasoning_links.event_id = Some(format!("{}:reasoning", row.id));
                    emit(Record {
                        source: SourceKind::Hermes,
                        doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                        ts: row.timestamp_ms,
                        project: project.clone(),
                        session_id: row.session_id.clone(),
                        turn_id: base_turn + component,
                        role: "reasoning".to_string(),
                        text: reasoning,
                        tool_name: None,
                        tool_input: None,
                        tool_output: None,
                        links: reasoning_links,
                        source_path: source_path.clone(),
                    })?;
                    component += 1;
                }

                let text = content_text(row.content.as_deref());
                if !text.trim().is_empty() {
                    emit(Record {
                        source: SourceKind::Hermes,
                        doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                        ts: row.timestamp_ms,
                        project: project.clone(),
                        session_id: row.session_id.clone(),
                        turn_id: base_turn + component,
                        role: "assistant".to_string(),
                        text,
                        tool_name: None,
                        tool_input: None,
                        tool_output: None,
                        links: links.clone(),
                        source_path: source_path.clone(),
                    })?;
                    component += 1;
                }

                for call in planned_calls.remove(&row.id).unwrap_or_default() {
                    let mut call_links = links.clone();
                    call_links.event_id = Some(call.id.clone());
                    call_links.parent_event_id = Some(row.id.to_string());
                    let doc_id = next_doc_id.fetch_add(1, Ordering::SeqCst);
                    let replaced = pending_tool_calls.insert(
                        call.id.clone(),
                        super::common::pending_tool_call(
                            call.name.clone(),
                            Some(call.id.clone()),
                            doc_id,
                            row.timestamp_ms,
                            Some(&call.arguments),
                            &call_links,
                            &row.session_id,
                        ),
                    );
                    if replaced.is_some() {
                        diagnostics.duplicate_tool_calls += 1;
                    }
                    emit(Record {
                        source: SourceKind::Hermes,
                        doc_id,
                        ts: row.timestamp_ms,
                        project: project.clone(),
                        session_id: row.session_id.clone(),
                        turn_id: base_turn + component,
                        role: "tool_use".to_string(),
                        text: call.arguments.clone(),
                        tool_name: call.name,
                        tool_input: Some(call.arguments),
                        tool_output: None,
                        links: call_links,
                        source_path: source_path.clone(),
                    })?;
                    component += 1;
                }
            }
            "tool" => {
                let native_call_id = row.tool_call_id.as_deref().filter(|id| !id.is_empty());
                let matched_id = native_call_id
                    .filter(|id| pending_tool_calls.contains_key(*id))
                    .map(str::to_string)
                    .or_else(|| {
                        find_pending_call(
                            &pending_tool_calls,
                            &row.session_id,
                            row.tool_name.as_deref(),
                        )
                    });
                let pending = matched_id
                    .as_ref()
                    .and_then(|id| pending_tool_calls.remove(id));
                if native_call_id.is_some() && pending.is_none() {
                    diagnostics.orphan_tool_results += 1;
                }
                let tool_name = row
                    .tool_name
                    .clone()
                    .or_else(|| pending.as_ref().and_then(|call| call.tool_name.clone()));
                let output = content_text(row.content.as_deref());
                let mut result_links = links;
                if let Some(parent_id) = matched_id.or_else(|| native_call_id.map(str::to_string)) {
                    result_links.parent_event_id = Some(parent_id.clone());
                    result_links.parent_tool_use_id = Some(parent_id);
                }
                emit(Record {
                    source: SourceKind::Hermes,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: row.timestamp_ms,
                    project,
                    session_id: row.session_id.clone(),
                    turn_id: base_turn,
                    role: "tool_result".to_string(),
                    text: output.clone(),
                    tool_name,
                    tool_input: None,
                    tool_output: Some(output),
                    links: result_links,
                    source_path: source_path.clone(),
                })?;
            }
            "system" => {}
            role => diagnostics.increment_unknown_semantic(role),
        }
    }

    Ok(IndexParseOutput {
        offset: max_message_id,
        turn_id: 0,
        pending_tool_calls,
        session_id: None,
        diagnostics,
    })
}

fn open_read_only(path: &Path) -> Result<Connection> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open Hermes store {}", path.display()))?;
    connection.busy_timeout(Duration::from_secs(1))?;
    Ok(connection)
}

fn table_columns(connection: &Connection, table: &str) -> Result<HashSet<String>> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;
    rows.collect::<std::result::Result<HashSet<_>, _>>()
        .map_err(Into::into)
}

fn optional_column(columns: &HashSet<String>, column: &str, fallback: &str) -> String {
    if columns.contains(column) {
        column.to_string()
    } else {
        fallback.to_string()
    }
}

fn load_rows(connection: &Connection, after_id: u64) -> Result<Vec<MessageRow>> {
    let message_columns = table_columns(connection, "messages")?;
    let session_columns = table_columns(connection, "sessions")?;
    let active = if message_columns.contains("active") {
        "COALESCE(m.active, 1) != 0"
    } else {
        "1"
    };
    let sql = format!(
        "SELECT m.id, m.session_id, m.role, m.content, m.tool_call_id, m.tool_calls,
                m.tool_name, m.timestamp,
                {}, {}, {}, {}
         FROM messages m
         LEFT JOIN sessions s ON s.id = m.session_id
         WHERE m.id > ?1 AND {active}
         ORDER BY m.id",
        optional_column(&message_columns, "reasoning", "NULL"),
        optional_column(&message_columns, "reasoning_content", "NULL"),
        optional_column(&session_columns, "cwd", "NULL"),
        optional_column(&session_columns, "parent_session_id", "NULL"),
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params![after_id.min(i64::MAX as u64) as i64], |row| {
        Ok(MessageRow {
            id: row.get::<_, i64>(0)?.max(0) as u64,
            session_id: row.get(1)?,
            role: row.get(2)?,
            content: row.get(3)?,
            tool_call_id: row.get(4)?,
            tool_calls: row.get(5)?,
            tool_name: row.get(6)?,
            timestamp_ms: timestamp_millis(row.get::<_, f64>(7)?),
            reasoning: row.get(8)?,
            reasoning_content: row.get(9)?,
            cwd: row.get(10)?,
            parent_session_id: row.get(11)?,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn plan_tool_calls(
    rows: &[MessageRow],
    diagnostics: &mut ParseDiagnostics,
) -> HashMap<u64, Vec<ToolCall>> {
    let mut planned = HashMap::new();
    for (index, row) in rows.iter().enumerate() {
        if row.role != "assistant" {
            continue;
        }
        let Some(raw) = row
            .tool_calls
            .as_deref()
            .filter(|raw| !raw.trim().is_empty())
        else {
            continue;
        };
        let value: Value = match serde_json::from_str(raw) {
            Ok(value) => value,
            Err(_) => {
                diagnostics.malformed_json_lines += 1;
                continue;
            }
        };
        let Some(entries) = value.as_array() else {
            diagnostics.increment_unknown_semantic("tool_calls_not_array");
            continue;
        };
        let mut calls = entries
            .iter()
            .enumerate()
            .filter_map(|(position, entry)| parse_tool_call(row.id, position, entry))
            .collect::<Vec<_>>();
        let idless_positions = calls
            .iter()
            .enumerate()
            .filter_map(|(position, call)| call.id.starts_with("hermes:").then_some(position))
            .collect::<Vec<_>>();
        if !idless_positions.is_empty() {
            let available = rows[index + 1..]
                .iter()
                .take_while(|next| next.role == "tool")
                .filter_map(|next| next.tool_call_id.clone())
                .collect::<Vec<_>>();
            if available.len() == idless_positions.len() {
                for (position, adopted) in idless_positions.into_iter().zip(available) {
                    calls[position].id = adopted;
                }
            }
        }
        if !calls.is_empty() {
            planned.insert(row.id, calls);
        }
    }
    planned
}

fn parse_tool_call(row_id: u64, position: usize, value: &Value) -> Option<ToolCall> {
    let object = value.as_object()?;
    let function = object.get("function").and_then(Value::as_object);
    let name = function
        .and_then(|function| function.get("name"))
        .or_else(|| object.get("name"))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(str::to_string);
    let native_id = object
        .get("id")
        .or_else(|| object.get("call_id"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty());
    let arguments = function
        .and_then(|function| function.get("arguments"))
        .or_else(|| object.get("arguments"))
        .map(json_text)
        .unwrap_or_default();
    Some(ToolCall {
        id: native_id
            .map(str::to_string)
            .unwrap_or_else(|| format!("hermes:{row_id}:{position}")),
        name,
        arguments,
    })
}

fn find_pending_call(
    pending: &HashMap<String, PendingToolCall>,
    session_id: &str,
    tool_name: Option<&str>,
) -> Option<String> {
    let matches = pending
        .iter()
        .filter(|(_, call)| call.session_id.as_deref() == Some(session_id))
        .filter(|(_, call)| {
            tool_name.is_none_or(|name| call.tool_name.as_deref().is_none_or(|value| value == name))
        })
        .map(|(id, _)| id.clone())
        .collect::<Vec<_>>();
    (matches.len() == 1).then(|| matches[0].clone())
}

fn row_links(row: &MessageRow) -> RecordLinks {
    RecordLinks {
        event_id: Some(row.id.to_string()),
        parent_session_id: row.parent_session_id.clone(),
        thread_source: row.parent_session_id.as_ref().map(|_| "fork".to_string()),
        conversation_kind: Some(
            if row.parent_session_id.is_some() {
                ConversationKind::Fork
            } else {
                ConversationKind::Main
            }
            .as_str()
            .to_string(),
        ),
        ..RecordLinks::default()
    }
}

fn reasoning_text(row: &MessageRow, diagnostics: &mut ParseDiagnostics) -> Option<String> {
    let text = row
        .reasoning_content
        .as_deref()
        .filter(|text| !text.trim().is_empty())
        .or_else(|| {
            row.reasoning
                .as_deref()
                .filter(|text| !text.trim().is_empty())
        })
        .map(str::trim)?;
    if is_encrypted_reasoning(text) {
        diagnostics.encrypted_reasoning_dropped += 1;
        None
    } else {
        Some(text.to_string())
    }
}

fn is_encrypted_reasoning(text: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(text) else {
        return false;
    };
    let Some(object) = value.as_object() else {
        return false;
    };
    matches!(
        object.get("type").and_then(Value::as_str),
        Some("redacted_thinking" | "encrypted_reasoning")
    ) || object.contains_key("encrypted_content")
}

fn content_text(content: Option<&str>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    let value = if let Some(encoded) = content.strip_prefix(CONTENT_JSON_PREFIX) {
        match serde_json::from_str::<Value>(encoded) {
            Ok(value) => value,
            Err(_) => return encoded.to_string(),
        }
    } else {
        return content.to_string();
    };
    match value {
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                let object = block.as_object()?;
                let block_type = object.get("type").and_then(Value::as_str).unwrap_or("");
                matches!(block_type, "text" | "input_text" | "output_text")
                    .then(|| object.get("text").and_then(Value::as_str))
                    .flatten()
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::String(text) => text,
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn json_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn timestamp_millis(value: f64) -> u64 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let millis = if value > 100_000_000_000.0 {
        value
    } else {
        value * 1_000.0
    };
    millis.round().clamp(0.0, u64::MAX as f64) as u64
}

pub(crate) fn parse_usage_file(path: &Path) -> Result<Vec<UsageEvent>> {
    let connection = open_read_only(path)?;
    let columns = table_columns(&connection, "sessions")?;
    let expression = |column: &str, fallback: &str| {
        if columns.contains(column) {
            column.to_string()
        } else {
            fallback.to_string()
        }
    };
    let sql = format!(
        "SELECT id, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
         FROM sessions
         ORDER BY started_at, id",
        if columns.contains("ended_at") {
            "COALESCE(ended_at, started_at)".to_string()
        } else {
            "started_at".to_string()
        },
        expression("cwd", "NULL"),
        expression("billing_provider", "NULL"),
        expression("model", "NULL"),
        expression("input_tokens", "0"),
        expression("cache_read_tokens", "0"),
        expression("cache_write_tokens", "0"),
        expression("output_tokens", "0"),
        expression("reasoning_tokens", "0"),
        expression("actual_cost_usd", "NULL"),
        expression("estimated_cost_usd", "NULL"),
        expression("parent_session_id", "NULL"),
    );
    let source_path: Arc<str> = Arc::from(path.to_string_lossy());
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map([], |row| {
        let session_id: String = row.get(0)?;
        let input = row.get::<_, i64>(5)?.max(0) as u64;
        let cache_read = row.get::<_, i64>(6)?.max(0) as u64;
        let cache_write = row.get::<_, i64>(7)?.max(0) as u64;
        let output = row.get::<_, i64>(8)?.max(0) as u64;
        let reasoning = row.get::<_, i64>(9)?.max(0) as u64;
        let mut tokens = TokenBuckets::disjoint(input, cache_read, cache_write, output);
        tokens.reasoning = reasoning.min(output);
        Ok(UsageEvent {
            source: "hermes",
            source_path: source_path.clone(),
            source_record_id: Some(format!("session:{session_id}")),
            session_id: Some(session_id),
            request_id: None,
            message_id: None,
            timestamp_ms: timestamp_millis(row.get::<_, f64>(1)?),
            project: row.get(2)?,
            provider: row.get(3)?,
            model: row.get(4)?,
            tokens,
            source_cost_usd: row.get::<_, Option<f64>>(10)?.or(row.get(11)?),
            dedupe_confidence: "exact",
            conservative_undercount: false,
            sidechain: row.get::<_, Option<String>>(12)?.is_some(),
            source_order: 0,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map(|events| {
            events
                .into_iter()
                .filter(|event| event.tokens.additive_total() > 0)
                .collect()
        })
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn fixture_store() -> (tempfile::TempDir, PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("state.db");
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(include_str!("../../fixtures/trajectory_parity/hermes.sql"))
            .unwrap();
        drop(connection);
        (temp, path)
    }

    #[test]
    fn fixture_projects_active_rows_tools_usage_and_opt_in_reasoning() {
        let (_temp, path) = fixture_store();
        let mut records = Vec::new();
        let parsed = parse_index_records(
            &path,
            IndexParseState::default(),
            false,
            &AtomicU64::new(1),
            |record| {
                records.push(record);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(parsed.offset, 107);
        assert!(!records.iter().any(|record| record.role == "reasoning"));
        assert!(!records.iter().any(|record| record.text.contains("rewound")));
        assert!(
            !records
                .iter()
                .any(|record| record.text.contains("ciphertext-must-never-be-indexed"))
        );
        assert!(records.iter().any(|record| {
            record.role == "tool_use"
                && record.tool_name.as_deref() == Some("terminal")
                && record.links.event_id.as_deref() == Some("call-hermes-1")
        }));
        assert!(records.iter().any(|record| {
            record.role == "tool_result"
                && record.links.parent_tool_use_id.as_deref() == Some("call-hermes-1")
        }));

        let mut reasoning = Vec::new();
        let reasoning_parse = parse_index_records(
            &path,
            IndexParseState::default(),
            true,
            &AtomicU64::new(1),
            |record| {
                reasoning.push(record);
                Ok(())
            },
        )
        .unwrap();
        assert!(reasoning.iter().any(|record| {
            record.role == "reasoning" && record.text == "Check the working directory."
        }));
        assert_eq!(reasoning_parse.diagnostics.encrypted_reasoning_dropped, 1);

        let usage = parse_usage_file(&path).unwrap();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0].tokens.total(), 175);
        assert_eq!(usage[0].source_cost_usd, Some(0.03));
        assert_eq!(usage[0].timestamp_ms, 1_783_000_000_000);
    }

    #[test]
    fn checkpoint_tracks_appends_and_rewinds_separately() {
        let (_temp, path) = fixture_store();
        let before = checkpoint(&path).unwrap();
        assert_eq!(before.max_message_id, 107);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "INSERT INTO messages(id, session_id, role, content, timestamp, active)
                 VALUES (108, 'hermes-session', 'user', 'next', 1783000011.0, 1)",
                [],
            )
            .unwrap();
        let appended = checkpoint(&path).unwrap();
        assert_eq!(appended.max_message_id, 108);
        assert_eq!(appended.generation, before.generation);

        connection
            .execute(
                "UPDATE sessions SET rewind_count = rewind_count + 1
                 WHERE id = 'hermes-session'",
                [],
            )
            .unwrap();
        let rewound = checkpoint(&path).unwrap();
        assert_ne!(rewound.generation, appended.generation);
    }

    #[test]
    fn incremental_parse_reads_only_new_active_messages() {
        let (_temp, path) = fixture_store();
        let mut records = Vec::new();
        let parsed = parse_index_records(
            &path,
            IndexParseState {
                offset: 104,
                ..IndexParseState::default()
            },
            false,
            &AtomicU64::new(1),
            |record| {
                records.push(record);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(parsed.offset, 107);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].text, "Visible multimodal text");
        assert_eq!(records[1].text, "Visible encrypted-reasoning answer");
    }
}
