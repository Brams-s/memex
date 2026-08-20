//! Stable record identity and bounded context retrieval.
//!
//! This module deliberately sits above the current Tantivy projection.  It does not require a
//! catalog migration: records are read through `SearchIndex`, while the canonical identity is
//! reconstructed from source-native fields.  A future catalog can use the same identity and
//! ordering contract without changing callers of this module.

use crate::index::SearchIndex;
use crate::types::{Record, SourceKind};
use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const CANONICAL_RECORD_ID_VERSION: &str = "rid1";

/// Maximum number of records that may be requested on either side of a context anchor.
///
/// Context is intentionally bounded at the API boundary so an accidental or hostile CLI value
/// cannot turn a neighborhood request into an unbounded transcript dump.
pub const MAX_CONTEXT_WINDOW: usize = 1_000;

/// A selector for a context anchor.  Session and source scopes are optional for all selector
/// kinds; they are especially useful for event IDs, which are commonly only unique within one
/// source/session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextSelector {
    RecordId {
        id: String,
        session_id: Option<String>,
        source: Option<SourceKind>,
    },
    DocId {
        id: u64,
        session_id: Option<String>,
        source: Option<SourceKind>,
    },
    EventId {
        id: String,
        session_id: Option<String>,
        source: Option<SourceKind>,
    },
}

impl ContextSelector {
    pub fn record_id(id: impl Into<String>) -> Self {
        Self::RecordId {
            id: id.into(),
            session_id: None,
            source: None,
        }
    }

    pub fn doc_id(id: u64) -> Self {
        Self::DocId {
            id,
            session_id: None,
            source: None,
        }
    }

    pub fn event_id(id: impl Into<String>) -> Self {
        Self::EventId {
            id: id.into(),
            session_id: None,
            source: None,
        }
    }

    pub fn with_scope(self, session_id: Option<String>, source: Option<SourceKind>) -> Self {
        match self {
            Self::RecordId { id, .. } => Self::RecordId {
                id,
                session_id,
                source,
            },
            Self::DocId { id, .. } => Self::DocId {
                id,
                session_id,
                source,
            },
            Self::EventId { id, .. } => Self::EventId {
                id,
                session_id,
                source,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextOptions {
    pub before: usize,
    pub after: usize,
    pub expand_interactions: bool,
}

impl ContextOptions {
    fn validate(self) -> Result<()> {
        if self.before > MAX_CONTEXT_WINDOW {
            bail!(
                "context before window {} exceeds maximum {}",
                self.before,
                MAX_CONTEXT_WINDOW
            );
        }
        if self.after > MAX_CONTEXT_WINDOW {
            bail!(
                "context after window {} exceeds maximum {}",
                self.after,
                MAX_CONTEXT_WINDOW
            );
        }
        Ok(())
    }
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            before: 5,
            after: 5,
            expand_interactions: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ContextRelation {
    Before,
    Anchor,
    After,
    Interaction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRecord {
    pub record_id: String,
    pub relation: ContextRelation,
    /// Signed distance from the anchor. Interaction-expanded records have zero distance because
    /// they are not selected by the linear before/after window.
    pub distance: i64,
    pub record: Record,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResult {
    pub anchor_record_id: String,
    pub records: Vec<ContextRecord>,
}

/// Compute a stable, versioned identity for a projected record.
///
/// The identity includes source/session and immutable source-native event identities, plus
/// role/tool subtype and a source-file/turn fallback when no native event identity exists. It
/// intentionally excludes optional relationship metadata, `doc_id`, text, and tool payloads:
/// local index allocation, parser enrichment, and projection content may change while the source
/// event remains the same.
pub fn canonical_record_id(record: &Record) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, "version", CANONICAL_RECORD_ID_VERSION);
    hash_field(&mut hasher, "source", record.source.storage_label());
    let has_native_event = record.links.event_id.is_some();
    let has_native_tool = record.role == "tool_use" && record.links.source_tool_use_id.is_some();
    if has_native_event || has_native_tool {
        // Event/tool IDs are source-native identities.  Do not include optional parent/thread
        // metadata here: parsers may learn to populate those fields later without changing the
        // identity of an already-indexed source event.
        hash_field(
            &mut hasher,
            "session",
            if record.session_id.is_empty() {
                &record.source_path
            } else {
                &record.session_id
            },
        );
        if let Some(event_id) = record.links.event_id.as_deref() {
            hash_field(&mut hasher, "event", event_id);
        }
        if !has_native_event && let Some(tool_id) = record.links.source_tool_use_id.as_deref() {
            hash_field(&mut hasher, "source-tool-use", tool_id);
        }
    } else {
        // Without a native event identity, a session ID alone is insufficient: imported or
        // legacy source files can reuse it. Include the source path and turn so those records
        // remain distinct and deterministic across local index rebuilds.
        hash_field(&mut hasher, "session", &record.session_id);
        hash_field(&mut hasher, "source-path", &record.source_path);
        hash_field(&mut hasher, "turn", &record.turn_id.to_string());
    }
    hash_field(&mut hasher, "role", &record.role);
    hash_field(
        &mut hasher,
        "tool",
        record.tool_name.as_deref().unwrap_or_default(),
    );
    format!("{CANONICAL_RECORD_ID_VERSION}_{:x}", hasher.finalize())
}

fn hash_field(hasher: &mut Sha256, label: &str, value: &str) {
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label.as_bytes());
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

/// Resolve an anchor and return a bounded, deterministic context neighborhood.
pub fn context_records(
    index: &SearchIndex,
    selector: &ContextSelector,
    options: ContextOptions,
) -> Result<ContextResult> {
    options.validate()?;
    let mut all_records = Vec::new();
    index.for_each_record(|record| {
        all_records.push(record);
        Ok(())
    })?;

    // A current index can contain duplicate projections from older versions.  Collapse those by
    // canonical identity before resolving an anchor, preferring the greatest local doc ID as the
    // deterministic latest projection.
    let mut unique = HashMap::<String, Record>::new();
    for record in all_records {
        let key = canonical_record_id(&record);
        match unique.get(&key) {
            Some(previous) if projection_order(previous, &record).is_ge() => {}
            _ => {
                unique.insert(key, record);
            }
        }
    }
    let mut records: Vec<Record> = unique.into_values().collect();
    records.sort_by(record_order);

    let candidates: Vec<Record> = records
        .iter()
        .filter(|record| selector_matches(record, selector))
        .cloned()
        .collect();
    let anchor = match candidates.as_slice() {
        [] => bail!("context anchor not found"),
        [anchor] => anchor.clone(),
        _ => {
            let ids = candidates
                .iter()
                .map(canonical_record_id)
                .collect::<Vec<_>>();
            return Err(anyhow!(
                "context selector is ambiguous; matching record IDs: {}",
                ids.join(", ")
            ));
        }
    };

    let anchor_id = canonical_record_id(&anchor);
    let session_records: Vec<Record> = records
        .into_iter()
        .filter(|record| {
            record.session_id == anchor.session_id
                && record.source_path == anchor.source_path
                && record.source == anchor.source
        })
        .collect();
    let anchor_position = session_records
        .iter()
        .position(|record| canonical_record_id(record) == anchor_id)
        .ok_or_else(|| anyhow!("resolved context anchor is not in its session"))?;

    let first = anchor_position.saturating_sub(options.before);
    let last = (anchor_position + options.after + 1).min(session_records.len());
    let mut selected = HashMap::<String, (Record, ContextRelation, i64)>::new();
    for (position, record) in session_records[first..last].iter().enumerate() {
        let absolute = first + position;
        let key = canonical_record_id(record);
        let (relation, distance) = if absolute == anchor_position {
            (ContextRelation::Anchor, 0)
        } else if absolute < anchor_position {
            (
                ContextRelation::Before,
                -((anchor_position - absolute) as i64),
            )
        } else {
            (ContextRelation::After, (absolute - anchor_position) as i64)
        };
        selected.insert(key, (record.clone(), relation, distance));
    }

    if options.expand_interactions {
        let mut interaction_ids = identifier_set(&anchor);
        let mut changed = true;
        while changed {
            changed = false;
            for record in &session_records {
                if !identifier_set(record)
                    .iter()
                    .any(|identifier| interaction_ids.contains(identifier))
                {
                    continue;
                }
                for identifier in identifier_set(record) {
                    changed |= interaction_ids.insert(identifier);
                }
            }
        }
        for record in &session_records {
            let key = canonical_record_id(record);
            if selected.contains_key(&key)
                || !identifier_set(record)
                    .iter()
                    .any(|identifier| interaction_ids.contains(identifier))
            {
                continue;
            }
            selected.insert(key, (record.clone(), ContextRelation::Interaction, 0));
        }
    }

    let mut selected: Vec<(Record, ContextRelation, i64)> = selected.into_values().collect();
    selected.sort_by(|left, right| record_order(&left.0, &right.0));
    let records = selected
        .into_iter()
        .map(|(record, relation, distance)| ContextRecord {
            record_id: canonical_record_id(&record),
            relation,
            distance,
            record,
        })
        .collect();
    Ok(ContextResult {
        anchor_record_id: anchor_id,
        records,
    })
}

fn selector_matches(record: &Record, selector: &ContextSelector) -> bool {
    let (session_id, source, matches) = match selector {
        ContextSelector::RecordId {
            id,
            session_id,
            source,
        } => (session_id, source, canonical_record_id(record) == *id),
        ContextSelector::DocId {
            id,
            session_id,
            source,
        } => (session_id, source, record.doc_id == *id),
        ContextSelector::EventId {
            id,
            session_id,
            source,
        } => (
            session_id,
            source,
            record.links.event_id.as_deref() == Some(id.as_str()),
        ),
    };
    matches
        && session_id
            .as_deref()
            .is_none_or(|session_id| record.session_id == session_id)
        && source.is_none_or(|source| record.source == source)
}

fn record_order(left: &Record, right: &Record) -> std::cmp::Ordering {
    left.turn_id
        .cmp(&right.turn_id)
        .then_with(|| left.ts.cmp(&right.ts))
        .then_with(|| left.doc_id.cmp(&right.doc_id))
        .then_with(|| canonical_record_id(left).cmp(&canonical_record_id(right)))
}

fn projection_order(left: &Record, right: &Record) -> std::cmp::Ordering {
    left.ts
        .cmp(&right.ts)
        .then_with(|| left.turn_id.cmp(&right.turn_id))
        .then_with(|| left.doc_id.cmp(&right.doc_id))
}

fn identifier_set(record: &Record) -> HashSet<String> {
    let identifiers: Vec<String> = [
        record.links.event_id.as_deref(),
        record.links.parent_event_id.as_deref(),
        record.links.logical_parent_event_id.as_deref(),
        record.links.parent_tool_use_id.as_deref(),
        record.links.source_tool_use_id.as_deref(),
        record.links.source_tool_assistant_uuid.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(str::to_string)
    .collect();
    let mut result: HashSet<String> = identifiers.iter().cloned().collect();

    // Conversation labels are not identifiers: using one by itself would join every record in
    // a session. When both labels are present, pair them with a real event/tool identifier so
    // they only strengthen an existing interaction edge.
    if let (Some(thread_source), Some(conversation_kind)) = (
        record.links.thread_source.as_deref(),
        record.links.conversation_kind.as_deref(),
    ) {
        for identifier in identifiers {
            result.insert(format!(
                "__interaction_scope:{thread_source}:{conversation_kind}:{identifier}"
            ));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RecordLinks, SourceKind};

    fn record(doc_id: u64, turn_id: u32, text: &str) -> Record {
        Record {
            source: SourceKind::Codex,
            doc_id,
            ts: turn_id as u64 * 10,
            project: "memex".to_string(),
            session_id: "session".to_string(),
            turn_id,
            role: "assistant".to_string(),
            text: text.to_string(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links: RecordLinks::default(),
            source_path: "/tmp/session.jsonl".to_string(),
        }
    }

    fn indexed(records: &[Record]) -> SearchIndex {
        let tmp = tempfile::tempdir().expect("tempdir");
        // Leak the directory for the duration of this test's index handle.  SearchIndex owns the
        // mmap but not the parent TempDir, and tests need the path to outlive this helper.
        let path = tmp.keep();
        let index = SearchIndex::open_or_create(&path).expect("open index");
        let mut writer = index.writer().expect("writer");
        for record in records {
            index.add_record(&mut writer, record).expect("add record");
        }
        writer.commit().expect("commit");
        index
    }

    #[test]
    fn canonical_id_ignores_doc_id_path_when_session_and_event_are_stable() {
        let mut first = record(1, 2, "old body");
        first.links.event_id = Some("event-1".to_string());
        let mut rebuilt = first.clone();
        rebuilt.doc_id = 900;
        rebuilt.turn_id = 99;
        rebuilt.source_path = "/moved/session.jsonl".to_string();
        rebuilt.text = "new body".to_string();
        assert_eq!(canonical_record_id(&first), canonical_record_id(&rebuilt));
        assert!(canonical_record_id(&first).starts_with("rid1_"));
    }

    #[test]
    fn canonical_id_uses_path_only_when_session_is_empty() {
        let mut first = record(1, 2, "same");
        first.session_id.clear();
        let mut moved = first.clone();
        moved.source_path = "/tmp/other.jsonl".to_string();
        assert_ne!(canonical_record_id(&first), canonical_record_id(&moved));
    }

    #[test]
    fn canonical_id_distinguishes_repeated_parent_only_updates_by_turn() {
        let mut first = record(1, 2, "first");
        first.role = "tool_result".to_string();
        first.links.parent_tool_use_id = Some("call-1".to_string());
        let mut second = first.clone();
        second.doc_id = 2;
        second.turn_id = 3;
        second.text = "second".to_string();
        assert_ne!(canonical_record_id(&first), canonical_record_id(&second));
    }

    #[test]
    fn canonical_id_includes_link_identity_without_content() {
        let mut first = record(1, 2, "body-a");
        first.links.event_id = Some("event-a".to_string());
        let mut changed = first.clone();
        changed.text = "body-b".to_string();
        assert_eq!(canonical_record_id(&first), canonical_record_id(&changed));
        changed.links.event_id = Some("event-b".to_string());
        assert_ne!(canonical_record_id(&first), canonical_record_id(&changed));
    }

    #[test]
    fn canonical_id_ignores_optional_link_enrichment_for_native_events() {
        let mut first = record(1, 2, "body");
        first.links.event_id = Some("event-a".to_string());
        let mut enriched = first.clone();
        enriched.links.parent_event_id = Some("parent".to_string());
        enriched.links.parent_session_id = Some("parent-session".to_string());
        enriched.links.thread_source = Some("subagent".to_string());
        enriched.links.conversation_kind = Some("subagent".to_string());
        assert_eq!(canonical_record_id(&first), canonical_record_id(&enriched));
    }

    #[test]
    fn canonical_id_fallback_distinguishes_source_files_with_same_session() {
        let first = record(1, 2, "body");
        let mut moved = first.clone();
        moved.source_path = "/tmp/other.jsonl".to_string();
        assert_ne!(canonical_record_id(&first), canonical_record_id(&moved));
    }

    #[test]
    fn context_selects_bounded_ordered_window_and_includes_anchor() {
        let records: Vec<Record> = (1..=5)
            .map(|turn| record(turn as u64, turn, &format!("r{turn}")))
            .collect();
        let index = indexed(&records);
        let result = context_records(
            &index,
            &ContextSelector::doc_id(3),
            ContextOptions {
                before: 1,
                after: 1,
                expand_interactions: false,
            },
        )
        .expect("context");
        assert_eq!(
            result
                .records
                .iter()
                .map(|item| item.record.turn_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert_eq!(result.records[0].relation, ContextRelation::Before);
        assert_eq!(result.records[1].relation, ContextRelation::Anchor);
        assert_eq!(result.records[2].relation, ContextRelation::After);
        assert_eq!(result.anchor_record_id, result.records[1].record_id);
    }

    #[test]
    fn context_event_id_requires_scope_when_ambiguous() {
        let mut first = record(1, 1, "first");
        first.links.event_id = Some("same".to_string());
        let mut second = record(2, 1, "second");
        second.session_id = "other".to_string();
        second.source_path = "/tmp/other.jsonl".to_string();
        second.links.event_id = Some("same".to_string());
        let index = indexed(&[first, second]);
        let error = context_records(
            &index,
            &ContextSelector::event_id("same"),
            ContextOptions::default(),
        )
        .expect_err("ambiguous event ID");
        assert!(error.to_string().contains("ambiguous"));

        let scoped = ContextSelector::event_id("same")
            .with_scope(Some("other".to_string()), Some(SourceKind::Codex));
        let result =
            context_records(&index, &scoped, ContextOptions::default()).expect("scoped context");
        assert_eq!(result.records[0].record.session_id, "other");
    }

    #[test]
    fn context_neighborhood_isolated_by_source() {
        let mut anchor = record(1, 1, "codex");
        anchor.links.event_id = Some("same".to_string());
        let mut other = anchor.clone();
        other.doc_id = 2;
        other.source = SourceKind::Claude;
        other.text = "claude".to_string();
        let index = indexed(&[anchor, other]);
        let result = context_records(
            &index,
            &ContextSelector::event_id("same").with_scope(None, Some(SourceKind::Codex)),
            ContextOptions {
                before: 1,
                after: 1,
                expand_interactions: false,
            },
        )
        .expect("source-scoped context");
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.records[0].record.source, SourceKind::Codex);
    }

    #[test]
    fn context_rejects_unbounded_window() {
        let index = indexed(&[]);
        let error = context_records(
            &index,
            &ContextSelector::doc_id(1),
            ContextOptions {
                before: MAX_CONTEXT_WINDOW + 1,
                after: 0,
                expand_interactions: false,
            },
        )
        .expect_err("oversized context window");
        assert!(error.to_string().contains("exceeds maximum"));
    }

    #[test]
    fn context_dedup_prefers_latest_timestamp_before_doc_id() {
        let mut old = record(9, 2, "old");
        old.ts = 10;
        let mut latest = old.clone();
        latest.doc_id = 1;
        latest.ts = 20;
        let id = canonical_record_id(&old);
        let index = indexed(&[old, latest]);
        let result = context_records(
            &index,
            &ContextSelector::record_id(id),
            ContextOptions::default(),
        )
        .expect("deduplicated context");
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.records[0].record.text, "old");
        assert_eq!(result.records[0].record.ts, 20);
    }

    #[test]
    fn interaction_expansion_is_transitive_and_stays_in_session_path() {
        let mut anchor = record(1, 1, "assistant");
        anchor.links.event_id = Some("assistant".to_string());
        let mut call = record(2, 2, "call");
        call.role = "tool_use".to_string();
        call.links.event_id = Some("call".to_string());
        call.links.parent_event_id = Some("assistant".to_string());
        let mut result = record(3, 3, "result");
        result.role = "tool_result".to_string();
        result.links.parent_tool_use_id = Some("call".to_string());
        let mut unrelated = record(4, 4, "unrelated");
        unrelated.links.event_id = Some("different".to_string());
        let mut other_path = record(5, 5, "other path");
        other_path.source_path = "/tmp/other.jsonl".to_string();
        other_path.links.parent_event_id = Some("assistant".to_string());
        let index = indexed(&[anchor, call, result, unrelated, other_path]);

        let result = context_records(
            &index,
            &ContextSelector::event_id("assistant"),
            ContextOptions {
                before: 0,
                after: 0,
                expand_interactions: true,
            },
        )
        .expect("expanded context");
        assert_eq!(
            result
                .records
                .iter()
                .map(|item| item.record.text.as_str())
                .collect::<Vec<_>>(),
            vec!["assistant", "call", "result"]
        );
        assert!(
            result
                .records
                .iter()
                .all(|item| item.record.text != "other path")
        );
        assert!(
            result
                .records
                .iter()
                .filter(|item| item.record.text == "call" || item.record.text == "result")
                .all(|item| item.relation == ContextRelation::Interaction)
        );
    }

    #[test]
    fn interaction_expansion_deduplicates_window_members() {
        let mut anchor = record(1, 1, "anchor");
        anchor.links.event_id = Some("a".to_string());
        let mut linked = record(2, 2, "linked");
        linked.links.parent_event_id = Some("a".to_string());
        let index = indexed(&[anchor, linked]);
        let result = context_records(
            &index,
            &ContextSelector::event_id("a"),
            ContextOptions {
                before: 0,
                after: 1,
                expand_interactions: true,
            },
        )
        .expect("context");
        assert_eq!(result.records.len(), 2);
        assert_eq!(
            result
                .records
                .iter()
                .filter(|item| item.record.text == "linked")
                .count(),
            1
        );
    }

    #[test]
    fn interaction_expansion_does_not_join_on_parent_session_id() {
        let mut anchor = record(1, 1, "anchor");
        anchor.links.event_id = Some("anchor-event".to_string());
        anchor.links.parent_session_id = Some("parent-session".to_string());

        let mut linked = record(2, 2, "linked");
        linked.links.parent_event_id = Some("anchor-event".to_string());
        linked.links.parent_session_id = Some("parent-session".to_string());

        let mut unrelated = record(3, 3, "unrelated");
        unrelated.links.event_id = Some("unrelated-event".to_string());
        unrelated.links.parent_session_id = Some("parent-session".to_string());

        let index = indexed(&[anchor, linked, unrelated]);
        let result = context_records(
            &index,
            &ContextSelector::event_id("anchor-event"),
            ContextOptions {
                before: 0,
                after: 0,
                expand_interactions: true,
            },
        )
        .expect("context");

        assert_eq!(
            result
                .records
                .iter()
                .map(|item| item.record.text.as_str())
                .collect::<Vec<_>>(),
            vec!["anchor", "linked"]
        );
    }
}
