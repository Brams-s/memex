use crate::types::{Record, RecordLinks, SourceFilter};
use anyhow::{Context, Result, anyhow, bail};
use std::fs;
use std::io::{self, Write};
use std::ops::Bound;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use tantivy::collector::{Count, TopDocs};
use tantivy::directory::error::{DeleteError, LockError, OpenReadError, OpenWriteError};
use tantivy::directory::{
    Directory, DirectoryLock, FileHandle, Lock, MmapDirectory, WatchCallback, WatchHandle, WritePtr,
};
use tantivy::query::{AllQuery, BooleanQuery, Occur, Query, RangeQuery, TermQuery};
use tantivy::schema::Value;
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, STORED, STRING, Schema, SchemaBuilder, TEXT,
    TextFieldIndexing, TextOptions,
};
use tantivy::{Index, IndexReader, IndexWriter, Order, ReloadPolicy, TantivyDocument, Term};

#[derive(Clone)]
pub struct IndexFields {
    pub doc_id: Field,
    pub ts: Field,
    pub project: Field,
    pub session_id: Field,
    pub turn_id: Field,
    pub role: Field,
    pub text: Field,
    pub source: Option<Field>,
    pub tool_name: Field,
    pub tool_input: Field,
    pub tool_output: Field,
    pub event_id: Field,
    pub parent_event_id: Field,
    pub logical_parent_event_id: Field,
    pub parent_session_id: Field,
    pub thread_source: Field,
    pub conversation_kind: Field,
    pub parent_tool_use_id: Field,
    pub source_tool_use_id: Field,
    pub source_tool_assistant_uuid: Field,
    pub source_path: Field,
}

#[derive(Clone)]
pub struct SearchIndex {
    pub index: Index,
    pub fields: IndexFields,
    writable: bool,
    pending_generation: Option<Arc<PendingGeneration>>,
}

const GENERATIONS_DIR: &str = "generations";
const CURRENT_FILE: &str = "CURRENT";

#[derive(Debug)]
struct PendingGeneration {
    index_root: PathBuf,
    staging_dir: PathBuf,
    generation_name: String,
    replaces_published_generation: bool,
    published: AtomicBool,
}

impl Drop for PendingGeneration {
    fn drop(&mut self) {
        if !self.published.load(AtomicOrdering::Acquire) {
            let _ = fs::remove_dir_all(&self.staging_dir);
        }
    }
}

/// Tantivy normally takes a metadata lock every time it opens segment readers so its own
/// garbage collector cannot remove a segment concurrently. Published generations are immutable,
/// so Tantivy cannot remove their segments and the lock is unnecessary for sealed readers.
#[derive(Clone, Debug)]
struct SealedDirectory(MmapDirectory);

impl Directory for SealedDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.0.get_file_handle(path)
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.0.delete(path)
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.0.exists(path)
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        self.0.open_write(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.0.atomic_read(path)
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.0.atomic_write(path, data)
    }

    fn sync_directory(&self) -> io::Result<()> {
        self.0.sync_directory()
    }

    fn acquire_lock(&self, _lock: &Lock) -> Result<DirectoryLock, LockError> {
        Ok(DirectoryLock::from(Box::new(())))
    }

    fn watch(&self, callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.0.watch(callback)
    }
}

#[derive(Debug, Clone)]
pub struct QueryOptions {
    pub query: String,
    pub project: Option<String>,
    pub role: Option<String>,
    pub tool: Option<String>,
    pub session_id: Option<String>,
    pub source: Option<crate::types::SourceFilter>,
    pub since: Option<u64>,
    pub until: Option<u64>,
    pub limit: usize,
}

impl SearchIndex {
    pub fn exists(dir: &Path) -> bool {
        resolve_current_generation(dir)
            .is_some_and(|generation| generation.join("meta.json").exists())
            || dir.join("meta.json").exists()
    }

    pub fn open_or_create(dir: &Path) -> Result<Self> {
        if let Some(generation) = resolve_current_generation(dir) {
            return open_sealed_generation(&generation);
        }
        Self::open_or_create_with_policy(dir, StaleSchemaPolicy::Error)
    }

    pub fn open_or_create_for_ingest(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let generations = dir.join(GENERATIONS_DIR);
        fs::create_dir_all(&generations)?;
        let generation_name = new_generation_name();
        let staging_dir = generations.join(format!(".{generation_name}.tmp"));

        let current = resolve_current_generation(dir);
        if let Some(current) = &current {
            clone_generation(current, &staging_dir)?;
        } else if dir.join("meta.json").exists() {
            clone_generation(dir, &staging_dir)?;
        } else {
            fs::create_dir_all(&staging_dir)?;
        }

        let mut index =
            Self::open_or_create_with_policy(&staging_dir, StaleSchemaPolicy::Recreate)?;
        index.pending_generation = Some(Arc::new(PendingGeneration {
            index_root: dir.to_path_buf(),
            staging_dir,
            generation_name,
            replaces_published_generation: current.is_some(),
            published: AtomicBool::new(false),
        }));
        Ok(index)
    }

    fn open_or_create_with_policy(
        dir: &Path,
        stale_schema_policy: StaleSchemaPolicy,
    ) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let meta_path = dir.join("meta.json");
        if meta_path.exists() {
            let index = Index::open_in_dir(dir)?;
            if !schema_is_current(&index.schema()) {
                return match stale_schema_policy {
                    StaleSchemaPolicy::Error => Err(stale_schema_error(dir)),
                    StaleSchemaPolicy::Recreate => {
                        drop(index);
                        recreate_index_dir(dir)
                    }
                };
            }
            let fields = load_fields(index.schema())?;
            Ok(Self {
                index,
                fields,
                writable: true,
                pending_generation: None,
            })
        } else {
            create_index_in_dir(dir)
        }
    }

    pub fn writer(&self) -> Result<IndexWriter> {
        if !self.writable {
            bail!("cannot create a writer for a sealed index generation");
        }
        Ok(self.index.writer(256_000_000)?)
    }

    pub fn reader(&self) -> Result<IndexReader> {
        Ok(self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?)
    }

    pub(crate) fn publish_generation(&self) -> Result<()> {
        let Some(pending) = &self.pending_generation else {
            return Ok(());
        };
        if pending.published.load(AtomicOrdering::Acquire) {
            return Ok(());
        }

        let final_dir = pending
            .index_root
            .join(GENERATIONS_DIR)
            .join(&pending.generation_name);
        if pending.staging_dir.exists() {
            fs::rename(&pending.staging_dir, &final_dir)
                .with_context(|| format!("publish index generation {}", pending.generation_name))?;
        } else if !final_dir.exists() {
            bail!(
                "index generation {} has neither staging nor published data",
                pending.generation_name
            );
        }
        sync_directory(&pending.index_root.join(GENERATIONS_DIR))?;
        atomic_write_current(&pending.index_root, &pending.generation_name)?;
        pending.published.store(true, AtomicOrdering::Release);
        Ok(())
    }

    pub(crate) fn publish_generation_if_uninitialized(&self) -> Result<()> {
        if self
            .pending_generation
            .as_ref()
            .is_some_and(|pending| !pending.replaces_published_generation)
        {
            self.publish_generation()?;
        }
        Ok(())
    }

    pub fn delete_by_source_path(&self, writer: &mut IndexWriter, path: &str) {
        let term = Term::from_field_text(self.fields.source_path, path);
        writer.delete_term(term);
    }

    pub fn add_record(&self, writer: &mut IndexWriter, record: &Record) -> Result<()> {
        let mut doc = TantivyDocument::default();
        doc.add_u64(self.fields.doc_id, record.doc_id);
        doc.add_u64(self.fields.ts, record.ts);
        doc.add_text(self.fields.project, &record.project);
        doc.add_text(self.fields.session_id, &record.session_id);
        doc.add_u64(self.fields.turn_id, record.turn_id as u64);
        doc.add_text(self.fields.role, &record.role);
        doc.add_text(self.fields.text, &record.text);
        if let Some(field) = self.fields.source {
            doc.add_text(field, record.source.storage_label());
        }
        if let Some(tool_name) = &record.tool_name {
            doc.add_text(self.fields.tool_name, tool_name);
        }
        if let Some(tool_input) = &record.tool_input {
            doc.add_text(self.fields.tool_input, tool_input);
        }
        if let Some(tool_output) = &record.tool_output {
            doc.add_text(self.fields.tool_output, tool_output);
        }
        add_optional_text(&mut doc, self.fields.event_id, &record.links.event_id);
        add_optional_text(
            &mut doc,
            self.fields.parent_event_id,
            &record.links.parent_event_id,
        );
        add_optional_text(
            &mut doc,
            self.fields.logical_parent_event_id,
            &record.links.logical_parent_event_id,
        );
        add_optional_text(
            &mut doc,
            self.fields.parent_session_id,
            &record.links.parent_session_id,
        );
        add_optional_text(
            &mut doc,
            self.fields.thread_source,
            &record.links.thread_source,
        );
        add_optional_text(
            &mut doc,
            self.fields.conversation_kind,
            &record.links.conversation_kind,
        );
        add_optional_text(
            &mut doc,
            self.fields.parent_tool_use_id,
            &record.links.parent_tool_use_id,
        );
        add_optional_text(
            &mut doc,
            self.fields.source_tool_use_id,
            &record.links.source_tool_use_id,
        );
        add_optional_text(
            &mut doc,
            self.fields.source_tool_assistant_uuid,
            &record.links.source_tool_assistant_uuid,
        );
        doc.add_text(self.fields.source_path, &record.source_path);
        writer.add_document(doc)?;
        Ok(())
    }

    pub fn get_by_doc_id(&self, doc_id: u64) -> Result<Option<Record>> {
        let reader = self.reader()?;
        let searcher = reader.searcher();
        let term = Term::from_field_u64(self.fields.doc_id, doc_id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let top = searcher.search(&query, &TopDocs::with_limit(1))?;
        let Some((_, addr)) = top.first() else {
            return Ok(None);
        };
        let doc = searcher.doc::<TantivyDocument>(*addr)?;
        Ok(Some(record_from_doc(&self.fields, &doc)))
    }

    pub fn search(&self, options: &QueryOptions) -> Result<Vec<(f32, Record)>> {
        let reader = self.reader()?;
        let searcher = reader.searcher();
        let query = build_query(&self.fields, options, &self.index)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(options.limit))?;
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, addr) in top_docs {
            let doc = searcher.doc::<TantivyDocument>(addr)?;
            results.push((score, record_from_doc(&self.fields, &doc)));
        }
        Ok(results)
    }

    pub fn records_by_session_id(&self, session_id: &str) -> Result<Vec<Record>> {
        let reader = self.reader()?;
        let searcher = reader.searcher();
        let term = Term::from_field_text(self.fields.session_id, session_id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let limit = searcher.num_docs() as usize;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
        let mut records = Vec::with_capacity(top_docs.len());
        for (_score, addr) in top_docs {
            let doc = searcher.doc::<TantivyDocument>(addr)?;
            records.push(record_from_doc(&self.fields, &doc));
        }
        Ok(records)
    }

    pub fn records_by_session_id_page(
        &self,
        session_id: &str,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<Record>, usize)> {
        let reader = self.reader()?;
        let searcher = reader.searcher();
        let term = Term::from_field_text(self.fields.session_id, session_id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let total = searcher.search(&query, &Count)?;
        if offset >= total {
            return Ok((Vec::new(), total));
        }
        let page_limit = limit.max(1).min(total - offset);
        let collector = TopDocs::with_limit(page_limit)
            .and_offset(offset)
            .order_by_fast_field::<u64>("turn_id", Order::Asc);
        let top_docs: Vec<(u64, tantivy::DocAddress)> = searcher.search(&query, &collector)?;
        let mut records = Vec::with_capacity(top_docs.len());
        for (_turn_id, addr) in top_docs {
            let doc = searcher.doc::<TantivyDocument>(addr)?;
            records.push(record_from_doc(&self.fields, &doc));
        }
        records.sort_by(|a, b| {
            a.turn_id
                .cmp(&b.turn_id)
                .then_with(|| a.ts.cmp(&b.ts))
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        Ok((records, total))
    }

    pub fn recent_records(&self, limit: usize) -> Result<Vec<Record>> {
        self.recent_records_filtered(limit, None, None)
    }

    pub fn recent_records_for_source(
        &self,
        limit: usize,
        source: Option<SourceFilter>,
    ) -> Result<Vec<Record>> {
        self.recent_records_filtered(limit, source, None)
    }

    pub fn recent_records_filtered(
        &self,
        limit: usize,
        source: Option<SourceFilter>,
        project: Option<&str>,
    ) -> Result<Vec<Record>> {
        let reader = self.reader()?;
        let searcher = reader.searcher();
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        if let Some(source) = source
            && let Some(field) = self.fields.source
        {
            let terms = source
                .storage_labels()
                .iter()
                .map(|label| {
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(
                            Term::from_field_text(field, label),
                            IndexRecordOption::Basic,
                        )) as Box<dyn Query>,
                    )
                })
                .collect();
            clauses.push((Occur::Must, Box::new(BooleanQuery::new(terms))));
        }
        if let Some(project) = project {
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(
                    Term::from_field_text(self.fields.project, project),
                    IndexRecordOption::Basic,
                )),
            ));
        }
        let query: Box<dyn Query> = if clauses.is_empty() {
            Box::new(AllQuery)
        } else {
            Box::new(BooleanQuery::new(clauses))
        };
        let collector =
            TopDocs::with_limit(limit.max(1)).order_by_fast_field::<u64>("ts", Order::Desc);
        let top_docs: Vec<(u64, tantivy::DocAddress)> =
            searcher.search(query.as_ref(), &collector)?;
        let mut records = Vec::with_capacity(top_docs.len());
        for (_ts, addr) in top_docs {
            let doc = searcher.doc::<TantivyDocument>(addr)?;
            records.push(record_from_doc(&self.fields, &doc));
        }
        Ok(records)
    }

    pub fn doc_count(&self) -> Result<usize> {
        let reader = self.reader()?;
        Ok(reader.searcher().num_docs() as usize)
    }

    pub fn for_each_record<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(Record) -> Result<()>,
    {
        let reader = self.reader()?;
        let searcher = reader.searcher();
        for segment_reader in searcher.segment_readers() {
            let store = segment_reader.get_store_reader(0)?;
            for doc in store.iter::<TantivyDocument>(segment_reader.alive_bitset()) {
                let doc = doc?;
                let record = record_from_doc(&self.fields, &doc);
                f(record)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum StaleSchemaPolicy {
    Error,
    Recreate,
}

fn stale_schema_error(dir: &Path) -> anyhow::Error {
    anyhow!(
        "index schema at {} is stale; run `memex index` or `memex reindex` to rebuild it",
        dir.display()
    )
}

fn recreate_index_dir(dir: &Path) -> Result<SearchIndex> {
    std::fs::remove_dir_all(dir)?;
    std::fs::create_dir_all(dir)?;
    create_index_in_dir(dir)
}

fn create_index_in_dir(dir: &Path) -> Result<SearchIndex> {
    let schema = build_schema()?;
    let index = Index::create_in_dir(dir, schema.clone())?;
    let fields = load_fields(schema)?;
    Ok(SearchIndex {
        index,
        fields,
        writable: true,
        pending_generation: None,
    })
}

fn open_sealed_generation(dir: &Path) -> Result<SearchIndex> {
    let directory = MmapDirectory::open(dir)
        .with_context(|| format!("open sealed index generation {}", dir.display()))?;
    let index = Index::open(SealedDirectory(directory))?;
    if !schema_is_current(&index.schema()) {
        return Err(stale_schema_error(dir));
    }
    let fields = load_fields(index.schema())?;
    Ok(SearchIndex {
        index,
        fields,
        writable: false,
        pending_generation: None,
    })
}

fn resolve_current_generation(index_root: &Path) -> Option<PathBuf> {
    let name = fs::read_to_string(index_root.join(CURRENT_FILE)).ok()?;
    let name = name.trim();
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return None;
    }
    Some(index_root.join(GENERATIONS_DIR).join(name))
}

fn new_generation_name() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{nanos:032x}-{:08x}", std::process::id())
}

fn clone_generation(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name_text = name.to_string_lossy();
        if name_text == ".tantivy-meta.lock" || name_text == ".tantivy-writer.lock" {
            continue;
        }
        let target = destination.join(&name);
        if should_copy_generation_file(&name_text) || fs::hard_link(entry.path(), &target).is_err()
        {
            fs::copy(entry.path(), &target).with_context(|| {
                format!(
                    "copy index generation file {} to {}",
                    entry.path().display(),
                    target.display()
                )
            })?;
        }
    }
    Ok(())
}

fn should_copy_generation_file(name: &str) -> bool {
    matches!(name, "meta.json" | ".managed.json")
}

fn atomic_write_current(index_root: &Path, generation_name: &str) -> Result<()> {
    let mut temp = tempfile::NamedTempFile::new_in(index_root)?;
    temp.write_all(format!("{generation_name}\n").as_bytes())?;
    temp.as_file_mut().sync_all()?;
    temp.persist(index_root.join(CURRENT_FILE))?;
    sync_directory(index_root)?;
    Ok(())
}

#[cfg(unix)]
fn sync_directory(dir: &Path) -> io::Result<()> {
    use std::fs::File;
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_dir: &Path) -> io::Result<()> {
    Ok(())
}

fn build_schema() -> Result<Schema> {
    let mut builder = SchemaBuilder::default();

    builder.add_u64_field("doc_id", INDEXED | STORED | FAST);
    builder.add_u64_field("ts", INDEXED | STORED | FAST);
    builder.add_text_field("project", STRING | STORED);
    builder.add_text_field("session_id", STRING | STORED);
    builder.add_u64_field("turn_id", INDEXED | STORED | FAST);
    builder.add_text_field("role", STRING | STORED);
    builder.add_text_field("source", STRING | STORED);

    let text_indexing = TextFieldIndexing::default()
        .set_tokenizer("default")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text_options = TextOptions::default()
        .set_indexing_options(text_indexing)
        .set_stored();
    builder.add_text_field("text", text_options);

    builder.add_text_field("tool_name", STRING | STORED);
    builder.add_text_field("tool_input", TEXT | STORED);
    builder.add_text_field("tool_output", TEXT | STORED);
    builder.add_text_field("event_id", STRING | STORED);
    builder.add_text_field("parent_event_id", STRING | STORED);
    builder.add_text_field("logical_parent_event_id", STRING | STORED);
    builder.add_text_field("parent_session_id", STRING | STORED);
    builder.add_text_field("thread_source", STRING | STORED);
    builder.add_text_field("conversation_kind", STRING | STORED);
    builder.add_text_field("parent_tool_use_id", STRING | STORED);
    builder.add_text_field("source_tool_use_id", STRING | STORED);
    builder.add_text_field("source_tool_assistant_uuid", STRING | STORED);
    builder.add_text_field("source_path", STRING | STORED);

    Ok(builder.build())
}

fn schema_is_current(schema: &Schema) -> bool {
    [
        "doc_id",
        "ts",
        "project",
        "session_id",
        "turn_id",
        "role",
        "source",
        "text",
        "tool_name",
        "tool_input",
        "tool_output",
        "event_id",
        "parent_event_id",
        "logical_parent_event_id",
        "parent_session_id",
        "thread_source",
        "conversation_kind",
        "parent_tool_use_id",
        "source_tool_use_id",
        "source_tool_assistant_uuid",
        "source_path",
    ]
    .into_iter()
    .all(|field| schema.get_field(field).is_ok())
}

fn load_fields(schema: Schema) -> Result<IndexFields> {
    let get = |name: &str| {
        schema
            .get_field(name)
            .map_err(|_| anyhow!(format!("missing field {name}")))
    };
    Ok(IndexFields {
        doc_id: get("doc_id")?,
        ts: get("ts")?,
        project: get("project")?,
        session_id: get("session_id")?,
        turn_id: get("turn_id")?,
        role: get("role")?,
        text: get("text")?,
        source: schema.get_field("source").ok(),
        tool_name: get("tool_name")?,
        tool_input: get("tool_input")?,
        tool_output: get("tool_output")?,
        event_id: get("event_id")?,
        parent_event_id: get("parent_event_id")?,
        logical_parent_event_id: get("logical_parent_event_id")?,
        parent_session_id: get("parent_session_id")?,
        thread_source: get("thread_source")?,
        conversation_kind: get("conversation_kind")?,
        parent_tool_use_id: get("parent_tool_use_id")?,
        source_tool_use_id: get("source_tool_use_id")?,
        source_tool_assistant_uuid: get("source_tool_assistant_uuid")?,
        source_path: get("source_path")?,
    })
}

fn build_query(
    fields: &IndexFields,
    options: &QueryOptions,
    index: &Index,
) -> Result<Box<dyn Query>> {
    let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    if options.query.trim().is_empty() {
        clauses.push((Occur::Must, Box::new(AllQuery)));
    } else {
        let parser = tantivy::query::QueryParser::for_index(index, vec![fields.text]);
        let text_query = parser.parse_query(&options.query)?;
        clauses.push((Occur::Must, text_query));
    }

    if let Some(project) = &options.project {
        let term = Term::from_field_text(fields.project, project);
        clauses.push((
            Occur::Must,
            Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
        ));
    }

    if let Some(role) = &options.role {
        let term = Term::from_field_text(fields.role, role);
        clauses.push((
            Occur::Must,
            Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
        ));
    }

    if let Some(tool) = &options.tool {
        let term = Term::from_field_text(fields.tool_name, tool);
        clauses.push((
            Occur::Must,
            Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
        ));
    }

    if let Some(source) = options.source
        && let Some(field) = fields.source
    {
        let source_terms = source
            .storage_labels()
            .iter()
            .map(|label| {
                (
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(field, label),
                        IndexRecordOption::Basic,
                    )) as Box<dyn Query>,
                )
            })
            .collect::<Vec<_>>();
        clauses.push((Occur::Must, Box::new(BooleanQuery::new(source_terms))));
    }

    if let Some(session_id) = &options.session_id {
        let term = Term::from_field_text(fields.session_id, session_id);
        clauses.push((
            Occur::Must,
            Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
        ));
    }

    if options.since.is_some() || options.until.is_some() {
        let start = options.since.unwrap_or(0);
        let end = options.until.unwrap_or(u64::MAX);
        let range = RangeQuery::new_u64_bounds(
            "ts".to_string(),
            Bound::Included(start),
            Bound::Included(end),
        );
        clauses.push((Occur::Must, Box::new(range)));
    }

    Ok(Box::new(BooleanQuery::new(clauses)))
}

fn record_from_doc(fields: &IndexFields, doc: &TantivyDocument) -> Record {
    let get_str = |field: Field| -> Option<String> {
        doc.get_first(field)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };

    let get_u64 =
        |field: Field| -> u64 { doc.get_first(field).and_then(|v| v.as_u64()).unwrap_or(0) };

    let source_path = get_str(fields.source_path).unwrap_or_default();
    let source = fields
        .source
        .and_then(&get_str)
        .and_then(|label| crate::types::SourceKind::from_label(&label))
        .unwrap_or_else(|| crate::types::SourceKind::from_path(&source_path));
    Record {
        source,
        doc_id: get_u64(fields.doc_id),
        ts: get_u64(fields.ts),
        project: get_str(fields.project).unwrap_or_default(),
        session_id: get_str(fields.session_id).unwrap_or_default(),
        turn_id: get_u64(fields.turn_id) as u32,
        role: get_str(fields.role).unwrap_or_default(),
        text: get_str(fields.text).unwrap_or_default(),
        tool_name: get_str(fields.tool_name),
        tool_input: get_str(fields.tool_input),
        tool_output: get_str(fields.tool_output),
        links: RecordLinks {
            event_id: get_str(fields.event_id),
            parent_event_id: get_str(fields.parent_event_id),
            logical_parent_event_id: get_str(fields.logical_parent_event_id),
            parent_session_id: get_str(fields.parent_session_id),
            thread_source: get_str(fields.thread_source),
            conversation_kind: get_str(fields.conversation_kind),
            parent_tool_use_id: get_str(fields.parent_tool_use_id),
            source_tool_use_id: get_str(fields.source_tool_use_id),
            source_tool_assistant_uuid: get_str(fields.source_tool_assistant_uuid),
        },
        source_path,
    }
}

fn add_optional_text(doc: &mut TantivyDocument, field: Field, value: &Option<String>) {
    if let Some(value) = value {
        doc.add_text(field, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_record(doc_id: u64, text: &str) -> Record {
        Record {
            source: crate::types::SourceKind::Codex,
            doc_id,
            ts: doc_id,
            project: "memex".to_string(),
            session_id: "session".to_string(),
            turn_id: doc_id as u32,
            role: "user".to_string(),
            text: text.to_string(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links: RecordLinks::default(),
            source_path: "session.jsonl".to_string(),
        }
    }

    fn create_stale_schema_index(dir: &Path) {
        let mut builder = SchemaBuilder::default();
        builder.add_u64_field("doc_id", INDEXED | STORED);
        builder.add_u64_field("ts", FAST | STORED | INDEXED);
        builder.add_text_field("project", STRING | STORED);
        builder.add_text_field("session_id", STRING | STORED);
        builder.add_u64_field("turn_id", FAST | STORED);
        builder.add_text_field("role", STRING | STORED);
        builder.add_text_field("source", STRING | STORED);
        builder.add_text_field("text", TEXT | STORED);
        builder.add_text_field("tool_name", STRING | STORED);
        builder.add_text_field("tool_input", TEXT | STORED);
        builder.add_text_field("tool_output", TEXT | STORED);
        builder.add_text_field("source_path", STRING | STORED);

        let index = Index::create_in_dir(dir, builder.build()).expect("create stale index");
        drop(index);
        std::fs::write(dir.join("sentinel"), "keep").expect("write sentinel");
    }

    #[test]
    fn read_only_open_preserves_stale_schema_index() {
        let tmp = tempfile::tempdir().expect("tempdir");
        create_stale_schema_index(tmp.path());

        let err = match SearchIndex::open_or_create(tmp.path()) {
            Ok(_) => panic!("stale index unexpectedly opened"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("index schema"));
        assert!(tmp.path().join("meta.json").exists());
        assert!(tmp.path().join("sentinel").exists());
    }

    #[test]
    fn ingest_open_recreates_stale_schema_index() {
        let tmp = tempfile::tempdir().expect("tempdir");
        create_stale_schema_index(tmp.path());

        let index =
            SearchIndex::open_or_create_for_ingest(tmp.path()).expect("recreate stale index");

        assert_eq!(index.doc_count().expect("doc count"), 0);
        index.publish_generation().expect("publish generation");
        assert!(SearchIndex::exists(tmp.path()));
        assert_eq!(
            SearchIndex::open_or_create(tmp.path())
                .expect("open published generation")
                .doc_count()
                .expect("published doc count"),
            0
        );
    }

    #[test]
    fn publishing_generation_atomically_advances_new_readers() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let first = SearchIndex::open_or_create_for_ingest(tmp.path()).expect("first generation");
        let mut writer = first.writer().expect("first writer");
        first
            .add_record(&mut writer, &test_record(1, "first"))
            .expect("add first");
        writer.commit().expect("commit first");
        writer.wait_merging_threads().expect("finish first writer");
        first.publish_generation().expect("publish first");

        let old_reader = SearchIndex::open_or_create(tmp.path()).expect("old reader");
        assert_eq!(old_reader.doc_count().expect("old count"), 1);

        let second = SearchIndex::open_or_create_for_ingest(tmp.path()).expect("second generation");
        let mut writer = second.writer().expect("second writer");
        second
            .add_record(&mut writer, &test_record(2, "second"))
            .expect("add second");
        writer.commit().expect("commit second");
        writer.wait_merging_threads().expect("finish second writer");

        assert_eq!(
            SearchIndex::open_or_create(tmp.path())
                .expect("reader before publish")
                .doc_count()
                .expect("count before publish"),
            1
        );
        second.publish_generation().expect("publish second");
        assert_eq!(
            SearchIndex::open_or_create(tmp.path())
                .expect("reader after publish")
                .doc_count()
                .expect("count after publish"),
            2
        );
        assert_eq!(old_reader.doc_count().expect("old reader remains valid"), 1);
    }

    #[test]
    fn publishing_waits_for_merges_without_losing_segments() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let index = SearchIndex::open_or_create_for_ingest(tmp.path()).expect("generation");
        let mut writer = index.writer().expect("writer");

        for doc_id in 1..=4 {
            index
                .add_record(
                    &mut writer,
                    &test_record(doc_id, &format!("unique{doc_id}")),
                )
                .expect("add record");
            writer.commit().expect("commit segment");
        }

        let segment_ids = index
            .index
            .searchable_segment_ids()
            .expect("segments before merge");
        assert!(segment_ids.len() > 1);
        writer.merge(&segment_ids).wait().expect("merge segments");
        writer.wait_merging_threads().expect("finish writer");
        index.publish_generation().expect("publish generation");

        let published = SearchIndex::open_or_create(tmp.path()).expect("published generation");
        assert_eq!(published.doc_count().expect("published count"), 4);
        assert_eq!(
            published
                .index
                .searchable_segment_ids()
                .expect("published segments")
                .len(),
            1
        );
        for doc_id in 1..=4 {
            assert_eq!(
                published
                    .search(&QueryOptions {
                        query: format!("unique{doc_id}"),
                        project: None,
                        role: None,
                        tool: None,
                        session_id: None,
                        source: None,
                        since: None,
                        until: None,
                        limit: 10,
                    })
                    .expect("search merged segment")
                    .len(),
                1
            );
        }
    }

    #[test]
    fn legacy_index_is_adopted_without_rebuilding() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let legacy = SearchIndex::open_or_create(tmp.path()).expect("legacy index");
        let mut writer = legacy.writer().expect("legacy writer");
        legacy
            .add_record(&mut writer, &test_record(1, "preserved"))
            .expect("add legacy record");
        writer.commit().expect("commit legacy index");
        writer.wait_merging_threads().expect("finish legacy writer");

        let adopted =
            SearchIndex::open_or_create_for_ingest(tmp.path()).expect("adopt legacy index");
        assert_eq!(adopted.doc_count().expect("adopted count"), 1);
        adopted.publish_generation().expect("publish adoption");

        let published = SearchIndex::open_or_create(tmp.path()).expect("published generation");
        assert_eq!(published.doc_count().expect("published count"), 1);
        assert_eq!(
            published
                .search(&QueryOptions {
                    query: "preserved".to_string(),
                    project: None,
                    role: None,
                    tool: None,
                    session_id: None,
                    source: None,
                    since: None,
                    until: None,
                    limit: 10,
                })
                .expect("search adopted generation")
                .len(),
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn sealed_generation_can_be_searched_without_directory_write_access() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().expect("tempdir");
        let writable = SearchIndex::open_or_create_for_ingest(tmp.path()).expect("generation");
        let mut writer = writable.writer().expect("writer");
        writable
            .add_record(&mut writer, &test_record(1, "needle"))
            .expect("add record");
        writer.commit().expect("commit");
        writer.wait_merging_threads().expect("finish writer");
        writable.publish_generation().expect("publish");

        let generation = resolve_current_generation(tmp.path()).expect("current generation");
        let original_permissions = fs::metadata(&generation).expect("metadata").permissions();
        fs::set_permissions(&generation, fs::Permissions::from_mode(0o555))
            .expect("seal directory");
        let result = SearchIndex::open_or_create(tmp.path())
            .expect("open sealed generation")
            .search(&QueryOptions {
                query: "needle".to_string(),
                project: None,
                role: None,
                tool: None,
                session_id: None,
                source: None,
                since: None,
                until: None,
                limit: 10,
            })
            .expect("search sealed generation");
        fs::set_permissions(&generation, original_permissions).expect("restore permissions");
        assert_eq!(result.len(), 1);
    }
}
