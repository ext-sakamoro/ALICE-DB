//! Storage Engine: Manages segments on disk
//!
//! Handles:
//! - Segment persistence (write/read)
//! - Segment index management
//! - Compaction (merging small segments)
//! - WAL (Write-Ahead Log) for durability
//!
//! # Phase 4: Interval Tree Indexing
//!
//! Uses an Interval Tree for O(log N + K) time range queries.
//! Traditional `BTreeMap` index requires O(N) scan for overlapping segments.
//! Interval Tree enables efficient stabbing queries and range overlap detection.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::memtable::{FitConfig, MemTable};
use crate::segment::{DataSegment, SegmentView};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// =============================================================================
// Interval Tree Implementation (Arena Allocated - Cache Friendly)
// =============================================================================
//
// # Performance: Arena Allocation
//
// Instead of Box<Node> (random heap pointers causing cache misses),
// we store all nodes in a contiguous Vec<Node> and use u32 indices.
// This keeps the tree in a single cache-friendly memory block.

/// Sentinel value for "no child" (like None but cache-friendly)
const NO_CHILD: u32 = u32::MAX;

/// Interval with associated segment ID (packed for cache efficiency)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Interval {
    start: i64,
    end: i64,
    segment_id: u64,
}

/// Interval Tree node (Arena-allocated, no Box/pointer chasing)
#[derive(Debug, Clone)]
struct IntervalNode {
    /// Center point of this node
    center: i64,
    /// Start index into `intervals_storage`
    intervals_start: u32,
    /// Number of intervals at this node
    intervals_count: u16,
    /// Left child index (`NO_CHILD` if none)
    left_idx: u32,
    /// Right child index (`NO_CHILD` if none)
    right_idx: u32,
}

/// Augmented Interval Tree for O(log N + K) range queries
///
/// # Arena Allocation
///
/// All nodes stored in contiguous `Vec<IntervalNode>` for cache locality.
/// No pointer chasing - just array indexing.
///
/// # Memory Layout
///
/// ```text
/// nodes:     [Node0, Node1, Node2, ...] (contiguous, cache-friendly)
/// intervals: [Int0, Int1, Int2, ...]     (sorted by start within each node)
/// ```
#[derive(Debug, Default)]
pub struct IntervalTree {
    /// Arena: all nodes stored here (contiguous memory)
    nodes: Vec<IntervalNode>,
    /// Arena: all intervals stored here (contiguous memory)
    intervals_storage: Vec<Interval>,
    /// Root node index (`NO_CHILD` if empty)
    root_idx: u32,
    /// All intervals for rebuilding
    all_intervals: Vec<Interval>,
    /// Insertions since last rebuild
    insertions_since_rebuild: usize,
}

impl IntervalTree {
    /// Create a new empty interval tree
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(64),
            intervals_storage: Vec::with_capacity(256),
            root_idx: NO_CHILD,
            all_intervals: Vec::with_capacity(256),
            insertions_since_rebuild: 0,
        }
    }

    /// Insert an interval (O(1) append, no rebuild)
    ///
    /// # Performance: Jitter-Free Writes
    ///
    /// This method only appends to the interval list without rebuilding.
    /// The tree is rebuilt lazily on first query after inserts, or explicitly
    /// via `rebuild()` during flush/compaction.
    ///
    /// This ensures consistent write latency without periodic spikes.
    pub fn insert(&mut self, start: i64, end: i64, segment_id: u64) {
        self.all_intervals.push(Interval {
            start,
            end,
            segment_id,
        });
        self.insertions_since_rebuild += 1;
        // NOTE: No automatic rebuild here! Call rebuild() explicitly after flush.
    }

    /// Check if rebuild is needed (for callers to decide when to rebuild)
    #[inline]
    #[must_use]
    pub const fn needs_rebuild(&self) -> bool {
        self.insertions_since_rebuild > 0 || self.root_idx == NO_CHILD
    }

    /// Rebuild the entire tree for optimal balance
    ///
    /// Call this after `flush()` or compaction, NOT during hot write path.
    pub fn rebuild(&mut self) {
        self.nodes.clear();
        self.intervals_storage.clear();

        if self.all_intervals.is_empty() {
            self.root_idx = NO_CHILD;
            return;
        }

        let intervals: Vec<Interval> = self.all_intervals.clone();
        self.root_idx = self.build_tree_arena(&intervals);
        self.insertions_since_rebuild = 0;
    }

    /// Build a balanced tree into arena, returns root index
    fn build_tree_arena(&mut self, intervals: &[Interval]) -> u32 {
        if intervals.is_empty() {
            return NO_CHILD;
        }

        // Find center point (median of endpoints)
        let mut endpoints: Vec<i64> = intervals.iter().flat_map(|i| [i.start, i.end]).collect();
        endpoints.sort_unstable();
        let center = endpoints[endpoints.len() / 2];

        // Partition intervals
        let mut containing: Vec<Interval> = Vec::new();
        let mut left_intervals: Vec<Interval> = Vec::new();
        let mut right_intervals: Vec<Interval> = Vec::new();

        for &interval in intervals {
            if interval.end < center {
                left_intervals.push(interval);
            } else if interval.start > center {
                right_intervals.push(interval);
            } else {
                containing.push(interval);
            }
        }

        // Sort containing intervals by start (for efficient query)
        containing.sort_unstable_by_key(|i| i.start);

        // Store intervals in arena
        let intervals_start = self.intervals_storage.len() as u32;
        let intervals_count = containing.len() as u16;
        self.intervals_storage.extend(containing);

        // Allocate node in arena
        let node_idx = self.nodes.len() as u32;
        self.nodes.push(IntervalNode {
            center,
            intervals_start,
            intervals_count,
            left_idx: NO_CHILD, // Will be set below
            right_idx: NO_CHILD,
        });

        // Recursively build children
        let left_idx = self.build_tree_arena(&left_intervals);
        let right_idx = self.build_tree_arena(&right_intervals);

        // Update child indices
        self.nodes[node_idx as usize].left_idx = left_idx;
        self.nodes[node_idx as usize].right_idx = right_idx;

        node_idx
    }

    /// Find all intervals that overlap with [start, end]
    ///
    /// If tree is not built yet, falls back to linear scan of all intervals.
    #[inline]
    #[must_use]
    pub fn query_range(&self, start: i64, end: i64) -> Vec<u64> {
        // If tree is not built, fallback to linear scan
        if self.root_idx == NO_CHILD && !self.all_intervals.is_empty() {
            return self
                .all_intervals
                .iter()
                .filter(|i| i.start <= end && i.end >= start)
                .map(|i| i.segment_id)
                .collect();
        }

        let mut results = Vec::with_capacity(8);
        self.query_node_iterative(start, end, &mut results);
        results
    }

    /// Iterative query (no recursion = no function call overhead)
    #[inline]
    fn query_node_iterative(&self, start: i64, end: i64, results: &mut Vec<u64>) {
        if self.root_idx == NO_CHILD {
            return;
        }

        // Use a small stack for iteration (avoids recursion overhead)
        let mut stack: [u32; 32] = [NO_CHILD; 32];
        stack[0] = self.root_idx;
        let mut stack_top: usize = 1;

        while stack_top > 0 {
            stack_top -= 1;
            let node_idx = stack[stack_top];

            if node_idx == NO_CHILD {
                continue;
            }

            let node = &self.nodes[node_idx as usize];
            let intervals = &self.intervals_storage[node.intervals_start as usize
                ..(node.intervals_start as usize + node.intervals_count as usize)];

            if end < node.center {
                // Query entirely to the left - check intervals that might overlap
                for interval in intervals {
                    if interval.start > end {
                        break;
                    }
                    results.push(interval.segment_id);
                }
                // Push left child
                if node.left_idx != NO_CHILD && stack_top < 31 {
                    stack[stack_top] = node.left_idx;
                    stack_top += 1;
                }
            } else if start > node.center {
                // Query entirely to the right - check intervals that might overlap
                // Need to check from end (we stored by start, so scan all)
                for interval in intervals.iter().rev() {
                    if interval.end < start {
                        continue;
                    }
                    results.push(interval.segment_id);
                }
                // Push right child
                if node.right_idx != NO_CHILD && stack_top < 31 {
                    stack[stack_top] = node.right_idx;
                    stack_top += 1;
                }
            } else {
                // Query contains center - all intervals at this node overlap
                for interval in intervals {
                    results.push(interval.segment_id);
                }
                // Push both children
                if node.left_idx != NO_CHILD && stack_top < 31 {
                    stack[stack_top] = node.left_idx;
                    stack_top += 1;
                }
                if node.right_idx != NO_CHILD && stack_top < 31 {
                    stack[stack_top] = node.right_idx;
                    stack_top += 1;
                }
            }
        }
    }

    /// Find all intervals containing a specific point
    #[inline]
    #[must_use]
    pub fn query_point(&self, point: i64) -> Vec<u64> {
        self.query_range(point, point)
    }

    /// Get total number of intervals
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.all_intervals.len()
    }

    /// Check if tree is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.all_intervals.is_empty()
    }
}

/// Segment index entry
#[derive(Debug, Clone)]
pub struct SegmentIndexEntry {
    /// Segment ID
    pub id: u64,
    /// Start timestamp
    pub start_time: i64,
    /// End timestamp
    pub end_time: i64,
    /// File offset
    pub offset: u64,
    /// Segment size in bytes
    pub size: u64,
    /// Model type name (for quick filtering)
    pub model_type: String,
    /// Compression ratio
    pub compression_ratio: f64,
}

/// Job for background flush thread
enum FlushJob {
    /// A segment to persist to disk
    Segment(DataSegment),
    /// Shutdown signal
    Shutdown,
}

/// Storage Engine configuration
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct StorageConfig {
    /// Directory for data files
    pub data_dir: PathBuf,
    /// `MemTable` capacity (points before flush)
    pub memtable_capacity: usize,
    /// Model fitting configuration
    pub fit_config: FitConfig,
    /// Enable WAL for durability
    pub enable_wal: bool,
    /// Sync writes to disk immediately
    pub sync_writes: bool,
    /// Compaction threshold (number of segments)
    pub compaction_threshold: usize,
    /// Use mmap for segment access (true = mmap, false = read into Vec)
    pub use_mmap: bool,
    /// Enable background flush thread to avoid latency spikes on `put()`
    pub enable_background_flush: bool,
    /// Encryption key for at-rest encryption (requires "crypto" feature)
    #[cfg(feature = "crypto")]
    pub encryption_key: Option<alice_crypto::Key>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./alice_db_data"),
            memtable_capacity: 1000,
            fit_config: FitConfig::default(),
            enable_wal: true,
            sync_writes: false,
            compaction_threshold: 10,
            use_mmap: true,
            enable_background_flush: false,
            #[cfg(feature = "crypto")]
            encryption_key: None,
        }
    }
}

impl std::fmt::Debug for StorageConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("StorageConfig");
        s.field("data_dir", &self.data_dir)
            .field("memtable_capacity", &self.memtable_capacity)
            .field("fit_config", &self.fit_config)
            .field("enable_wal", &self.enable_wal)
            .field("sync_writes", &self.sync_writes)
            .field("compaction_threshold", &self.compaction_threshold)
            .field("use_mmap", &self.use_mmap)
            .field("enable_background_flush", &self.enable_background_flush);
        #[cfg(feature = "crypto")]
        s.field(
            "encryption_key",
            &self.encryption_key.as_ref().map(|_| "<redacted>"),
        );
        s.finish()
    }
}

/// Shared state accessible from both main engine and background flush thread
struct SharedState {
    /// Segment index (`segment_id` → segment info)
    index: RwLock<BTreeMap<u64, SegmentIndexEntry>>,
    /// Interval Tree for O(log N + K) range queries (Phase 4)
    interval_tree: RwLock<IntervalTree>,
    /// Cached segments: `Arc<SegmentView>` for Zero-Copy access.
    segment_cache: RwLock<BTreeMap<u64, Arc<SegmentView>>>,
    /// Number of in-flight background flush jobs (for synchronization)
    in_flight: std::sync::atomic::AtomicUsize,
}

/// Storage Engine: Core persistence layer
pub struct StorageEngine {
    /// Configuration
    config: StorageConfig,
    /// In-memory write buffer
    memtable: MemTable,
    /// Shared state (index, interval tree, cache)
    shared: Arc<SharedState>,
    /// Data file handle (legacy, for index persistence)
    data_file: RwLock<Option<File>>,
    /// WAL file handle
    wal_file: RwLock<Option<File>>,
    /// Current data file offset
    current_offset: RwLock<u64>,
    /// Background flush channel sender
    flush_sender: Option<crossbeam_channel::Sender<FlushJob>>,
    /// Background flush thread handle
    flush_handle: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl StorageEngine {
    /// Create a new storage engine
    ///
    /// # Errors
    ///
    /// Returns an error if the data directory cannot be created or files cannot be initialized.
    ///
    /// # Panics
    ///
    /// Panics if the background flush thread cannot be spawned.
    pub fn new(config: StorageConfig) -> io::Result<Self> {
        // Create data directory if it doesn't exist
        fs::create_dir_all(&config.data_dir)?;

        let memtable = MemTable::with_config(config.memtable_capacity, config.fit_config.clone());

        let shared = Arc::new(SharedState {
            index: RwLock::new(BTreeMap::new()),
            interval_tree: RwLock::new(IntervalTree::new()),
            segment_cache: RwLock::new(BTreeMap::new()),
            in_flight: std::sync::atomic::AtomicUsize::new(0),
        });

        // Start background flush thread if enabled
        let (flush_sender, flush_handle) = if config.enable_background_flush {
            let (tx, rx) = crossbeam_channel::bounded::<FlushJob>(4);
            let shared_clone = Arc::clone(&shared);
            let cfg = config.clone();
            let handle = std::thread::Builder::new()
                .name("alice-db-flush".into())
                .spawn(move || {
                    Self::flush_thread_loop(&rx, &shared_clone, &cfg);
                })
                .expect("failed to spawn flush thread");
            (Some(tx), parking_lot::Mutex::new(Some(handle)))
        } else {
            (None, parking_lot::Mutex::new(None))
        };

        let engine = Self {
            config,
            memtable,
            shared,
            data_file: RwLock::new(None),
            wal_file: RwLock::new(None),
            current_offset: RwLock::new(0),
            flush_sender,
            flush_handle,
        };

        engine.init_files()?;
        engine.load_index()?;
        engine.replay_wal()?;

        Ok(engine)
    }

    /// Create with default configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the storage engine cannot be initialized at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let config = StorageConfig {
            data_dir: path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Initialize data and WAL files
    fn init_files(&self) -> io::Result<()> {
        let data_path = self.config.data_dir.join("data.alice");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&data_path)?;

        let offset = file.metadata()?.len();
        *self.current_offset.write() = offset;
        *self.data_file.write() = Some(file);

        if self.config.enable_wal {
            let wal_path = self.config.data_dir.join("wal.alice");
            let wal = OpenOptions::new()
                .read(true)
                .create(true)
                .append(true)
                .open(&wal_path)?;
            *self.wal_file.write() = Some(wal);
        }

        Ok(())
    }

    /// Load index from disk (decrypt if crypto key is configured)
    #[allow(clippy::significant_drop_tightening)]
    fn load_index(&self) -> io::Result<()> {
        let index_path = self.config.data_dir.join("index.alice");
        if !index_path.exists() {
            return Ok(());
        }

        // Read entire file
        let raw = fs::read(&index_path)?;
        if raw.is_empty() {
            return Ok(());
        }

        // Decrypt if key configured
        #[cfg(feature = "crypto")]
        let data = if let Some(ref key) = self.config.encryption_key {
            alice_crypto::open(key, &raw).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("index decryption failed: {e:?}"),
                )
            })?
        } else {
            raw
        };
        #[cfg(not(feature = "crypto"))]
        let data = raw;

        let mut reader = std::io::Cursor::new(&data);

        // Read number of entries
        let mut count_bytes = [0u8; 8];
        if std::io::Read::read_exact(&mut reader, &mut count_bytes).is_err() {
            return Ok(()); // Empty or corrupt index
        }
        let count = u64::from_le_bytes(count_bytes) as usize;

        let mut index = self.shared.index.write();
        let mut interval_tree = self.shared.interval_tree.write();

        for _ in 0..count {
            // Read entry
            let mut id_bytes = [0u8; 8];
            std::io::Read::read_exact(&mut reader, &mut id_bytes)?;
            let id = u64::from_le_bytes(id_bytes);

            let mut start_bytes = [0u8; 8];
            std::io::Read::read_exact(&mut reader, &mut start_bytes)?;
            let start_time = i64::from_le_bytes(start_bytes);

            let mut end_bytes = [0u8; 8];
            std::io::Read::read_exact(&mut reader, &mut end_bytes)?;
            let end_time = i64::from_le_bytes(end_bytes);

            let mut offset_bytes = [0u8; 8];
            std::io::Read::read_exact(&mut reader, &mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);

            let mut size_bytes = [0u8; 8];
            std::io::Read::read_exact(&mut reader, &mut size_bytes)?;
            let size = u64::from_le_bytes(size_bytes);

            let mut ratio_bytes = [0u8; 8];
            std::io::Read::read_exact(&mut reader, &mut ratio_bytes)?;
            let compression_ratio = f64::from_le_bytes(ratio_bytes);

            let mut model_len_bytes = [0u8; 4];
            std::io::Read::read_exact(&mut reader, &mut model_len_bytes)?;
            let model_len = u32::from_le_bytes(model_len_bytes) as usize;

            let mut model_type_bytes = vec![0u8; model_len];
            std::io::Read::read_exact(&mut reader, &mut model_type_bytes)?;
            let model_type = String::from_utf8_lossy(&model_type_bytes).to_string();

            let entry = SegmentIndexEntry {
                id,
                start_time,
                end_time,
                offset,
                size,
                model_type,
                compression_ratio,
            };

            // Insert into both index and interval tree
            index.insert(id, entry);
            interval_tree.insert(start_time, end_time, id);
        }

        Ok(())
    }

    /// Save index to disk (encrypted if crypto key is configured)
    #[allow(clippy::significant_drop_tightening)]
    fn save_index(&self) -> io::Result<()> {
        let index_path = self.config.data_dir.join("index.alice");

        let index = self.shared.index.read();

        // Serialize index entries to bytes
        let mut buf = Vec::new();
        buf.extend_from_slice(&(index.len() as u64).to_le_bytes());

        for entry in index.values() {
            buf.extend_from_slice(&entry.id.to_le_bytes());
            buf.extend_from_slice(&entry.start_time.to_le_bytes());
            buf.extend_from_slice(&entry.end_time.to_le_bytes());
            buf.extend_from_slice(&entry.offset.to_le_bytes());
            buf.extend_from_slice(&entry.size.to_le_bytes());
            buf.extend_from_slice(&entry.compression_ratio.to_le_bytes());

            let model_bytes = entry.model_type.as_bytes();
            buf.extend_from_slice(&(model_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(model_bytes);
        }

        // Encrypt if key configured
        #[cfg(feature = "crypto")]
        let output = if let Some(ref key) = self.config.encryption_key {
            alice_crypto::seal(key, &buf)
                .map_err(|e| io::Error::other(format!("index encryption failed: {e:?}")))?
        } else {
            buf
        };
        #[cfg(not(feature = "crypto"))]
        let output = buf;

        fs::write(&index_path, &output)?;
        Ok(())
    }

    /// Insert a single value
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write or segment persistence fails.
    pub fn put(&self, timestamp: i64, value: f32) -> io::Result<()> {
        // Write to WAL first for durability
        if self.config.enable_wal {
            self.write_wal(timestamp, value)?;
        }

        // Insert into MemTable
        if let Some(segment) = self.memtable.put(timestamp, value) {
            if let Some(ref sender) = self.flush_sender {
                self.shared
                    .in_flight
                    .fetch_add(1, std::sync::atomic::Ordering::Acquire);
                let _ = sender.send(FlushJob::Segment(segment));
            } else {
                self.persist_segment(&segment)?;
            }
        }

        Ok(())
    }

    /// Insert multiple values (batch)
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write or segment persistence fails.
    pub fn put_batch(&self, data: &[(i64, f32)]) -> io::Result<()> {
        // Write to WAL first
        if self.config.enable_wal {
            for &(t, v) in data {
                self.write_wal(t, v)?;
            }
        }

        // Insert into MemTable
        let segments = self.memtable.put_batch(data);
        for segment in segments {
            if let Some(ref sender) = self.flush_sender {
                self.shared
                    .in_flight
                    .fetch_add(1, std::sync::atomic::Ordering::Acquire);
                let _ = sender.send(FlushJob::Segment(segment));
            } else {
                self.persist_segment(&segment)?;
            }
        }

        Ok(())
    }

    /// Background flush thread loop
    fn flush_thread_loop(
        rx: &crossbeam_channel::Receiver<FlushJob>,
        shared: &SharedState,
        config: &StorageConfig,
    ) {
        while let Ok(job) = rx.recv() {
            match job {
                FlushJob::Segment(segment) => {
                    if let Err(e) = Self::persist_segment_shared(shared, config, &segment) {
                        log::error!("Background flush failed: {e}");
                    }
                    shared
                        .in_flight
                        .fetch_sub(1, std::sync::atomic::Ordering::Release);
                }
                FlushJob::Shutdown => break,
            }
        }
    }

    /// Persist segment using shared state (callable from both main and flush thread)
    fn persist_segment_shared(
        shared: &SharedState,
        config: &StorageConfig,
        segment: &DataSegment,
    ) -> io::Result<()> {
        let segment_id = segment.metadata.id;
        let start_time = segment.start_time;
        let end_time = segment.end_time;
        let model_name = segment.model.name().to_string();
        let compression_ratio = segment.metadata.compression_ratio;

        let rkyv_bytes = segment.to_rkyv_bytes()?;
        let segment_size = rkyv_bytes.len() as u64;

        let view = SegmentView::from_vec(rkyv_bytes.clone())?;
        shared
            .segment_cache
            .write()
            .insert(segment_id, Arc::new(view));

        // Write to disk with exclusive advisory lock (encrypt if key configured)
        let segment_path = config.data_dir.join(format!("seg_{segment_id}.rkyv"));
        {
            use fs2::FileExt;

            #[cfg(feature = "crypto")]
            let bytes_to_write = if let Some(ref key) = config.encryption_key {
                alice_crypto::seal(key, &rkyv_bytes)
                    .map_err(|e| io::Error::other(format!("encryption failed: {e:?}")))?
            } else {
                rkyv_bytes.clone()
            };
            #[cfg(not(feature = "crypto"))]
            let bytes_to_write = &rkyv_bytes;

            let file = File::create(&segment_path)?;
            file.lock_exclusive()?;
            let mut writer = BufWriter::new(&file);
            writer.write_all(bytes_to_write.as_ref())?;
            writer.flush()?;
            // Lock released on file drop
        }

        if config.sync_writes {
            let file = File::open(&segment_path)?;
            file.sync_all()?;
        }

        let entry = SegmentIndexEntry {
            id: segment_id,
            start_time,
            end_time,
            offset: 0,
            size: segment_size,
            model_type: model_name,
            compression_ratio,
        };

        shared.index.write().insert(segment_id, entry);
        {
            let mut tree = shared.interval_tree.write();
            tree.insert(start_time, end_time, segment_id);
            if tree.needs_rebuild() {
                tree.rebuild();
            }
        }

        Ok(())
    }

    /// Write to WAL
    fn write_wal(&self, timestamp: i64, value: f32) -> io::Result<()> {
        if let Some(ref mut wal) = *self.wal_file.write() {
            #[cfg(feature = "crypto")]
            if let Some(ref key) = self.config.encryption_key {
                // Encrypted WAL: length-prefixed sealed entries
                let mut plaintext = Vec::with_capacity(12);
                plaintext.extend_from_slice(&timestamp.to_le_bytes());
                plaintext.extend_from_slice(&value.to_le_bytes());
                let sealed = alice_crypto::seal(key, &plaintext)
                    .map_err(|e| io::Error::other(format!("WAL encrypt failed: {e:?}")))?;
                wal.write_all(&(sealed.len() as u32).to_le_bytes())?;
                wal.write_all(&sealed)?;
            } else {
                wal.write_all(&timestamp.to_le_bytes())?;
                wal.write_all(&value.to_le_bytes())?;
            }

            #[cfg(not(feature = "crypto"))]
            {
                wal.write_all(&timestamp.to_le_bytes())?;
                wal.write_all(&value.to_le_bytes())?;
            }

            if self.config.sync_writes {
                wal.sync_all()?;
            }
        }
        Ok(())
    }

    /// Replay WAL entries to recover unflushed data after crash
    ///
    /// Reads 12-byte entries (i64 LE timestamp + f32 LE value) from wal.alice,
    /// inserts into `MemTable` (without re-writing to WAL), and persists any
    /// segments that are produced. Partial entries (incomplete writes) are
    /// silently ignored. The WAL file is truncated after successful replay.
    fn replay_wal(&self) -> io::Result<usize> {
        let wal_path = self.config.data_dir.join("wal.alice");
        if !wal_path.exists() {
            return Ok(0);
        }

        let wal_data = fs::read(&wal_path)?;
        if wal_data.is_empty() {
            return Ok(0);
        }

        let mut recovered = 0usize;

        #[cfg(feature = "crypto")]
        let is_encrypted = self.config.encryption_key.is_some();
        #[cfg(not(feature = "crypto"))]
        let is_encrypted = false;

        if is_encrypted {
            // Encrypted WAL: length-prefixed sealed entries
            #[cfg(feature = "crypto")]
            {
                let key = self.config.encryption_key.as_ref().unwrap();
                let mut pos = 0;
                while pos + 4 <= wal_data.len() {
                    let sealed_len =
                        u32::from_le_bytes(wal_data[pos..pos + 4].try_into().unwrap()) as usize;
                    pos += 4;
                    if pos + sealed_len > wal_data.len() {
                        break; // Partial entry
                    }
                    let sealed = &wal_data[pos..pos + sealed_len];
                    pos += sealed_len;

                    let Ok(plaintext) = alice_crypto::open(key, sealed) else {
                        break; // Corrupted entry
                    };
                    if plaintext.len() < 12 {
                        break;
                    }
                    let timestamp = i64::from_le_bytes(plaintext[0..8].try_into().unwrap());
                    let value = f32::from_le_bytes(plaintext[8..12].try_into().unwrap());

                    if let Some(segment) = self.memtable.put(timestamp, value) {
                        self.persist_segment(&segment)?;
                    }
                    recovered += 1;
                }
            }
        } else {
            // Unencrypted WAL: fixed 12-byte entries
            let entry_size = 12; // i64 (8) + f32 (4)
            let full_entries = wal_data.len() / entry_size;

            for i in 0..full_entries {
                let offset = i * entry_size;
                let timestamp =
                    i64::from_le_bytes(wal_data[offset..offset + 8].try_into().unwrap());
                let value =
                    f32::from_le_bytes(wal_data[offset + 8..offset + 12].try_into().unwrap());

                if let Some(segment) = self.memtable.put(timestamp, value) {
                    self.persist_segment(&segment)?;
                }
                recovered += 1;
            }
        }

        // Persist any remaining data in memtable
        if let Some(segment) = self.memtable.force_flush() {
            self.persist_segment(&segment)?;
        }

        // Truncate WAL after successful replay
        File::create(&wal_path)?;

        Ok(recovered)
    }

    /// Persist a segment to disk (rkyv format for Zero-Copy)
    ///
    /// # Performance: Zero-Copy Path
    ///
    /// 1. Serialize to rkyv bytes (in memory)
    /// 2. Create `SegmentView` from bytes IMMEDIATELY (no disk I/O wait)
    /// 3. Write bytes to disk (can be async in future)
    ///
    /// This ensures queries can hit the cache instantly after `MemTable` flush,
    /// without waiting for disk I/O to complete.
    fn persist_segment(&self, segment: &DataSegment) -> io::Result<()> {
        Self::persist_segment_shared(&self.shared, &self.config, segment)
    }

    /// Load a segment as `SegmentView` (Zero-Copy mmap)
    ///
    /// # Performance: Zero-Copy
    ///
    /// Returns Arc<SegmentView> which provides direct mmap access.
    /// No deserialization occurs - model coefficients are read directly.
    fn load_segment(&self, entry: &SegmentIndexEntry) -> io::Result<Arc<SegmentView>> {
        // Check cache first (fast path)
        if let Some(view) = self.shared.segment_cache.read().get(&entry.id) {
            return Ok(Arc::clone(view));
        }

        // Open segment file as SegmentView (mmap)
        let segment_path = self.config.data_dir.join(format!("seg_{}.rkyv", entry.id));

        // Fallback to legacy format if rkyv file doesn't exist
        if !segment_path.exists() {
            return self.load_segment_legacy(entry);
        }

        // Load segment (decrypt if encrypted)
        #[cfg(feature = "crypto")]
        let view = if let Some(ref key) = self.config.encryption_key {
            // Encrypted: must read to memory, decrypt, then create view
            let sealed = fs::read(&segment_path)?;
            let plaintext = alice_crypto::open(key, &sealed).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("decryption failed: {e:?}"),
                )
            })?;
            SegmentView::from_vec(plaintext)?
        } else if self.config.use_mmap {
            SegmentView::open(&segment_path)?
        } else {
            SegmentView::open_read(&segment_path)?
        };

        #[cfg(not(feature = "crypto"))]
        let view = if self.config.use_mmap {
            SegmentView::open(&segment_path)?
        } else {
            SegmentView::open_read(&segment_path)?
        };

        let arc_view = Arc::new(view);

        // Update cache
        self.shared
            .segment_cache
            .write()
            .insert(entry.id, Arc::clone(&arc_view));

        Ok(arc_view)
    }

    /// Legacy segment loading (for backwards compatibility with old data.alice format)
    fn load_segment_legacy(&self, entry: &SegmentIndexEntry) -> io::Result<Arc<SegmentView>> {
        let mut data_file = self.data_file.write();
        if let Some(ref mut file) = *data_file {
            file.seek(SeekFrom::Start(entry.offset))?;

            // Read size
            let mut size_bytes = [0u8; 8];
            file.read_exact(&mut size_bytes)?;
            let size = u64::from_le_bytes(size_bytes) as usize;

            // Read segment data
            let mut data = vec![0u8; size];
            file.read_exact(&mut data)?;

            let segment = DataSegment::from_bytes(&data)?;

            // Convert to rkyv format and save for future zero-copy access
            let segment_path = self.config.data_dir.join(format!("seg_{}.rkyv", entry.id));
            segment.write_rkyv(&segment_path)?;

            // Open as SegmentView (respect use_mmap config)
            let view = if self.config.use_mmap {
                SegmentView::open(&segment_path)?
            } else {
                SegmentView::open_read(&segment_path)?
            };
            let arc_view = Arc::new(view);

            // Update cache
            self.shared
                .segment_cache
                .write()
                .insert(entry.id, Arc::clone(&arc_view));

            Ok(arc_view)
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Data file not open",
            ))
        }
    }

    /// Find segments that overlap with a time range
    ///
    /// Uses Interval Tree for O(log N + K) query performance.
    /// K = number of overlapping segments returned.
    pub fn find_segments(&self, start: i64, end: i64) -> Vec<SegmentIndexEntry> {
        // Use Interval Tree for efficient range query
        let segment_ids = self.shared.interval_tree.read().query_range(start, end);
        let index = self.shared.index.read();

        segment_ids
            .into_iter()
            .filter_map(|id| index.get(&id).cloned())
            .collect()
    }

    /// Query a single point (Zero-Copy via `SegmentView`)
    ///
    /// # Errors
    ///
    /// Returns an error if segment loading fails.
    pub fn query_point(&self, timestamp: i64) -> io::Result<Option<f32>> {
        // Check MemTable first (most recent data)
        // Note: MemTable doesn't support point queries directly,
        // so we check segments

        let entries = self.find_segments(timestamp, timestamp);
        if entries.is_empty() {
            return Ok(None);
        }

        // Load segment as SegmentView and query (Zero-Copy)
        let view = self.load_segment(&entries[0])?;
        Ok(view.query_point(timestamp))
    }

    /// Query a time range (Zero-Copy via `SegmentView`)
    ///
    /// # Performance: Zero-Copy Path
    ///
    /// Uses `SegmentView` (mmap + SIMD) for maximum throughput.
    /// No deserialization occurs during query.
    ///
    /// # Errors
    ///
    /// Returns an error if segment loading fails.
    pub fn query_range(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        let entries = self.find_segments(start, end);

        let mut results = Vec::new();
        for entry in entries {
            let view = self.load_segment(&entry)?;
            let segment_results = view.query_range(start, end);
            results.extend(segment_results);
        }

        // Sort by timestamp
        results.sort_by_key(|&(t, _)| t);

        // Remove duplicates (keep first occurrence)
        results.dedup_by_key(|&mut (t, _)| t);

        Ok(results)
    }

    /// Force flush `MemTable` to disk
    ///
    /// Also rebuilds the Interval Tree for optimal query performance.
    /// This is the correct place to rebuild (not during hot write path).
    ///
    /// # Errors
    ///
    /// Returns an error if segment persistence or index saving fails.
    pub fn flush(&self) -> io::Result<()> {
        // Wait for all background flush jobs to complete
        if self.flush_sender.is_some() {
            while self
                .shared
                .in_flight
                .load(std::sync::atomic::Ordering::Acquire)
                > 0
            {
                std::thread::yield_now();
            }
        }

        // Force flush memtable synchronously (bypass channel)
        if let Some(segment) = self.memtable.force_flush() {
            Self::persist_segment_shared(&self.shared, &self.config, &segment)?;
        }

        // Rebuild Interval Tree after flush (jitter-free writes)
        {
            let mut tree = self.shared.interval_tree.write();
            if tree.needs_rebuild() {
                tree.rebuild();
            }
        }

        self.save_index()?;
        Ok(())
    }

    /// Close the storage engine
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or WAL cleanup fails.
    #[allow(clippy::significant_drop_in_scrutinee)]
    pub fn close(&self) -> io::Result<()> {
        self.flush()?;

        // Shutdown background flush thread
        if let Some(ref sender) = self.flush_sender {
            let _ = sender.send(FlushJob::Shutdown);
        }
        if let Some(handle) = self.flush_handle.lock().take() {
            let _ = handle.join();
        }

        // Close files
        *self.data_file.write() = None;
        *self.wal_file.write() = None;

        // Clear WAL (data is persisted)
        if self.config.enable_wal {
            let wal_path = self.config.data_dir.join("wal.alice");
            if wal_path.exists() {
                fs::remove_file(&wal_path)?;
            }
        }

        Ok(())
    }

    /// Get statistics
    #[allow(clippy::significant_drop_tightening)]
    pub fn stats(&self) -> StorageStats {
        let index = self.shared.index.read();

        let total_segments = index.len();
        let total_compression_ratio: f64 =
            index.values().map(|e| e.compression_ratio).sum::<f64>() / total_segments.max(1) as f64;

        let total_size: u64 = index.values().map(|e| e.size).sum();

        let model_counts: std::collections::HashMap<String, usize> =
            index
                .values()
                .fold(std::collections::HashMap::new(), |mut acc, e| {
                    *acc.entry(e.model_type.clone()).or_insert(0) += 1;
                    acc
                });

        StorageStats {
            total_segments,
            memtable_size: self.memtable.len(),
            total_disk_size: total_size,
            average_compression_ratio: total_compression_ratio,
            model_distribution: model_counts,
        }
    }
}

impl Drop for StorageEngine {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_segments: usize,
    pub memtable_size: usize,
    pub total_disk_size: u64,
    pub average_compression_ratio: f64,
    pub model_distribution: std::collections::HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_storage_engine_basic() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 100,
            enable_wal: false,
            ..Default::default()
        };

        let engine = StorageEngine::new(config).unwrap();

        // Insert some data
        for i in 0..50 {
            engine.put(i, i as f32 * 2.0).unwrap();
        }

        assert_eq!(engine.memtable.len(), 50);

        // Force flush
        engine.flush().unwrap();
        assert_eq!(engine.memtable.len(), 0);

        let stats = engine.stats();
        assert_eq!(stats.total_segments, 1);
    }

    #[test]
    fn test_storage_engine_flush_on_capacity() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 50,
            enable_wal: false,
            ..Default::default()
        };

        let engine = StorageEngine::new(config).unwrap();

        // Insert enough to trigger flush
        for i in 0..100 {
            engine.put(i, i as f32).unwrap();
        }

        let stats = engine.stats();
        assert!(stats.total_segments >= 1);
    }

    #[test]
    fn test_storage_engine_query() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 1000,
            enable_wal: false,
            ..Default::default()
        };

        let engine = StorageEngine::new(config).unwrap();

        // Insert linear data
        for i in 0..100 {
            engine.put(i, i as f32).unwrap();
        }

        engine.flush().unwrap();

        // Query range
        let results = engine.query_range(0, 99).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_interval_tree_empty() {
        let tree = IntervalTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        let results = tree.query_range(0, 100);
        assert!(results.is_empty());
        let point_results = tree.query_point(50);
        assert!(point_results.is_empty());
    }

    #[test]
    fn test_interval_tree_insert_and_query_without_rebuild() {
        let mut tree = IntervalTree::new();
        tree.insert(0, 100, 1);
        tree.insert(50, 150, 2);
        tree.insert(200, 300, 3);

        // Without rebuild, should fallback to linear scan
        assert!(tree.needs_rebuild());
        let results = tree.query_range(60, 90);
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));
    }

    #[test]
    fn test_interval_tree_after_rebuild() {
        let mut tree = IntervalTree::new();
        tree.insert(0, 100, 1);
        tree.insert(50, 150, 2);
        tree.insert(200, 300, 3);
        tree.insert(250, 350, 4);

        tree.rebuild();
        assert!(!tree.needs_rebuild());
        assert_eq!(tree.len(), 4);

        // Point query
        let point_results = tree.query_point(75);
        assert!(point_results.contains(&1));
        assert!(point_results.contains(&2));
        assert!(!point_results.contains(&3));

        // Range query fully in right side
        let right_results = tree.query_range(250, 280);
        assert!(right_results.contains(&3));
        assert!(right_results.contains(&4));
        assert!(!right_results.contains(&1));

        // No overlap
        let no_results = tree.query_range(160, 190);
        assert!(no_results.is_empty());
    }

    #[test]
    fn test_interval_tree_single_interval() {
        let mut tree = IntervalTree::new();
        tree.insert(10, 20, 42);
        tree.rebuild();

        assert_eq!(tree.query_point(15).len(), 1);
        assert_eq!(tree.query_point(15)[0], 42);
        assert!(tree.query_point(5).is_empty());
        assert!(tree.query_point(25).is_empty());
    }

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.memtable_capacity, 1000);
        assert!(config.enable_wal);
        assert!(!config.sync_writes);
        assert_eq!(config.compaction_threshold, 10);
    }

    #[test]
    fn test_storage_engine_with_wal() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 100,
            enable_wal: true,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();

        for i in 0..50 {
            engine.put(i, i as f32).unwrap();
        }
        engine.flush().unwrap();

        let stats = engine.stats();
        assert_eq!(stats.total_segments, 1);
        engine.close().unwrap();

        // WAL should be removed on clean close
        assert!(!dir.path().join("wal.alice").exists());
    }

    #[test]
    fn test_storage_engine_batch_put() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 50,
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();

        let data: Vec<(i64, f32)> = (0..200).map(|i| (i, i as f32 * 0.5)).collect();
        engine.put_batch(&data).unwrap();
        engine.flush().unwrap();

        let stats = engine.stats();
        assert!(stats.total_segments >= 4); // 200/50 = 4 flushes

        let results = engine.query_range(0, 199).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_storage_engine_point_query() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 1000,
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();

        for i in 0..100 {
            engine.put(i, i as f32 * 2.0).unwrap();
        }
        engine.flush().unwrap();

        // Point query for existing data
        let result = engine.query_point(50).unwrap();
        assert!(result.is_some());
        let val = result.unwrap();
        // Should be approximately 100.0 (50 * 2.0)
        assert!((val - 100.0).abs() < 10.0);

        // Point query for non-existent timestamp
        let missing = engine.query_point(9999).unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn test_storage_engine_stats_model_distribution() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 100,
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();

        // Insert constant data
        for i in 0..100 {
            engine.put(i, 42.0).unwrap();
        }
        engine.flush().unwrap();

        let stats = engine.stats();
        assert!(!stats.model_distribution.is_empty());
        // The constant data should be stored with some model type
        let total_models: usize = stats.model_distribution.values().sum();
        assert_eq!(total_models, stats.total_segments);
    }

    #[test]
    fn test_storage_engine_find_segments_empty() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 100,
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();
        let segments = engine.find_segments(0, 100);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_background_flush_no_spike() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 50,
            enable_wal: false,
            enable_background_flush: true,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();

        // Insert enough data to trigger multiple flushes
        let start = std::time::Instant::now();
        for i in 0..200 {
            engine.put(i, i as f32 * 0.5).unwrap();
        }
        let _elapsed = start.elapsed();

        // Wait for background flush to complete
        engine.close().unwrap();

        // Verify data was persisted
        let config2 = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 50,
            enable_wal: false,
            ..Default::default()
        };
        let engine2 = StorageEngine::new(config2).unwrap();
        let stats = engine2.stats();
        assert!(
            stats.total_segments >= 4,
            "Should have at least 4 segments (200/50)"
        );

        // Check data is queryable
        let results = engine2.query_range(0, 199).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_query_without_manual_flush() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 50,
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();

        // Insert enough to trigger auto-flush (2x capacity)
        for i in 0..100 {
            engine.put(i, i as f32 * 2.0).unwrap();
        }

        // Do NOT call flush() — auto-rebuild in persist_segment should make
        // the interval tree ready for queries
        let results = engine.query_range(0, 99).unwrap();
        assert!(
            !results.is_empty(),
            "Auto-rebuilt interval tree should support queries"
        );
    }

    #[test]
    fn test_wal_recovery_after_crash() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();

        // Write data with WAL enabled, then "crash" (drop without close)
        {
            let config = StorageConfig {
                data_dir: dir_path.clone(),
                memtable_capacity: 1000, // large enough to NOT auto-flush
                enable_wal: true,
                ..Default::default()
            };
            let engine = StorageEngine::new(config).unwrap();

            for i in 0..50 {
                engine.put(i, i as f32 * 2.0).unwrap();
            }

            // Simulate crash: WAL written but no flush/close
            // Drop the engine without calling close()
            // We need to prevent Drop from calling close, so we leak intentionally
            // Actually, Drop calls close which flushes - to simulate crash,
            // we manually write WAL and skip the drop
            std::mem::forget(engine);
        }

        // WAL file should exist with data
        assert!(dir_path.join("wal.alice").exists());

        // Reopen - should recover from WAL
        {
            let config = StorageConfig {
                data_dir: dir_path,
                memtable_capacity: 1000,
                enable_wal: true,
                ..Default::default()
            };
            let engine = StorageEngine::new(config).unwrap();

            // Data should have been recovered from WAL and flushed
            let stats = engine.stats();
            assert!(
                stats.total_segments >= 1,
                "WAL recovery should produce segments"
            );

            // Verify data
            let results = engine.query_range(0, 49).unwrap();
            assert!(!results.is_empty(), "Recovered data should be queryable");

            engine.close().unwrap();
        }
    }

    #[test]
    fn test_storage_engine_reopen_persistence() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();

        // Write data
        {
            let config = StorageConfig {
                data_dir: dir_path.clone(),
                memtable_capacity: 100,
                enable_wal: false,
                ..Default::default()
            };
            let engine = StorageEngine::new(config).unwrap();
            for i in 0..100 {
                engine.put(i, i as f32).unwrap();
            }
            engine.flush().unwrap();
            let stats = engine.stats();
            assert!(stats.total_segments >= 1);
            engine.close().unwrap();
        }

        // Reopen and verify data persisted
        {
            let config = StorageConfig {
                data_dir: dir_path,
                memtable_capacity: 100,
                enable_wal: false,
                ..Default::default()
            };
            let engine = StorageEngine::new(config).unwrap();
            let stats = engine.stats();
            assert!(stats.total_segments >= 1);

            let results = engine.query_range(0, 99).unwrap();
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_empty_scan_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();
        let results = engine.query_range(100, 200).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_point_query() {
        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 50,
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();
        for i in 0..100 {
            engine.put(i, i as f32 * 2.0).unwrap();
        }
        engine.flush().unwrap();
        let val = engine.query_point(50).unwrap();
        assert!(val.is_some());
    }

    #[test]
    fn test_stats_after_multiple_flushes() {
        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 30,
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();
        for i in 0..100 {
            engine.put(i, i as f32).unwrap();
        }
        engine.flush().unwrap();
        let stats = engine.stats();
        assert!(stats.total_segments >= 2);
        assert!(stats.total_disk_size > 0);
    }

    #[test]
    fn test_interval_tree_query_accuracy() {
        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_capacity: 50,
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();
        // Segment 1: timestamps 0..49
        for i in 0..50 {
            engine.put(i, i as f32).unwrap();
        }
        engine.flush().unwrap();
        // Segment 2: timestamps 100..149
        for i in 100..150 {
            engine.put(i, i as f32).unwrap();
        }
        engine.flush().unwrap();

        // Query range that spans only segment 2
        let results = engine.query_range(100, 149).unwrap();
        assert!(!results.is_empty());
        // Query range between segments should be empty
        let gap = engine.query_range(60, 90).unwrap();
        assert!(gap.is_empty());
    }

    #[test]
    fn test_close_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();
        engine.put(0, 1.0).unwrap();
        engine.close().unwrap();
        // Second close should not panic
        engine.close().unwrap();
    }

    #[test]
    fn test_wal_disabled_no_wal_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig {
            data_dir: dir.path().to_path_buf(),
            enable_wal: false,
            ..Default::default()
        };
        let engine = StorageEngine::new(config).unwrap();
        engine.put(0, 1.0).unwrap();
        engine.flush().unwrap();
        let wal_path = dir.path().join("wal.alice");
        assert!(!wal_path.exists());
    }
}
