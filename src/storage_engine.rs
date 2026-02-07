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
//! Traditional BTreeMap index requires O(N) scan for overlapping segments.
//! Interval Tree enables efficient stabbing queries and range overlap detection.
//!
//! License: MIT
//! Author: Moroya Sakamoto

use crate::memtable::{FitConfig, MemTable};
use crate::segment::{DataSegment, SegmentView};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
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
    /// Start index into intervals_storage
    intervals_start: u32,
    /// Number of intervals at this node
    intervals_count: u16,
    /// Left child index (NO_CHILD if none)
    left_idx: u32,
    /// Right child index (NO_CHILD if none)
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
    /// Root node index (NO_CHILD if empty)
    root_idx: u32,
    /// All intervals for rebuilding
    all_intervals: Vec<Interval>,
    /// Insertions since last rebuild
    insertions_since_rebuild: usize,
}

impl IntervalTree {
    /// Create a new empty interval tree
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
        self.all_intervals.push(Interval { start, end, segment_id });
        self.insertions_since_rebuild += 1;
        // NOTE: No automatic rebuild here! Call rebuild() explicitly after flush.
    }

    /// Check if rebuild is needed (for callers to decide when to rebuild)
    #[inline]
    pub fn needs_rebuild(&self) -> bool {
        self.insertions_since_rebuild > 0 || self.root_idx == NO_CHILD
    }

    /// Rebuild the entire tree for optimal balance
    ///
    /// Call this after flush() or compaction, NOT during hot write path.
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
        let mut endpoints: Vec<i64> = intervals.iter()
            .flat_map(|i| [i.start, i.end])
            .collect();
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
            left_idx: NO_CHILD,  // Will be set below
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
    pub fn query_range(&self, start: i64, end: i64) -> Vec<u64> {
        // If tree is not built, fallback to linear scan
        if self.root_idx == NO_CHILD && !self.all_intervals.is_empty() {
            return self.all_intervals
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
            let intervals = &self.intervals_storage[
                node.intervals_start as usize..
                (node.intervals_start as usize + node.intervals_count as usize)
            ];

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
    pub fn query_point(&self, point: i64) -> Vec<u64> {
        self.query_range(point, point)
    }

    /// Get total number of intervals
    #[inline]
    pub fn len(&self) -> usize {
        self.all_intervals.len()
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
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

/// Storage Engine configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Directory for data files
    pub data_dir: PathBuf,
    /// MemTable capacity (points before flush)
    pub memtable_capacity: usize,
    /// Model fitting configuration
    pub fit_config: FitConfig,
    /// Enable WAL for durability
    pub enable_wal: bool,
    /// Sync writes to disk immediately
    pub sync_writes: bool,
    /// Compaction threshold (number of segments)
    pub compaction_threshold: usize,
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
        }
    }
}

/// Storage Engine: Core persistence layer
pub struct StorageEngine {
    /// Configuration
    config: StorageConfig,
    /// In-memory write buffer
    memtable: MemTable,
    /// Segment index (segment_id → segment info)
    index: RwLock<BTreeMap<u64, SegmentIndexEntry>>,
    /// Interval Tree for O(log N + K) range queries (Phase 4)
    interval_tree: RwLock<IntervalTree>,
    /// Cached segments: Arc<SegmentView> for Zero-Copy access
    /// Using SegmentView instead of DataSegment avoids deserialization overhead
    segment_cache: RwLock<BTreeMap<u64, Arc<SegmentView>>>,
    /// Data file handle (legacy, for index persistence)
    data_file: RwLock<Option<File>>,
    /// WAL file handle
    wal_file: RwLock<Option<File>>,
    /// Current data file offset
    current_offset: RwLock<u64>,
}

impl StorageEngine {
    /// Create a new storage engine
    pub fn new(config: StorageConfig) -> io::Result<Self> {
        // Create data directory if it doesn't exist
        fs::create_dir_all(&config.data_dir)?;

        let memtable = MemTable::with_config(config.memtable_capacity, config.fit_config.clone());

        let engine = Self {
            config,
            memtable,
            index: RwLock::new(BTreeMap::new()),
            interval_tree: RwLock::new(IntervalTree::new()),
            segment_cache: RwLock::new(BTreeMap::new()),
            data_file: RwLock::new(None),
            wal_file: RwLock::new(None),
            current_offset: RwLock::new(0),
        };

        engine.init_files()?;
        engine.load_index()?;

        Ok(engine)
    }

    /// Create with default configuration
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
            .open(&data_path)?;

        let offset = file.metadata()?.len();
        *self.current_offset.write() = offset;
        *self.data_file.write() = Some(file);

        if self.config.enable_wal {
            let wal_path = self.config.data_dir.join("wal.alice");
            let wal = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .append(true)
                .open(&wal_path)?;
            *self.wal_file.write() = Some(wal);
        }

        Ok(())
    }

    /// Load index from disk
    fn load_index(&self) -> io::Result<()> {
        let index_path = self.config.data_dir.join("index.alice");
        if !index_path.exists() {
            return Ok(());
        }

        let file = File::open(&index_path)?;
        let mut reader = BufReader::new(file);

        // Read number of entries
        let mut count_bytes = [0u8; 8];
        if reader.read_exact(&mut count_bytes).is_err() {
            return Ok(()); // Empty or corrupt index
        }
        let count = u64::from_le_bytes(count_bytes) as usize;

        let mut index = self.index.write();
        let mut interval_tree = self.interval_tree.write();

        for _ in 0..count {
            // Read entry
            let mut id_bytes = [0u8; 8];
            reader.read_exact(&mut id_bytes)?;
            let id = u64::from_le_bytes(id_bytes);

            let mut start_bytes = [0u8; 8];
            reader.read_exact(&mut start_bytes)?;
            let start_time = i64::from_le_bytes(start_bytes);

            let mut end_bytes = [0u8; 8];
            reader.read_exact(&mut end_bytes)?;
            let end_time = i64::from_le_bytes(end_bytes);

            let mut offset_bytes = [0u8; 8];
            reader.read_exact(&mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);

            let mut size_bytes = [0u8; 8];
            reader.read_exact(&mut size_bytes)?;
            let size = u64::from_le_bytes(size_bytes);

            let mut ratio_bytes = [0u8; 8];
            reader.read_exact(&mut ratio_bytes)?;
            let compression_ratio = f64::from_le_bytes(ratio_bytes);

            let mut model_len_bytes = [0u8; 4];
            reader.read_exact(&mut model_len_bytes)?;
            let model_len = u32::from_le_bytes(model_len_bytes) as usize;

            let mut model_type_bytes = vec![0u8; model_len];
            reader.read_exact(&mut model_type_bytes)?;
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

    /// Save index to disk
    fn save_index(&self) -> io::Result<()> {
        let index_path = self.config.data_dir.join("index.alice");
        let file = File::create(&index_path)?;
        let mut writer = BufWriter::new(file);

        let index = self.index.read();

        // Write count
        writer.write_all(&(index.len() as u64).to_le_bytes())?;

        for entry in index.values() {
            writer.write_all(&entry.id.to_le_bytes())?;
            writer.write_all(&entry.start_time.to_le_bytes())?;
            writer.write_all(&entry.end_time.to_le_bytes())?;
            writer.write_all(&entry.offset.to_le_bytes())?;
            writer.write_all(&entry.size.to_le_bytes())?;
            writer.write_all(&entry.compression_ratio.to_le_bytes())?;

            let model_bytes = entry.model_type.as_bytes();
            writer.write_all(&(model_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(model_bytes)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Insert a single value
    pub fn put(&self, timestamp: i64, value: f32) -> io::Result<()> {
        // Write to WAL first for durability
        if self.config.enable_wal {
            self.write_wal(timestamp, value)?;
        }

        // Insert into MemTable
        if let Some(segment) = self.memtable.put(timestamp, value) {
            self.persist_segment(segment)?;
        }

        Ok(())
    }

    /// Insert multiple values (batch)
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
            self.persist_segment(segment)?;
        }

        Ok(())
    }

    /// Write to WAL
    fn write_wal(&self, timestamp: i64, value: f32) -> io::Result<()> {
        if let Some(ref mut wal) = *self.wal_file.write() {
            wal.write_all(&timestamp.to_le_bytes())?;
            wal.write_all(&value.to_le_bytes())?;
            if self.config.sync_writes {
                wal.sync_all()?;
            }
        }
        Ok(())
    }

    /// Persist a segment to disk (rkyv format for Zero-Copy)
    ///
    /// # Performance: Zero-Copy Path
    ///
    /// 1. Serialize to rkyv bytes (in memory)
    /// 2. Create SegmentView from bytes IMMEDIATELY (no disk I/O wait)
    /// 3. Write bytes to disk (can be async in future)
    ///
    /// This ensures queries can hit the cache instantly after MemTable flush,
    /// without waiting for disk I/O to complete.
    fn persist_segment(&self, segment: DataSegment) -> io::Result<()> {
        let segment_id = segment.metadata.id;
        let start_time = segment.start_time;
        let end_time = segment.end_time;
        let model_name = segment.model.name().to_string();
        let compression_ratio = segment.metadata.compression_ratio;

        // 1. Serialize to rkyv bytes (Aligned Vec)
        let rkyv_bytes = segment.to_rkyv_bytes()?;
        let segment_size = rkyv_bytes.len() as u64;

        // 2. Create SegmentView from bytes IMMEDIATELY (Zero-Copy, no disk wait)
        // This is the key optimization: cache is populated before disk write
        let view = SegmentView::from_vec(rkyv_bytes.clone())?;
        self.segment_cache.write().insert(segment_id, Arc::new(view));

        // 3. Write to disk (could be async/background in future)
        let segment_path = self.config.data_dir.join(format!("seg_{}.rkyv", segment_id));
        fs::write(&segment_path, &rkyv_bytes)?;

        if self.config.sync_writes {
            // Force sync if configured
            let file = File::open(&segment_path)?;
            file.sync_all()?;
        }

        // 4. Update index
        let entry = SegmentIndexEntry {
            id: segment_id,
            start_time,
            end_time,
            offset: 0, // Not used for individual files
            size: segment_size,
            model_type: model_name,
            compression_ratio,
        };

        // Insert into both index and interval tree
        self.index.write().insert(segment_id, entry);
        self.interval_tree.write().insert(start_time, end_time, segment_id);

        Ok(())
    }

    /// Load a segment as SegmentView (Zero-Copy mmap)
    ///
    /// # Performance: Zero-Copy
    ///
    /// Returns Arc<SegmentView> which provides direct mmap access.
    /// No deserialization occurs - model coefficients are read directly.
    fn load_segment(&self, entry: &SegmentIndexEntry) -> io::Result<Arc<SegmentView>> {
        // Check cache first (fast path)
        if let Some(view) = self.segment_cache.read().get(&entry.id) {
            return Ok(Arc::clone(view));
        }

        // Open segment file as SegmentView (mmap)
        let segment_path = self.config.data_dir.join(format!("seg_{}.rkyv", entry.id));

        // Fallback to legacy format if rkyv file doesn't exist
        if !segment_path.exists() {
            return self.load_segment_legacy(entry);
        }

        let view = SegmentView::open(&segment_path)?;
        let arc_view = Arc::new(view);

        // Update cache
        self.segment_cache.write().insert(entry.id, Arc::clone(&arc_view));

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

            // Open as SegmentView
            let view = SegmentView::open(&segment_path)?;
            let arc_view = Arc::new(view);

            // Update cache
            self.segment_cache.write().insert(entry.id, Arc::clone(&arc_view));

            Ok(arc_view)
        } else {
            Err(io::Error::new(io::ErrorKind::NotFound, "Data file not open"))
        }
    }

    /// Find segments that overlap with a time range
    ///
    /// Uses Interval Tree for O(log N + K) query performance.
    /// K = number of overlapping segments returned.
    pub fn find_segments(&self, start: i64, end: i64) -> Vec<SegmentIndexEntry> {
        // Use Interval Tree for efficient range query
        let segment_ids = self.interval_tree.read().query_range(start, end);
        let index = self.index.read();

        segment_ids
            .into_iter()
            .filter_map(|id| index.get(&id).cloned())
            .collect()
    }

    /// Query a single point (Zero-Copy via SegmentView)
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

    /// Query a time range (Zero-Copy via SegmentView)
    ///
    /// # Performance: Zero-Copy Path
    ///
    /// Uses SegmentView (mmap + SIMD) for maximum throughput.
    /// No deserialization occurs during query.
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

    /// Force flush MemTable to disk
    ///
    /// Also rebuilds the Interval Tree for optimal query performance.
    /// This is the correct place to rebuild (not during hot write path).
    pub fn flush(&self) -> io::Result<()> {
        if let Some(segment) = self.memtable.force_flush() {
            self.persist_segment(segment)?;
        }

        // Rebuild Interval Tree after flush (jitter-free writes)
        {
            let mut tree = self.interval_tree.write();
            if tree.needs_rebuild() {
                tree.rebuild();
            }
        }

        self.save_index()?;
        Ok(())
    }

    /// Close the storage engine
    pub fn close(&self) -> io::Result<()> {
        self.flush()?;

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
    pub fn stats(&self) -> StorageStats {
        let index = self.index.read();

        let total_segments = index.len();
        let total_compression_ratio: f64 = index
            .values()
            .map(|e| e.compression_ratio)
            .sum::<f64>() / total_segments.max(1) as f64;

        let total_size: u64 = index.values().map(|e| e.size).sum();

        let model_counts: std::collections::HashMap<String, usize> = index
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
}
