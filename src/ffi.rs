//! FFI bindings for ALICE-DB (C/C++/C# interop)
//!
//! 16 exported functions for cross-language integration.
//!
//! Author: Moroya Sakamoto

use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::sync::Mutex;

use crate::{Aggregation, AliceDB, StorageConfig};

// ============================================================================
// Types
// ============================================================================

/// Opaque handle to an ALICE-DB instance
pub type DbHandle = *mut std::ffi::c_void;

/// Null handle constant
pub const DB_HANDLE_NULL: DbHandle = ptr::null_mut();

/// Result code for FFI operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbResult {
    /// Operation succeeded
    Ok = 0,
    /// Invalid handle provided
    InvalidHandle = 1,
    /// Null pointer provided
    NullPointer = 2,
    /// Invalid parameter value
    InvalidParameter = 3,
    /// I/O error
    IoError = 4,
    /// Database already closed
    Closed = 5,
    /// Unknown error
    Unknown = 99,
}

/// Aggregation type for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    /// Sum of all values
    Sum = 0,
    /// Average of all values
    Avg = 1,
    /// Minimum value
    Min = 2,
    /// Maximum value
    Max = 3,
    /// Number of data points
    Count = 4,
    /// First value in range
    First = 5,
    /// Last value in range
    Last = 6,
    /// Standard deviation
    StdDev = 7,
    /// Variance
    Variance = 8,
}

/// Database statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DbStats {
    /// Total number of segments on disk
    pub total_segments: u64,
    /// Number of entries in the memtable
    pub memtable_size: u64,
    /// Total disk usage in bytes
    pub total_disk_size: u64,
    /// Average compression ratio
    pub average_compression_ratio: f64,
}

/// Point result (timestamp + value)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PointResult {
    /// Timestamp
    pub timestamp: i64,
    /// Value
    pub value: f32,
    /// Whether the query found a result
    pub found: bool,
}

/// Version information
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VersionInfo {
    /// Major version number
    pub major: u16,
    /// Minor version number
    pub minor: u16,
    /// Patch version number
    pub patch: u16,
}

// ============================================================================
// Internal helpers
// ============================================================================

struct DbWrapper {
    db: Mutex<Option<AliceDB>>,
}

const fn agg_from_ffi(agg: AggregationType) -> Aggregation {
    match agg {
        AggregationType::Sum => Aggregation::Sum,
        AggregationType::Avg => Aggregation::Avg,
        AggregationType::Min => Aggregation::Min,
        AggregationType::Max => Aggregation::Max,
        AggregationType::Count => Aggregation::Count,
        AggregationType::First => Aggregation::First,
        AggregationType::Last => Aggregation::Last,
        AggregationType::StdDev => Aggregation::StdDev,
        AggregationType::Variance => Aggregation::Variance,
    }
}

fn with_db<F, T>(handle: DbHandle, f: F) -> Result<T, DbResult>
where
    F: FnOnce(&AliceDB) -> Result<T, DbResult>,
{
    if handle.is_null() {
        return Err(DbResult::InvalidHandle);
    }
    // SAFETY: `handle` has been checked non-null above. Callers must
    // provide a handle previously returned by `alice_db_open`, which
    // guarantees the pointer originates from `Box::into_raw(DbWrapper)`.
    let wrapper = unsafe { &*(handle as *const DbWrapper) };
    let guard = wrapper.db.lock().map_err(|_| DbResult::Unknown)?;
    let db = guard.as_ref().ok_or(DbResult::Closed)?;
    f(db)
}

// ============================================================================
// Exported Functions
// ============================================================================

/// Get version info
#[no_mangle]
pub const extern "C" fn alice_db_version() -> VersionInfo {
    VersionInfo {
        major: 0,
        minor: 1,
        patch: 0,
    }
}

/// Open a database at the specified path
///
/// # Safety
///
/// `path` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn alice_db_open(path: *const c_char) -> DbHandle {
    if path.is_null() {
        return DB_HANDLE_NULL;
    }
    // SAFETY: `path` has been checked non-null. The `# Safety` contract
    // requires the caller to pass a valid null-terminated UTF-8 string.
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return DB_HANDLE_NULL,
    };
    match AliceDB::open(path_str) {
        Ok(db) => {
            let wrapper = Box::new(DbWrapper {
                db: Mutex::new(Some(db)),
            });
            Box::into_raw(wrapper) as DbHandle
        }
        Err(_) => DB_HANDLE_NULL,
    }
}

/// Open with custom configuration
///
/// # Safety
///
/// `path` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn alice_db_open_with_config(
    path: *const c_char,
    memtable_capacity: u32,
    enable_wal: bool,
    sync_writes: bool,
) -> DbHandle {
    if path.is_null() {
        return DB_HANDLE_NULL;
    }
    // SAFETY: `path` has been checked non-null. The `# Safety` contract
    // requires the caller to pass a valid null-terminated UTF-8 string.
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return DB_HANDLE_NULL,
    };
    let config = StorageConfig {
        data_dir: std::path::PathBuf::from(path_str),
        memtable_capacity: memtable_capacity as usize,
        enable_wal,
        sync_writes,
        ..Default::default()
    };
    match AliceDB::with_config(config) {
        Ok(db) => {
            let wrapper = Box::new(DbWrapper {
                db: Mutex::new(Some(db)),
            });
            Box::into_raw(wrapper) as DbHandle
        }
        Err(_) => DB_HANDLE_NULL,
    }
}

/// Close and free a database handle
///
/// # Safety
///
/// `handle` must be a valid handle returned by `alice_db_open`.
#[no_mangle]
pub unsafe extern "C" fn alice_db_close(handle: DbHandle) -> DbResult {
    if handle.is_null() {
        return DbResult::InvalidHandle;
    }
    // SAFETY: `handle` has been checked non-null. The `# Safety` contract
    // requires this to be a handle returned by `alice_db_open`. We reclaim
    // the `Box` to drop the `DbWrapper` after closing.
    let wrapper = unsafe { Box::from_raw(handle.cast::<DbWrapper>()) };
    let mut guard = match wrapper.db.lock() {
        Ok(g) => g,
        Err(_) => return DbResult::Unknown,
    };
    if let Some(db) = guard.take() {
        match db.close() {
            Ok(()) => DbResult::Ok,
            Err(_) => DbResult::IoError,
        }
    } else {
        DbResult::Closed
    }
}

/// Insert a single value
#[no_mangle]
pub extern "C" fn alice_db_put(handle: DbHandle, timestamp: i64, value: f32) -> DbResult {
    match with_db(handle, |db| {
        db.put(timestamp, value).map_err(|_| DbResult::IoError)
    }) {
        Ok(()) => DbResult::Ok,
        Err(e) => e,
    }
}

/// Insert multiple values (batch)
///
/// # Safety
///
/// `timestamps` and `values` must point to arrays of at least `count` elements.
#[no_mangle]
pub unsafe extern "C" fn alice_db_put_batch(
    handle: DbHandle,
    timestamps: *const i64,
    values: *const f32,
    count: u32,
) -> DbResult {
    if timestamps.is_null() || values.is_null() {
        return DbResult::NullPointer;
    }
    // SAFETY: `timestamps` and `values` have been checked non-null. The
    // `# Safety` contract requires them to point to arrays of `count` elements.
    let ts = unsafe { std::slice::from_raw_parts(timestamps, count as usize) };
    let vs = unsafe { std::slice::from_raw_parts(values, count as usize) };
    let data: Vec<(i64, f32)> = ts.iter().zip(vs.iter()).map(|(&t, &v)| (t, v)).collect();
    match with_db(handle, |db| {
        db.put_batch(&data).map_err(|_| DbResult::IoError)
    }) {
        Ok(()) => DbResult::Ok,
        Err(e) => e,
    }
}

/// Query a single point
#[no_mangle]
pub extern "C" fn alice_db_get(handle: DbHandle, timestamp: i64) -> PointResult {
    let not_found = PointResult {
        timestamp,
        value: 0.0,
        found: false,
    };
    match with_db(handle, |db| {
        db.get(timestamp).map_err(|_| DbResult::IoError)
    }) {
        Ok(Some(v)) => PointResult {
            timestamp,
            value: v,
            found: true,
        },
        _ => not_found,
    }
}

/// Query a time range, writing results to caller-allocated buffers
///
/// Returns the number of results written. If `out_timestamps` or `out_values` is null,
/// returns the number of results available (for pre-allocation).
///
/// # Safety
///
/// Output buffers must have capacity for at least `max_count` elements.
#[no_mangle]
pub unsafe extern "C" fn alice_db_scan(
    handle: DbHandle,
    start: i64,
    end: i64,
    out_timestamps: *mut i64,
    out_values: *mut f32,
    max_count: u32,
) -> i32 {
    let results = match with_db(handle, |db| {
        db.scan(start, end).map_err(|_| DbResult::IoError)
    }) {
        Ok(r) => r,
        Err(_) => return -1,
    };
    if out_timestamps.is_null() || out_values.is_null() {
        return results.len() as i32;
    }
    let n = results.len().min(max_count as usize);
    // SAFETY: We checked non-null above. The `# Safety` contract requires
    // output buffers to have capacity for at least `max_count` elements.
    // `n` is clamped to `max_count` so we stay within bounds.
    let ts_out = unsafe { std::slice::from_raw_parts_mut(out_timestamps, n) };
    let vs_out = unsafe { std::slice::from_raw_parts_mut(out_values, n) };
    for (i, &(t, v)) in results.iter().take(n).enumerate() {
        ts_out[i] = t;
        vs_out[i] = v;
    }
    n as i32
}

/// Aggregation query
///
/// # Safety
///
/// `out_value` must point to a valid f64.
#[no_mangle]
pub unsafe extern "C" fn alice_db_aggregate(
    handle: DbHandle,
    start: i64,
    end: i64,
    agg: AggregationType,
    out_value: *mut f64,
) -> DbResult {
    if out_value.is_null() {
        return DbResult::NullPointer;
    }
    match with_db(handle, |db| {
        db.aggregate(start, end, agg_from_ffi(agg))
            .map_err(|_| DbResult::IoError)
    }) {
        Ok(val) => {
            // SAFETY: `out_value` has been checked non-null above.
            unsafe { *out_value = val };
            DbResult::Ok
        }
        Err(e) => e,
    }
}

/// Downsampling query
///
/// # Safety
///
/// Output buffers must have capacity for at least `max_count` elements.
#[no_mangle]
pub unsafe extern "C" fn alice_db_downsample(
    handle: DbHandle,
    start: i64,
    end: i64,
    interval: i64,
    agg: AggregationType,
    out_timestamps: *mut i64,
    out_values: *mut f64,
    max_count: u32,
) -> i32 {
    let results = match with_db(handle, |db| {
        db.downsample(start, end, interval, agg_from_ffi(agg))
            .map_err(|_| DbResult::IoError)
    }) {
        Ok(r) => r,
        Err(_) => return -1,
    };
    if out_timestamps.is_null() || out_values.is_null() {
        return results.len() as i32;
    }
    let n = results.len().min(max_count as usize);
    // SAFETY: We checked non-null above. The `# Safety` contract requires
    // output buffers to have capacity for at least `max_count` elements.
    let ts_out = unsafe { std::slice::from_raw_parts_mut(out_timestamps, n) };
    let vs_out = unsafe { std::slice::from_raw_parts_mut(out_values, n) };
    for (i, &(t, v)) in results.iter().take(n).enumerate() {
        ts_out[i] = t;
        vs_out[i] = v;
    }
    n as i32
}

/// Force flush memtable to disk
#[no_mangle]
pub extern "C" fn alice_db_flush(handle: DbHandle) -> DbResult {
    match with_db(handle, |db| db.flush().map_err(|_| DbResult::IoError)) {
        Ok(()) => DbResult::Ok,
        Err(e) => e,
    }
}

/// Get database statistics
#[no_mangle]
pub extern "C" fn alice_db_stats(handle: DbHandle) -> DbStats {
    let empty = DbStats {
        total_segments: 0,
        memtable_size: 0,
        total_disk_size: 0,
        average_compression_ratio: 0.0,
    };
    match with_db(handle, |db| {
        let s = db.stats();
        Ok(DbStats {
            total_segments: s.total_segments as u64,
            memtable_size: s.memtable_size as u64,
            total_disk_size: s.total_disk_size,
            average_compression_ratio: s.average_compression_ratio,
        })
    }) {
        Ok(s) => s,
        Err(_) => empty,
    }
}

/// Check if a handle is valid
#[no_mangle]
pub extern "C" fn alice_db_is_valid(handle: DbHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: `handle` has been checked non-null above. Callers must
    // provide a handle previously returned by `alice_db_open`.
    let wrapper = unsafe { &*(handle as *const DbWrapper) };
    match wrapper.db.lock() {
        Ok(guard) => guard.is_some(),
        Err(_) => false,
    }
}

/// Free a C string returned by ALICE-DB
///
/// # Safety
///
/// `s` must be a string allocated by ALICE-DB FFI functions.
#[no_mangle]
pub unsafe extern "C" fn alice_db_free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: `s` has been checked non-null. The `# Safety` contract
        // requires it to be a string allocated by ALICE-DB FFI functions.
        drop(unsafe { CString::from_raw(s) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn open_temp_db() -> (tempfile::TempDir, DbHandle) {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let handle = unsafe { alice_db_open(path.as_ptr()) };
        assert!(!handle.is_null());
        (dir, handle)
    }

    #[test]
    fn test_version() {
        let v = alice_db_version();
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_open_and_close() {
        let (_dir, handle) = open_temp_db();
        assert!(alice_db_is_valid(handle));
        let result = unsafe { alice_db_close(handle) };
        assert_eq!(result, DbResult::Ok);
    }

    #[test]
    fn test_open_null_path() {
        let handle = unsafe { alice_db_open(ptr::null()) };
        assert!(handle.is_null());
    }

    #[test]
    fn test_open_with_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let handle = unsafe { alice_db_open_with_config(path.as_ptr(), 50, true, false) };
        assert!(!handle.is_null());
        assert!(alice_db_is_valid(handle));
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_is_valid_null() {
        assert!(!alice_db_is_valid(ptr::null_mut()));
    }

    #[test]
    fn test_put_single() {
        let (_dir, handle) = open_temp_db();
        let result = alice_db_put(handle, 100, 42.0);
        assert_eq!(result, DbResult::Ok);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_put_invalid_handle() {
        let result = alice_db_put(ptr::null_mut(), 100, 42.0);
        assert_eq!(result, DbResult::InvalidHandle);
    }

    #[test]
    fn test_put_batch() {
        let (_dir, handle) = open_temp_db();
        let timestamps: Vec<i64> = (0..100).collect();
        let values: Vec<f32> = (0..100).map(|i| i as f32 * 2.0).collect();
        let result =
            unsafe { alice_db_put_batch(handle, timestamps.as_ptr(), values.as_ptr(), 100) };
        assert_eq!(result, DbResult::Ok);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_put_batch_null_pointer() {
        let (_dir, handle) = open_temp_db();
        let result = unsafe { alice_db_put_batch(handle, ptr::null(), ptr::null(), 10) };
        assert_eq!(result, DbResult::NullPointer);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_get_point() {
        let (_dir, handle) = open_temp_db();
        for i in 0..100 {
            alice_db_put(handle, i, i as f32 * 3.0);
        }
        let flush_result = alice_db_flush(handle);
        assert_eq!(flush_result, DbResult::Ok);

        let point = alice_db_get(handle, 50);
        assert!(point.found);
        assert_eq!(point.timestamp, 50);
        assert!((point.value - 150.0).abs() < 20.0);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_get_missing_point() {
        let (_dir, handle) = open_temp_db();
        for i in 0..100 {
            alice_db_put(handle, i, i as f32);
        }
        alice_db_flush(handle);

        let point = alice_db_get(handle, 9999);
        assert!(!point.found);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_scan_count_only() {
        let (_dir, handle) = open_temp_db();
        for i in 0..100 {
            alice_db_put(handle, i, i as f32);
        }
        alice_db_flush(handle);

        let count = unsafe { alice_db_scan(handle, 0, 99, ptr::null_mut(), ptr::null_mut(), 0) };
        assert!(count > 0);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_scan_with_buffers() {
        let (_dir, handle) = open_temp_db();
        for i in 0..100 {
            alice_db_put(handle, i, i as f32);
        }
        alice_db_flush(handle);

        let mut ts = vec![0i64; 200];
        let mut vs = vec![0f32; 200];
        let n = unsafe { alice_db_scan(handle, 0, 99, ts.as_mut_ptr(), vs.as_mut_ptr(), 200) };
        assert!(n > 0);
        // 最初のデータポイントの検証
        assert!(ts[0] >= 0 && ts[0] <= 99);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_aggregate() {
        let (_dir, handle) = open_temp_db();
        for i in 0..100 {
            alice_db_put(handle, i, i as f32);
        }
        alice_db_flush(handle);

        let mut val: f64 = 0.0;
        let result =
            unsafe { alice_db_aggregate(handle, 0, 99, AggregationType::Avg, &mut val as *mut _) };
        assert_eq!(result, DbResult::Ok);
        // Average of 0..99 ≈ 49.5
        assert!(val > 30.0 && val < 70.0);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_aggregate_null_out() {
        let (_dir, handle) = open_temp_db();
        let result =
            unsafe { alice_db_aggregate(handle, 0, 99, AggregationType::Sum, ptr::null_mut()) };
        assert_eq!(result, DbResult::NullPointer);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_downsample() {
        let (_dir, handle) = open_temp_db();
        for i in 0..100 {
            alice_db_put(handle, i, i as f32);
        }
        alice_db_flush(handle);

        let mut ts = vec![0i64; 10];
        let mut vs = vec![0f64; 10];
        let n = unsafe {
            alice_db_downsample(
                handle,
                0,
                99,
                25,
                AggregationType::Avg,
                ts.as_mut_ptr(),
                vs.as_mut_ptr(),
                10,
            )
        };
        assert!(n >= 0);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_flush() {
        let (_dir, handle) = open_temp_db();
        alice_db_put(handle, 0, 1.0);
        let result = alice_db_flush(handle);
        assert_eq!(result, DbResult::Ok);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_stats() {
        let (_dir, handle) = open_temp_db();
        for i in 0..100 {
            alice_db_put(handle, i, i as f32);
        }
        alice_db_flush(handle);

        let stats = alice_db_stats(handle);
        assert!(stats.total_segments >= 1);
        assert!(stats.average_compression_ratio > 1.0);
        unsafe { alice_db_close(handle) };
    }

    #[test]
    fn test_stats_invalid_handle() {
        let stats = alice_db_stats(ptr::null_mut());
        assert_eq!(stats.total_segments, 0);
        assert_eq!(stats.average_compression_ratio, 0.0);
    }

    #[test]
    fn test_close_null_handle() {
        let result = unsafe { alice_db_close(ptr::null_mut()) };
        assert_eq!(result, DbResult::InvalidHandle);
    }

    #[test]
    fn test_full_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let handle = unsafe { alice_db_open_with_config(path.as_ptr(), 50, false, false) };
        assert!(!handle.is_null());
        assert!(alice_db_is_valid(handle));

        // Batch insert
        let timestamps: Vec<i64> = (0..200).collect();
        let values: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
        let result =
            unsafe { alice_db_put_batch(handle, timestamps.as_ptr(), values.as_ptr(), 200) };
        assert_eq!(result, DbResult::Ok);

        // Flush
        assert_eq!(alice_db_flush(handle), DbResult::Ok);

        // Stats
        let stats = alice_db_stats(handle);
        assert!(stats.total_segments >= 1);

        // Point query
        let point = alice_db_get(handle, 100);
        assert!(point.found);

        // Scan
        let count = unsafe { alice_db_scan(handle, 0, 199, ptr::null_mut(), ptr::null_mut(), 0) };
        assert!(count > 0);

        // Aggregate
        let mut avg: f64 = 0.0;
        let agg_result =
            unsafe { alice_db_aggregate(handle, 0, 199, AggregationType::Avg, &mut avg as *mut _) };
        assert_eq!(agg_result, DbResult::Ok);

        // Close
        assert_eq!(unsafe { alice_db_close(handle) }, DbResult::Ok);
    }
}
