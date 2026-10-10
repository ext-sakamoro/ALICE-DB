//! Exact series: numeric values keyed by `i64`, stored bit for bit
//!
//! The time-series store ([`AliceDB::put`] / [`AliceDB::get`] /
//! [`AliceDB::scan`]) keeps a fitted model per segment and reads its samples
//! back on an even grid between the segment's first and last key. That is
//! the right trade for dense, regularly sampled signals, and the wrong one
//! for records that must come back exactly: metric keys, simulation step
//! counters and other sparse or irregular keys.
//!
//! An exact series keeps every value as one record of the blob key-value
//! store (the path [`crate::law_store`] uses), so:
//!
//! - every value reads back with the bits it was written with (`f32` and
//!   `f64` alike, NaN payloads and signed zeros included)
//! - [`Series::scan`] returns exactly the keys that were written, in
//!   ascending key order, and nothing else
//! - no model is fitted; storage is 8 key bytes plus 4 or 8 value bytes per
//!   point, plus the series name
//! - writing a key again overwrites it; [`Series::delete`] removes it
//!
//! Series are namespaced by name, so separate sinks sharing one database
//! never see each other's points. Both backends are supported: records go
//! to the blob WAL / `SSTable`s on the file backend (durable on return with
//! the default [`crate::blob_wal::SyncPolicy::EveryWrite`]; otherwise call
//! [`AliceDB::flush_blobs`]), stay in memory with [`AliceDB::in_memory`],
//! and are carried by [`AliceDB::to_bytes`] / [`AliceDB::from_bytes`].
//!
//! ```
//! use alice_db::{AliceDB, StorageConfig};
//!
//! let db = AliceDB::in_memory(StorageConfig::default())?;
//! let energy = db.series("energy")?;
//! for step in [0i64, 1, 1 << 20, 1 << 40, (1 << 53) - 1, -7] {
//!     energy.put_f32(step, 0.1 + step as f32)?;
//! }
//! assert_eq!(energy.get_f32(1 << 40)?, Some(0.1 + (1i64 << 40) as f32));
//! let keys: Vec<i64> = energy.scan_f32(0, 1 << 41)?.iter().map(|p| p.0).collect();
//! assert_eq!(keys, [0, 1, 1 << 20, 1 << 40]);
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! # Layout
//!
//! ```text
//! key   = "\0alice-series\0" name "\0" (key as u64 ^ 2^63, 8 bytes big-endian)
//! value = f32 bits (4 bytes little-endian) | f64 bits (8 bytes little-endian)
//! ```
//!
//! Flipping the sign bit makes the byte order of the encoded keys equal to
//! the signed order of the keys, so a prefix scan is already in key order.
//! The value width records the type: a 4-byte record is an `f32`, an 8-byte
//! record an `f64`. Names are non-empty UTF-8 without a NUL byte; the NUL
//! that ends the name keeps one name's keys from being a prefix of another
//! name's. [`SERIES_KEY_PREFIX`] differs from
//! [`crate::law_store::LAW_KEY_PREFIX`] at its eighth byte, so the two key
//! families never overlap.
//!
//! Keys are `i64` because the rest of the crate (`put` / `get` / `scan`)
//! and its callers (simulation steps, packed metric keys, epoch timestamps)
//! use signed 64-bit keys; the encoding keeps their signed order.

use crate::AliceDB;
use std::io;

/// Blob key prefix reserved for exact series records
pub const SERIES_KEY_PREFIX: &[u8] = b"\0alice-series\0";

/// A value stored in an exact series, with the width it was written with
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeriesValue {
    /// Written with an `f32` put (4-byte record)
    F32(f32),
    /// Written with an `f64` put (8-byte record)
    F64(f64),
}

impl SeriesValue {
    /// The value as `f64` (exact for both widths)
    #[must_use]
    pub fn to_f64(self) -> f64 {
        match self {
            Self::F32(v) => f64::from(v),
            Self::F64(v) => v,
        }
    }
}

/// Encode a key so that byte order equals signed numeric order
#[inline]
#[must_use]
pub const fn encode_key(key: i64) -> [u8; 8] {
    ((key as u64) ^ (1 << 63)).to_be_bytes()
}

/// Inverse of [`encode_key`]
#[inline]
#[must_use]
pub const fn decode_key(bytes: [u8; 8]) -> i64 {
    (u64::from_be_bytes(bytes) ^ (1 << 63)) as i64
}

fn decode_value(bytes: &[u8]) -> io::Result<SeriesValue> {
    match bytes.len() {
        4 => {
            let mut b = [0u8; 4];
            b.copy_from_slice(bytes);
            Ok(SeriesValue::F32(f32::from_bits(u32::from_le_bytes(b))))
        }
        8 => {
            let mut b = [0u8; 8];
            b.copy_from_slice(bytes);
            Ok(SeriesValue::F64(f64::from_bits(u64::from_le_bytes(b))))
        }
        n => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("series record holds {n} bytes, expected 4 (f32) or 8 (f64)"),
        )),
    }
}

fn want_f32(v: SeriesValue) -> io::Result<f32> {
    match v {
        SeriesValue::F32(x) => Ok(x),
        SeriesValue::F64(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "series record is an f64; read it with get_f64 / scan_f64 / get",
        )),
    }
}

fn want_f64(v: SeriesValue) -> io::Result<f64> {
    match v {
        SeriesValue::F64(x) => Ok(x),
        SeriesValue::F32(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "series record is an f32; read it with get_f32 / scan_f32 / get",
        )),
    }
}

/// Handle on one named exact series of an [`AliceDB`] (see the module docs)
#[derive(Clone)]
pub struct Series<'a> {
    db: &'a AliceDB,
    /// `SERIES_KEY_PREFIX name "\0"`
    prefix: Vec<u8>,
}

impl std::fmt::Debug for Series<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Series")
            .field("name", &self.name())
            .finish()
    }
}

/// The blob key of `key` in series `name` (see the module docs for the
/// layout)
///
/// # Errors
///
/// `InvalidInput` when `name` is empty or contains a NUL byte.
pub fn series_key(name: &str, key: i64) -> io::Result<Vec<u8>> {
    let mut k = series_prefix(name)?;
    k.extend_from_slice(&encode_key(key));
    Ok(k)
}

fn series_prefix(name: &str) -> io::Result<Vec<u8>> {
    if name.is_empty() || name.as_bytes().contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "series name must be non-empty and contain no NUL byte",
        ));
    }
    let mut p = Vec::with_capacity(SERIES_KEY_PREFIX.len() + name.len() + 1 + 8);
    p.extend_from_slice(SERIES_KEY_PREFIX);
    p.extend_from_slice(name.as_bytes());
    p.push(0);
    Ok(p)
}

impl<'a> Series<'a> {
    pub(crate) fn new(db: &'a AliceDB, name: &str) -> io::Result<Self> {
        Ok(Self {
            db,
            prefix: series_prefix(name)?,
        })
    }

    /// The series name
    #[must_use]
    pub fn name(&self) -> &str {
        let name = &self.prefix[SERIES_KEY_PREFIX.len()..self.prefix.len() - 1];
        // built from a `&str` in `new`
        std::str::from_utf8(name).unwrap_or_default()
    }

    fn key(&self, key: i64) -> Vec<u8> {
        let mut k = Vec::with_capacity(self.prefix.len() + 8);
        k.extend_from_slice(&self.prefix);
        k.extend_from_slice(&encode_key(key));
        k
    }

    /// Store `value` at `key` with its exact `f32` bits
    ///
    /// # Errors
    ///
    /// Returns the error from [`AliceDB::put_blob`].
    pub fn put_f32(&self, key: i64, value: f32) -> io::Result<()> {
        self.db
            .put_blob(&self.key(key), &value.to_bits().to_le_bytes())
    }

    /// Store `value` at `key` with its exact `f64` bits
    ///
    /// # Errors
    ///
    /// Returns the error from [`AliceDB::put_blob`].
    pub fn put_f64(&self, key: i64, value: f64) -> io::Result<()> {
        self.db
            .put_blob(&self.key(key), &value.to_bits().to_le_bytes())
    }

    /// The value stored at `key`, with the width it was written with
    ///
    /// # Errors
    ///
    /// Returns the error from [`AliceDB::get_blob`], or `InvalidData` when
    /// the record is neither 4 nor 8 bytes long.
    pub fn get(&self, key: i64) -> io::Result<Option<SeriesValue>> {
        self.db
            .get_blob(&self.key(key))?
            .map(|b| decode_value(&b))
            .transpose()
    }

    /// The `f32` stored at `key`
    ///
    /// # Errors
    ///
    /// As [`Self::get`], and `InvalidData` when the record is an `f64`.
    pub fn get_f32(&self, key: i64) -> io::Result<Option<f32>> {
        self.get(key)?.map(want_f32).transpose()
    }

    /// The `f64` stored at `key`
    ///
    /// # Errors
    ///
    /// As [`Self::get`], and `InvalidData` when the record is an `f32`.
    pub fn get_f64(&self, key: i64) -> io::Result<Option<f64>> {
        self.get(key)?.map(want_f64).transpose()
    }

    /// Every stored point with `start <= key <= end`, in ascending key order
    ///
    /// Returns exactly the keys that were written (and not deleted); an
    /// empty range (`start > end`) returns nothing.
    ///
    /// # Errors
    ///
    /// Returns the error from [`AliceDB::scan_blob_prefix`], or
    /// `InvalidData` when a record is neither 4 nor 8 bytes long.
    pub fn scan(&self, start: i64, end: i64) -> io::Result<Vec<(i64, SeriesValue)>> {
        if start > end {
            return Ok(Vec::new());
        }
        // Narrow the prefix scan to the bytes the two ends share
        let (lo, hi) = (encode_key(start), encode_key(end));
        let shared = lo.iter().zip(&hi).take_while(|(a, b)| a == b).count();
        let mut prefix = self.prefix.clone();
        prefix.extend_from_slice(&lo[..shared]);
        let mut out = Vec::new();
        for (k, v) in self.db.scan_blob_prefix(&prefix)? {
            let Some(raw) = k
                .strip_prefix(self.prefix.as_slice())
                .and_then(|r| <[u8; 8]>::try_from(r).ok())
            else {
                continue;
            };
            let key = decode_key(raw);
            if key < start || key > end {
                continue;
            }
            out.push((key, decode_value(&v)?));
        }
        Ok(out)
    }

    /// [`Self::scan`] for a series written with `f32` values
    ///
    /// # Errors
    ///
    /// As [`Self::scan`], and `InvalidData` when a record in the range is an
    /// `f64`.
    pub fn scan_f32(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.scan(start, end)?
            .into_iter()
            .map(|(k, v)| want_f32(v).map(|x| (k, x)))
            .collect()
    }

    /// [`Self::scan`] for a series written with `f64` values
    ///
    /// # Errors
    ///
    /// As [`Self::scan`], and `InvalidData` when a record in the range is an
    /// `f32`.
    pub fn scan_f64(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f64)>> {
        self.scan(start, end)?
            .into_iter()
            .map(|(k, v)| want_f64(v).map(|x| (k, x)))
            .collect()
    }

    /// Remove the point at `key` (no-op when absent)
    ///
    /// # Errors
    ///
    /// Returns the error from [`AliceDB::delete_blob`].
    pub fn delete(&self, key: i64) -> io::Result<()> {
        self.db.delete_blob(&self.key(key))
    }
}

impl AliceDB {
    /// Handle on the exact series `name` (see [`crate::series`])
    ///
    /// # Errors
    ///
    /// `InvalidInput` when `name` is empty or contains a NUL byte.
    pub fn series(&self, name: &str) -> io::Result<Series<'_>> {
        Series::new(self, name)
    }
}
