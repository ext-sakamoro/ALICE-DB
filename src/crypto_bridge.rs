//! `ALICE-Crypto` bridge: Encryption at rest for `ALICE-DB`
//!
//! Wraps an `AliceDB` instance to transparently encrypt values before
//! writing and decrypt on read, providing encryption at rest.

use crate::storage_engine::StorageConfig;
use crate::AliceDB;
use alice_crypto::{self as crypto, CipherError, Key};
use std::io;
use std::path::Path;

/// Encrypted wrapper around [`AliceDB`].
///
/// Encryption is handled transparently at the storage layer:
/// - Segments are encrypted before writing to disk
/// - WAL entries are encrypted with length-prefixed sealed format
/// - In-memory data ([`MemTable`](crate::memtable::MemTable), segment cache) remains in plaintext
/// - Timestamps remain in cleartext in the index for range queries
pub struct EncryptedDB {
    inner: AliceDB,
    key: Key,
}

impl EncryptedDB {
    /// Open an encrypted database.
    ///
    /// The encryption key is passed to `StorageConfig` so that the storage
    /// engine handles encryption/decryption transparently at the I/O boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the database directory cannot be created or opened.
    pub fn open<P: AsRef<Path>>(path: P, key: Key) -> io::Result<Self> {
        let config = StorageConfig {
            data_dir: path.as_ref().to_path_buf(),
            encryption_key: Some(key.clone()),
            use_mmap: false, // mmap not compatible with encrypted segments
            ..Default::default()
        };
        let inner = AliceDB::with_config(config)?;
        Ok(Self { inner, key })
    }

    /// Insert a single value.
    /// Encryption is handled transparently by the storage layer.
    ///
    /// # Errors
    ///
    /// Returns an error if the write or WAL operation fails.
    pub fn put(&self, timestamp: i64, value: f32) -> io::Result<()> {
        self.inner.put(timestamp, value)
    }

    /// Insert multiple values.
    ///
    /// # Errors
    ///
    /// Returns an error if any write or WAL operation fails.
    pub fn put_batch(&self, data: &[(i64, f32)]) -> io::Result<()> {
        self.inner.put_batch(data)
    }

    /// Query a single point.
    ///
    /// # Errors
    ///
    /// Returns an error if segment decryption or I/O fails.
    pub fn get(&self, timestamp: i64) -> io::Result<Option<f32>> {
        self.inner.get(timestamp)
    }

    /// Query a time range.
    ///
    /// # Errors
    ///
    /// Returns an error if segment decryption or I/O fails.
    pub fn scan(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.inner.scan(start, end)
    }

    /// Flush to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if segment persistence fails.
    pub fn flush(&self) -> io::Result<()> {
        self.inner.flush()
    }

    /// Get the encryption key (for key rotation).
    #[must_use]
    pub fn key(&self) -> &Key {
        &self.key
    }
}

/// Encrypt arbitrary data for external storage via `ALICE-DB`.
///
/// # Errors
///
/// Returns a `CipherError` if encryption fails.
pub fn seal_blob(key: &Key, data: &[u8]) -> Result<Vec<u8>, CipherError> {
    crypto::seal(key, data)
}

/// Decrypt data retrieved from external storage.
///
/// # Errors
///
/// Returns a `CipherError` if decryption fails (e.g. wrong key or corrupted data).
pub fn open_blob(key: &Key, sealed: &[u8]) -> Result<Vec<u8>, CipherError> {
    crypto::open(key, sealed)
}

/// Derive a database encryption key from a passphrase.
#[must_use]
pub fn derive_db_key(passphrase: &[u8]) -> Key {
    let raw = crypto::derive_key("alice-db-encryption-v1", passphrase);
    Key::from_bytes(raw)
}

/// Content hash for integrity verification (no key needed).
#[must_use]
pub fn content_hash(data: &[u8]) -> crypto::Hash {
    crypto::hash(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_encrypted_roundtrip() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();
        let key = Key::generate().unwrap();

        // Write data and close
        {
            let db = EncryptedDB::open(&dir_path, key.clone()).unwrap();
            for i in 0..50 {
                db.put(i, i as f32 * 0.5).unwrap();
            }
            db.flush().unwrap();

            let results = db.scan(0, 49).unwrap();
            assert!(!results.is_empty());
        }

        // Reopen with same key and verify
        {
            let db = EncryptedDB::open(&dir_path, key).unwrap();
            let results = db.scan(0, 49).unwrap();
            assert!(
                !results.is_empty(),
                "Should read back encrypted data after reopen"
            );
        }
    }

    #[test]
    fn test_encrypted_wal_replay() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();
        let key = Key::generate().unwrap();

        // Write data with WAL, simulate crash
        {
            let config = StorageConfig {
                data_dir: dir_path.clone(),
                memtable_capacity: 10000, // big enough to not auto-flush
                enable_wal: true,
                encryption_key: Some(key.clone()),
                use_mmap: false,
                ..Default::default()
            };
            let db = AliceDB::with_config(config).unwrap();
            for i in 0..20 {
                db.put(i, i as f32 * 3.0).unwrap();
            }
            // Simulate crash: forget without close
            std::mem::forget(db);
        }

        // Reopen with same key - WAL should replay
        {
            let db = EncryptedDB::open(&dir_path, key).unwrap();
            let stats = db.inner.stats();
            assert!(
                stats.total_segments >= 1,
                "WAL replay should produce segments"
            );

            let results = db.scan(0, 19).unwrap();
            assert!(!results.is_empty(), "WAL replay data should be queryable");
        }
    }

    #[test]
    fn test_wrong_key_fails() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();
        let key1 = Key::generate().unwrap();
        let key2 = Key::generate().unwrap();

        // Write with key1
        {
            let db = EncryptedDB::open(&dir_path, key1).unwrap();
            for i in 0..100 {
                db.put(i, i as f32).unwrap();
            }
            db.flush().unwrap();
        }

        // Try to read with key2 - should fail
        {
            let db = EncryptedDB::open(&dir_path, key2).unwrap();
            let result = db.scan(0, 99);
            assert!(result.is_err(), "Wrong key should cause decryption error");
        }
    }

    #[test]
    fn test_seal_open_blob() {
        let key = Key::generate().unwrap();
        let data = b"sensitive database record";
        let sealed = seal_blob(&key, data).unwrap();
        let opened = open_blob(&key, &sealed).unwrap();
        assert_eq!(&opened, data);
    }

    #[test]
    fn test_derive_db_key_deterministic() {
        let k1 = derive_db_key(b"my-passphrase");
        let k2 = derive_db_key(b"my-passphrase");
        assert_eq!(k1.as_bytes(), k2.as_bytes());
    }
}
