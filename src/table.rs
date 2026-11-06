//! Records with secondary indexes on the blob store.
//!
//! A [`Table`] keeps each row and its index entries in one
//! [`BlobStorage::write_batch`](crate::blob::BlobStorage::write_batch), so they change
//! together or not at all, and checks a unique index in the same exclusive step as
//! the write, so two writers cannot both take a value.
//!
//! # Keys
//!
//! ```text
//! row    NS name 0x01 primary-key
//! index  NS name 0x02 index key primary-key
//! ```
//!
//! `NS` is `\0alice-table\0`; `name`, `index` and `key` are escaped (0x00 becomes
//! 0x00 0xFF) and end with 0x00 0x01, which keeps byte order and lets no field run
//! into the next. An index entry's value is empty.
//!
//! # Rows
//!
//! ```text
//! version u8 (= 1), entry-count u32 LE, then per entry:
//!   unique u8, name-len u32 LE, index name, key-len u32 LE, index key
//! then the value
//! ```
//!
//! The row lists its entries, so an upsert or delete removes exactly the ones it
//! wrote. Index keys and primary keys are stored as given, never sealed.

use crate::blob::{prefix_end, BlobStorage, ScanOrder, WriteBatch};
use std::fmt;
use std::io;
use std::ops::Bound;

const NAMESPACE: &[u8] = b"\0alice-table\0";
const ROWS: u8 = 1;
const INDEX: u8 = 2;
const ROW_VERSION: u8 = 1;
const TERMINATOR: [u8; 2] = [0x00, 0x01];
/// Sorts after every key equal to the field it follows and before any longer one.
const PAST_TERMINATOR: [u8; 2] = [0x00, 0x02];

/// `(primary key, value)` rows, or `(index key, primary key)` entries.
pub type Pairs = Vec<(Vec<u8>, Vec<u8>)>;

/// One index entry of a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexKey<'a> {
    index: &'a str,
    key: &'a [u8],
    unique: bool,
}

impl<'a> IndexKey<'a> {
    /// An entry of an index in which only one row may hold a key.
    #[must_use]
    pub const fn unique(index: &'a str, key: &'a [u8]) -> Self {
        Self {
            index,
            key,
            unique: true,
        }
    }

    /// An entry of an index in which many rows may hold a key.
    #[must_use]
    pub const fn non_unique(index: &'a str, key: &'a [u8]) -> Self {
        Self {
            index,
            key,
            unique: false,
        }
    }
}

/// Why a table operation did not take effect. Nothing was changed.
#[derive(Debug)]
#[non_exhaustive]
pub enum TableError {
    /// [`Table::insert`] found a row with this primary key.
    AlreadyExists,
    /// Another row holds this key of a unique index.
    UniqueViolation {
        /// The name of the index
        index: String,
    },
    /// The table name is empty.
    EmptyName,
    /// A stored row does not parse.
    Corrupt,
    /// A sealed value does not open for this row and key.
    #[cfg(feature = "crypto")]
    Seal,
    /// The key provider did not give a key.
    #[cfg(feature = "crypto")]
    Key(alice_crypto::keystore::KeyProviderError),
    /// The store failed; [`io::ErrorKind::PermissionDenied`] on a read-only store.
    Io(io::Error),
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyExists => f.write_str("a row with this primary key exists"),
            Self::UniqueViolation { index } => {
                write!(
                    f,
                    "another row holds this key of the unique index `{index}`"
                )
            }
            Self::EmptyName => f.write_str("a table needs a name"),
            Self::Corrupt => f.write_str("a stored row does not parse"),
            #[cfg(feature = "crypto")]
            Self::Seal => f.write_str("a sealed value does not open for this row"),
            #[cfg(feature = "crypto")]
            Self::Key(_) => f.write_str("the key provider did not give a key"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TableError {}

impl From<io::Error> for TableError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

fn push_escaped(out: &mut Vec<u8>, field: &[u8]) {
    for &b in field {
        out.push(b);
        if b == 0 {
            out.push(0xFF);
        }
    }
}

fn push_field(out: &mut Vec<u8>, field: &[u8]) {
    push_escaped(out, field);
    out.extend_from_slice(&TERMINATOR);
}

/// Split an escaped, terminated field off the front of `bytes`.
fn split_field(bytes: &[u8]) -> Option<(Vec<u8>, &[u8])> {
    let mut field = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0 {
            match *bytes.get(i + 1)? {
                0xFF => field.push(0),
                0x01 => return Some((field, &bytes[i + 2..])),
                _ => return None,
            }
            i += 2;
        } else {
            field.push(bytes[i]);
            i += 1;
        }
    }
    None
}

/// A named set of rows in a [`BlobStorage`]. It holds no state of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    rows: Vec<u8>,
    indexes: Vec<u8>,
}

impl Table {
    /// The table with this name.
    ///
    /// # Errors
    /// [`TableError::EmptyName`] for an empty name.
    pub fn new(name: &str) -> Result<Self, TableError> {
        if name.is_empty() {
            return Err(TableError::EmptyName);
        }
        let mut base = NAMESPACE.to_vec();
        push_field(&mut base, name.as_bytes());
        let mut rows = base.clone();
        rows.push(ROWS);
        let mut indexes = base;
        indexes.push(INDEX);
        Ok(Self { rows, indexes })
    }

    fn row_key(&self, pk: &[u8]) -> Vec<u8> {
        [self.rows.as_slice(), pk].concat()
    }

    fn index_prefix(&self, index: &str) -> Vec<u8> {
        let mut out = self.indexes.clone();
        push_field(&mut out, index.as_bytes());
        out
    }

    /// Every entry of `index` holding exactly `key` starts with this.
    fn entry_prefix(&self, index: &str, key: &[u8]) -> Vec<u8> {
        let mut out = self.index_prefix(index);
        push_field(&mut out, key);
        out
    }

    /// Add a row; a row with this primary key is an error.
    ///
    /// # Errors
    /// [`TableError::AlreadyExists`], [`TableError::UniqueViolation`], or a store error.
    pub fn insert(
        &self,
        store: &BlobStorage,
        pk: &[u8],
        value: &[u8],
        entries: &[IndexKey<'_>],
    ) -> Result<(), TableError> {
        store.exclusive(|| {
            if store.get(&self.row_key(pk))?.is_some() {
                return Err(TableError::AlreadyExists);
            }
            self.write_row(store, pk, value, entries, None)
        })
    }

    /// Add or replace a row, moving its index entries.
    ///
    /// # Errors
    /// [`TableError::UniqueViolation`], [`TableError::Corrupt`], or a store error.
    pub fn upsert(
        &self,
        store: &BlobStorage,
        pk: &[u8],
        value: &[u8],
        entries: &[IndexKey<'_>],
    ) -> Result<(), TableError> {
        store.exclusive(|| {
            let old = store.get(&self.row_key(pk))?;
            self.write_row(store, pk, value, entries, old.as_deref())
        })
    }

    fn write_row(
        &self,
        store: &BlobStorage,
        pk: &[u8],
        value: &[u8],
        entries: &[IndexKey<'_>],
        old_row: Option<&[u8]>,
    ) -> Result<(), TableError> {
        for entry in entries.iter().filter(|e| e.unique) {
            let prefix = self.entry_prefix(entry.index, entry.key);
            let end = prefix_end(&prefix);
            // Two rows at most can matter: this one and another.
            let holders = store.scan_range(
                Bound::Included(&prefix),
                end.as_deref().map_or(Bound::Unbounded, Bound::Excluded),
                ScanOrder::Ascending,
                2,
            )?;
            if holders.iter().any(|(k, _)| &k[prefix.len()..] != pk) {
                return Err(TableError::UniqueViolation {
                    index: entry.index.to_string(),
                });
            }
        }
        let mut batch = WriteBatch::new();
        if let Some(old) = old_row {
            self.delete_entries(&mut batch, pk, old)?;
        }
        for entry in entries {
            let mut key = self.entry_prefix(entry.index, entry.key);
            key.extend_from_slice(pk);
            batch.put(&key, &[]);
        }
        batch.put(&self.row_key(pk), &encode_row(entries, value)?);
        store.apply_exclusive(&batch)?;
        Ok(())
    }

    fn delete_entries(
        &self,
        batch: &mut WriteBatch,
        pk: &[u8],
        row: &[u8],
    ) -> Result<(), TableError> {
        let (old_entries, _) = decode_row(row).ok_or(TableError::Corrupt)?;
        for (index, key) in old_entries {
            let index = std::str::from_utf8(index).map_err(|_| TableError::Corrupt)?;
            let mut entry = self.entry_prefix(index, key);
            entry.extend_from_slice(pk);
            batch.delete(&entry);
        }
        Ok(())
    }

    /// Remove a row and its index entries; returns whether there was one.
    ///
    /// # Errors
    /// [`TableError::Corrupt`] or a store error.
    pub fn delete(&self, store: &BlobStorage, pk: &[u8]) -> Result<bool, TableError> {
        store.exclusive(|| {
            let row_key = self.row_key(pk);
            let Some(old) = store.get(&row_key)? else {
                // a read-only store refuses this too, whatever it holds
                store.apply_exclusive(&WriteBatch::new())?;
                return Ok(false);
            };
            let mut batch = WriteBatch::new();
            self.delete_entries(&mut batch, pk, &old)?;
            batch.delete(&row_key);
            store.apply_exclusive(&batch)?;
            Ok(true)
        })
    }

    /// The value of a row.
    ///
    /// # Errors
    /// [`TableError::Corrupt`] or a store error.
    pub fn get(&self, store: &BlobStorage, pk: &[u8]) -> Result<Option<Vec<u8>>, TableError> {
        store
            .get(&self.row_key(pk))?
            .map(|row| {
                decode_row(&row)
                    .map(|(_, value)| value.to_vec())
                    .ok_or(TableError::Corrupt)
            })
            .transpose()
    }

    /// Rows between two primary keys, `(primary key, value)`, at most `limit`.
    ///
    /// # Errors
    /// [`TableError::Corrupt`] or a store error.
    pub fn scan(
        &self,
        store: &BlobStorage,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        order: ScanOrder,
        limit: usize,
    ) -> Result<Pairs, TableError> {
        let inside = |bound: Bound<&[u8]>| bound.map(|pk| self.row_key(pk));
        let (lower, upper) = (inside(start), inside(end));
        let table_end = prefix_end(&self.rows);
        let lower = match &lower {
            Bound::Unbounded => Bound::Included(self.rows.as_slice()),
            other => as_slices(other),
        };
        let upper = match &upper {
            Bound::Unbounded => table_end
                .as_deref()
                .map_or(Bound::Unbounded, Bound::Excluded),
            other => as_slices(other),
        };
        store
            .scan_range(lower, upper, order, limit)?
            .into_iter()
            .map(|(k, row)| {
                let (_, value) = decode_row(&row).ok_or(TableError::Corrupt)?;
                Ok((k[self.rows.len()..].to_vec(), value.to_vec()))
            })
            .collect()
    }

    /// Whole stored rows from `start`, ascending, at most `limit`.
    #[cfg(feature = "crypto")]
    fn scan_rows(
        &self,
        store: &BlobStorage,
        start: Bound<&[u8]>,
        limit: usize,
    ) -> Result<Pairs, TableError> {
        let lower = start.map(|pk| self.row_key(pk));
        let lower = match &lower {
            Bound::Unbounded => Bound::Included(self.rows.as_slice()),
            other => as_slices(other),
        };
        let end = prefix_end(&self.rows);
        Ok(store
            .scan_range(
                lower,
                end.as_deref().map_or(Bound::Unbounded, Bound::Excluded),
                ScanOrder::Ascending,
                limit,
            )?
            .into_iter()
            .map(|(k, row)| (k[self.rows.len()..].to_vec(), row))
            .collect())
    }

    /// How many rows the table holds.
    ///
    /// # Errors
    /// A store error.
    pub fn count(&self, store: &BlobStorage) -> Result<usize, TableError> {
        Ok(store.count_prefix(&self.rows)?)
    }

    /// The primary key of the row holding `key` in `index`.
    ///
    /// # Errors
    /// A store error.
    pub fn find_unique(
        &self,
        store: &BlobStorage,
        index: &str,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, TableError> {
        Ok(self
            .find(store, index, key, ScanOrder::Ascending, 1)?
            .into_iter()
            .next())
    }

    /// The primary keys of the rows holding `key` in `index`, at most `limit`.
    ///
    /// # Errors
    /// A store error.
    pub fn find(
        &self,
        store: &BlobStorage,
        index: &str,
        key: &[u8],
        order: ScanOrder,
        limit: usize,
    ) -> Result<Vec<Vec<u8>>, TableError> {
        let prefix = self.entry_prefix(index, key);
        let end = prefix_end(&prefix);
        Ok(store
            .scan_range(
                Bound::Included(&prefix),
                end.as_deref().map_or(Bound::Unbounded, Bound::Excluded),
                order,
                limit,
            )?
            .into_iter()
            .map(|(k, _)| k[prefix.len()..].to_vec())
            .collect())
    }

    /// Entries of `index` between two keys, `(index key, primary key)`, at most `limit`.
    ///
    /// # Errors
    /// [`TableError::Corrupt`] for an entry that does not parse, or a store error.
    pub fn index_range(
        &self,
        store: &BlobStorage,
        index: &str,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        order: ScanOrder,
        limit: usize,
    ) -> Result<Pairs, TableError> {
        let prefix = self.index_prefix(index);
        let field = |key: &[u8], tail: [u8; 2]| {
            let mut out = prefix.clone();
            push_escaped(&mut out, key);
            out.extend_from_slice(&tail);
            out
        };
        // Included(k) starts at the first entry of k, Excluded(k) after its last; ends likewise.
        let lower = match start {
            Bound::Unbounded => Bound::Included(prefix.clone()),
            Bound::Included(k) => Bound::Included(field(k, TERMINATOR)),
            Bound::Excluded(k) => Bound::Included(field(k, PAST_TERMINATOR)),
        };
        let upper = match end {
            Bound::Unbounded => prefix_end(&prefix).map_or(Bound::Unbounded, Bound::Excluded),
            Bound::Included(k) => Bound::Excluded(field(k, PAST_TERMINATOR)),
            Bound::Excluded(k) => Bound::Excluded(field(k, TERMINATOR)),
        };
        store
            .scan_range(as_slices(&lower), as_slices(&upper), order, limit)?
            .into_iter()
            .map(|(k, _)| {
                let (key, pk) = split_field(&k[prefix.len()..]).ok_or(TableError::Corrupt)?;
                Ok((key, pk.to_vec()))
            })
            .collect()
    }
}

fn as_slices(bound: &Bound<Vec<u8>>) -> Bound<&[u8]> {
    match bound {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(k) => Bound::Included(k),
        Bound::Excluded(k) => Bound::Excluded(k),
    }
}

fn encode_row(entries: &[IndexKey<'_>], value: &[u8]) -> Result<Vec<u8>, TableError> {
    let len = |n: usize| {
        u32::try_from(n)
            .map(u32::to_le_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "length exceeds u32::MAX"))
    };
    let mut out = vec![ROW_VERSION];
    out.extend_from_slice(&len(entries.len())?);
    for entry in entries {
        out.push(u8::from(entry.unique));
        out.extend_from_slice(&len(entry.index.len())?);
        out.extend_from_slice(entry.index.as_bytes());
        out.extend_from_slice(&len(entry.key.len())?);
        out.extend_from_slice(entry.key);
    }
    out.extend_from_slice(value);
    Ok(out)
}

/// `(index name, index key)` of each entry, and the value.
#[allow(clippy::type_complexity, reason = "one private parser")]
fn decode_row(row: &[u8]) -> Option<(Vec<(&[u8], &[u8])>, &[u8])> {
    fn take<'a>(rest: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
        if rest.len() < n {
            return None;
        }
        let (head, tail) = rest.split_at(n);
        *rest = tail;
        Some(head)
    }
    fn take_len(rest: &mut &[u8]) -> Option<usize> {
        let bytes: [u8; 4] = take(rest, 4)?.try_into().ok()?;
        usize::try_from(u32::from_le_bytes(bytes)).ok()
    }
    let mut rest = row;
    if take(&mut rest, 1)? != [ROW_VERSION] {
        return None;
    }
    let count = take_len(&mut rest)?;
    let mut entries = Vec::new();
    for _ in 0..count {
        take(&mut rest, 1)?;
        let name_len = take_len(&mut rest)?;
        let name = take(&mut rest, name_len)?;
        let key_len = take_len(&mut rest)?;
        entries.push((name, take(&mut rest, key_len)?));
    }
    Some((entries, rest))
}

/// A [`Table`] whose values are sealed with `alice_crypto::envelope` before they
/// reach the store. The associated data names the table and the primary key, so a
/// value copied to another row does not open. Primary keys and index keys are not
/// sealed: they are what the store sorts by.
#[cfg(feature = "crypto")]
pub struct SealedTable {
    table: Table,
    keys: std::sync::Arc<dyn alice_crypto::keystore::KeyProvider>,
}

#[cfg(feature = "crypto")]
impl fmt::Debug for SealedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SealedTable")
            .field("table", &self.table)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "crypto")]
impl SealedTable {
    /// How many rows one atomic step of [`Self::reseal`] rewrites.
    const RESEAL_STEP: usize = 256;

    /// The sealed table with this name.
    ///
    /// # Errors
    /// [`TableError::EmptyName`] for an empty name.
    pub fn new(
        name: &str,
        keys: std::sync::Arc<dyn alice_crypto::keystore::KeyProvider>,
    ) -> Result<Self, TableError> {
        Ok(Self {
            table: Table::new(name)?,
            keys,
        })
    }

    /// The table underneath, for what needs no seal: its index lookups, `delete`, `count`.
    #[must_use]
    pub const fn table(&self) -> &Table {
        &self.table
    }

    fn aad(&self, pk: &[u8]) -> Vec<u8> {
        self.table.row_key(pk)
    }

    fn seal(&self, pk: &[u8], value: &[u8]) -> Result<Vec<u8>, TableError> {
        let (generation, key) = self.keys.current().map_err(TableError::Key)?;
        alice_crypto::envelope::seal_envelope(&key, generation, &self.aad(pk), value)
            .map_err(|_| TableError::Seal)
    }

    fn open(&self, pk: &[u8], sealed: &[u8]) -> Result<Vec<u8>, TableError> {
        let generation =
            alice_crypto::envelope::envelope_key_id(sealed).map_err(|_| TableError::Seal)?;
        let key = self
            .keys
            .by_generation(generation)
            .map_err(TableError::Key)?;
        alice_crypto::envelope::open_envelope(&key, generation, &self.aad(pk), sealed)
            .map_err(|_| TableError::Seal)
    }

    /// [`Table::insert`] with a sealed value.
    ///
    /// # Errors
    /// As [`Table::insert`], and [`TableError::Key`] / [`TableError::Seal`].
    pub fn insert(
        &self,
        store: &BlobStorage,
        pk: &[u8],
        value: &[u8],
        entries: &[IndexKey<'_>],
    ) -> Result<(), TableError> {
        self.table
            .insert(store, pk, &self.seal(pk, value)?, entries)
    }

    /// [`Table::upsert`] with a sealed value.
    ///
    /// # Errors
    /// As [`Table::upsert`], and [`TableError::Key`] / [`TableError::Seal`].
    pub fn upsert(
        &self,
        store: &BlobStorage,
        pk: &[u8],
        value: &[u8],
        entries: &[IndexKey<'_>],
    ) -> Result<(), TableError> {
        self.table
            .upsert(store, pk, &self.seal(pk, value)?, entries)
    }

    /// The opened value of a row.
    ///
    /// # Errors
    /// As [`Table::get`], and [`TableError::Key`] / [`TableError::Seal`].
    pub fn get(&self, store: &BlobStorage, pk: &[u8]) -> Result<Option<Vec<u8>>, TableError> {
        self.table
            .get(store, pk)?
            .map(|sealed| self.open(pk, &sealed))
            .transpose()
    }

    /// [`Table::scan`] with opened values.
    ///
    /// # Errors
    /// As [`Table::scan`], and [`TableError::Key`] / [`TableError::Seal`].
    pub fn scan(
        &self,
        store: &BlobStorage,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        order: ScanOrder,
        limit: usize,
    ) -> Result<Pairs, TableError> {
        self.table
            .scan(store, start, end, order, limit)?
            .into_iter()
            .map(|(pk, sealed)| {
                let value = self.open(&pk, &sealed)?;
                Ok((pk, value))
            })
            .collect()
    }

    /// Move every row to the current key and return how many moved. Each step of
    /// 256 rows is atomic; after an interruption, running it again
    /// finishes the rest. A row already at the current key is not rewritten.
    ///
    /// # Errors
    /// [`TableError::Key`] when a row's key is gone (the rows before it have moved),
    /// [`TableError::Seal`], [`TableError::Corrupt`], or a store error.
    pub fn reseal(&self, store: &BlobStorage) -> Result<usize, TableError> {
        let mut moved = 0;
        let mut after: Option<Vec<u8>> = None;
        loop {
            let start = after.as_deref().map_or(Bound::Unbounded, Bound::Excluded);
            let (step_moved, last) = store.exclusive(|| self.reseal_step(store, start))?;
            moved += step_moved;
            match last {
                Some(pk) => after = Some(pk),
                None => return Ok(moved),
            }
        }
    }

    /// One step from `start`; the last primary key it looked at, `None` at the end.
    fn reseal_step(
        &self,
        store: &BlobStorage,
        start: Bound<&[u8]>,
    ) -> Result<(usize, Option<Vec<u8>>), TableError> {
        let (current, current_key) = self.keys.current().map_err(TableError::Key)?;
        let rows = self.table.scan_rows(store, start, Self::RESEAL_STEP)?;
        let mut batch = WriteBatch::new();
        for (pk, row) in &rows {
            let (_, sealed) = decode_row(row).ok_or(TableError::Corrupt)?;
            let generation =
                alice_crypto::envelope::envelope_key_id(sealed).map_err(|_| TableError::Seal)?;
            if generation == current {
                continue;
            }
            let old_key = self
                .keys
                .by_generation(generation)
                .map_err(TableError::Key)?;
            let resealed = alice_crypto::envelope::reseal_envelope(
                &old_key,
                generation,
                &current_key,
                current,
                &self.aad(pk),
                sealed,
            )
            .map_err(|_| TableError::Seal)?;
            let head = row.len() - sealed.len();
            batch.put(
                &self.table.row_key(pk),
                &[&row[..head], resealed.as_slice()].concat(),
            );
        }
        store.apply_exclusive(&batch)?;
        Ok((batch.len(), rows.last().map(|(pk, _)| pk.clone())))
    }
}
