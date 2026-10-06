//! Byte snapshot of a whole database ([`crate::AliceDB::to_bytes`] /
//! [`crate::AliceDB::from_bytes`]).
//!
//! The snapshot carries the stored files of the time-series engine (the
//! segment index and every `seg_<id>.rkyv`, byte-for-byte as the backend
//! stores them) plus the live blob key-value pairs. A caller that wants
//! persistence without a filesystem — for example a browser page keeping
//! the bytes in `IndexedDB` — stores the snapshot and restores it later.
//!
//! # Layout (all integers little-endian)
//!
//! ```text
//! magic        8 bytes   b"ALICEDB\x01"
//! format       u32       SNAPSHOT_FORMAT (= 1)
//! file_count   u32
//!   name_len   u32, name (UTF-8), data_len u64, data      × file_count
//! blob_count   u64
//!   key_len    u32, key, value_len u64, value             × blob_count
//! crc32        u32       CRC-32 (IEEE) of every preceding byte
//! ```
//!
//! Decoding rejects a wrong magic / format, a checksum mismatch, a
//! truncated buffer and trailing bytes, so a snapshot that lost or gained
//! a byte never restores silently.

use std::collections::BTreeMap;
use std::io;

/// Leading bytes of every snapshot.
pub const SNAPSHOT_MAGIC: [u8; 8] = *b"ALICEDB\x01";

/// Snapshot layout version written by this build.
pub const SNAPSHOT_FORMAT: u32 = 1;

/// Stored files of the time-series engine (name → bytes).
pub type SnapshotFiles = BTreeMap<String, Vec<u8>>;

/// Live blob pairs in ascending key order.
pub type SnapshotBlobs = Vec<(Vec<u8>, Vec<u8>)>;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn len_u32(len: usize, what: &str) -> io::Result<u32> {
    u32::try_from(len).map_err(|_| invalid(format!("{what} length {len} exceeds u32")))
}

/// Serialize engine files and blob pairs into one snapshot buffer.
///
/// # Errors
/// Returns `InvalidData` if a name or key is longer than `u32::MAX` bytes,
/// or there are more than `u32::MAX` files.
pub fn encode(files: &SnapshotFiles, blobs: &[(Vec<u8>, Vec<u8>)]) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&SNAPSHOT_MAGIC);
    out.extend_from_slice(&SNAPSHOT_FORMAT.to_le_bytes());
    out.extend_from_slice(&len_u32(files.len(), "file count")?.to_le_bytes());
    for (name, data) in files {
        out.extend_from_slice(&len_u32(name.len(), "file name")?.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
        out.extend_from_slice(data);
    }
    out.extend_from_slice(&(blobs.len() as u64).to_le_bytes());
    for (key, value) in blobs {
        out.extend_from_slice(&len_u32(key.len(), "blob key")?.to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value);
    }
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(out)
}

/// Cursor over the checksummed body of a snapshot.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.buf.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "snapshot is truncated"))?;
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> io::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> io::Result<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn len_u64(&mut self) -> io::Result<usize> {
        let n = self.u64()?;
        usize::try_from(n).map_err(|_| invalid(format!("length {n} exceeds usize")))
    }
}

/// Parse a snapshot produced by [`encode`].
///
/// # Errors
/// Returns `InvalidData` for a wrong magic, an unknown format, a checksum
/// mismatch, a non-UTF-8 file name, a duplicate file name or trailing
/// bytes, and `UnexpectedEof` for a truncated buffer.
pub fn decode(bytes: &[u8]) -> io::Result<(SnapshotFiles, SnapshotBlobs)> {
    if bytes.len() < SNAPSHOT_MAGIC.len() + 4 + 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "snapshot is shorter than its header",
        ));
    }
    let (body, crc_bytes) = bytes.split_at(bytes.len() - 4);
    let stored_crc = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
    if body.get(..SNAPSHOT_MAGIC.len()) != Some(&SNAPSHOT_MAGIC[..]) {
        return Err(invalid("not an ALICE-DB snapshot (magic mismatch)"));
    }
    let actual_crc = crc32fast::hash(body);
    if actual_crc != stored_crc {
        return Err(invalid(format!(
            "snapshot checksum mismatch (stored {stored_crc:#010x}, computed {actual_crc:#010x})"
        )));
    }

    let mut r = Reader {
        buf: body,
        pos: SNAPSHOT_MAGIC.len(),
    };
    let format = r.u32()?;
    if format != SNAPSHOT_FORMAT {
        return Err(invalid(format!(
            "unsupported snapshot format {format} (this build reads {SNAPSHOT_FORMAT})"
        )));
    }

    let file_count = r.u32()?;
    let mut files = SnapshotFiles::new();
    for _ in 0..file_count {
        let name_len = r.u32()? as usize;
        let name = std::str::from_utf8(r.take(name_len)?)
            .map_err(|e| invalid(format!("snapshot file name is not UTF-8: {e}")))?
            .to_string();
        let data_len = r.len_u64()?;
        let data = r.take(data_len)?.to_vec();
        if files.insert(name.clone(), data).is_some() {
            return Err(invalid(format!("snapshot repeats file {name}")));
        }
    }

    let blob_count = r.u64()?;
    let mut blobs = SnapshotBlobs::new();
    for _ in 0..blob_count {
        let key_len = r.u32()? as usize;
        let key = r.take(key_len)?.to_vec();
        let value_len = r.len_u64()?;
        let value = r.take(value_len)?.to_vec();
        blobs.push((key, value));
    }

    if r.pos != body.len() {
        return Err(invalid(format!(
            "snapshot has {} trailing bytes",
            body.len() - r.pos
        )));
    }
    Ok((files, blobs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (SnapshotFiles, SnapshotBlobs) {
        let mut files = SnapshotFiles::new();
        files.insert("index.alice".into(), vec![1, 2, 3]);
        files.insert("seg_1.rkyv".into(), (0..=255).collect());
        let blobs = vec![
            (b"a".to_vec(), b"xyz".to_vec()),
            (b"b".to_vec(), Vec::new()),
        ];
        (files, blobs)
    }

    #[test]
    fn encode_decode_roundtrip_is_exact() {
        let (files, blobs) = sample();
        let bytes = encode(&files, &blobs).unwrap();
        let (f2, b2) = decode(&bytes).unwrap();
        assert_eq!(f2, files);
        assert_eq!(b2, blobs);
    }

    #[test]
    fn every_single_byte_truncation_and_flip_is_rejected() {
        let (files, blobs) = sample();
        let bytes = encode(&files, &blobs).unwrap();
        let mut checked = 0usize;
        for cut in 0..bytes.len() {
            assert!(
                decode(&bytes[..cut]).is_err(),
                "truncated to {cut} accepted"
            );
            let mut dropped = bytes.clone();
            dropped.remove(cut);
            assert!(decode(&dropped).is_err(), "byte {cut} removed accepted");
            let mut flipped = bytes.clone();
            flipped[cut] ^= 0x01;
            assert!(decode(&flipped).is_err(), "byte {cut} flipped accepted");
            checked += 1;
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(decode(&extra).is_err(), "trailing byte accepted");
        assert_eq!(checked, bytes.len());
        assert!(checked > 0);
    }
}
