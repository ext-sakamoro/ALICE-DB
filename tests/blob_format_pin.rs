//! Pins the on-disk bytes of the blob `SSTable` and WAL to their documented
//! layout, and the module docs to the code.
//!
//! The checksum is CRC-32 (IEEE 802.3, reflected polynomial `0xEDB88320`,
//! `crc32fast`), computed here by an independent bitwise reference. A
//! Castagnoli (CRC-32C) trailer, a different header version or a moved
//! field makes these tests fail.

#![cfg(feature = "fs")]

use alice_db::blob::BlobValue;
use alice_db::blob_sstable::BlobSstable;
use alice_db::blob_wal::BlobWal;
use tempfile::TempDir;

/// One `SSTable` record for key `k`, value `v`: 9-byte header, key, value.
const RECORD_LEN: usize = 9 + 1 + 1;
/// One WAL Delete record for key `k`: 10-byte header, key.
const DEL_LEN: usize = 10 + 1;

/// CRC-32 (IEEE), bit by bit.
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

#[test]
fn reference_crc_is_ieee_not_castagnoli() {
    // Standard check values for the input "123456789".
    assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
    assert_ne!(crc32_ieee(b"123456789"), 0xE306_9283); // CRC-32C
}

#[test]
fn sstable_bytes_follow_the_documented_v3_layout() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.sst");
    let raw = BlobValue::Raw(b"v".to_vec());
    BlobSstable::write_from_iter(&path, [(b"k".as_slice(), &raw)]).unwrap();
    let b = std::fs::read(&path).unwrap();

    // Header (24 bytes).
    assert_eq!(&b[0..8], b"ALICEBBS");
    assert_eq!(u32_at(&b, 8), 3, "header version");
    assert_eq!(u64_at(&b, 12), 1, "num_records");
    assert_eq!(u32_at(&b, 20), 0, "reserved");

    // One record: key_len, value_len, value_kind, key, value, CRC-32.
    let record = &b[24..24 + RECORD_LEN];
    assert_eq!(record, &[1, 0, 0, 0, 1, 0, 0, 0, 0x00, b'k', b'v']);
    let records_end = 24 + record.len() + 4;
    assert_eq!(u32_at(&b, 24 + record.len()), crc32_ieee(record));

    // Footer (24 bytes): records_size, bloom_size, magic.
    let footer = b.len() - 24;
    assert_eq!(&b[footer + 16..], b"ALICEEND");
    assert_eq!(u64_at(&b, footer), (records_end - 24) as u64);
    let bloom_size = usize::try_from(u64_at(&b, footer + 8)).unwrap();
    assert_eq!(
        records_end + bloom_size,
        footer,
        "bloom section fills the gap"
    );

    // Bloom section: num_bits u64, num_hashes u32, bits_len u64, bits, CRC-32.
    let bloom = &b[records_end..footer];
    let bits_len = usize::try_from(u64_at(bloom, 12)).unwrap();
    assert_eq!(bloom.len(), 20 + bits_len + 4);
    assert!(bits_len > 0 && u64_at(bloom, 0) > 0 && u32_at(bloom, 8) > 0);
    assert_eq!(
        u32_at(bloom, 20 + bits_len),
        crc32_ieee(&bloom[..20 + bits_len])
    );
}

#[test]
fn wal_record_bytes_follow_the_documented_layout() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("blob.wal");
    {
        let wal = BlobWal::open(&path).unwrap();
        wal.append_put(b"k", &BlobValue::Raw(b"v".to_vec()))
            .unwrap();
        wal.append_delete(b"k").unwrap();
    }
    let b = std::fs::read(&path).unwrap();

    let put = &b[0..10 + 2];
    assert_eq!(put, &[0x01, 1, 0, 0, 0, 1, 0, 0, 0, 0x00, b'k', b'v']);
    assert_eq!(u32_at(&b, put.len()), crc32_ieee(put));

    let at = put.len() + 4;
    let del = &b[at..at + DEL_LEN];
    assert_eq!(del[0], 0x02, "Delete");
    assert_eq!(u32_at(del, 5), 0, "value_len is 0 on Delete");
    assert_eq!(del[10], b'k');
    assert_eq!(u32_at(&b, at + del.len()), crc32_ieee(del));
    assert_eq!(b.len(), at + del.len() + 4, "two records and nothing else");
}

fn module_doc(src: &str) -> String {
    src.lines()
        .filter_map(|l| l.trim_start().strip_prefix("//!"))
        .collect::<Vec<_>>()
        .join("\n")
}

const SSTABLE_SRC: &str = include_str!("../src/blob_sstable.rs");
const WAL_SRC: &str = include_str!("../src/blob_wal.rs");

#[test]
fn docs_name_the_checksum_that_the_code_computes() {
    for (name, src) in [("blob_sstable.rs", SSTABLE_SRC), ("blob_wal.rs", WAL_SRC)] {
        // The one allowed mention says what the checksum is not.
        let lower = src.to_lowercase().replace("not crc-32c", "");
        assert!(
            !lower.contains("crc32c") && !lower.contains("crc-32c"),
            "{name} still names CRC-32C; the code computes CRC-32 (IEEE) via crc32fast"
        );
        assert!(
            module_doc(src).contains("CRC-32 (IEEE"),
            "{name} module doc must name the checksum"
        );
    }
}

#[test]
fn sstable_doc_states_the_version_the_writer_stamps() {
    let doc = module_doc(SSTABLE_SRC);
    assert!(
        doc.contains("version        u32 LE (written: 3; read: 1, 2, 3)"),
        "header version line in the module doc"
    );
    for field in [
        "bloom_size",
        "trailer  4             CRC-32 over the entire record above",
        "trailer  4             CRC-32 over the bloom section above",
        "Footer, v2 and v3 (24 bytes, at end of file)",
        "Footer, v1 (16 bytes; v1 files have no bloom section)",
        "Versions 2 and 3 have the same layout and are read the same way",
        "value_kind     u8  (0x00 = Raw, 0x01 = Compressed, 0x02 = Tombstone)",
    ] {
        assert!(doc.contains(field), "module doc lacks `{field}`");
    }
}

#[test]
fn wal_doc_states_the_record_trailer() {
    assert!(
        module_doc(WAL_SRC)
            .contains("trailer 4     crc32           CRC-32 over everything from offset 0 upward"),
        "WAL record trailer line in the module doc"
    );
}
