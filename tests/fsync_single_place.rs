//! Every fsync in `src/` goes through `fs_audit::sync_all` / `fs_audit::sync_data`, so the recorded fsyncs
//! (see `tests/wal_recovery_crash.rs`) are all of the crate's fsyncs. A direct `.sync_all(` / `.sync_data(`
//! anywhere else in `src/` (outside `#[cfg(test)]` modules) fails here, and so does scanning nothing.

use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.extend([path]);
        }
    }
}

/// The production part of a source file: everything before its `#[cfg(test)]` module.
fn production(text: &str) -> &str {
    text.find("#[cfg(test)]\nmod tests")
        .map_or(text, |i| &text[..i])
}

#[test]
fn every_fsync_in_src_goes_through_the_recording_helpers() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    files.sort();
    assert!(
        files.len() > 10,
        "scanned {} files under src/ (compared nothing)",
        files.len()
    );

    let mut direct = Vec::new();
    let mut in_helpers = 0;
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        for (n, line) in production(&text).lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for call in [".sync_all(", ".sync_data("] {
                if code.contains(call) {
                    if file.ends_with("fs_audit.rs") {
                        in_helpers += 1;
                    } else {
                        direct.extend([format!("{}:{}: {}", file.display(), n + 1, line.trim())]);
                    }
                }
            }
        }
    }
    assert!(
        direct.is_empty(),
        "fsync outside fs_audit::sync_all / sync_data:\n{}",
        direct.join("\n")
    );
    assert_eq!(
        in_helpers, 2,
        "fs_audit.rs must call the fsync methods exactly once each, in its two helpers"
    );
}
