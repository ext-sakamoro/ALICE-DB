use alice_db::{AliceDB, FitConfig, StorageConfig};
fn main() -> std::io::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("dir"));
    let db = AliceDB::with_config(StorageConfig {
        data_dir: dir,
        enable_wal: false,
        enable_background_flush: false,
        fit_config: FitConfig { lossless: true, ..FitConfig::default() },
        ..StorageConfig::default()
    })?;
    for k in 0..64_i64 {
        let x = 0.3_f64 * k as f64 + 0.4;
        db.put(k, (2.0 * x.sin() + 0.25) as f32)?;
    }
    db.flush()?;
    db.close()
}
