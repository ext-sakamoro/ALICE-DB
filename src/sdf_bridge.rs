//! SDF Spatial Data Storage Bridge
//!
//! Stores SDF coefficients and SVO data using Morton code (Z-order curve)
//! for efficient 3D spatial queries. Each spatial cell is a 24-bit key
//! encoding a voxel in the world grid.
//!
//! Author: Moroya Sakamoto

use crate::AliceDB;
use std::io;
use std::path::Path;

const MORTON_SPREAD_LUT: [u32; 256] = {
    let mut lut = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut v = 0u32;
        let mut j = 0u32;
        while j < 8 {
            v |= ((i >> j) & 1) << (j * 3);
            j += 1;
        }
        lut[i as usize] = v;
        i += 1;
    }
    lut
};

const MORTON_COMPACT_LUT: [u8; 512] = {
    let mut lut = [0u8; 512];
    let mut i = 0u32;
    while i < 512 {
        let mut v = 0u8;
        let mut j = 0u32;
        while j < 3 {
            v |= (((i >> (j * 3)) & 1) as u8) << j;
            j += 1;
        }
        lut[i as usize] = v;
        i += 1;
    }
    lut
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MortonCode(pub u32);

impl MortonCode {
    #[inline]
    pub fn encode(x: u8, y: u8, z: u8) -> Self {
        let code = MORTON_SPREAD_LUT[x as usize]
            | (MORTON_SPREAD_LUT[y as usize] << 1)
            | (MORTON_SPREAD_LUT[z as usize] << 2);
        MortonCode(code)
    }

    #[inline]
    pub fn decode(self) -> (u8, u8, u8) {
        let c = self.0;
        let x = MORTON_COMPACT_LUT[(c & 0x1FF) as usize]
            | (MORTON_COMPACT_LUT[((c >> 9) & 0x1FF) as usize] << 3)
            | (MORTON_COMPACT_LUT[((c >> 18) & 0x1FF) as usize] << 6);
        let y = MORTON_COMPACT_LUT[((c >> 1) & 0x1FF) as usize]
            | (MORTON_COMPACT_LUT[((c >> 10) & 0x1FF) as usize] << 3)
            | (MORTON_COMPACT_LUT[((c >> 19) & 0x1FF) as usize] << 6);
        let z = MORTON_COMPACT_LUT[((c >> 2) & 0x1FF) as usize]
            | (MORTON_COMPACT_LUT[((c >> 11) & 0x1FF) as usize] << 3)
            | (MORTON_COMPACT_LUT[((c >> 20) & 0x1FF) as usize] << 6);
        (x, y, z)
    }

    #[inline]
    pub fn from_world(
        wx: f32, wy: f32, wz: f32,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> Self {
        let inv_rx = 1.0 / (world_max[0] - world_min[0]);
        let inv_ry = 1.0 / (world_max[1] - world_min[1]);
        let inv_rz = 1.0 / (world_max[2] - world_min[2]);

        let nx = ((wx - world_min[0]) * inv_rx).clamp(0.0, 1.0);
        let ny = ((wy - world_min[1]) * inv_ry).clamp(0.0, 1.0);
        let nz = ((wz - world_min[2]) * inv_rz).clamp(0.0, 1.0);

        let gx = (nx * 255.0) as u8;
        let gy = (ny * 255.0) as u8;
        let gz = (nz * 255.0) as u8;

        Self::encode(gx, gy, gz)
    }

    /// Get the Morton code as i64 timestamp for AliceDB storage
    pub fn as_key(self) -> i64 {
        self.0 as i64
    }
}

/// SDF storage for spatial data
///
/// Wraps AliceDB with Morton-code spatial indexing for SDF coefficients.
pub struct SdfStorage {
    /// Database for SDF coefficient storage (keyframes)
    keyframe_db: AliceDB,
    /// Database for delta accumulation
    delta_db: AliceDB,
    /// World bounds for coordinate mapping
    world_min: [f32; 3],
    world_max: [f32; 3],
}

impl SdfStorage {
    /// Open or create SDF storage at the given directory
    pub fn open<P: AsRef<Path>>(
        path: P,
        world_min: [f32; 3],
        world_max: [f32; 3],
    ) -> io::Result<Self> {
        let base = path.as_ref();
        std::fs::create_dir_all(base.join("keyframes"))?;
        std::fs::create_dir_all(base.join("deltas"))?;

        let keyframe_db = AliceDB::open(base.join("keyframes"))?;
        let delta_db = AliceDB::open(base.join("deltas"))?;

        Ok(Self {
            keyframe_db,
            delta_db,
            world_min,
            world_max,
        })
    }

    /// Store a keyframe SDF value at a spatial cell
    ///
    /// The value is stored using the Morton code as key in the time-series DB.
    /// Multiple values per cell are appended with incrementing sub-keys.
    pub fn store_keyframe(
        &self,
        cell: MortonCode,
        scene_version: u32,
        value: f32,
    ) -> io::Result<()> {
        // Combine Morton code with scene version for unique key
        let key = (cell.0 as i64) << 16 | (scene_version as i64 & 0xFFFF);
        self.keyframe_db.put(key, value)
    }

    /// Store a batch of SDF coefficients for a spatial region
    pub fn store_keyframe_batch(
        &self,
        cells: &[(MortonCode, f32)],
        scene_version: u32,
    ) -> io::Result<()> {
        let batch: Vec<(i64, f32)> = cells.iter()
            .map(|(cell, value)| {
                let key = (cell.0 as i64) << 16 | (scene_version as i64 & 0xFFFF);
                (key, *value)
            })
            .collect();

        self.keyframe_db.put_batch(&batch)
    }

    /// Store a delta update for a spatial cell
    pub fn store_delta(
        &self,
        cell: MortonCode,
        delta_version: u32,
        value: f32,
    ) -> io::Result<()> {
        let key = (cell.0 as i64) << 16 | (delta_version as i64 & 0xFFFF);
        self.delta_db.put(key, value)
    }

    /// Query SDF values in a spatial region defined by min/max world coordinates
    ///
    /// Returns all stored values within the Morton code range.
    pub fn query_spatial_region(
        &self,
        region_min: [f32; 3],
        region_max: [f32; 3],
    ) -> io::Result<Vec<(MortonCode, f32)>> {
        let morton_min = MortonCode::from_world(
            region_min[0], region_min[1], region_min[2],
            self.world_min, self.world_max,
        );
        let morton_max = MortonCode::from_world(
            region_max[0], region_max[1], region_max[2],
            self.world_min, self.world_max,
        );

        // Range scan using Morton code bounds
        let start_key = (morton_min.0 as i64) << 16;
        let end_key = ((morton_max.0 as i64) + 1) << 16;

        let results = self.keyframe_db.scan(start_key, end_key)?;

        Ok(results.into_iter().map(|(key, value)| {
            let morton = MortonCode((key >> 16) as u32);
            (morton, value)
        }).collect())
    }

    /// Flush all data to disk
    pub fn flush(&self) -> io::Result<()> {
        self.keyframe_db.flush()?;
        self.delta_db.flush()?;
        Ok(())
    }

    /// Close the storage
    pub fn close(self) -> io::Result<()> {
        self.keyframe_db.close()?;
        self.delta_db.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_morton_encode_decode() {
        let code = MortonCode::encode(10, 20, 30);
        let (x, y, z) = code.decode();
        assert_eq!((x, y, z), (10, 20, 30));
    }

    #[test]
    fn test_morton_origin() {
        let code = MortonCode::encode(0, 0, 0);
        assert_eq!(code.0, 0);
    }

    #[test]
    fn test_morton_max() {
        let code = MortonCode::encode(255, 255, 255);
        let (x, y, z) = code.decode();
        assert_eq!((x, y, z), (255, 255, 255));
    }

    #[test]
    fn test_morton_from_world() {
        let code = MortonCode::from_world(
            0.5, 0.5, 0.5,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let (x, y, z) = code.decode();
        // Should be approximately center (127-128)
        assert!(x >= 126 && x <= 128);
        assert!(y >= 126 && y <= 128);
        assert!(z >= 126 && z <= 128);
    }

    #[test]
    fn test_sdf_storage_basic() {
        let dir = tempdir().unwrap();
        let storage = SdfStorage::open(
            dir.path().join("sdf"),
            [-10.0, -10.0, -10.0],
            [10.0, 10.0, 10.0],
        ).unwrap();

        let cell = MortonCode::encode(128, 128, 128);
        storage.store_keyframe(cell, 1, 0.5).unwrap();
        storage.flush().unwrap();

        let results = storage.query_spatial_region(
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ).unwrap();

        assert!(!results.is_empty());
        storage.close().unwrap();
    }

    #[test]
    fn test_morton_spatial_locality() {
        // Adjacent cells should have similar Morton codes
        let code1 = MortonCode::encode(10, 10, 10);
        let code2 = MortonCode::encode(11, 10, 10);
        let code_far = MortonCode::encode(200, 200, 200);

        let diff_near = (code1.0 as i64 - code2.0 as i64).unsigned_abs();
        let diff_far = (code1.0 as i64 - code_far.0 as i64).unsigned_abs();
        assert!(diff_near < diff_far);
    }
}
