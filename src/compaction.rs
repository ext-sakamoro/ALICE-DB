//! LSM スタイル高度コンパクション
//!
//! Leveled / Tiered / `SizeTiered` コンパクション戦略。
//! セグメントの階層管理、コンパクション候補選択、マージ実行。

use std::collections::BTreeMap;

/// コンパクション戦略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Leveled コンパクション (`RocksDB` スタイル)。
    ///
    /// 各レベルにサイズ上限があり、超過したらマージして次レベルへ。
    Leveled,
    /// Tiered コンパクション (`Cassandra` スタイル)。
    ///
    /// 同レベルのセグメントが一定数に達したらマージ。
    Tiered,
    /// サイズベース Tiered コンパクション。
    ///
    /// 類似サイズのセグメントをグループ化してマージ。
    SizeTiered,
}

impl std::fmt::Display for CompactionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Leveled => write!(f, "Leveled"),
            Self::Tiered => write!(f, "Tiered"),
            Self::SizeTiered => write!(f, "SizeTiered"),
        }
    }
}

/// コンパクション設定。
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// 戦略。
    pub strategy: CompactionStrategy,
    /// 最大レベル数。
    pub max_levels: usize,
    /// レベル間のサイズ比率 (例: 10 = 次のレベルは10倍)。
    pub level_size_ratio: usize,
    /// L0 の最大セグメント数 (超過でコンパクション発動)。
    pub l0_compaction_trigger: usize,
    /// 1セグメントの最大サイズ (バイト)。
    pub max_segment_size: u64,
    /// Tiered: マージトリガーのセグメント数。
    pub tiered_min_merge: usize,
    /// `SizeTiered`: サイズ類似度閾値 (0.0-1.0)。
    pub size_ratio_threshold: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            strategy: CompactionStrategy::Leveled,
            max_levels: 7,
            level_size_ratio: 10,
            l0_compaction_trigger: 4,
            max_segment_size: 64 * 1024 * 1024, // 64 MB
            tiered_min_merge: 4,
            size_ratio_threshold: 0.5,
        }
    }
}

/// セグメント情報 (コンパクション用)。
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// セグメント ID。
    pub id: u64,
    /// サイズ (バイト)。
    pub size: u64,
    /// 開始タイムスタンプ。
    pub start_time: i64,
    /// 終了タイムスタンプ。
    pub end_time: i64,
    /// データポイント数。
    pub point_count: usize,
    /// レベル。
    pub level: usize,
}

/// コンパクションタスク。
#[derive(Debug, Clone)]
pub struct CompactionTask {
    /// マージ対象セグメント ID。
    pub input_segments: Vec<u64>,
    /// 入力レベル。
    pub input_level: usize,
    /// 出力レベル。
    pub output_level: usize,
    /// 推定出力サイズ。
    pub estimated_output_size: u64,
}

/// レベル別セグメント管理。
#[derive(Debug)]
pub struct LevelState {
    /// レベル → セグメント一覧。
    levels: BTreeMap<usize, Vec<SegmentInfo>>,
    /// 設定。
    config: CompactionConfig,
}

impl LevelState {
    /// 新しいレベル状態を作成。
    #[must_use]
    pub fn new(config: CompactionConfig) -> Self {
        let mut levels = BTreeMap::new();
        for i in 0..config.max_levels {
            levels.insert(i, Vec::new());
        }
        Self { levels, config }
    }

    /// セグメントを追加。
    pub fn add_segment(&mut self, segment: SegmentInfo) {
        let level = segment.level.min(self.config.max_levels - 1);
        self.levels.entry(level).or_default().push(segment);
    }

    /// レベルのセグメント一覧。
    #[must_use]
    pub fn segments_at_level(&self, level: usize) -> &[SegmentInfo] {
        self.levels.get(&level).map_or(&[], Vec::as_slice)
    }

    /// レベルの合計サイズ。
    #[must_use]
    pub fn level_size(&self, level: usize) -> u64 {
        self.levels
            .get(&level)
            .map_or(0, |segs| segs.iter().map(|s| s.size).sum())
    }

    /// レベルのセグメント数。
    #[must_use]
    pub fn segment_count_at_level(&self, level: usize) -> usize {
        self.levels.get(&level).map_or(0, Vec::len)
    }

    /// 全セグメント数。
    #[must_use]
    pub fn total_segments(&self) -> usize {
        self.levels.values().map(Vec::len).sum()
    }

    /// セグメントを削除。
    pub fn remove_segment(&mut self, level: usize, segment_id: u64) -> bool {
        if let Some(segs) = self.levels.get_mut(&level) {
            let before = segs.len();
            segs.retain(|s| s.id != segment_id);
            return segs.len() < before;
        }
        false
    }
}

/// コンパクションプランナー。
#[derive(Debug)]
pub struct CompactionPlanner {
    /// 設定。
    config: CompactionConfig,
}

impl CompactionPlanner {
    /// 新しいプランナーを作成。
    #[must_use]
    pub const fn new(config: CompactionConfig) -> Self {
        Self { config }
    }

    /// コンパクションが必要か判定し、タスクを返す。
    #[must_use]
    pub fn plan(&self, state: &LevelState) -> Option<CompactionTask> {
        match self.config.strategy {
            CompactionStrategy::Leveled => self.plan_leveled(state),
            CompactionStrategy::Tiered => self.plan_tiered(state),
            CompactionStrategy::SizeTiered => self.plan_size_tiered(state),
        }
    }

    /// Leveled コンパクション計画。
    fn plan_leveled(&self, state: &LevelState) -> Option<CompactionTask> {
        // L0 がトリガー閾値を超えたら L0 → L1 マージ
        let l0_count = state.segment_count_at_level(0);
        if l0_count >= self.config.l0_compaction_trigger {
            let ids: Vec<u64> = state.segments_at_level(0).iter().map(|s| s.id).collect();
            let total_size: u64 = state.segments_at_level(0).iter().map(|s| s.size).sum();
            return Some(CompactionTask {
                input_segments: ids,
                input_level: 0,
                output_level: 1,
                estimated_output_size: total_size,
            });
        }

        // 各レベルのサイズ上限チェック
        let base_size = self.config.max_segment_size;
        for level in 1..self.config.max_levels - 1 {
            let max_size = base_size * (self.config.level_size_ratio as u64).pow(level as u32);
            let current_size = state.level_size(level);

            if current_size > max_size {
                // 最大サイズのセグメントを選択
                if let Some(segs) = state.levels.get(&level) {
                    if let Some(largest) = segs.iter().max_by_key(|s| s.size) {
                        return Some(CompactionTask {
                            input_segments: vec![largest.id],
                            input_level: level,
                            output_level: level + 1,
                            estimated_output_size: largest.size,
                        });
                    }
                }
            }
        }

        None
    }

    /// Tiered コンパクション計画。
    fn plan_tiered(&self, state: &LevelState) -> Option<CompactionTask> {
        for level in 0..self.config.max_levels - 1 {
            let count = state.segment_count_at_level(level);
            if count >= self.config.tiered_min_merge {
                let ids: Vec<u64> = state
                    .segments_at_level(level)
                    .iter()
                    .map(|s| s.id)
                    .collect();
                let total_size: u64 = state.segments_at_level(level).iter().map(|s| s.size).sum();
                return Some(CompactionTask {
                    input_segments: ids,
                    input_level: level,
                    output_level: level + 1,
                    estimated_output_size: total_size,
                });
            }
        }
        None
    }

    /// `SizeTiered` コンパクション計画。
    fn plan_size_tiered(&self, state: &LevelState) -> Option<CompactionTask> {
        // 各レベルでサイズが類似のセグメントをグループ化
        for level in 0..self.config.max_levels {
            let segs = state.segments_at_level(level);
            if segs.len() < 2 {
                continue;
            }

            // サイズでソート
            let mut sorted: Vec<&SegmentInfo> = segs.iter().collect();
            sorted.sort_by_key(|s| s.size);

            // 類似サイズのグループを探索
            let mut group = vec![sorted[0].id];
            let mut group_size = sorted[0].size;

            for seg in &sorted[1..] {
                let avg = group_size / group.len() as u64;
                let ratio = if avg > 0 {
                    (seg.size as f64 - avg as f64).abs() / avg as f64
                } else {
                    0.0
                };

                if ratio <= self.config.size_ratio_threshold {
                    group.push(seg.id);
                    group_size += seg.size;
                } else if group.len() >= self.config.tiered_min_merge {
                    break;
                } else {
                    group.clear();
                    group.push(seg.id);
                    group_size = seg.size;
                }
            }

            if group.len() >= self.config.tiered_min_merge {
                return Some(CompactionTask {
                    input_segments: group,
                    input_level: level,
                    output_level: level,
                    estimated_output_size: group_size,
                });
            }
        }
        None
    }

    /// コンパクションスコアを計算。
    ///
    /// スコアが 1.0 以上ならコンパクションが推奨される。
    #[must_use]
    pub fn score(&self, state: &LevelState) -> f64 {
        match self.config.strategy {
            CompactionStrategy::Leveled => {
                let l0_score = state.segment_count_at_level(0) as f64
                    / self.config.l0_compaction_trigger as f64;

                let mut max_level_score = 0.0f64;
                let base_size = self.config.max_segment_size;
                for level in 1..self.config.max_levels {
                    let max_size =
                        base_size * (self.config.level_size_ratio as u64).pow(level as u32);
                    let current_size = state.level_size(level);
                    let score = current_size as f64 / max_size as f64;
                    max_level_score = max_level_score.max(score);
                }

                l0_score.max(max_level_score)
            }
            CompactionStrategy::Tiered | CompactionStrategy::SizeTiered => {
                let mut max_score = 0.0f64;
                for level in 0..self.config.max_levels {
                    let count = state.segment_count_at_level(level);
                    let score = count as f64 / self.config.tiered_min_merge as f64;
                    max_score = max_score.max(score);
                }
                max_score
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(id: u64, size: u64, level: usize) -> SegmentInfo {
        SegmentInfo {
            id,
            size,
            start_time: id as i64 * 100,
            end_time: id as i64 * 100 + 99,
            point_count: 100,
            level,
        }
    }

    #[test]
    fn default_config() {
        let config = CompactionConfig::default();
        assert_eq!(config.strategy, CompactionStrategy::Leveled);
        assert_eq!(config.max_levels, 7);
        assert_eq!(config.l0_compaction_trigger, 4);
    }

    #[test]
    fn strategy_display() {
        assert_eq!(CompactionStrategy::Leveled.to_string(), "Leveled");
        assert_eq!(CompactionStrategy::Tiered.to_string(), "Tiered");
        assert_eq!(CompactionStrategy::SizeTiered.to_string(), "SizeTiered");
    }

    #[test]
    fn level_state_add_and_query() {
        let mut state = LevelState::new(CompactionConfig::default());
        state.add_segment(make_segment(1, 1000, 0));
        state.add_segment(make_segment(2, 2000, 0));
        state.add_segment(make_segment(3, 5000, 1));

        assert_eq!(state.segment_count_at_level(0), 2);
        assert_eq!(state.segment_count_at_level(1), 1);
        assert_eq!(state.level_size(0), 3000);
        assert_eq!(state.total_segments(), 3);
    }

    #[test]
    fn level_state_remove() {
        let mut state = LevelState::new(CompactionConfig::default());
        state.add_segment(make_segment(1, 1000, 0));
        state.add_segment(make_segment(2, 2000, 0));

        assert!(state.remove_segment(0, 1));
        assert_eq!(state.segment_count_at_level(0), 1);
        assert!(!state.remove_segment(0, 99));
    }

    #[test]
    fn leveled_compaction_l0_trigger() {
        let config = CompactionConfig {
            l0_compaction_trigger: 4,
            ..CompactionConfig::default()
        };
        let mut state = LevelState::new(config.clone());

        // L0 に 4 セグメント追加 (トリガー)
        for i in 0..4 {
            state.add_segment(make_segment(i, 1000, 0));
        }

        let planner = CompactionPlanner::new(config);
        let task = planner.plan(&state);
        assert!(task.is_some());
        let task = task.unwrap();
        assert_eq!(task.input_level, 0);
        assert_eq!(task.output_level, 1);
        assert_eq!(task.input_segments.len(), 4);
    }

    #[test]
    fn leveled_no_compaction_below_trigger() {
        let config = CompactionConfig::default();
        let mut state = LevelState::new(config.clone());
        state.add_segment(make_segment(1, 1000, 0));

        let planner = CompactionPlanner::new(config);
        assert!(planner.plan(&state).is_none());
    }

    #[test]
    fn tiered_compaction() {
        let config = CompactionConfig {
            strategy: CompactionStrategy::Tiered,
            tiered_min_merge: 3,
            ..CompactionConfig::default()
        };
        let mut state = LevelState::new(config.clone());

        for i in 0..3 {
            state.add_segment(make_segment(i, 1000, 0));
        }

        let planner = CompactionPlanner::new(config);
        let task = planner.plan(&state);
        assert!(task.is_some());
        assert_eq!(task.unwrap().input_segments.len(), 3);
    }

    #[test]
    fn size_tiered_compaction() {
        let config = CompactionConfig {
            strategy: CompactionStrategy::SizeTiered,
            tiered_min_merge: 3,
            size_ratio_threshold: 0.5,
            ..CompactionConfig::default()
        };
        let mut state = LevelState::new(config.clone());

        // 類似サイズのセグメント
        state.add_segment(make_segment(1, 1000, 0));
        state.add_segment(make_segment(2, 1100, 0));
        state.add_segment(make_segment(3, 1200, 0));

        let planner = CompactionPlanner::new(config);
        let task = planner.plan(&state);
        assert!(task.is_some());
    }

    #[test]
    fn compaction_score_below_threshold() {
        let config = CompactionConfig::default();
        let state = LevelState::new(config.clone());
        let planner = CompactionPlanner::new(config);
        let score = planner.score(&state);
        assert!(score < 1.0);
    }

    #[test]
    fn compaction_score_above_threshold() {
        let config = CompactionConfig {
            l0_compaction_trigger: 4,
            ..CompactionConfig::default()
        };
        let mut state = LevelState::new(config.clone());
        for i in 0..5 {
            state.add_segment(make_segment(i, 1000, 0));
        }

        let planner = CompactionPlanner::new(config);
        assert!(planner.score(&state) >= 1.0);
    }

    #[test]
    fn compaction_task_fields() {
        let task = CompactionTask {
            input_segments: vec![1, 2, 3],
            input_level: 0,
            output_level: 1,
            estimated_output_size: 3000,
        };
        assert_eq!(task.input_segments.len(), 3);
        assert_eq!(task.estimated_output_size, 3000);
    }

    #[test]
    fn level_state_empty_level() {
        let state = LevelState::new(CompactionConfig::default());
        assert!(state.segments_at_level(0).is_empty());
        assert_eq!(state.level_size(99), 0);
    }
}
