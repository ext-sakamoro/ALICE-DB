//! Raft ベースレプリケーション。
//!
//! 単一ノード ALICE-DB を複数ノードに拡張するための
//! Raft コンセンサスプロトコル実装。
//! ログ複製 + リーダー選出 + スナップショット転送。
//!
//! Author: Moroya Sakamoto

use std::collections::HashMap;

// ============================================================================
// 型定義
// ============================================================================

/// ノード ID。
pub type NodeId = u64;

/// Raft の任期番号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term(pub u64);

/// ログエントリのインデックス。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogIndex(pub u64);

// ============================================================================
// ログエントリ
// ============================================================================

/// レプリケーションログの 1 エントリ。
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// エントリのインデックス。
    pub index: LogIndex,
    /// エントリが書き込まれた任期。
    pub term: Term,
    /// コマンド種別。
    pub command: Command,
}

/// ストレージに適用するコマンド。
#[derive(Debug, Clone)]
pub enum Command {
    /// 単一値の挿入。
    Put { timestamp: i64, value: f32 },
    /// バッチ挿入。
    PutBatch { data: Vec<(i64, f32)> },
    /// フラッシュ要求。
    Flush,
    /// 設定変更 (メンバーシップ)。
    ConfigChange {
        add: Vec<NodeId>,
        remove: Vec<NodeId>,
    },
    /// 空コマンド (リーダー確認用)。
    Noop,
}

// ============================================================================
// ノード状態
// ============================================================================

/// Raft ノードの役割。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// フォロワー: リーダーからログを受信。
    Follower,
    /// 候補者: 選挙中。
    Candidate,
    /// リーダー: クライアント要求を処理しログを複製。
    Leader,
}

/// Raft ノードの永続状態。
#[derive(Debug)]
pub struct PersistentState {
    /// 現在の任期。
    pub current_term: Term,
    /// この任期で投票したノード (None = 未投票)。
    pub voted_for: Option<NodeId>,
    /// ログエントリ列。
    pub log: Vec<LogEntry>,
}

impl PersistentState {
    /// 新しい永続状態を初期化。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current_term: Term(0),
            voted_for: None,
            log: Vec::new(),
        }
    }

    /// ログの最後のインデックスと任期を取得。
    #[must_use]
    pub fn last_log_info(&self) -> (LogIndex, Term) {
        self.log
            .last()
            .map_or((LogIndex(0), Term(0)), |e| (e.index, e.term))
    }

    /// 指定インデックスのエントリを取得。
    #[must_use]
    pub fn get_entry(&self, index: LogIndex) -> Option<&LogEntry> {
        self.log.iter().find(|e| e.index == index)
    }

    /// 指定インデックス以降のエントリを取得。
    #[must_use]
    pub fn entries_from(&self, index: LogIndex) -> &[LogEntry] {
        let pos = self.log.iter().position(|e| e.index >= index);
        match pos {
            Some(p) => &self.log[p..],
            None => &[],
        }
    }

    /// ログにエントリを追加。
    pub fn append(&mut self, entry: LogEntry) {
        // 同一インデックスの既存エントリを削除 (任期不一致時)
        if let Some(pos) = self.log.iter().position(|e| e.index == entry.index) {
            if self.log[pos].term == entry.term {
                return; // 既に存在
            }
            self.log.truncate(pos);
        }
        self.log.push(entry);
    }
}

impl Default for PersistentState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Raft ノード
// ============================================================================

/// Raft コンセンサスノード。
pub struct RaftNode {
    /// このノードの ID。
    pub id: NodeId,
    /// 現在の役割。
    pub role: Role,
    /// 永続状態。
    pub state: PersistentState,
    /// コミット済みインデックス。
    pub commit_index: LogIndex,
    /// 適用済みインデックス。
    pub last_applied: LogIndex,
    /// クラスタメンバー。
    pub peers: Vec<NodeId>,
    /// リーダーが管理: 各フォロワーの次送信インデックス。
    pub next_index: HashMap<NodeId, LogIndex>,
    /// リーダーが管理: 各フォロワーの一致確認済みインデックス。
    pub match_index: HashMap<NodeId, LogIndex>,
    /// 受信した投票数 (候補者時)。
    pub votes_received: usize,
    /// 現在のリーダー ID。
    pub leader_id: Option<NodeId>,
}

impl RaftNode {
    /// 新しい Raft ノードを作成。
    #[must_use]
    pub fn new(id: NodeId, peers: Vec<NodeId>) -> Self {
        Self {
            id,
            role: Role::Follower,
            state: PersistentState::new(),
            commit_index: LogIndex(0),
            last_applied: LogIndex(0),
            peers,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            votes_received: 0,
            leader_id: None,
        }
    }

    /// クラスタの過半数ノード数。
    #[must_use]
    pub const fn majority(&self) -> usize {
        self.peers.len().div_ceil(2) + 1
    }

    /// リーダー選挙を開始。
    pub fn start_election(&mut self) -> RequestVote {
        self.role = Role::Candidate;
        self.state.current_term = Term(self.state.current_term.0 + 1);
        self.state.voted_for = Some(self.id);
        self.votes_received = 1; // 自分に投票

        let (last_log_index, last_log_term) = self.state.last_log_info();

        RequestVote {
            term: self.state.current_term,
            candidate_id: self.id,
            last_log_index,
            last_log_term,
        }
    }

    /// 投票要求を処理。
    #[must_use]
    pub fn handle_vote_request(&mut self, req: &RequestVote) -> VoteResponse {
        // 任期が古い場合は拒否
        if req.term < self.state.current_term {
            return VoteResponse {
                term: self.state.current_term,
                vote_granted: false,
            };
        }

        // 任期が新しい場合はフォロワーに戻る
        if req.term > self.state.current_term {
            self.state.current_term = req.term;
            self.state.voted_for = None;
            self.role = Role::Follower;
        }

        // 投票判定: 未投票 or 同一候補、かつログが最新以上
        let (my_last_index, my_last_term) = self.state.last_log_info();
        let log_ok = req.last_log_term > my_last_term
            || (req.last_log_term == my_last_term && req.last_log_index >= my_last_index);

        let can_vote =
            self.state.voted_for.is_none() || self.state.voted_for == Some(req.candidate_id);

        let vote_granted = can_vote && log_ok;
        if vote_granted {
            self.state.voted_for = Some(req.candidate_id);
        }

        VoteResponse {
            term: self.state.current_term,
            vote_granted,
        }
    }

    /// 投票応答を処理。
    pub fn handle_vote_response(&mut self, resp: &VoteResponse) {
        if resp.term > self.state.current_term {
            self.state.current_term = resp.term;
            self.role = Role::Follower;
            return;
        }

        if self.role != Role::Candidate {
            return;
        }

        if resp.vote_granted {
            self.votes_received += 1;
            if self.votes_received >= self.majority() {
                self.become_leader();
            }
        }
    }

    /// リーダーに昇格。
    fn become_leader(&mut self) {
        self.role = Role::Leader;
        self.leader_id = Some(self.id);
        let (last_index, _) = self.state.last_log_info();
        let next = LogIndex(last_index.0 + 1);

        for &peer in &self.peers {
            self.next_index.insert(peer, next);
            self.match_index.insert(peer, LogIndex(0));
        }

        // Noop エントリを追加してコミットインデックスを更新
        let entry = LogEntry {
            index: next,
            term: self.state.current_term,
            command: Command::Noop,
        };
        self.state.append(entry);
    }

    /// クライアントコマンドを追加 (リーダーのみ)。
    ///
    /// # Errors
    ///
    /// リーダーでない場合は `Err` を返す。
    pub fn propose(&mut self, command: Command) -> Result<LogIndex, ReplicationError> {
        if self.role != Role::Leader {
            return Err(ReplicationError::NotLeader {
                leader_id: self.leader_id,
            });
        }

        let (last_index, _) = self.state.last_log_info();
        let index = LogIndex(last_index.0 + 1);
        let entry = LogEntry {
            index,
            term: self.state.current_term,
            command,
        };
        self.state.append(entry);
        Ok(index)
    }

    /// `AppendEntries` RPC を生成 (リーダー→フォロワー)。
    #[must_use]
    pub fn build_append_entries(&self, peer: NodeId) -> Option<AppendEntries> {
        if self.role != Role::Leader {
            return None;
        }

        let next = self.next_index.get(&peer)?;
        let prev_index = LogIndex(next.0.saturating_sub(1));
        let prev_term = self.state.get_entry(prev_index).map_or(Term(0), |e| e.term);
        let entries = self.state.entries_from(*next).to_vec();

        Some(AppendEntries {
            term: self.state.current_term,
            leader_id: self.id,
            prev_log_index: prev_index,
            prev_log_term: prev_term,
            entries,
            leader_commit: self.commit_index,
        })
    }

    /// `AppendEntries` RPC を処理 (フォロワー側)。
    #[must_use]
    pub fn handle_append_entries(&mut self, req: &AppendEntries) -> AppendEntriesResponse {
        // 任期チェック
        if req.term < self.state.current_term {
            return AppendEntriesResponse {
                term: self.state.current_term,
                success: false,
                match_index: LogIndex(0),
            };
        }

        // 任期更新
        if req.term > self.state.current_term {
            self.state.current_term = req.term;
            self.state.voted_for = None;
        }
        self.role = Role::Follower;
        self.leader_id = Some(req.leader_id);

        // 前エントリの整合性チェック
        if req.prev_log_index.0 > 0 {
            match self.state.get_entry(req.prev_log_index) {
                Some(e) if e.term != req.prev_log_term => {
                    return AppendEntriesResponse {
                        term: self.state.current_term,
                        success: false,
                        match_index: LogIndex(0),
                    };
                }
                None if req.prev_log_index.0 > 0 => {
                    return AppendEntriesResponse {
                        term: self.state.current_term,
                        success: false,
                        match_index: LogIndex(0),
                    };
                }
                _ => {}
            }
        }

        // エントリを追加
        for entry in &req.entries {
            self.state.append(entry.clone());
        }

        // コミットインデックス更新
        if req.leader_commit > self.commit_index {
            let (last_index, _) = self.state.last_log_info();
            self.commit_index = if req.leader_commit.0 < last_index.0 {
                req.leader_commit
            } else {
                last_index
            };
        }

        let (last_index, _) = self.state.last_log_info();
        AppendEntriesResponse {
            term: self.state.current_term,
            success: true,
            match_index: last_index,
        }
    }

    /// `AppendEntries` 応答を処理 (リーダー側)。
    pub fn handle_append_response(&mut self, peer: NodeId, resp: &AppendEntriesResponse) {
        if resp.term > self.state.current_term {
            self.state.current_term = resp.term;
            self.role = Role::Follower;
            return;
        }

        if resp.success {
            self.match_index.insert(peer, resp.match_index);
            self.next_index
                .insert(peer, LogIndex(resp.match_index.0 + 1));
            self.try_advance_commit();
        } else {
            // next_index を 1 つ下げてリトライ
            if let Some(next) = self.next_index.get_mut(&peer) {
                next.0 = next.0.saturating_sub(1).max(1);
            }
        }
    }

    /// 過半数が一致したインデックスまでコミットを進める。
    fn try_advance_commit(&mut self) {
        let (last_index, _) = self.state.last_log_info();

        for idx in (self.commit_index.0 + 1)..=last_index.0 {
            // 現在の任期のエントリのみコミット可能
            if let Some(entry) = self.state.get_entry(LogIndex(idx)) {
                if entry.term != self.state.current_term {
                    continue;
                }
            }

            let mut count = 1usize; // リーダー自身
            for &peer in &self.peers {
                if self.match_index.get(&peer).is_some_and(|m| m.0 >= idx) {
                    count += 1;
                }
            }

            if count >= self.majority() {
                self.commit_index = LogIndex(idx);
            }
        }
    }
}

// ============================================================================
// RPC メッセージ
// ============================================================================

/// 投票要求。
#[derive(Debug, Clone)]
pub struct RequestVote {
    pub term: Term,
    pub candidate_id: NodeId,
    pub last_log_index: LogIndex,
    pub last_log_term: Term,
}

/// 投票応答。
#[derive(Debug, Clone)]
pub struct VoteResponse {
    pub term: Term,
    pub vote_granted: bool,
}

/// ログ追加要求。
#[derive(Debug, Clone)]
pub struct AppendEntries {
    pub term: Term,
    pub leader_id: NodeId,
    pub prev_log_index: LogIndex,
    pub prev_log_term: Term,
    pub entries: Vec<LogEntry>,
    pub leader_commit: LogIndex,
}

/// ログ追加応答。
#[derive(Debug, Clone)]
pub struct AppendEntriesResponse {
    pub term: Term,
    pub success: bool,
    pub match_index: LogIndex,
}

// ============================================================================
// エラー
// ============================================================================

/// レプリケーションエラー。
#[derive(Debug)]
pub enum ReplicationError {
    /// このノードはリーダーではない。
    NotLeader { leader_id: Option<NodeId> },
    /// ログ不整合。
    LogInconsistency,
}

impl std::fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotLeader { leader_id } => {
                write!(f, "Not leader (leader: {leader_id:?})")
            }
            Self::LogInconsistency => write!(f, "Log inconsistency"),
        }
    }
}

impl std::error::Error for ReplicationError {}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster() -> (RaftNode, RaftNode, RaftNode) {
        let n1 = RaftNode::new(1, vec![2, 3]);
        let n2 = RaftNode::new(2, vec![1, 3]);
        let n3 = RaftNode::new(3, vec![1, 2]);
        (n1, n2, n3)
    }

    #[test]
    fn new_node_is_follower() {
        let node = RaftNode::new(1, vec![2, 3]);
        assert_eq!(node.role, Role::Follower);
        assert_eq!(node.state.current_term, Term(0));
    }

    #[test]
    fn majority_3_nodes() {
        let node = RaftNode::new(1, vec![2, 3]);
        assert_eq!(node.majority(), 2);
    }

    #[test]
    fn majority_5_nodes() {
        let node = RaftNode::new(1, vec![2, 3, 4, 5]);
        assert_eq!(node.majority(), 3);
    }

    #[test]
    fn start_election_becomes_candidate() {
        let mut node = RaftNode::new(1, vec![2, 3]);
        let req = node.start_election();
        assert_eq!(node.role, Role::Candidate);
        assert_eq!(node.state.current_term, Term(1));
        assert_eq!(node.votes_received, 1);
        assert_eq!(req.candidate_id, 1);
    }

    #[test]
    fn vote_granted_for_valid_request() {
        let (mut n1, mut n2, _) = make_cluster();
        let req = n1.start_election();
        let resp = n2.handle_vote_request(&req);
        assert!(resp.vote_granted);
    }

    #[test]
    fn vote_denied_for_old_term() {
        let (mut n1, mut n2, _) = make_cluster();
        n2.state.current_term = Term(5);
        let req = n1.start_election(); // term=1
        let resp = n2.handle_vote_request(&req);
        assert!(!resp.vote_granted);
    }

    #[test]
    fn leader_election_with_majority() {
        let (mut n1, mut n2, mut n3) = make_cluster();
        let req = n1.start_election();

        let resp2 = n2.handle_vote_request(&req);
        n1.handle_vote_response(&resp2);

        // n1 は自分 + n2 = 2票 → 過半数 (3ノード中2)
        assert_eq!(n1.role, Role::Leader);

        let resp3 = n3.handle_vote_request(&req);
        assert!(resp3.vote_granted);
    }

    #[test]
    fn propose_on_leader() {
        let (mut n1, mut n2, _) = make_cluster();
        let req = n1.start_election();
        let resp = n2.handle_vote_request(&req);
        n1.handle_vote_response(&resp);
        assert_eq!(n1.role, Role::Leader);

        let idx = n1
            .propose(Command::Put {
                timestamp: 100,
                value: 42.0,
            })
            .unwrap();
        assert!(idx.0 > 0);
    }

    #[test]
    fn propose_on_follower_fails() {
        let node = RaftNode::new(1, vec![2, 3]);
        let result = node.clone_for_test().propose(Command::Noop);
        assert!(result.is_err());
    }

    #[test]
    fn append_entries_basic() {
        let (mut n1, mut n2, _) = make_cluster();
        // n1 をリーダーにする
        let req = n1.start_election();
        let resp = n2.handle_vote_request(&req);
        n1.handle_vote_response(&resp);

        // コマンドを追加
        n1.propose(Command::Put {
            timestamp: 1,
            value: 10.0,
        })
        .unwrap();

        // AppendEntries を生成して n2 に送信
        let ae = n1.build_append_entries(2).unwrap();
        let ae_resp = n2.handle_append_entries(&ae);
        assert!(ae_resp.success);
    }

    #[test]
    fn append_entries_updates_commit() {
        let (mut n1, mut n2, mut n3) = make_cluster();
        let req = n1.start_election();
        let r2 = n2.handle_vote_request(&req);
        n1.handle_vote_response(&r2);
        let _ = n3.handle_vote_request(&req);

        // Put コマンド
        n1.propose(Command::Put {
            timestamp: 1,
            value: 1.0,
        })
        .unwrap();

        // n2 に複製
        let ae2 = n1.build_append_entries(2).unwrap();
        let resp2 = n2.handle_append_entries(&ae2);
        n1.handle_append_response(2, &resp2);

        // n3 に複製
        let ae3 = n1.build_append_entries(3).unwrap();
        let resp3 = n3.handle_append_entries(&ae3);
        n1.handle_append_response(3, &resp3);

        // 過半数が一致 → コミット進行
        assert!(n1.commit_index.0 > 0);
    }

    #[test]
    fn follower_rejects_stale_term() {
        let mut n2 = RaftNode::new(2, vec![1, 3]);
        n2.state.current_term = Term(5);

        let ae = AppendEntries {
            term: Term(1),
            leader_id: 1,
            prev_log_index: LogIndex(0),
            prev_log_term: Term(0),
            entries: vec![],
            leader_commit: LogIndex(0),
        };
        let resp = n2.handle_append_entries(&ae);
        assert!(!resp.success);
    }

    #[test]
    fn persistent_state_default() {
        let ps = PersistentState::default();
        assert_eq!(ps.current_term, Term(0));
        assert!(ps.voted_for.is_none());
        assert!(ps.log.is_empty());
    }

    #[test]
    fn persistent_state_append_and_get() {
        let mut ps = PersistentState::new();
        ps.append(LogEntry {
            index: LogIndex(1),
            term: Term(1),
            command: Command::Noop,
        });
        ps.append(LogEntry {
            index: LogIndex(2),
            term: Term(1),
            command: Command::Flush,
        });
        assert_eq!(ps.log.len(), 2);
        assert!(ps.get_entry(LogIndex(1)).is_some());
        assert!(ps.get_entry(LogIndex(3)).is_none());
    }

    #[test]
    fn entries_from_returns_suffix() {
        let mut ps = PersistentState::new();
        for i in 1..=5 {
            ps.append(LogEntry {
                index: LogIndex(i),
                term: Term(1),
                command: Command::Noop,
            });
        }
        let entries = ps.entries_from(LogIndex(3));
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn replication_error_display() {
        let err = ReplicationError::NotLeader { leader_id: Some(2) };
        assert!(err.to_string().contains("Not leader"));

        let err2 = ReplicationError::LogInconsistency;
        assert!(err2.to_string().contains("Log inconsistency"));
    }

    #[test]
    fn double_vote_same_candidate() {
        let (mut n1, mut n2, _) = make_cluster();
        let req = n1.start_election();
        let resp1 = n2.handle_vote_request(&req);
        assert!(resp1.vote_granted);
        // 同一候補への再投票は許可
        let resp2 = n2.handle_vote_request(&req);
        assert!(resp2.vote_granted);
    }

    #[test]
    fn config_change_command() {
        let (mut n1, mut n2, _) = make_cluster();
        let req = n1.start_election();
        let resp = n2.handle_vote_request(&req);
        n1.handle_vote_response(&resp);

        let idx = n1
            .propose(Command::ConfigChange {
                add: vec![4],
                remove: vec![],
            })
            .unwrap();
        assert!(idx.0 > 0);
    }
}

// テスト用ヘルパー (propose でムーブを避けるため)
impl RaftNode {
    #[cfg(test)]
    fn clone_for_test(&self) -> Self {
        Self {
            id: self.id,
            role: self.role,
            state: PersistentState {
                current_term: self.state.current_term,
                voted_for: self.state.voted_for,
                log: self.state.log.clone(),
            },
            commit_index: self.commit_index,
            last_applied: self.last_applied,
            peers: self.peers.clone(),
            next_index: self.next_index.clone(),
            match_index: self.match_index.clone(),
            votes_received: self.votes_received,
            leader_id: self.leader_id,
        }
    }
}
