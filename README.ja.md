# ALICE-DB

**モデルベースLSMツリーデータベース** - [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) 手続き的生成エンジン搭載

<p align="center">
  <img src="assets/concept.png" alt="ALICE-DB Concept" width="600">
</p>

## 革命：「関数に問い合わせる」

**従来のデータベースは生データを保存：**
```
1000件のセンサー読み取り → ディスク上に4KB
クエリ: ディスクから4KBを読み取り → 解凍 → 返却
```

**ALICE-DBはデータを生成する数学モデルを保存：**
```
1000件のセンサー読み取り（線形トレンド） → "y = 0.5x + 10" → ディスク上に16バイト
クエリ: f(500) = 0.5 * 500 + 10 = 260.0 を計算 → 返却
```

これはコルモゴロフ複雑性に基づいています：*出力を生成する最短のプログラムが最適な表現である。*

## 主な機能

| 機能 | 説明 |
|------|------|
| **極限圧縮** | 構造化時系列データで50〜1000倍 |
| **O(1)ポイントクエリ** | ディスク読み取りの代わりにf(x)を計算 |
| **自動モデル選択** | 多項式、フーリエ、パーリン、LZMAフォールバック |
| **LSMツリーアーキテクチャ** | モデルベースSSTableによる書き込み最適化 |
| **Python + Rust** | 高レベルPython API、ベアメタルRustコア |

## クイックスタート

### Python

```bash
pip install alice-db
# またはソースからビルド:
cd ALICE-DB && pip install maturin && maturin develop --release
```

```python
import alice_db
import numpy as np

# データベースを開く
db = alice_db.open("./my_timeseries")

# 時系列データを挿入
for i in range(10000):
    db.put(timestamp=i, value=np.sin(i * 0.01) * 100)

# またはnumpyでバッチ挿入（ゼロコピー）
timestamps = np.arange(10000, dtype=np.int64)
values = np.sin(timestamps * 0.01).astype(np.float32) * 100
db.put_numpy(timestamps, values)

# クエリ - モデルからsin(5000 * 0.01)を計算、ディスク読み取りなし！
value = db.get(5000)

# 範囲クエリ
points = db.scan(0, 9999)

# 集計
avg = db.aggregate(0, 9999, "avg")
total = db.aggregate(0, 9999, "sum")

# ダウンサンプリング（時間間隔でGROUP BY）
hourly = db.downsample(0, 9999, interval=3600, agg="avg")

# 圧縮統計を確認
stats = db.stats()
print(f"圧縮率: {stats.average_compression_ratio:.1f}x")
print(f"使用モデル: {stats.model_distribution}")

db.close()
```

### Rust

```rust
use alice_db::{AliceDB, Aggregation};

fn main() -> std::io::Result<()> {
    let db = AliceDB::open("./my_timeseries")?;

    // 挿入
    for i in 0..10000 {
        let value = (i as f32 * 0.01).sin() * 100.0;
        db.put(i, value)?;
    }

    // クエリ（モデルから計算！）
    if let Some(value) = db.get(5000)? {
        println!("5000での値: {}", value);
    }

    // 集計
    let avg = db.aggregate(0, 9999, Aggregation::Avg)?;
    println!("平均: {}", avg);

    // 統計
    let stats = db.stats();
    println!("圧縮率: {:.1}x", stats.average_compression_ratio);

    db.close()
}
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                       ALICE-DB                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │  MemTable   │───▶│   Fitter    │───▶│  Segment   │  │
│  │  (Vec)      │    │ Competition │    │  (Model)   │  │
│  └─────────────┘    └─────────────┘    └────────────┘  │
│         │                  │                  │        │
│         │         多項式、フーリエ、           │        │
│         │         正弦波、パーリン、LZMA      │        │
│         ▼                  ▼                  ▼        │
│  ┌─────────────────────────────────────────────────┐   │
│  │                 Storage Engine                   │   │
│  │  • Interval Tree（時間 → モデル）              │   │
│  │  • WAL（耐久性）                               │   │
│  │  • Zero-Copy（rkyv + mmap）                    │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                             │
│                          ▼                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   libalice                       │   │
│  │  • 多項式フィッティング（ホーナー法）          │   │
│  │  • フーリエ解析（FFT）                         │   │
│  │  • パーリンノイズ生成                          │   │
│  │  • LZMA圧縮（フォールバック）                  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## モデルタイプ

ALICE-DBはデータに最適なモデルを自動選択します：

| モデル | ユースケース | 圧縮率 |
|--------|--------------|--------|
| **Constant** | 平坦な線 | ∞（合計8バイト） |
| **Linear** | トレンド、傾斜 | 約250倍 |
| **Polynomial** | 曲線、ドリフト | 約50〜200倍 |
| **Fourier** | 周期信号 | 約20〜100倍 |
| **SineWave** | 単純な振動 | 約250倍 |
| **Perlin** | ノイズパターン | 約40倍 |
| **RawLZMA** | ランダムデータ（フォールバック） | 約2〜5倍 |

## パフォーマンス

### 書き込み性能

| 操作 | スループット |
|------|--------------|
| 単一挿入 | 約500K ops/sec |
| バッチ挿入 | 約2M points/sec |
| フラッシュ（モデルフィッティング） | 1000ポイントあたり約1ms |

### クエリ性能

| 操作 | レイテンシ |
|------|------------|
| ポイントクエリ | **約120ns**（f(x)を計算） |
| 範囲クエリ（1000ポイント） | 約5µs |
| 集計（10Kポイント） | 約50µs |

### 圧縮率

| データタイプ | 圧縮率 |
|--------------|--------|
| 線形センサーデータ | **100〜500倍** |
| 正弦波（温度） | **50〜200倍** |
| 多項式トレンド | **50〜150倍** |
| ランダムノイズ | 2〜5倍（LZMAフォールバック） |

## ソースからビルド

### 要件

- Rust 1.75以上
- Python 3.9以上（Pythonバインディング用）
- maturin（Pythonパッケージ用）

### ビルド

```bash
# クローン
git clone https://github.com/ext-sakamoro/ALICE-DB.git
cd ALICE-DB

# Rustライブラリをビルド
cargo build --release

# Pythonパッケージをビルド
pip install maturin
maturin develop --release

# テストを実行
cargo test
pytest tests/

# ベンチマークを実行
cargo bench
```

## 最適化技術

ALICE-DBは以下の最適化を実装しています：

| 技術 | 効果 |
|------|------|
| **Loop Unswitching** | 分岐予測ミスを排除 |
| **SIMD (f64x4)** | 多項式計算を4並列化 |
| **Zero-Copy (rkyv + mmap)** | デシリアライズコストゼロ |
| **Arena Allocation** | L1キャッシュヒット率最大化 |
| **Streaming Aggregation** | O(1)メモリ使用量 |

## 関連プロジェクト

| プロジェクト | 説明 |
|--------------|------|
| [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) | コア手続き的生成エンジン |
| [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) | 超低帯域幅ビデオストリーミング |

すべてのプロジェクトは共通の哲学を共有：**データ自体ではなく、生成プロセスをエンコードする。**

## ライセンス

ALICE-DBはオープンコアライセンスモデルを採用：

| 製品 | ライセンス | 説明 |
|------|------------|------|
| **ALICE-DB Core** | MIT | 無料・オープンソース |
| **ALICE-DB Server** | BSL 1.1 | ソース公開・DBaaS制限 |
| **ALICE-DB Enterprise** | 商用 | 年間契約 |

詳細は [LICENSE](LICENSE) を参照してください。

## 著者

坂本師哉 (Moroya Sakamoto)

---

*「最高の圧縮とは、料理ではなくレシピを保存することである。」*
