# data/ 以下のデータ仕様（Daily TAQ）

一次情報:

- `documents/Daily_TAQ_Client_Spec_v4.1b.pdf`
- `documents/ISM202208_dist.pdf`（クリーニング例）

## 共通フォーマット（BBO / TRADE）

- 区切り文字: `|`（パイプ）
- 1行目: ヘッダ（列名）
- 末尾: **トレーラ行**が付く（`END|YYYYMMDD|<record_count>|...` 形式）
  - パース時はこの `END|...` 行を除外すること（例: Polars の `scan_csv(..., comment_prefix="END")`）。
- 欠損: 空欄（`||` の間）が入り得る（列によっては `" "` のようなスペースが入ることもある）。

### 時刻フィールドのエンコード

`Time` や `Participant Timestamp` 等は、概ね以下の **整数**表現です（先頭ゼロはファイル上はあり得ますが、数値として読むと落ちます）:

- `HHMMSSxxxxxxxxx`（サブ秒が 9 桁のケースが多い）

秒（`time_sec`）へは、整数演算で次のように変換できます:

- `hh = Time // 10_000_000_000_000`
- `mm = (Time // 100_000_000_000) % 100`
- `ss = (Time // 1_000_000_000) % 100`
- `time_sec = 3600*hh + 60*mm + ss`

## BBO（Best Bid and Offer）

### 入力ファイル

- 置き場所: `data/dailyTAQ/`
- ファイル名パターン（例）:
  - `SPLITS_US_ALL_BBO_A_20251103`
  - `SPLITS_US_ALL_BBO_A_20251103.gz`

### 列（ヘッダ順）

`data/dailyTAQ/SPLITS_US_ALL_BBO_*.{,gz}` のヘッダは次の列を持ちます（例: `SPLITS_US_ALL_BBO_A_20251103`）:

1. `Time`
2. `Exchange`
3. `Symbol`
4. `Bid_Price`
5. `Bid_Size`
6. `Offer_Price`
7. `Offer_Size`
8. `Quote_Condition`
9. `Sequence_Number`
10. `National_BBO_Ind`
11. `FINRA_BBO_Indicator`
12. `FINRA_ADF_MPID_Indicator`
13. `Quote_Cancel_Correction`
14. `Source_Of_Quote`
15. `Retail_Interest_Indicator`
16. `Short_Sale_Restriction_Indicator`
17. `LULD_BBO_Indicator`
18. `SIP_Generated_Message_Identifier`
19. `National_BBO_LULD_Indicator`
20. `Participant_Timestamp`
21. `FINRA_ADF_Timestamp`
22. `FINRA_ADF_Market_Participant_Quote_Indicator`
23. `Security_Status_Indicator`

### 典型的なクリーニング（研究用の推奨）

`documents/ISM202208_dist.pdf` の例に合わせる場合:

- `Bid_Price > 0`
- `Offer_Price > 0`
- `Bid_Price < Offer_Price`（正スプレッド）
- 正規時間（例: 09:30–16:00）に限定する場合は `time_sec` で `34200 <= time_sec <= 57600`

## TRADE（Trades）

### 入力ファイル

- 置き場所: `data/dailyTAQ/`
- ファイル名パターン（例）:
  - `EQY_US_ALL_TRADE_20251103`
  - `EQY_US_ALL_TRADE_20251103.gz`

### 列（ヘッダ順）

`data/dailyTAQ/EQY_US_ALL_TRADE_*.{,gz}` のヘッダは次の列を持ちます（例: `EQY_US_ALL_TRADE_20251103`）:

1. `Time`
2. `Exchange`
3. `Symbol`
4. `Sale Condition`
5. `Trade Volume`
6. `Trade Price`
7. `Trade Stop Stock Indicator`
8. `Trade Correction Indicator`
9. `Sequence Number`
10. `Trade Id`
11. `Source of Trade`
12. `Trade Reporting Facility`
13. `Participant Timestamp`
14. `Trade Reporting Facility TRF Timestamp`
15. `Trade Through Exempt Indicator`

注意: 列名にスペースを含む（例: `Sale Condition`）ため、データ処理では列参照にそのままの文字列が必要です。

## 派生データ（例）

- `data/derived/daily_taq_bbo_parquet/`
  - `src/bench_daily_taq_bbo_filter_parquet.py` が生成する BBO フィルタ済み Parquet の出力先（デフォルト）。
