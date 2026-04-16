# TAQ Lead-Lag Analysis

Daily TAQ の raw data を filtering 済み parquet に変換し、その後の trade-quote mode summary や CPCF 図示に使うための研究用リポジトリです。

- 実行は repository root から `uv run ...` で行います。
- 依存関係は最初に `uv sync` でそろえます。
- 名前空間 package として実行するときは `uv run --with-editable . ...` を使います。
- raw data の仕様メモは [data/README.md](data/README.md) を参照してください。
- 一次仕様は `documents/Daily_TAQ_Client_Spec_v4.1b.pdf` です。

## Raw Data Layout

raw data は `/Users/shiotanitenshou/Research/datasets/dailytaq2` 配下にあります。

- master:
  - `EQY_US_ALL_REF_MASTER_{yyyy}/EQY_US_ALL_REF_MASTER_{yyyymm}/EQY_US_ALL_REF_MASTER_{yyyymmdd}.gz`
- trade:
  - `EQY_US_ALL_TRADE_{yyyy}/EQY_US_ALL_TRADE_{yyyymm}/EQY_US_ALL_TRADE_{yyyymmdd}/EQY_US_ALL_TRADE_{yyyymmdd}.gz`
- quote:
  - `SPLITS_US_ALL_BBO_{yyyy}/SPLITS_US_ALL_BBO_{yyyymm}/SPLITS_US_ALL_BBO_{yyyymmdd}/SPLITS_US_ALL_BBO_{shard}_{yyyymmdd}.gz`

quote は `Symbol` の先頭文字に対応する shard を自動で解決します。`A` で始まる symbol は `A` shard、`M` で始まる symbol は `M` shard を読みます。

## Main Commands

filtered parquet の CLI は trade / quote で分かれています。

trade parquet の生成:

```bash
uv run --with-editable . -m taq_llag_analysis.preprocess.trade_cli
```

quote parquet の生成:

```bash
uv run --with-editable . -m taq_llag_analysis.preprocess.quote_cli
```

- どちらも全銘柄を対象に書き出します。
- 対象日付は既定で `TARGET_DATES=("20251031", "20251103")` を使います。
- 既定の出力列は最小構成です。
  - trade: `Exchange`, `Participant Timestamp`
  - quote: `Exchange`, `Participant_Timestamp`

mode summary の生成:

```bash
uv run --with-editable . -m taq_llag_analysis.write_trade_quote_mode_summary --date 20251031 --date 20251103 --n-jobs 1
```

CPCF example 図の生成:

```bash
uv run --with-editable . -m taq_llag_analysis.write_trade_quote_cpcf_examples --run-id <mode-summary-run-id> --symbol MORN --symbol MAR
```

yfinance の平均終値テーブル生成:

```bash
uv run --with-editable . -m taq_llag_analysis.write_yfinance_avg_close_table --symbols path/to/symbols.csv --column ticker
```

## Outputs

filtered parquet と監査 JSON の出力先:

- trade parquet: `data/filtered/trade/<SYMBOL>/trade_<YYYYMMDD>.parquet`
- quote parquet: `data/filtered/quote/<SYMBOL>/quote_<YYYYMMDD>.parquet`
- trade audit: `data/filtered/trade/_audit/trade_<YYYYMMDD>.json`
- quote audit: `data/filtered/quote/_audit/quote_<YYYYMMDD>.json`

mode summary の出力先:

- `data/derived/trade_quote_modes/<run_id>/`
- 主な生成物:
  - `run_config.json`
  - `pair_summary.csv`
  - `modes.csv`

監査 JSON には少なくとも次が残ります。

- 入力 raw path
- requested symbols
- requested columns
- 実際に parquet に残した列
- 書き出せた symbol
- filter 後の総行数
- collect / write / total の処理時間

## Column Selection

trade / quote の個別 writer はライブラリ関数として使います。

- `taq_llag_analysis.preprocess.write_filtered_trade_parquet.write_filtered_trade_parquets`
- `taq_llag_analysis.preprocess.write_filtered_quote_parquet.write_filtered_quote_parquets`

`columns=None` なら filtered 後の raw 全列を保持します。`columns` を渡すと、その raw 列だけを parquet に残します。派生列は作りません。

個別 writer を Python から呼ぶ例:

```bash
uv run --with-editable . python - <<'PY'
from taq_llag_analysis.preprocess.write_filtered_trade_parquet import (
    write_filtered_trade_parquets,
)
from taq_llag_analysis.preprocess.write_filtered_quote_parquet import (
    write_filtered_quote_parquets,
)

write_filtered_trade_parquets(
    ["MAA", "MBB"],
    "20251031",
    columns=("Exchange", "Participant Timestamp"),
)
write_filtered_quote_parquets(
    ["MAA", "MBB"],
    "20251031",
    columns=("Exchange", "Participant_Timestamp"),
)
PY
```

trade CLI helper を Python から使って日付や列を変える例:

```bash
uv run --with-editable . python - <<'PY'
from taq_llag_analysis.preprocess.trade_cli import build_trade_summary

summary = build_trade_summary(
    target_dates=("20251031",),
    symbol_prefix="M",
    columns=("Exchange", "Participant Timestamp"),
)
print(summary)
PY
```

quote CLI helper を Python から使う例:

```bash
uv run --with-editable . python - <<'PY'
from taq_llag_analysis.preprocess.quote_cli import build_quote_summary

summary = build_quote_summary(
    target_dates=("20251031",),
    symbol_prefix="M",
    columns=("Exchange", "Participant_Timestamp"),
)
print(summary)
PY
```

## Logging

- writer は `INFO` ログで開始時、collect 完了時、書き出し完了時の概要を出します。
- trade / quote CLI はそれぞれ `logging.basicConfig(level=logging.INFO, ...)` を設定するため、標準出力に進行状況が出ます。
- CLI では少なくとも次を出します。
  - batch 開始
  - date ごとの開始と完了
  - writer の collect / write

## Validation

変更後は次を順に実行します。

```bash
uv run --with-editable . pytest
uv run --with-editable . ruff format
uv run --with-editable . ruff check
uv run --with-editable . ty check
```
