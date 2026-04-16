# Daily TAQ Preprocess Module

This module converts raw Daily TAQ files into filtered per-symbol parquet files.
It is organized so that filtering rules are easy to inspect separately from file
system concerns such as path resolution, logging, audit output, and CLI entrypoints.

## Module Layout

- `trade_logic.py`
  - Trade-side filtering rules.
  - This is the main review surface for trade business logic.
- `quote_logic.py`
  - Quote-side filtering rules.
  - This is the main review surface for quote business logic.
- `trade_pipeline.py`
  - Raw scan + trade filter application + final column selection.
- `quote_pipeline.py`
  - Raw scan + quote filter application + final column selection.
- `write_filtered_trade_parquet.py`
  - Trade writer orchestration, parquet output, logging, and audit JSON.
- `write_filtered_quote_parquet.py`
  - Quote writer orchestration, shard-first quote writing, logging, and audit JSON.
- `trade_cli.py`
  - CLI entrypoint for batch trade parquet generation.
- `quote_cli.py`
  - CLI entrypoint for batch quote parquet generation.
- `inventory.py`
  - Symbol discovery from the Daily TAQ master file.
- `daily_taq_paths.py`
  - Raw input paths, filtered output paths, and audit paths.
- `schemas.py`
  - Raw column lists and Polars schema overrides.
- `audit.py`
  - Audit payload construction and JSON writing.
- `runtime.py`
  - Small runtime helpers such as elapsed-time measurement.

## Design Intent

The main design goal is reviewability.

- Business logic lives in `trade_logic.py` and `quote_logic.py`.
- I/O, logging, path layout, and audit writing live outside those logic modules.
- Filter conditions are exposed as ordered named condition lists through
  `filter_conditions(...)`.
- The writer modules should answer "where and how do we write?",
  while the logic modules should answer "what rows survive?".

If you want to verify the filtering rules, start with:

- `trade_logic.filter_conditions(...)`
- `quote_logic.filter_conditions(...)`

## Public Python API

The module currently exposes these public entrypoints:

```python
from taq_llag_analysis.preprocess import (
    symbols_with_prefix,
    write_filtered_quote_parquets,
    write_filtered_trade_parquets,
)
```

### `symbols_with_prefix(date_yyyymmdd, symbol_prefix)`

Loads symbol names from the Daily TAQ master file for one trading date.

### `write_filtered_trade_parquets(symbols, date_yyyymmdd, *, columns=None)`

Writes filtered trade parquet files for one date.

### `write_filtered_quote_parquets(symbols, date_yyyymmdd, *, columns=None)`

Writes filtered quote parquet files for one date.

## CLI Entry Points

Run commands from the repository root.

Trade batch CLI:

```bash
uv run --with-editable . -m taq_llag_analysis.preprocess.trade_cli
```

Quote batch CLI:

```bash
uv run --with-editable . -m taq_llag_analysis.preprocess.quote_cli
```

Both CLIs:

- resolve symbols from the master file using `SYMBOL_PREFIX`
- process `TARGET_DATES`
- print a JSON summary to stdout
- emit `INFO` logs during filtering and writing

## Output Behavior

Trade output:

- parquet: `data/filtered/trade/<SYMBOL>/trade_<YYYYMMDD>.parquet`
- audit: `data/filtered/trade/_audit/trade_<YYYYMMDD>.json`

Quote output:

- parquet: `data/filtered/quote/<SYMBOL>/quote_<YYYYMMDD>.parquet`
- audit: `data/filtered/quote/_audit/quote_<YYYYMMDD>.json`

Quote writing is shard-first: the module resolves quote shards from the first
letter of each requested symbol and writes each shard independently.

## Column Selection

Both writers accept `columns=None` or an explicit raw-column subset.

- `columns=None`
  - keep all filtered raw columns
- `columns=(...)`
  - keep only the requested raw columns in the final parquet output

Internal partitioning may temporarily require `Symbol`, but `Symbol` is not kept
in the final parquet file unless it is explicitly requested.

## Recommended Reading Order

For trade:

1. `trade_logic.py`
2. `trade_pipeline.py`
3. `write_filtered_trade_parquet.py`

For quote:

1. `quote_logic.py`
2. `quote_pipeline.py`
3. `write_filtered_quote_parquet.py`

## Related Documentation

- Repository-level usage: `README.md`
- Raw data layout notes: `data/README.md`
- Vendor specification: `documents/Daily_TAQ_Client_Spec_v4.1b.pdf`
