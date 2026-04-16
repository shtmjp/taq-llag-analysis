from __future__ import annotations

import json
from typing import TYPE_CHECKING

from . import runtime

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def build_filter_audit_payload(
    *,
    dataset: str,
    date_yyyymmdd: str,
    input_paths: Sequence[Path],
    requested_symbols: Sequence[str],
    requested_columns: Sequence[str] | None,
    resolved_output_columns: Sequence[str],
    written_symbols: Sequence[str],
    n_filtered_rows_total: int,
    collect_elapsed_sec: float,
    write_elapsed_sec: float,
    total_elapsed_sec: float,
) -> dict[str, object]:
    """Build the standard audit payload for one filter-write run.

    Parameters
    ----------
    dataset
        Dataset label such as ``"trade"`` or ``"quote"``.
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.
    input_paths
        Raw input paths used for the run.
    requested_symbols
        Symbols requested by the caller.
    requested_columns
        Optional caller-specified output columns.
    resolved_output_columns
        Final output columns written to parquet.
    written_symbols
        Symbols that produced parquet output.
    n_filtered_rows_total
        Total number of rows surviving the filters.
    collect_elapsed_sec
        Elapsed seconds spent counting or collecting filtered rows.
    write_elapsed_sec
        Elapsed seconds spent writing parquet outputs.
    total_elapsed_sec
        End-to-end elapsed seconds for the run.

    Returns
    -------
    dict[str, object]
        JSON-serializable audit payload.

    """
    requested_symbols_list = list(requested_symbols)
    written_symbols_list = list(written_symbols)
    return {
        "created_at_utc": runtime.utc_now_timestamp(),
        "dataset": dataset,
        "date_yyyymmdd": date_yyyymmdd,
        "input_paths": [str(path) for path in input_paths],
        "requested_symbols": requested_symbols_list,
        "requested_columns": list(requested_columns) if requested_columns is not None else None,
        "resolved_output_columns": list(resolved_output_columns),
        "written_symbols": written_symbols_list,
        "n_requested_symbols": len(requested_symbols_list),
        "n_written_symbols": len(written_symbols_list),
        "n_filtered_rows_total": n_filtered_rows_total,
        "stage_elapsed_sec": {
            "collect": collect_elapsed_sec,
            "write": write_elapsed_sec,
            "total": total_elapsed_sec,
        },
    }


def write_audit_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Write an audit JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
