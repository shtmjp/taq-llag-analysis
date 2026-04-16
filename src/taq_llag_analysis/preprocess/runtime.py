from __future__ import annotations

import time
from datetime import UTC, datetime


def elapsed_seconds(start_time: float) -> float:
    """Return elapsed wall-clock seconds from a perf-counter start value."""
    return time.perf_counter() - start_time


def utc_now_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 ``Z`` form."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
