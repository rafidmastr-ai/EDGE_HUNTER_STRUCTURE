from __future__ import annotations

"""
SMC MTF RESEARCH V15 — Expanded Candidate Generator + ML Feature Layer
================================================
Purpose
-------
Keep the existing Tick Data/cache and execution layer intact while extending candidate capture for future per-symbol ML.
V15 is a single research version: it expands candidate capture modestly while preserving the SMC structural sequence and execution layer.
No previous-version result matrix is executed, loaded, compared, or written by this file.

Research period (UTC): 2020-01-01 -> 2026-09-30
Data source: data/tick_cache/<SYMBOL>.parquet (preferred) or CSV fallback, or M1 OHLC
files (data/tick_cache/<SYMBOL>_M1*.csv with timestamp,open,high,low,close,tick_volume,spread)
replayed conservatively (stop before target when one bar touches both). See DATA_MODE.
Execution: M1 decision -> next M1 executable quote; SL/TP resolved on ticks.
Higher timeframes: M5 and M15 are mapped only after their bars are complete.
Context features (session, today's/Asian range, stop size) come from M1 bars and the
clock only; they are ML features and never change which candidates are produced.
Holding horizon: maximum 120 minutes.
Cost model: GROSS price performance only. Broker spread and commission are
excluded. Tick Data are used only to reconstruct a broker-neutral market price
(mid when available, otherwise Last/fallback). Optional explicit exit slippage
is kept at 0 by default and is not a broker spread/commission charge.

SMC setup families
------------------
A_OB_RETEST_SEQUENCE
    Liquidity sweep -> displacement -> BOS -> order-block first retest -> confirmation.

B_FVG_RETEST_SEQUENCE
    Liquidity sweep -> displacement -> BOS -> FVG first retest -> confirmation.

C_DISPLACEMENT_MID_RETEST
    Liquidity sweep -> displacement -> BOS -> displacement-mid first retest -> confirmation.

The script preserves the existing SMC event/execution sequence and evaluates only V15. V15
modestly relaxes selected late-stage candidate gates while retaining the structural safeguards.
No previous-version result matrix is executed or written by this file.
Results are gross/before-broker-costs by design; no spread or commission is included.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from collections import deque
import re
import sys
import json
import math

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Fixed period / symbols
# ---------------------------------------------------------------------------
# SMC MTF RESEARCH V15
# The historical-data/cache layer below is intentionally left unchanged.
PERIOD_START = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
PERIOD_END = datetime(2026, 9, 30, 23, 59, 59, 999000, tzinfo=timezone.utc)
SYMBOLS = ("XAUUSD", "EURUSD", "AUDJPY")

M1_RULE = "1min"
M5_RULE = "5min"
M15_RULE = "15min"

# M1 structure is made slightly more responsive to raise trade count,
# while higher-timeframe structure is still confirmed before use.
SWING = 2
OB_LOOKBACK = 10
MAX_SWEEP_AGE = 8
ZONE_MAX_AGE = 15
ATR_PERIOD = 14
M5_FAST = 20
M5_SLOW = 50
M15_FAST = 20
M15_SLOW = 50
M15_LIQ_LOOKBACK = 20
M15_VETO_ATR = 0.90

# Stop model: structural level + adaptive buffer, bounded by ATR.
SL_BUFFER_POINTS = 3.0
SL_BUFFER_ATR = 0.10
ATR_FLOOR = 0.60
ATR_CAP = 2.50

# Short-horizon trading.
MAX_HOLD_MINUTES = 120

# Feature-layer revision only; strategy rules and execution logic are unchanged.
FEATURE_LAYER_VERSION = "V15_SMC_EVENT_FEATURES_3_M1_CONTEXT"

# Broker-neutral cost model. Spread and commission are intentionally excluded.
# The user applies broker-specific spread/commission separately after the research.

RR_VALUES = (1.5, 2.0, 2.5)
SL_MODES = ("STRUCTURAL", "STRUCTURAL_ATR_BOUNDED")
FAMILIES = (
    "A_OB_RETEST_SEQUENCE",
    "B_FVG_RETEST_SEQUENCE",
    "C_DISPLACEMENT_MID_RETEST",
)

# V4 event-sequence parameters. These are deliberately kept compact so the
# next research round can test their effect without changing the data layer.
SWEEP_MIN_ATR = 0.05
DISP_BODY_ATR = 0.55
DISP_RANGE_ATR = 0.75
DISP_CLOSE_LOCATION = 0.65
BOS_BUFFER_ATR = 0.05
CONFIRM_BODY_RATIO = 0.35
MAX_RETEST_BARS = 15

# Optional research-only explicit exit slippage. Default 0 keeps pure gross price
# performance. This is NOT a spread or commission model.
EXIT_SLIPPAGE_POINTS = 0.0

# ---------------------------------------------------------------------------
# V15 quality-preserving candidate expansion
# ---------------------------------------------------------------------------
# The core SMC structure is unchanged. V15 adds candidates only when exactly
# one late-stage soft gate needs relaxation AND the setup is compensated by
# stronger displacement/confirmation and full M5+M15 alignment. ML indicators
# below remain FEATURES ONLY; they are never used to reject a candidate.
V15_PROFILE = "V15_QUALITY_PRESERVING_EXPANDED"
V15_FILTERS = {
    # A/B receive most of the candidate expansion because V14 showed they
    # retain better raw quality than C on the main gold test period.
    "A_OB_RETEST_SEQUENCE": {
        "sweep_min": 0.45, "second_max": 4.0, "second_name": "retest_age_bars",
        "expanded_sweep_min": 0.35, "expanded_second_max": 6.0,
    },
    "B_FVG_RETEST_SEQUENCE": {
        "sweep_min": 0.45, "second_max": 0.22, "second_name": "bos_distance_atr",
        "expanded_sweep_min": 0.40, "expanded_second_max": 0.28,
    },
    # C is already the highest-count / weakest-quality family in V14 on XAUUSD,
    # so V15 only relaxes its sweep gate slightly and keeps BOS tolerance unchanged.
    "C_DISPLACEMENT_MID_RETEST": {
        "sweep_min": 0.45, "second_max": 0.40, "second_name": "bos_distance_atr",
        "expanded_sweep_min": 0.40, "expanded_second_max": 0.40,
    },
}

# Compensation required for an expanded candidate. Existing causal SMC/HTF
# features are used; their calculation is not changed.
EXP_DISP_BODY_ATR = 0.75
EXP_DISP_CLOSE_LOCATION = 0.75
EXP_DISP_BODY_RATIO = 0.62
EXP_MIN_HTF_ALIGNMENT = 2

FAMILIES = (
    "A_OB_RETEST_SEQUENCE",
    "B_FVG_RETEST_SEQUENCE",
    "C_DISPLACEMENT_MID_RETEST",
)

# No V8/V9/V10/V11/V12/V13 comparison is executed or written by V15.

def _resolve_filter_cfg(filter_profile: str, symbol: str, family: str) -> dict:
    """Resolve the single V15 expanded candidate-generation thresholds."""
    if filter_profile != V15_PROFILE:
        raise KeyError(filter_profile)
    try:
        return V15_FILTERS[family]
    except KeyError as exc:
        raise ValueError(f"Unknown family: {family}") from exc


# ---------------------------------------------------------------------------
# V15 diagnostic quality / causal confluence layer (no filtering, no ML)
# ---------------------------------------------------------------------------
QUALITY_SCORE_VERSION = "V15_Q2"
QUALITY_THRESHOLDS = {"htf_alignment_count":2,"displacement_body_atr":0.50,"displacement_body_ratio":0.40,"displacement_close_location":0.85}
CONFLUENCE_WINDOW_MINUTES=15
FAMILY_LABELS={"A_OB_RETEST_SEQUENCE":"A","B_FVG_RETEST_SEQUENCE":"B","C_DISPLACEMENT_MID_RETEST":"C"}


def _add_quality_score_frame(df: pd.DataFrame)->pd.DataFrame:
    x=df.copy(); htf=pd.to_numeric(x.get("htf_alignment_count"),errors="coerce"); body=pd.to_numeric(x.get("displacement_body_atr"),errors="coerce"); br=pd.to_numeric(x.get("displacement_body_ratio"),errors="coerce"); cl=pd.to_numeric(x.get("displacement_close_location"),errors="coerce")
    x["quality_score"]=((htf>=2).astype(int)+(body>=0.50).astype(int)+(br>=0.40).astype(int)+(cl>=0.85).astype(int)).astype(int)
    x["quality_score_version"]=QUALITY_SCORE_VERSION; x["quality_band"]=pd.cut(x["quality_score"],bins=[-1,1,2,3,4],labels=["LOW","MEDIUM","HIGH","VERY_HIGH"]).astype(str); return x


def _candidate_level_view(df: pd.DataFrame)->pd.DataFrame:
    if df.empty:return df.copy()
    cols=["symbol","filter_profile","family","setup_id","signal_time","entry_time","side","quality_score","quality_band"]+[c for c in FEATURE_NUMERIC if c in df.columns]
    return df[[c for c in cols if c in df.columns]].drop_duplicates(["symbol","filter_profile","family","setup_id"],keep="first").copy()


def _compute_causal_confluence(candidates: pd.DataFrame, window_minutes:int=15)->pd.DataFrame:
    if candidates.empty:return candidates.copy()
    x=candidates.copy(); x["signal_time"]=pd.to_datetime(x["signal_time"],utc=True,errors="coerce"); x=x.sort_values(["symbol","side","signal_time","family"]).reset_index(drop=True)
    x["confluence_count"]=1; x["confluence_families"]=x["family"].map(FAMILY_LABELS).fillna(x["family"]); x["confluence_window_min"]=window_minutes; window=pd.Timedelta(minutes=window_minutes)
    for _,idxs in x.groupby(["symbol","side"],sort=False).groups.items():
        idxs=list(idxs); left=0; counts={}; fam=[x.at[j,"family"] for j in idxs]; times=[x.at[j,"signal_time"] for j in idxs]
        for right,idx in enumerate(idxs):
            while left<=right and times[right]-times[left]>window:
                old=fam[left]; counts[old]=counts.get(old,0)-1
                if counts[old]<=0: counts.pop(old,None)
                left+=1
            current=set(counts); current.add(fam[right]); ordered=sorted(current,key=lambda f:(FAMILY_LABELS.get(f,f),f)); x.at[idx,"confluence_count"]=len(ordered); x.at[idx,"confluence_families"]="+".join(FAMILY_LABELS.get(f,f) for f in ordered); counts[fam[right]]=counts.get(fam[right],0)+1
    return x


def _attach_from_candidate_labels(trades:pd.DataFrame,candidate_labels:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    if candidate_labels.empty:return trades.copy(),candidate_labels.copy()
    c=_compute_causal_confluence(_candidate_level_view(_add_quality_score_frame(candidate_labels))); keys=["symbol","filter_profile","family","setup_id"]; enrich=keys+["quality_score","quality_band","confluence_count","confluence_families","confluence_window_min"]; lookup=c[enrich].drop_duplicates(keys)
    x=trades.copy().drop(columns=[col for col in enrich if col not in keys and col in trades.columns],errors="ignore"); x=x.merge(lookup,on=keys,how="left",validate="many_to_one"); x["confluence_confirmed"]=x["confluence_count"].ge(2); x["triple_confluence"]=x["confluence_count"].eq(3); return x,c

@dataclass
class Candidate:
    symbol: str
    family: str
    signal_idx: int
    signal_time: pd.Timestamp
    entry_idx: int
    entry_time: pd.Timestamp
    side: str
    anchor_low: float
    anchor_high: float
    trigger_price: float
    setup_id: str = ""
    features: dict = field(default_factory=dict)



# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unified local tick-data layer (identical across all strategy files)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
TICK_CACHE_DIR = DATA_DIR / "tick_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TICK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    pa = None
    pq = None
    PARQUET_AVAILABLE = False

CANONICAL_COLUMNS = ("timestamp", "bid", "ask", "last", "volume", "flags")
CSV_CHUNK_ROWS = 500_000
MT5_INITIAL_CHUNK = timedelta(hours=6)
MT5_MIN_CHUNK = timedelta(minutes=5)
MT5_TICK_THRESHOLD = 180_000


def safe_symbol(symbol: str) -> str:
    return (
        symbol.upper().strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def symbol_csv_path(symbol: str) -> Path:
    # Legacy compatibility path. Canonical storage is under data/tick_cache/.
    return DATA_DIR / f"{safe_symbol(symbol)}.csv"


def symbol_meta_path(symbol: str) -> Path:
    return TICK_CACHE_DIR / f"{safe_symbol(symbol)}.meta.json"


def canonical_parquet_path(symbol: str) -> Path:
    return TICK_CACHE_DIR / f"{safe_symbol(symbol)}.parquet"


def canonical_csv_path(symbol: str) -> Path:
    return TICK_CACHE_DIR / f"{safe_symbol(symbol)}.csv"


def _meta_candidates(symbol: str) -> list[Path]:
    s = safe_symbol(symbol)
    return [
        TICK_CACHE_DIR / f"{s}.meta.json",
        DATA_DIR / f"{s}.meta.json",
    ]


def _read_metadata(symbol: str) -> dict:
    for p in _meta_candidates(symbol):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _write_metadata(symbol: str, metadata: dict) -> None:
    symbol_meta_path(symbol).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _fallback_point(symbol: str) -> float:
    s = safe_symbol(symbol)
    if "XAU" in s or "GOLD" in s:
        return 0.01
    if "BTC" in s:
        return 0.01
    return 0.00001


def _parse_time_msc(series: pd.Series) -> np.ndarray:
    raw = series
    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        sample = numeric.dropna()
        if sample.empty:
            raise ValueError("Timestamp column is empty.")
        unit = "ms" if abs(float(sample.iloc[0])) >= 1e11 else "s"
        ts = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError("Could not parse timestamps in symbol data.")
    # Explicit ms unit: newer pandas may parse text timestamps at s/us resolution,
    # so dividing the raw int64 by 1e6 would give wrong times.
    ts = pd.Series(ts).dt.as_unit("ms")
    return np.asarray(ts.array.asi8, dtype=np.int64)  # same length as input; NaT -> int64 min as before


def _normalize_tick_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "time_msc" in out.columns:
        out["timestamp"] = pd.to_numeric(out["time_msc"], errors="coerce")
    elif "timestamp" in out.columns:
        # Keep numeric timestamps numeric; parse string timestamps as UTC.
        if not pd.api.types.is_numeric_dtype(out["timestamp"]):
            out["timestamp"] = _parse_time_msc(out["timestamp"])
        else:
            out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    elif "time" in out.columns:
        out["timestamp"] = _parse_time_msc(out["time"])
    else:
        raise ValueError("Tick data has no timestamp/time_msc/time column.")

    for col in ("bid", "ask", "last"):
        if col not in out.columns:
            raise ValueError(
                f"Tick data is missing required column '{col}'. "
                "Expected MT5 tick columns: timestamp,bid,ask,last,volume,flags"
            )
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("volume", "flags"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out[list(CANONICAL_COLUMNS)].dropna(subset=["timestamp"])
    out["timestamp"] = out["timestamp"].astype(np.int64)
    out = out.sort_values("timestamp", kind="mergesort")
    return out


def _data_candidates(symbol: str) -> list[Path]:
    s = safe_symbol(symbol)
    return [
        canonical_parquet_path(s),
        canonical_csv_path(s),
        DATA_DIR / f"{s}.parquet",
        DATA_DIR / f"{s}.csv",
    ]


def data_path(symbol: str) -> Optional[Path]:
    for p in _data_candidates(symbol):
        if p.exists() and p.is_file():
            return p
    return None


def saved_symbols() -> list[str]:
    found: set[str] = set()
    for folder in (DATA_DIR, TICK_CACHE_DIR):
        for ext in ("*.csv", "*.parquet"):
            for p in folder.glob(ext):
                if p.is_file():
                    found.add(p.stem.upper())
    return sorted(found)


def _scan_file_bounds(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".parquet":
        if not PARQUET_AVAILABLE:
            raise RuntimeError(
                f"Parquet file found but pyarrow is not installed: {path}. "
                "Install it with: pip install pyarrow"
            )
        df = pd.read_parquet(path, columns=["timestamp"])
        times = _parse_time_msc(df["timestamp"])
        good = times[np.isfinite(times)]
        if len(good) == 0:
            raise ValueError(f"No valid timestamps in {path.name}.")
        return int(good.min()), int(good.max())

    first_ms: Optional[int] = None
    last_ms: Optional[int] = None
    for chunk in pd.read_csv(path, usecols=lambda c: c in {"timestamp", "time_msc", "time"}, chunksize=CSV_CHUNK_ROWS):
        col = "timestamp" if "timestamp" in chunk.columns else ("time_msc" if "time_msc" in chunk.columns else "time")
        values = _parse_time_msc(chunk[col])
        values = values[np.isfinite(values)]
        if len(values):
            first_ms = int(values.min()) if first_ms is None else min(first_ms, int(values.min()))
            last_ms = int(values.max()) if last_ms is None else max(last_ms, int(values.max()))
    if first_ms is None or last_ms is None:
        raise ValueError(f"No valid timestamps in {path.name}.")
    return first_ms, last_ms


def _file_covers_period(symbol: str, path: Path, start_dt: datetime, end_dt: datetime) -> bool:
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    meta = _read_metadata(symbol)
    try:
        ms = int(pd.Timestamp(meta["period_start_utc"]).timestamp() * 1000)
        me = int(pd.Timestamp(meta["period_end_utc"]).timestamp() * 1000)
        if ms <= start_ms and me >= end_ms:
            return True
    except Exception:
        pass

    try:
        lo, hi = _scan_file_bounds(path)
        if lo <= start_ms and hi >= (end_ms - 5 * 60 * 1000):
            meta = {
                **meta,
                "symbol": safe_symbol(symbol),
                "period_start_utc": pd.to_datetime(lo, unit="ms", utc=True).isoformat(),
                "period_end_utc": pd.to_datetime(hi, unit="ms", utc=True).isoformat(),
                "source_path": str(path),
            }
            _write_metadata(symbol, meta)
            return True
    except Exception:
        return False
    return False


def _convert_csv_to_parquet(source: Path, destination: Path) -> Path:
    if not PARQUET_AVAILABLE:
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    temp.unlink(missing_ok=True)
    writer = None
    try:
        for chunk in pd.read_csv(source, chunksize=CSV_CHUNK_ROWS):
            out = _normalize_tick_frame(chunk)
            if out.empty:
                continue
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            raise RuntimeError(f"No usable tick rows found in {source}.")
        writer.close()
        writer = None
        temp.replace(destination)
        return destination
    finally:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)


def _resolve_mt5_symbol(requested: str):
    actual = requested.upper().strip()
    info = mt5.symbol_info(actual)
    if info is not None:
        return actual, info
    symbols = mt5.symbols_get() or []
    norm_req = re.sub(r"[^A-Z0-9]", "", requested.upper())
    candidates = [s.name for s in symbols]
    exact = [n for n in candidates if re.sub(r"[^A-Z0-9]", "", n.upper()) == norm_req]
    prefix = [n for n in candidates if re.sub(r"[^A-Z0-9]", "", n.upper()).startswith(norm_req)]
    contains = [n for n in candidates if norm_req and norm_req in re.sub(r"[^A-Z0-9]", "", n.upper())]
    chosen = exact or prefix or contains
    if not chosen:
        raise RuntimeError(f"Symbol '{requested}' was not found in MT5 terminal.")
    actual = chosen[0]
    info = mt5.symbol_info(actual)
    if info is None:
        raise RuntimeError(f"Could not read MT5 symbol info for {actual}.")
    return actual, info


def _download_symbol(symbol_requested: str, start_dt: datetime, end_dt: datetime, force: bool = False) -> dict:
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 is required only when the requested symbol is missing "
            "or its local data does not cover the unified backtest period."
        )
    requested = safe_symbol(symbol_requested)
    existing = data_path(requested)
    if existing is not None and not force and _file_covers_period(requested, existing, start_dt, end_dt):
        if existing.suffix.lower() == ".csv" and PARQUET_AVAILABLE:
            try:
                existing = _convert_csv_to_parquet(existing, canonical_parquet_path(requested))
            except Exception as exc:
                print(f"[DATA] CSV->Parquet conversion skipped: {exc}")
        print(f"[DATA HIT] {existing}")
        return {"path": existing, "metadata": _read_metadata(requested)}

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    target = canonical_parquet_path(requested) if PARQUET_AVAILABLE else canonical_csv_path(requested)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    temp.unlink(missing_ok=True)
    writer = None
    total = 0
    try:
        actual, info = _resolve_mt5_symbol(requested)
        if not info.visible and not mt5.symbol_select(actual, True):
            raise RuntimeError(f"Could not enable MT5 symbol {actual}: {mt5.last_error()}")

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        current = start_dt
        span = MT5_INITIAL_CHUNK
        while current <= end_dt:
            chunk_end = min(current + span, end_dt)
            while True:
                arr = mt5.copy_ticks_range(actual, current, chunk_end, mt5.COPY_TICKS_ALL)
                if arr is None:
                    raise RuntimeError(f"copy_ticks_range failed for {actual}: {mt5.last_error()}")
                if len(arr) >= MT5_TICK_THRESHOLD and span > MT5_MIN_CHUNK:
                    span = max(MT5_MIN_CHUNK, span / 2)
                    chunk_end = min(current + span, end_dt)
                    continue
                break

            if len(arr):
                frame = _normalize_tick_frame(pd.DataFrame(arr))
                frame = frame[(frame["timestamp"] >= start_ms) & (frame["timestamp"] <= end_ms)]
                if not frame.empty:
                    if PARQUET_AVAILABLE:
                        table = pa.Table.from_pandas(frame, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                        writer.write_table(table)
                    else:
                        frame.to_csv(
                            temp,
                            mode="a",
                            header=not temp.exists(),
                            index=False,
                        )
                    total += len(frame)

            if chunk_end >= end_dt:
                break
            current = chunk_end + timedelta(milliseconds=1)
            # Expand again only when recent chunks are comfortably below the threshold.
            if len(arr) < (MT5_TICK_THRESHOLD // 4):
                span = min(MT5_INITIAL_CHUNK, span * 2)

        if writer is not None:
            writer.close()
            writer = None
        if total == 0 or not temp.exists():
            raise RuntimeError(f"No Tick Data returned for {actual} in {start_dt} -> {end_dt}.")
        temp.replace(target)
        meta = {
            "symbol": requested,
            "actual_symbol": actual,
            "period_start_utc": start_dt.isoformat(),
            "period_end_utc": end_dt.isoformat(),
            "tick_count": int(total),
            "source": "MT5 COPY_TICKS_ALL",
            "storage": "parquet" if PARQUET_AVAILABLE else "csv",
            "point": float(getattr(info, "point", 0.0) or 0.0),
            "tick_size": float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0) or 0.0),
            "digits": int(getattr(info, "digits", 0) or 0),
        }
        _write_metadata(requested, meta)
        print(f"[DATA SAVED] {target} | ticks={total:,} | MT5={actual}")
        return {"path": target, "metadata": meta}
    finally:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)
        mt5.shutdown()


def ensure_local_data(symbol: str, start_dt: datetime, end_dt: datetime, force: bool = False) -> Path:
    requested = safe_symbol(symbol)
    existing = data_path(requested)
    if existing is not None and not force and _file_covers_period(requested, existing, start_dt, end_dt):
        if existing.suffix.lower() == ".csv" and PARQUET_AVAILABLE:
            try:
                existing = _convert_csv_to_parquet(existing, canonical_parquet_path(requested))
            except Exception as exc:
                print(f"[DATA] CSV->Parquet conversion skipped: {exc}")
        print(f"[DATA HIT] {existing}")
        return existing
    return Path(_download_symbol(requested, start_dt, end_dt, force=force)["path"])


def _load_tick_frame(path: Path, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        if not PARQUET_AVAILABLE:
            raise RuntimeError(
                f"Cannot read {path.name}: pyarrow is not installed. Install it with: pip install pyarrow"
            )
        df = pd.read_parquet(path, columns=list(CANONICAL_COLUMNS))
    else:
        df = pd.read_csv(path)
    df = _normalize_tick_frame(df)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
    if df.empty:
        raise RuntimeError(f"{path.name} contains no ticks in the requested period.")
    return df.reset_index(drop=True)


def _load_symbol_csv(symbol: str, start_dt: datetime, end_dt: datetime) -> dict:
    path = ensure_local_data(symbol, start_dt, end_dt)
    df = _load_tick_frame(path, start_dt, end_dt)
    metadata = _read_metadata(symbol)
    point = float(metadata.get("point", 0.0) or 0.0)
    if point <= 0:
        point = _fallback_point(symbol)
    tick_size = float(metadata.get("tick_size", point) or point)
    return {
        "time_msc": df["timestamp"].to_numpy(np.int64),
        "bid": df["bid"].to_numpy(np.float64),
        "ask": df["ask"].to_numpy(np.float64),
        "last": df["last"].to_numpy(np.float64),
        "metadata": {
            **metadata,
            "symbol": safe_symbol(symbol),
            "point": point,
            "tick_size": tick_size,
        },
        "data_path": path,
    }


# ---------------------------------------------------------------------------
# M1 OHLC input (alternative to Tick Data). Strategy and execution are unchanged.
# ---------------------------------------------------------------------------
# DATA_MODE: "AUTO" uses M1 OHLC files when they exist for a symbol, otherwise
# Tick Data; "TICK" or "M1" forces one source.
DATA_MODE = "AUTO"
# Each M1 bar is replayed as 4 prices inside its minute (open, then the two
# extremes, then close) and fed to the existing tick engine. The order of the
# extremes inside a bar is unknown, so the engine runs twice: BUY trades see the
# low before the high and SELL trades the high before the low. When one bar
# touches both stop and target, the trade is therefore always counted as a loss.
M1_TICK_OFFSETS_MS = np.array([0, 20_000, 40_000, 59_999], dtype=np.int64)
M1_OHLC_COLUMN_ALIASES = {
    "time": "timestamp", "datetime": "timestamp", "date": "timestamp",
    "tick_volume": "volume", "tickvol": "volume", "real_volume": "real_volume",
}


def _is_ohlc_columns(columns) -> bool:
    cols = {str(c).strip().lower() for c in columns}
    return {"open", "high", "low", "close"} <= cols and "bid" not in cols


def _file_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        if not PARQUET_AVAILABLE:
            return []
        return list(pq.ParquetFile(path).schema.names)
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def m1_ohlc_files(symbol: str) -> list[Path]:
    """M1 OHLC files for a symbol: <SYMBOL>_M1*.csv/.parquet, or <SYMBOL>.csv/.parquet with OHLC columns."""
    s = safe_symbol(symbol)
    found: list[Path] = []
    for folder in (TICK_CACHE_DIR, DATA_DIR):
        for ext in ("csv", "parquet"):
            found.extend(sorted(p for p in folder.glob(f"{s}_M1*.{ext}") if p.is_file()))
    if not found:
        for p in _data_candidates(s):
            try:
                if p.exists() and p.is_file() and _is_ohlc_columns(_file_columns(p)):
                    found.append(p)
                    break
            except Exception:
                continue
    return list(dict.fromkeys(found))


def _infer_point(prices: np.ndarray) -> float:
    """Smallest price step implied by the number of decimals in the data."""
    sample = prices[np.isfinite(prices)][:20_000]
    for k in range(0, 7):
        scaled = sample * (10 ** k)
        if np.all(np.abs(scaled - np.round(scaled)) < 1e-6):
            return float(10 ** -k)
    return 1e-5


def _read_m1_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    df = df.rename(columns={k: v for k, v in M1_OHLC_COLUMN_ALIASES.items() if k in df.columns and v not in df.columns})
    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name}: M1 OHLC file needs a timestamp/time column.")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"{path.name}: M1 OHLC file is missing column '{col}'.")
    out = pd.DataFrame({"time_msc": _parse_time_msc(df["timestamp"])})
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float) if "volume" in df.columns else 1.0
    out["spread"] = pd.to_numeric(df["spread"], errors="coerce").to_numpy(float) if "spread" in df.columns else 0.0
    return out


def load_m1_ohlc(symbol: str, start_dt: datetime, end_dt: datetime) -> dict:
    """Load M1 OHLC bars and replay them as a tick-like stream for the unchanged engine."""
    files = m1_ohlc_files(symbol)
    if not files:
        raise RuntimeError(f"No M1 OHLC file found for {symbol} in {TICK_CACHE_DIR} or {DATA_DIR}.")
    bars = pd.concat([_read_m1_frame(p) for p in files], ignore_index=True)
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
    bars = bars[(bars["time_msc"] >= start_ms) & (bars["time_msc"] <= end_ms)]
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(axis=1)]
    bars = bars.sort_values("time_msc", kind="mergesort").drop_duplicates("time_msc", keep="last").reset_index(drop=True)
    if bars.empty:
        raise RuntimeError(f"M1 OHLC files for {symbol} contain no bars in the requested period.")
    o, c = bars["open"].to_numpy(float), bars["close"].to_numpy(float)
    h = np.maximum.reduce([bars["high"].to_numpy(float), o, c])
    l = np.minimum.reduce([bars["low"].to_numpy(float), o, c])

    metadata = _read_metadata(symbol)
    meta_point = float(metadata.get("point", 0.0) or 0.0)
    spread_point = meta_point if meta_point > 0 else _infer_point(c)
    point = meta_point if meta_point > 0 else _fallback_point(symbol)
    # MT5 M1 bars are Bid prices; half the bar spread gives the same mid price the tick path uses.
    spread_px = np.nan_to_num(bars["spread"].to_numpy(float), nan=0.0).clip(min=0.0) * spread_point
    volume = np.nan_to_num(bars["volume"].to_numpy(float), nan=0.0).clip(min=0.0)

    n = len(bars)
    time_msc = (bars["time_msc"].to_numpy(np.int64)[:, None] + M1_TICK_OFFSETS_MS[None, :]).ravel()
    buy_path = np.column_stack([o, l, h, c]).ravel()    # low first: worst case for BUY
    sell_path = np.column_stack([o, h, l, c]).ravel()   # high first: worst case for SELL
    # Round to the quote precision, as real Bid/Ask ticks are, so the mid price is
    # computed exactly like the tick path (no float noise at the ATR-cap boundary).
    decimals = int(round(-math.log10(spread_point))) if spread_point > 0 else 5
    buy_path, sell_path = np.round(buy_path, decimals), np.round(sell_path, decimals)
    spread_rep = np.repeat(spread_px, 4)
    buy_ask, sell_ask = np.round(buy_path + spread_rep, decimals), np.round(sell_path + spread_rep, decimals)
    weight = np.column_stack([volume, np.zeros(n), np.zeros(n), np.zeros(n)]).ravel()
    print(f"  M1 OHLC source: {len(files)} file(s), {n:,} bars, spread point={spread_point:g}", flush=True)
    return {
        "time_msc": time_msc,
        "bid": buy_path, "ask": buy_ask,
        "sell_bid": sell_path, "sell_ask": sell_ask,
        "last": np.zeros(len(time_msc)),
        "volume_weight": weight,
        "metadata": {**metadata, "symbol": safe_symbol(symbol), "point": point,
                     "tick_size": float(metadata.get("tick_size", point) or point)},
        "data_path": ", ".join(str(p) for p in files),
        "data_source": "M1_OHLC",
    }


def load_tick_cache(symbol: str) -> dict:
    mode = DATA_MODE.upper()
    if mode == "M1" or (mode == "AUTO" and m1_ohlc_files(symbol)):
        return load_m1_ohlc(symbol, PERIOD_START, PERIOD_END)
    t = _load_symbol_csv(symbol, PERIOD_START, PERIOD_END)
    t["data_source"] = "TICK"
    return t


# Tick -> bar construction
# ---------------------------------------------------------------------------
def tick_price_series(t: dict) -> pd.DataFrame:
    idx = pd.to_datetime(t["time_msc"], unit="ms", utc=True)
    bid = t["bid"].astype(float)
    ask = t["ask"].astype(float)
    last = t["last"].astype(float).copy()
    mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, np.nan)
    invalid = (last <= 0) | ~np.isfinite(last)
    last[invalid] = mid[invalid]
    invalid = (last <= 0) | ~np.isfinite(last)
    last[invalid & (bid > 0)] = bid[invalid & (bid > 0)]
    invalid = (last <= 0) | ~np.isfinite(last)
    last[invalid & (ask > 0)] = ask[invalid & (ask > 0)]
    out = pd.DataFrame({"price": last}, index=idx)
    if "volume_weight" in t:  # M1 OHLC replay: bar volume = tick_volume, not replayed price count
        out["weight"] = np.asarray(t["volume_weight"], dtype=float)
    out = out.dropna(subset=["price"])
    return out[out["price"] > 0]


def ticks_to_bars(t: dict, rule: str) -> pd.DataFrame:
    s = tick_price_series(t)
    bars = s["price"].resample(rule, label="left", closed="left").ohlc()
    if "weight" in s.columns:
        bars["volume"] = s["weight"].resample(rule, label="left", closed="left").sum()
    else:
        bars["volume"] = s["price"].resample(rule, label="left", closed="left").count()
    bars = bars.dropna(subset=["open", "high", "low", "close"]).reset_index(names="timestamp")
    return bars


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.replace([np.inf, -np.inf], np.nan)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    signal_line = ema(line, signal)
    return line, signal_line, line - signal_line


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high-low,(high-prev_close).abs(),(low-prev_close).abs()], axis=1).max(axis=1)
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_s = plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    minus_s = minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * plus_s / tr_s.replace(0.0, np.nan)
    minus_di = 100.0 * minus_s / tr_s.replace(0.0, np.nan)
    dx = 100.0 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean(), plus_di, minus_di


def stochastic_k(df: pd.DataFrame, n: int = 14) -> pd.Series:
    lo = df["low"].rolling(n, min_periods=n).min()
    hi = df["high"].rolling(n, min_periods=n).max()
    return 100.0 * (df["close"] - lo) / (hi - lo).replace(0.0, np.nan)


def add_ml_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute compact, causal ML features from bars only; V15 ML features are never entry filters."""
    d=df.copy()
    close=d["close"].astype(float); high=d["high"].astype(float); low=d["low"].astype(float); op=d["open"].astype(float); vol=d["volume"].astype(float)
    d["rsi14"]=rsi(close,14)
    d["macd_line"],d["macd_signal"],d["macd_hist"]=macd(close,12,26,9)
    d["adx14"],d["plus_di14"],d["minus_di14"]=adx(d,14)
    d["stoch_k14"]=stochastic_k(d,14)
    mid=close.rolling(20,min_periods=20).mean(); sd=close.rolling(20,min_periods=20).std(ddof=0)
    upper=mid+2*sd; lower=mid-2*sd
    d["bb_percent_b"]=(close-lower)/(upper-lower).replace(0.0,np.nan)
    d["bb_width_atr"]=(upper-lower)/d["atr"].replace(0.0,np.nan)
    for n in (1,3,5,15,30): d[f"ret_{n}"]=close.pct_change(n)
    d["ret_5_atr"]=(close-close.shift(5))/d["atr"].replace(0.0,np.nan)
    d["ret_15_atr"]=(close-close.shift(15))/d["atr"].replace(0.0,np.nan)
    d["range_atr"]=(high-low)/d["atr"].replace(0.0,np.nan)
    d["body_atr"]=(close-op).abs()/d["atr"].replace(0.0,np.nan)
    cr=pd.concat([op,close],axis=1); hi_oc=cr.max(axis=1); lo_oc=cr.min(axis=1)
    d["upper_wick_ratio"]=(high-hi_oc)/(high-low).replace(0.0,np.nan)
    d["lower_wick_ratio"]=(lo_oc-low)/(high-low).replace(0.0,np.nan)
    for ema_col in ("ema20","ema50","ema200"):
        d[f"dist_{ema_col}_atr"]=(close-d[ema_col])/d["atr"].replace(0.0,np.nan)
    d["ema20_50_gap_atr"]=(d["ema20"]-d["ema50"])/d["atr"].replace(0.0,np.nan)
    d["ema50_200_gap_atr"]=(d["ema50"]-d["ema200"])/d["atr"].replace(0.0,np.nan)
    d["ema20_slope_5_atr"]=(d["ema20"]-d["ema20"].shift(5))/d["atr"].replace(0.0,np.nan)
    d["ema50_slope_10_atr"]=(d["ema50"]-d["ema50"].shift(10))/d["atr"].replace(0.0,np.nan)
    typical=(high+low+close)/3.0; pv=typical*vol
    rv=vol.rolling(60,min_periods=20).sum()
    d["dist_vwap60_atr"]=(close-pv.rolling(60,min_periods=20).sum()/rv.replace(0.0,np.nan))/d["atr"].replace(0.0,np.nan)
    vm=vol.rolling(50,min_periods=20).mean(); vs=vol.rolling(50,min_periods=20).std(ddof=0)
    d["volume_z50"]=(vol-vm)/vs.replace(0.0,np.nan)
    am=d["atr"].rolling(50,min_periods=20).mean()
    d["atr_regime_ratio"]=d["atr"]/am.replace(0.0,np.nan)
    d["range_10_atr"]=(high.rolling(10,min_periods=10).max()-low.rolling(10,min_periods=10).min())/d["atr"].replace(0.0,np.nan)
    d["range_30_atr"]=(high.rolling(30,min_periods=30).max()-low.rolling(30,min_periods=30).min())/d["atr"].replace(0.0,np.nan)

    # V15 additions: compact, causal market-state features for future per-symbol ML.
    # None of these is an entry gate.
    d["rsi14_slope_3"] = d["rsi14"] - d["rsi14"].shift(3)
    d["macd_hist_slope_3"] = d["macd_hist"] - d["macd_hist"].shift(3)
    d["adx_slope_3"] = d["adx14"] - d["adx14"].shift(3)
    price_path = close.diff().abs()
    directional_20 = (close - close.shift(20)).abs()
    path_20 = price_path.rolling(20, min_periods=20).sum()
    d["trend_efficiency_20"] = directional_20 / path_20.replace(0.0, np.nan)
    rv_30 = d["ret_1"].rolling(30, min_periods=20).std(ddof=0)
    rv_120 = d["ret_1"].rolling(120, min_periods=60).std(ddof=0)
    d["realized_vol_ratio_30_120"] = rv_30 / rv_120.replace(0.0, np.nan)
    d["atr_vs_median_100"] = d["atr"] / d["atr"].rolling(100, min_periods=40).median().replace(0.0, np.nan)
    d["range_expansion_10_30"] = d["range_10_atr"] / d["range_30_atr"].replace(0.0, np.nan)
    candle_loc = (close-low)/(high-low).replace(0.0,np.nan)
    d["close_location_3"] = candle_loc.rolling(3, min_periods=3).mean()
    d["distance_high_60_atr"] = (high.rolling(60, min_periods=30).max() - close) / d["atr"].replace(0.0,np.nan)
    d["distance_low_60_atr"] = (close - low.rolling(60, min_periods=30).min()) / d["atr"].replace(0.0,np.nan)
    h=d["timestamp"].dt.hour.astype(float); dow=d["timestamp"].dt.dayofweek.astype(float)
    d["utc_hour_sin"]=np.sin(2*np.pi*h/24); d["utc_hour_cos"]=np.cos(2*np.pi*h/24)
    d["day_of_week_sin"]=np.sin(2*np.pi*dow/7); d["day_of_week_cos"]=np.cos(2*np.pi*dow/7)
    return d


def confirmed_swings(df: pd.DataFrame, w: int = SWING) -> tuple[pd.Series, pd.Series]:
    """Mark pivots on their occurrence bar; callers must delay usage by w bars."""
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    sh = np.zeros(len(df), dtype=bool)
    sl = np.zeros(len(df), dtype=bool)
    for i in range(w, len(df) - w):
        sh[i] = hi[i] >= np.max(hi[i-w:i+w+1])
        sl[i] = lo[i] <= np.min(lo[i-w:i+w+1])
    return pd.Series(sh, index=df.index), pd.Series(sl, index=df.index)


def fvg(df: pd.DataFrame, i: int, side: str) -> bool:
    if i < 2:
        return False
    if side == "BUY":
        return float(df.iloc[i]["low"]) > float(df.iloc[i-2]["high"])
    return float(df.iloc[i]["high"]) < float(df.iloc[i-2]["low"])


def displacement(df: pd.DataFrame, i: int, side: str) -> bool:
    """Strict but responsive displacement confirmation on the decision TF."""
    if i < 5:
        return False
    row = df.iloc[i]
    rng = max(float(row.high - row.low), 1e-12)
    body = abs(float(row.close - row.open))
    atr_v = float(row.atr) if np.isfinite(row.atr) else np.nan
    directional = (row.close > row.open) if side == "BUY" else (row.close < row.open)
    if not directional or not np.isfinite(atr_v) or atr_v <= 0:
        return False
    close_loc = ((float(row.close) - float(row.low)) / rng) if side == "BUY" else ((float(row.high) - float(row.close)) / rng)
    return bool(
        body >= DISP_BODY_ATR * atr_v
        and rng >= DISP_RANGE_ATR * atr_v
        and close_loc >= DISP_CLOSE_LOCATION
        and body / rng >= 0.55
    )


def last_opposite_between(df: pd.DataFrame, start_i: int, end_i: int, side: str) -> Optional[int]:
    if end_i < start_i:
        return None
    lo = max(0, start_i)
    hi = min(len(df) - 1, end_i)
    for j in range(hi, lo - 1, -1):
        row = df.iloc[j]
        if side == "BUY" and float(row.close) < float(row.open):
            return j
        if side == "SELL" and float(row.close) > float(row.open):
            return j
    return None


def _zone_intersects(row: pd.Series, low: float, high: float) -> bool:
    return bool(float(row.low) <= high and float(row.high) >= low)


def _confirmation_after_retest(row: pd.Series, side: str, zone_mid: float) -> bool:
    rng = max(float(row.high - row.low), 1e-12)
    body = abs(float(row.close - row.open))
    directional = float(row.close) > float(row.open) if side == "BUY" else float(row.close) < float(row.open)
    if not directional or body / rng < CONFIRM_BODY_RATIO:
        return False
    return float(row.close) >= zone_mid if side == "BUY" else float(row.close) <= zone_mid


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["swing_high_raw"], d["swing_low_raw"] = confirmed_swings(d, SWING)
    d["atr"] = atr(d, ATR_PERIOD)
    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    return add_ml_indicators(d)


def m5_direction(df5: pd.DataFrame) -> pd.Series:
    e20 = ema(df5["close"], M5_FAST)
    e50 = ema(df5["close"], M5_SLOW)
    out = pd.Series(0, index=df5.index, dtype=int)
    out[e20 > e50] = 1
    out[e20 < e50] = -1
    return out


def m5_trend_strength(df5: pd.DataFrame) -> pd.Series:
    e20 = ema(df5["close"], M5_FAST)
    e50 = ema(df5["close"], M5_SLOW)
    a = atr(df5, ATR_PERIOD)
    return (e20 - e50).abs() / a.replace(0.0, np.nan)


def _delayed_pivot_levels(df: pd.DataFrame, w: int) -> tuple[pd.Series, pd.Series]:
    sh_raw, sl_raw = confirmed_swings(df, w)
    hi = pd.Series(np.where(sh_raw, df["high"].to_numpy(float), np.nan), index=df.index)
    lo = pd.Series(np.where(sl_raw, df["low"].to_numpy(float), np.nan), index=df.index)
    return hi.shift(w), lo.shift(w)


def m15_context(df15: pd.DataFrame) -> pd.DataFrame:
    d = df15.copy()
    d["ema20"] = ema(d["close"], M15_FAST)
    d["ema50"] = ema(d["close"], M15_SLOW)
    d["atr"] = atr(d, ATR_PERIOD)
    sh_known, sl_known = _delayed_pivot_levels(d, SWING)
    d["last_swing_high"] = sh_known.ffill()
    d["last_swing_low"] = sl_known.ffill()
    d["m15_dir"] = np.where(d["ema20"] > d["ema50"], 1, np.where(d["ema20"] < d["ema50"], -1, 0))
    d["m15_trend_strength"] = (d["ema20"] - d["ema50"]).abs() / d["atr"].replace(0.0, np.nan)
    return d


def asof_feature(base: pd.DataFrame, higher: pd.DataFrame, higher_rule: str, decision_offset: str = "1min") -> pd.DataFrame:
    x = base[["timestamp"]].copy().sort_values("timestamp")
    x["decision_time"] = x["timestamp"] + pd.Timedelta(decision_offset)
    h = higher.sort_values("timestamp").copy()
    h["available_time"] = h["timestamp"] + pd.Timedelta(higher_rule)
    cols = [c for c in h.columns if c not in ("timestamp", "available_time")]
    z = pd.merge_asof(
        x.sort_values("decision_time"),
        h[["available_time"] + cols].sort_values("available_time"),
        left_on="decision_time", right_on="available_time", direction="backward"
    )
    z.index = base.index
    return z


# ---------------------------------------------------------------------------
# V15 context feature layer (FEATURES ONLY — never used as entry gates)
# ---------------------------------------------------------------------------
# Computed from M1 bars and the clock only; no H1/H4/D1 bars are used.
# Adds what the SMC setup features cannot see: trading session, today's range and
# the Asian range so far, where price sits inside them, the stop size, and the room
# to the first intraday liquidity (today's / Asian high-low) in units of the risk.
# Everything is causal: values include the signal bar but nothing after it.
ASIA_END_HOUR_UTC = 7
LONDON_TZ = "Europe/London"
NEW_YORK_TZ = "America/New_York"
SESSION_OPEN_MIN = 8 * 60           # 08:00 local time in London and New York
LONDON_CLOSE_MIN = 16 * 60 + 30
NEW_YORK_CLOSE_MIN = 17 * 60


def add_context_features(b1: pd.DataFrame) -> pd.DataFrame:
    """Attach raw ctx_* columns to M1 bars; side-aware features are derived per candidate."""
    d = b1.copy()
    ts = d["timestamp"]

    # Today so far (UTC day, includes the signal bar) and the Asian range.
    day = ts.dt.floor("D")
    g = d.groupby(day, sort=False)
    d["ctx_day_open"] = g["open"].transform("first")
    d["ctx_day_high"] = g["high"].cummax()
    d["ctx_day_low"] = g["low"].cummin()
    asia = ts.dt.hour < ASIA_END_HOUR_UTC
    d["ctx_asia_high"] = d["high"].where(asia).groupby(day).cummax().groupby(day).ffill()
    d["ctx_asia_low"] = d["low"].where(asia).groupby(day).cummin().groupby(day).ffill()

    # Trading session from local London/New York clocks (DST-aware).
    lon = ts.dt.tz_convert(LONDON_TZ); ny = ts.dt.tz_convert(NEW_YORK_TZ)
    lon_min = (lon.dt.hour * 60 + lon.dt.minute).to_numpy()
    ny_min = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    in_lon = (lon_min >= SESSION_OPEN_MIN) & (lon_min < LONDON_CLOSE_MIN)
    in_ny = (ny_min >= SESSION_OPEN_MIN) & (ny_min < NEW_YORK_CLOSE_MIN)
    d["ctx_session"] = np.select([in_lon & in_ny, in_lon, in_ny], ["LONDON_NY_OVERLAP", "LONDON", "NEW_YORK"], "ASIA")
    d["ctx_session_minutes"] = np.select(
        [in_ny, in_lon], [ny_min - SESSION_OPEN_MIN, lon_min - SESSION_OPEN_MIN], (ny_min - NEW_YORK_CLOSE_MIN) % 1440
    ).astype(float)

    vol = d["volume"].astype(float)
    d["ctx_activity_ratio"] = vol.rolling(60, min_periods=30).mean() / vol.rolling(1440, min_periods=300).mean().replace(0.0, np.nan)
    return d


def _ratio(num: float, den: float) -> float:
    return _feature_value(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan


def _discount_aligned(close: float, lo: float, hi: float, side: str) -> float:
    """0 = at the extreme against the trade (best SMC location), 1 = at the target-side extreme."""
    pos = _ratio(close - lo, hi - lo)
    if not np.isfinite(pos):
        return np.nan
    return float(np.clip(pos if side == "BUY" else 1.0 - pos, -1.0, 2.0))


def _context_candidate_features(row: pd.Series, side: str, atr_value: float, invalidation: float) -> dict:
    """Side-aware context features for one candidate, from the signal bar's ctx_* columns."""
    g = lambda k: float(row[k]) if k in row.index and pd.notna(row[k]) else np.nan
    sgn = 1.0 if side == "BUY" else -1.0
    close = float(row.close); a = float(atr_value) if np.isfinite(atr_value) and atr_value > 0 else np.nan
    risk = abs(close - invalidation) if np.isfinite(invalidation) else np.nan
    risk = risk if np.isfinite(risk) and risk > 0 else np.nan
    day_hi, day_lo = g("ctx_day_high"), g("ctx_day_low")
    asia_hi, asia_lo = g("ctx_asia_high"), g("ctx_asia_low")

    # First intraday liquidity in the trade direction (today's / Asian extreme), in R.
    targets = (day_hi, asia_hi) if side == "BUY" else (day_lo, asia_lo)
    room = [sgn * (x - close) for x in targets if np.isfinite(x)]
    room = [x for x in room if x > 0]
    return {
        "session": str(row["ctx_session"]) if "ctx_session" in row.index else "UNKNOWN",
        "session_minutes": g("ctx_session_minutes"),
        "risk_atr": _ratio(risk, a),
        "day_range_atr": _ratio(day_hi - day_lo, a),
        "day_discount_aligned": _discount_aligned(close, day_lo, day_hi, side),
        "day_open_move_atr": _ratio(sgn * (close - g("ctx_day_open")), a),
        "asia_range_atr": _ratio(asia_hi - asia_lo, a),
        "asia_discount_aligned": _discount_aligned(close, asia_lo, asia_hi, side),
        "first_liquidity_target_r": _ratio(min(room), risk) if room else np.nan,
        "activity_ratio_60_1440": _feature_value(g("ctx_activity_ratio")),
    }


CONTEXT_FEATURES = (
    "session_minutes", "risk_atr", "day_range_atr", "day_discount_aligned", "day_open_move_atr",
    "asia_range_atr", "asia_discount_aligned", "first_liquidity_target_r", "activity_ratio_60_1440",
)


FEATURE_NUMERIC = (
    # Existing SMC features — preserved.
    "sweep_penetration_atr","sweep_age_bars","displacement_range_atr","displacement_body_atr",
    "displacement_body_ratio","displacement_close_location","bos_distance_atr","fvg_size_atr",
    "ob_size_atr","zone_size_atr","retest_age_bars","retest_penetration_pct",
    "distance_sweep_to_zone_atr","distance_entry_to_sweep_atr","distance_entry_to_bos_atr",
    "m15_opposing_distance_atr","m5_dir","m15_dir","htf_alignment_count",
    # New causal SMC-event features — FEATURES ONLY; never used as strategy gates.
    "sweep_reclaim_strength_atr","sweep_wick_ratio","sweep_to_bos_bars","sweep_to_bos_move_atr",
    "bos_break_strength_atr","displacement_expansion_ratio","pre_displacement_compression",
    "sequence_age_bars","nearest_liquidity_distance_atr","opposing_liquidity_distance_atr",
    "liquidity_asymmetry","m5_trend_strength","m15_trend_strength","mtf_trend_consistency",
    # Additional causal ML features.
    "rsi14","macd_line","macd_signal","macd_hist","adx14","plus_di14","minus_di14","stoch_k14",
    "bb_percent_b","bb_width_atr","ret_1","ret_3","ret_5","ret_15","ret_30","ret_5_atr","ret_15_atr",
    "range_atr","body_atr","upper_wick_ratio","lower_wick_ratio","dist_ema20_atr","dist_ema50_atr",
    "dist_ema200_atr","ema20_50_gap_atr","ema50_200_gap_atr","ema20_slope_5_atr","ema50_slope_10_atr",
    "dist_vwap60_atr","volume_z50","atr_regime_ratio","range_10_atr","range_30_atr",
    "rsi14_slope_3","macd_hist_slope_3","adx_slope_3","trend_efficiency_20",
    "realized_vol_ratio_30_120","atr_vs_median_100","range_expansion_10_30","close_location_3",
    "distance_high_60_atr","distance_low_60_atr",
    "utc_hour_sin","utc_hour_cos","day_of_week_sin","day_of_week_cos","side_adjusted_ret_5",
    "side_adjusted_ret_15","side_adjusted_macd_hist","side_adjusted_di_gap","utc_hour","day_of_week",
    # Family-specific causal geometry; NaN for non-applicable families.
    "ob_body_quality","ob_to_bos_distance_atr","fvg_to_displacement_ratio","fvg_fill_depth",
    "midpoint_retest_depth",
    # Context layer (M1 + clock only): session, today's/Asian range, stop size, room to target.
    *CONTEXT_FEATURES,
)

FEATURE_CATEGORICAL = ("side","family","utc_block","candidate_tier","session")


def _utc_block(hour: int) -> str:
    # Pure UTC buckets; deliberately avoids broker/session assumptions.
    if 0 <= hour <= 5:
        return "UTC_00_05"
    if 6 <= hour <= 11:
        return "UTC_06_11"
    if 12 <= hour <= 17:
        return "UTC_12_17"
    return "UTC_18_23"


def _feature_value(x: float) -> float:
    return float(x) if np.isfinite(x) else np.nan


def _liquidity_distances_at_entry(
    close: float,
    side: str,
    atr_value: float,
    known_highs: deque,
    known_lows: deque,
    m15_last_hi: float,
    m15_last_lo: float,
) -> tuple[float, float, float]:
    """Causal liquidity distances using confirmed levels known before the signal."""
    a=float(atr_value)
    if not np.isfinite(a) or a<=0 or not np.isfinite(close):
        return np.nan,np.nan,np.nan
    highs=[float(x) for x in known_highs if np.isfinite(x)]
    lows=[float(x) for x in known_lows if np.isfinite(x)]
    if np.isfinite(m15_last_hi): highs.append(float(m15_last_hi))
    if np.isfinite(m15_last_lo): lows.append(float(m15_last_lo))
    if side=="BUY":
        favorable=[x-close for x in highs if x>close]
        opposing=[close-x for x in lows if x<close]
    else:
        favorable=[close-x for x in lows if x<close]
        opposing=[x-close for x in highs if x>close]
    fav=min(favorable)/a if favorable else np.nan
    opp=min(opposing)/a if opposing else np.nan
    asym=(opp-fav)/(opp+fav) if np.isfinite(fav) and np.isfinite(opp) and (opp+fav)>0 else np.nan
    return _feature_value(fav),_feature_value(opp),_feature_value(asym)


def _candidate_features(
    d: pd.DataFrame, i: int, side: str, setup: dict, family: str,
    atr_value: float, m5_value: int, m15_value: int, m15_hi: float, m15_lo: float,
    m5_strength: float = np.nan, m15_strength: float = np.nan,
    known_highs: Optional[deque] = None, known_lows: Optional[deque] = None,
) -> dict:
    """Build the causal ML feature vector; does not change strategy decisions."""
    row=d.iloc[i]; a=float(atr_value) if np.isfinite(atr_value) and atr_value>0 else np.nan
    signal_time=pd.Timestamp(row.timestamp); hour=int(signal_time.hour); close=float(row.close)
    fvg_low=float(setup["fvg_low"]) if np.isfinite(setup["fvg_low"]) else np.nan
    fvg_high=float(setup["fvg_high"]) if np.isfinite(setup["fvg_high"]) else np.nan
    ob_low=float(setup["ob_low"]); ob_high=float(setup["ob_high"])
    if family=="B_FVG_RETEST_SEQUENCE" and np.isfinite(fvg_low) and np.isfinite(fvg_high): zlo,zhi=min(fvg_low,fvg_high),max(fvg_low,fvg_high)
    elif family=="A_OB_RETEST_SEQUENCE": zlo,zhi=min(ob_low,ob_high),max(ob_low,ob_high)
    else:
        dl=float(setup["disp_low"]); dh=float(setup["disp_high"]); mid=(dl+dh)/2
        half=max(0.25*a,0.15*max(dh-dl,0.0)) if np.isfinite(a) else 0.0; zlo,zhi=mid-half,mid+half
    zwidth=max(zhi-zlo,1e-12)
    penetration=((zhi-min(float(row.low),zhi))/zwidth) if side=="BUY" else ((min(max(float(row.high),zlo),zhi)-zlo)/zwidth)
    penetration=float(np.clip(penetration,0,1))
    disp_range=max(float(setup["disp_high"]-setup["disp_low"]),0.0); disp_body=abs(float(row.close-row.open)); disp_ratio=disp_body/max(disp_range,1e-12)
    close_loc=((close-float(row.low))/max(float(row.high-row.low),1e-12)) if side=="BUY" else ((float(row.high)-close)/max(float(row.high-row.low),1e-12))
    bos=float(setup["bos_level"]); sweep_extreme=float(setup["sweep_extreme"]); sweep_idx=int(setup["sweep_idx"]); created=int(setup["created"])
    nearest=min(abs(sweep_extreme-zlo),abs(sweep_extreme-zhi))
    if side=="BUY" and m15_value==-1 and np.isfinite(m15_hi): opposing=abs(m15_hi-close)
    elif side=="SELL" and m15_value==1 and np.isfinite(m15_lo): opposing=abs(close-m15_lo)
    else: opposing=np.nan
    direction=1.0 if side=="BUY" else -1.0

    # New causal SMC-event features. Every referenced event is known by i.
    sweep_bar=d.iloc[sweep_idx]
    bos_idx=int(setup.get("created", i))
    bos_row=d.iloc[bos_idx]
    sweep_range=max(float(sweep_bar.high-sweep_bar.low),1e-12)
    bos_a=float(bos_row.atr) if np.isfinite(bos_row.atr) and bos_row.atr>0 else a
    if side=="BUY":
        sweep_wick=float(min(sweep_bar.open,sweep_bar.close)-sweep_bar.low)
        sweep_reclaim=float(sweep_bar.close-setup["sweep"])
        sweep_move_to_bos=float(bos_row.close-sweep_extreme)
        bos_break=max(float(bos_row.close-bos),0.0)
    else:
        sweep_wick=float(sweep_bar.high-max(sweep_bar.open,sweep_bar.close))
        sweep_reclaim=float(setup["sweep"]-sweep_bar.close)
        sweep_move_to_bos=float(sweep_extreme-bos_row.close)
        bos_break=max(float(bos-bos_row.close),0.0)

    # Compare the displacement candle with the ranges immediately BEFORE BOS.
    prior_ranges=(d["high"]-d["low"]).iloc[max(0,bos_idx-20):bos_idx].astype(float)
    med20=float(prior_ranges.median()) if len(prior_ranges.dropna())>=5 else np.nan
    mean5=float(prior_ranges.tail(5).mean()) if len(prior_ranges.tail(5).dropna())>=3 else np.nan
    mean20=float(prior_ranges.mean()) if len(prior_ranges.dropna())>=5 else np.nan
    displacement_expansion=disp_range/med20 if np.isfinite(med20) and med20>0 else np.nan
    pre_compression=mean5/mean20 if np.isfinite(mean5) and np.isfinite(mean20) and mean20>0 else np.nan

    ob_idx=int(setup.get("ob_idx",-1))
    if 0<=ob_idx<len(d):
        ob_row=d.iloc[ob_idx]; ob_rng=max(float(ob_row.high-ob_row.low),1e-12)
        ob_body_quality=abs(float(ob_row.close-ob_row.open))/ob_rng
        ob_to_bos=abs(((ob_low+ob_high)/2.0)-bos)/bos_a if np.isfinite(bos_a) else np.nan
    else: ob_body_quality=np.nan; ob_to_bos=np.nan

    fvg_ratio=(abs(fvg_high-fvg_low)/disp_range) if family=="B_FVG_RETEST_SEQUENCE" and np.isfinite(fvg_low) and np.isfinite(fvg_high) and disp_range>0 else np.nan
    fvg_fill=penetration if family=="B_FVG_RETEST_SEQUENCE" else np.nan

    if family=="C_DISPLACEMENT_MID_RETEST":
        disp_mid=(float(setup["disp_low"])+float(setup["disp_high"]))/2.0
        disp_half=max(float(setup["disp_high"]-setup["disp_low"])/2.0,1e-12)
        midpoint_depth=np.clip(1.0-abs(close-disp_mid)/disp_half,0.0,1.0)
    else: midpoint_depth=np.nan

    known_highs=known_highs if known_highs is not None else deque(maxlen=50)
    known_lows=known_lows if known_lows is not None else deque(maxlen=50)
    favorable_liq,opposing_liq,liquidity_asym=_liquidity_distances_at_entry(
        close,side,a,known_highs,known_lows,m15_hi,m15_lo
    )
    sgn=1 if side=="BUY" else -1
    signed_m5=(m5_strength if m5_value==sgn else (-m5_strength if m5_value==-sgn else 0.0)) if np.isfinite(m5_strength) else 0.0
    signed_m15=(m15_strength if m15_value==sgn else (-m15_strength if m15_value==-sgn else 0.0)) if np.isfinite(m15_strength) else 0.0
    mtf_consistency=float(np.tanh(signed_m5)*np.tanh(signed_m15))

    return {
        "side":side,"family":family,"utc_block":_utc_block(hour),
        "sweep_penetration_atr":_feature_value((float(setup.get("sweep",sweep_extreme))-sweep_extreme)/a) if side=="BUY" and np.isfinite(a) else _feature_value((sweep_extreme-float(setup.get("sweep",sweep_extreme)))/a) if side=="SELL" and np.isfinite(a) else np.nan,
        "sweep_age_bars":float(i-sweep_idx),"displacement_range_atr":_feature_value(disp_range/a) if np.isfinite(a) else np.nan,
        "displacement_body_atr":_feature_value(disp_body/a) if np.isfinite(a) else np.nan,"displacement_body_ratio":_feature_value(disp_ratio),
        "displacement_close_location":_feature_value(close_loc),"bos_distance_atr":_feature_value(abs(close-bos)/a) if np.isfinite(a) else np.nan,
        "fvg_size_atr":_feature_value((fvg_high-fvg_low)/a) if np.isfinite(a) and np.isfinite(fvg_low) and np.isfinite(fvg_high) else np.nan,
        "ob_size_atr":_feature_value((ob_high-ob_low)/a) if np.isfinite(a) else np.nan,"zone_size_atr":_feature_value(zwidth/a) if np.isfinite(a) else np.nan,
        "retest_age_bars":float(i-created),"retest_penetration_pct":penetration*100,"distance_sweep_to_zone_atr":_feature_value(nearest/a) if np.isfinite(a) else np.nan,
        "distance_entry_to_sweep_atr":_feature_value(abs(close-sweep_extreme)/a) if np.isfinite(a) else np.nan,"distance_entry_to_bos_atr":_feature_value(abs(close-bos)/a) if np.isfinite(a) else np.nan,
        "m15_opposing_distance_atr":_feature_value(opposing/a) if np.isfinite(a) and np.isfinite(opposing) else np.nan,
        "m5_dir":int(m5_value),"m15_dir":int(m15_value),"htf_alignment_count":int((m5_value==(1 if side=="BUY" else -1))+(m15_value==(1 if side=="BUY" else -1))),
        "sweep_reclaim_strength_atr":_feature_value(sweep_reclaim/a) if np.isfinite(a) else np.nan,
        "sweep_wick_ratio":_feature_value(sweep_wick/sweep_range),
        "sweep_to_bos_bars":float(max(bos_idx-sweep_idx,0)),
        "sweep_to_bos_move_atr":_feature_value(sweep_move_to_bos/bos_a) if np.isfinite(bos_a) else np.nan,
        "bos_break_strength_atr":_feature_value(bos_break/bos_a) if np.isfinite(bos_a) else np.nan,
        "displacement_expansion_ratio":_feature_value(displacement_expansion),
        "pre_displacement_compression":_feature_value(pre_compression),
        "sequence_age_bars":float(max(i-sweep_idx,0)),
        "nearest_liquidity_distance_atr":_feature_value(favorable_liq),
        "opposing_liquidity_distance_atr":_feature_value(opposing_liq),
        "liquidity_asymmetry":_feature_value(liquidity_asym),
        "m5_trend_strength":_feature_value(m5_strength),
        "m15_trend_strength":_feature_value(m15_strength),
        "mtf_trend_consistency":_feature_value(mtf_consistency),
        "rsi14":_feature_value(float(row.rsi14)),"macd_line":_feature_value(float(row.macd_line)),"macd_signal":_feature_value(float(row.macd_signal)),"macd_hist":_feature_value(float(row.macd_hist)),
        "adx14":_feature_value(float(row.adx14)),"plus_di14":_feature_value(float(row.plus_di14)),"minus_di14":_feature_value(float(row.minus_di14)),"stoch_k14":_feature_value(float(row.stoch_k14)),
        "bb_percent_b":_feature_value(float(row.bb_percent_b)),"bb_width_atr":_feature_value(float(row.bb_width_atr)),
        "ret_1":_feature_value(float(row.ret_1)),"ret_3":_feature_value(float(row.ret_3)),"ret_5":_feature_value(float(row.ret_5)),"ret_15":_feature_value(float(row.ret_15)),"ret_30":_feature_value(float(row.ret_30)),
        "ret_5_atr":_feature_value(float(row.ret_5_atr)),"ret_15_atr":_feature_value(float(row.ret_15_atr)),"range_atr":_feature_value(float(row.range_atr)),"body_atr":_feature_value(float(row.body_atr)),
        "upper_wick_ratio":_feature_value(float(row.upper_wick_ratio)),"lower_wick_ratio":_feature_value(float(row.lower_wick_ratio)),"dist_ema20_atr":_feature_value(float(row.dist_ema20_atr)),
        "dist_ema50_atr":_feature_value(float(row.dist_ema50_atr)),"dist_ema200_atr":_feature_value(float(row.dist_ema200_atr)),"ema20_50_gap_atr":_feature_value(float(row.ema20_50_gap_atr)),
        "ema50_200_gap_atr":_feature_value(float(row.ema50_200_gap_atr)),"ema20_slope_5_atr":_feature_value(float(row.ema20_slope_5_atr)),"ema50_slope_10_atr":_feature_value(float(row.ema50_slope_10_atr)),
        "dist_vwap60_atr":_feature_value(float(row.dist_vwap60_atr)),"volume_z50":_feature_value(float(row.volume_z50)),"atr_regime_ratio":_feature_value(float(row.atr_regime_ratio)),
        "range_10_atr":_feature_value(float(row.range_10_atr)),"range_30_atr":_feature_value(float(row.range_30_atr)),
        "rsi14_slope_3":_feature_value(float(row.rsi14_slope_3)),"macd_hist_slope_3":_feature_value(float(row.macd_hist_slope_3)),"adx_slope_3":_feature_value(float(row.adx_slope_3)),
        "trend_efficiency_20":_feature_value(float(row.trend_efficiency_20)),"realized_vol_ratio_30_120":_feature_value(float(row.realized_vol_ratio_30_120)),
        "atr_vs_median_100":_feature_value(float(row.atr_vs_median_100)),"range_expansion_10_30":_feature_value(float(row.range_expansion_10_30)),
        "close_location_3":_feature_value(float(row.close_location_3)),"distance_high_60_atr":_feature_value(float(row.distance_high_60_atr)),"distance_low_60_atr":_feature_value(float(row.distance_low_60_atr)),
        "utc_hour_sin":_feature_value(float(row.utc_hour_sin)),"utc_hour_cos":_feature_value(float(row.utc_hour_cos)),"day_of_week_sin":_feature_value(float(row.day_of_week_sin)),"day_of_week_cos":_feature_value(float(row.day_of_week_cos)),
        "side_adjusted_ret_5":_feature_value(direction*float(row.ret_5)),"side_adjusted_ret_15":_feature_value(direction*float(row.ret_15)),"side_adjusted_macd_hist":_feature_value(direction*float(row.macd_hist)),"side_adjusted_di_gap":_feature_value(direction*float(row.plus_di14-row.minus_di14)),
        "utc_hour":hour,"day_of_week":int(signal_time.dayofweek),
        "ob_body_quality":_feature_value(ob_body_quality) if family=="A_OB_RETEST_SEQUENCE" else np.nan,
        "ob_to_bos_distance_atr":_feature_value(ob_to_bos) if family=="A_OB_RETEST_SEQUENCE" else np.nan,
        "fvg_to_displacement_ratio":_feature_value(fvg_ratio) if family=="B_FVG_RETEST_SEQUENCE" else np.nan,
        "fvg_fill_depth":_feature_value(fvg_fill) if family=="B_FVG_RETEST_SEQUENCE" else np.nan,
        "midpoint_retest_depth":_feature_value(midpoint_depth) if family=="C_DISPLACEMENT_MID_RETEST" else np.nan,
    }


def _v15_candidate_gate(family: str, cfg: dict, sweep_pen: float, second_value: float, feat: dict) -> tuple[bool, str, str]:
    """Quality-preserving expansion: relax at most one late-stage soft gate."""
    core_sweep = float(cfg["sweep_min"])
    core_second = float(cfg["second_max"])
    exp_sweep = float(cfg["expanded_sweep_min"])
    exp_second = float(cfg["expanded_second_max"])

    if not (np.isfinite(sweep_pen) and np.isfinite(second_value)):
        return False, "REJECT", "NON_FINITE"

    if sweep_pen >= core_sweep and second_value <= core_second:
        return True, "CORE", "CORE_GATE"

    # Exactly one soft gate may be relaxed; never admit a setup weak on both.
    if sweep_pen < exp_sweep or second_value > exp_second:
        return False, "REJECT", "OUTSIDE_EXPANDED_BOUNDS"
    if not (sweep_pen >= core_sweep or second_value <= core_second):
        return False, "REJECT", "BOTH_SOFT_GATES_WEAK"

    strong = (
        float(feat.get("displacement_body_atr", np.nan)) >= EXP_DISP_BODY_ATR
        and float(feat.get("displacement_close_location", np.nan)) >= EXP_DISP_CLOSE_LOCATION
        and float(feat.get("displacement_body_ratio", np.nan)) >= EXP_DISP_BODY_RATIO
        and int(feat.get("htf_alignment_count", 0)) >= EXP_MIN_HTF_ALIGNMENT
    )
    if not strong:
        return False, "REJECT", "COMPENSATION_FAILED"

    relaxed = "SWEEP" if sweep_pen < core_sweep else "SECOND"
    return True, "EXPANDED", f"RELAXED_{relaxed}_WITH_STRONG_COMPENSATION"


def build_candidates(
    bars1: pd.DataFrame, bars5: pd.DataFrame, bars15: pd.DataFrame, family: str,
    filter_profile: str = V15_PROFILE, diagnostics: bool = False, symbol: str = "",
):
    """Build candidates and optionally return stage-by-stage filter diagnostics."""
    if "ctx_session" not in bars1.columns:
        bars1 = add_context_features(bars1)
    d = add_structure_features(bars1)
    sw_hi = d["swing_high_raw"].to_numpy(bool)
    sw_lo = d["swing_low_raw"].to_numpy(bool)
    hi = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    cl = d["close"].to_numpy(float)
    atr1 = d["atr"].to_numpy(float)

    h5_frame = bars5.assign(m5_dir=m5_direction(bars5), m5_trend_strength=m5_trend_strength(bars5))
    h5a = asof_feature(d, h5_frame, M5_RULE)
    m5_dir = h5a["m5_dir"].fillna(0).to_numpy(int)
    m5_trend_strength_arr = h5a["m5_trend_strength"].to_numpy(float)

    c15 = m15_context(bars15)
    c15a = asof_feature(d, c15, M15_RULE)
    m15_dir = c15a["m15_dir"].fillna(0).to_numpy(int)
    m15_trend_strength_arr = c15a["m15_trend_strength"].to_numpy(float)
    m15_last_hi = c15a["last_swing_high"].to_numpy(float)
    m15_last_lo = c15a["last_swing_low"].to_numpy(float)
    m15_atr = c15a["atr"].to_numpy(float)

    # Causal liquidity memory: confirmed M1 swings only; metadata for ML, not a gate.
    known_swing_highs = deque(maxlen=50)
    known_swing_lows = deque(maxlen=50)

    last_swing_high = None
    last_swing_high_idx = None
    last_swing_low = None
    last_swing_low_idx = None

    # Active setup state per side. A new sequence replaces an older one,
    # ensuring only the first retest of the freshest zone can trigger.
    sweep_buy = None
    sweep_sell = None
    setup_buy = None
    setup_sell = None

    out: list[Candidate] = []
    diag = {
        "family": family,
        "filter_profile": filter_profile,
        "sequence_ready": 0,
        "filter_1_pass": 0,
        "filter_2_pass": 0,
        "core_pass": 0,
        "expanded_pass": 0,
        "final_unique": 0,
    }
    try:
        filter_cfg = _resolve_filter_cfg(filter_profile, symbol, family)
    except KeyError as exc:
        raise ValueError(f"Unknown filter profile/family: {filter_profile} / {family} / {symbol}") from exc

    def htf_allows(side: str, i: int) -> bool:
        sgn = 1 if side == "BUY" else -1
        # Require at least one completed HTF trend alignment; do not force both
        # timeframes to agree, otherwise trade count collapses in transitions.
        if m5_dir[i] != sgn and m15_dir[i] != sgn:
            return False
        # Soft M15 veto only when price is close to a fully known opposing swing
        # while M15 trend is decisively against the trade.
        a15 = m15_atr[i]
        if not np.isfinite(a15) or a15 <= 0:
            return True
        p = cl[i]
        if side == "BUY" and m15_dir[i] == -1 and np.isfinite(m15_last_hi[i]):
            if 0 <= (m15_last_hi[i] - p) <= M15_VETO_ATR * a15:
                return False
        if side == "SELL" and m15_dir[i] == 1 and np.isfinite(m15_last_lo[i]):
            if 0 <= (p - m15_last_lo[i]) <= M15_VETO_ATR * a15:
                return False
        return True

    for i in range(len(d)):
        confirmed = i - SWING
        if confirmed >= SWING:
            if sw_hi[confirmed]:
                last_swing_high = hi[confirmed]
                last_swing_high_idx = confirmed
                known_swing_highs.append(float(last_swing_high))
            if sw_lo[confirmed]:
                last_swing_low = lo[confirmed]
                last_swing_low_idx = confirmed
                known_swing_lows.append(float(last_swing_low))

        a = atr1[i]
        if not np.isfinite(a) or a <= 0:
            continue

        # 1) Liquidity sweep: pierce a confirmed swing and close back through it.
        if last_swing_low is not None and lo[i] < last_swing_low and cl[i] > last_swing_low:
            penetration = (last_swing_low - lo[i]) / a
            if penetration >= SWEEP_MIN_ATR:
                sweep_buy = {"idx": i, "extreme": lo[i], "swing": last_swing_low, "swing_idx": last_swing_low_idx}
                setup_buy = None
        if last_swing_high is not None and hi[i] > last_swing_high and cl[i] < last_swing_high:
            penetration = (hi[i] - last_swing_high) / a
            if penetration >= SWEEP_MIN_ATR:
                sweep_sell = {"idx": i, "extreme": hi[i], "swing": last_swing_high, "swing_idx": last_swing_high_idx}
                setup_sell = None

        if sweep_buy and i - sweep_buy["idx"] > MAX_SWEEP_AGE:
            sweep_buy = None
        if sweep_sell and i - sweep_sell["idx"] > MAX_SWEEP_AGE:
            sweep_sell = None

        # 2) Displacement + BOS following the sweep.
        bull_bos = last_swing_high is not None and cl[i] > (last_swing_high + BOS_BUFFER_ATR * a)
        bear_bos = last_swing_low is not None and cl[i] < (last_swing_low - BOS_BUFFER_ATR * a)
        bull_disp = displacement(d, i, "BUY")
        bear_disp = displacement(d, i, "SELL")

        if sweep_buy and bull_bos and bull_disp:
            oi = last_opposite_between(d, sweep_buy["idx"] + 1, i - 1, "BUY")
            if oi is None:
                oi = last_opposite_between(d, max(0, i - OB_LOOKBACK), i - 1, "BUY")
            ob_low, ob_high = (lo[oi], hi[oi]) if oi is not None else (lo[i], hi[i])
            fvg_low = float(d.iloc[i - 2].high) if i >= 2 and fvg(d, i, "BUY") else np.nan
            fvg_high = float(d.iloc[i].low) if i >= 2 and fvg(d, i, "BUY") else np.nan
            setup_buy = {
                "created": i, "sweep_idx": sweep_buy["idx"], "sweep_extreme": sweep_buy["extreme"], "sweep": sweep_buy["swing"],
                "ob_low": float(ob_low), "ob_high": float(ob_high), "ob_idx": int(oi) if oi is not None else -1,
                "fvg_low": fvg_low, "fvg_high": fvg_high,
                "disp_low": float(lo[i]), "disp_high": float(hi[i]),
                "bos_level": float(last_swing_high),
            }
            sweep_buy = None

        if sweep_sell and bear_bos and bear_disp:
            oi = last_opposite_between(d, sweep_sell["idx"] + 1, i - 1, "SELL")
            if oi is None:
                oi = last_opposite_between(d, max(0, i - OB_LOOKBACK), i - 1, "SELL")
            ob_low, ob_high = (lo[oi], hi[oi]) if oi is not None else (lo[i], hi[i])
            fvg_low = float(d.iloc[i].high) if i >= 2 and fvg(d, i, "SELL") else np.nan
            fvg_high = float(d.iloc[i - 2].low) if i >= 2 and fvg(d, i, "SELL") else np.nan
            setup_sell = {
                "created": i, "sweep_idx": sweep_sell["idx"], "sweep_extreme": sweep_sell["extreme"], "sweep": sweep_sell["swing"],
                "ob_low": float(ob_low), "ob_high": float(ob_high), "ob_idx": int(oi) if oi is not None else -1,
                "fvg_low": fvg_low, "fvg_high": fvg_high,
                "disp_low": float(lo[i]), "disp_high": float(hi[i]),
                "bos_level": float(last_swing_low),
            }
            sweep_sell = None

        # 3) First retest + confirmation. Signal is known only at close of i;
        # executable entry is the next M1 bar/tick.
        for side in ("BUY", "SELL"):
            s = setup_buy if side == "BUY" else setup_sell
            if s is None:
                continue
            age = i - int(s["created"])
            if age <= 0:
                continue
            if age > MAX_RETEST_BARS:
                if side == "BUY": setup_buy = None
                else: setup_sell = None
                continue
            if not htf_allows(side, i):
                # A retest can remain active; a later bar may regain alignment.
                continue

            if family == "A_OB_RETEST_SEQUENCE":
                zlo, zhi = s["ob_low"], s["ob_high"]
            elif family == "B_FVG_RETEST_SEQUENCE":
                if not np.isfinite(s["fvg_low"]) or not np.isfinite(s["fvg_high"]):
                    continue
                zlo, zhi = s["fvg_low"], s["fvg_high"]
            else:
                mid = (s["disp_low"] + s["disp_high"]) / 2.0
                half = max(0.25 * a, 0.15 * (s["disp_high"] - s["disp_low"]))
                zlo, zhi = mid - half, mid + half

            if zlo > zhi:
                zlo, zhi = zhi, zlo
            if not _zone_intersects(d.iloc[i], zlo, zhi):
                continue
            zone_mid = (zlo + zhi) / 2.0
            if not _confirmation_after_retest(d.iloc[i], side, zone_mid):
                continue
            if i + 1 >= len(d):
                continue

            if side == "BUY":
                invalidation = min(float(s["sweep_extreme"]), float(zlo))
            else:
                invalidation = max(float(s["sweep_extreme"]), float(zhi))

            feature_payload = _candidate_features(
                d, i, side, s, family, a, int(m5_dir[i]), int(m15_dir[i]),
                float(m15_last_hi[i]) if np.isfinite(m15_last_hi[i]) else np.nan,
                float(m15_last_lo[i]) if np.isfinite(m15_last_lo[i]) else np.nan,
                float(m5_trend_strength_arr[i]) if np.isfinite(m5_trend_strength_arr[i]) else np.nan,
                float(m15_trend_strength_arr[i]) if np.isfinite(m15_trend_strength_arr[i]) else np.nan,
                known_swing_highs, known_swing_lows,
            )
            feature_payload.update(_context_candidate_features(d.iloc[i], side, a, invalidation))

            # V15 quality-preserving gate. Full SMC structure and existing
            # displacement/confirmation/HTF safeguards above are unchanged.
            diag["sequence_ready"] += 1
            sweep_pen = float(feature_payload.get("sweep_penetration_atr", np.nan))
            second_key = "retest_age_bars" if family == "A_OB_RETEST_SEQUENCE" else "bos_distance_atr"
            second_value = float(feature_payload.get(second_key, np.nan))

            accepted, candidate_tier, gate_reason = _v15_candidate_gate(
                family, filter_cfg, sweep_pen, second_value, feature_payload
            )
            if sweep_pen >= float(filter_cfg["sweep_min"]):
                diag["filter_1_pass"] += 1
            if np.isfinite(second_value) and second_value <= float(filter_cfg["second_max"]):
                diag["filter_2_pass"] += 1
            if not accepted:
                continue
            if candidate_tier == "CORE":
                diag["core_pass"] += 1
            else:
                diag["expanded_pass"] += 1

            feature_payload["candidate_tier"] = candidate_tier
            feature_payload["candidate_gate_reason"] = gate_reason

            setup_id = f"{family}|{side}|{pd.Timestamp(d.iloc[i].timestamp).isoformat()}"
            out.append(
                Candidate(
                    symbol="", family=family, signal_idx=i,
                    signal_time=pd.Timestamp(d.iloc[i].timestamp),
                    entry_idx=i + 1, entry_time=pd.Timestamp(d.iloc[i + 1].timestamp),
                    side=side, anchor_low=float(invalidation if side == "BUY" else zlo),
                    anchor_high=float(invalidation if side == "SELL" else zhi),
                    trigger_price=float(cl[i]), setup_id=setup_id, features=feature_payload,
                )
            )
            if side == "BUY": setup_buy = None
            else: setup_sell = None

    uniq = {}
    for c in out:
        uniq[(c.entry_idx, c.side)] = c
    final = sorted(uniq.values(), key=lambda x: (x.entry_idx, x.side))
    diag["final_unique"] = len(final)
    if diagnostics:
        return final, diag
    return final


def _neutral_price_arrays(t: dict) -> np.ndarray:
    """Return one broker-neutral market-price series for both long and short trades.

    Priority: Last when valid, otherwise midpoint of Bid/Ask, then Bid, then Ask.
    This intentionally removes side-specific spread from entry/exit PnL.
    """
    bid = np.asarray(t["bid"], dtype=np.float64)
    ask = np.asarray(t["ask"], dtype=np.float64)
    last = np.asarray(t["last"], dtype=np.float64)
    mid = np.where(
        (bid > 0) & np.isfinite(bid) & (ask > 0) & np.isfinite(ask),
        (bid + ask) / 2.0,
        np.nan,
    )
    px = last.copy()
    bad = (px <= 0) | ~np.isfinite(px)
    px[bad] = mid[bad]
    bad = (px <= 0) | ~np.isfinite(px)
    px[bad & (bid > 0) & np.isfinite(bid)] = bid[bad & (bid > 0) & np.isfinite(bid)]
    bad = (px <= 0) | ~np.isfinite(px)
    px[bad & (ask > 0) & np.isfinite(ask)] = ask[bad & (ask > 0) & np.isfinite(ask)]
    return np.ascontiguousarray(px, dtype=np.float64)


def tick_entry_price(
    t: dict,
    entry_time: pd.Timestamp,
    side: str = "BUY",
) -> Optional[float]:
    """Return the first broker-neutral market price at/after entry time.

    `side` is accepted for API compatibility, but does not change the price:
    this research version deliberately excludes Bid/Ask spread.
    """
    i = int(np.searchsorted(t["time_msc"], int(entry_time.timestamp() * 1000), side="left"))
    if i >= len(t["time_msc"]):
        return None
    px = float(_neutral_price_arrays(t)[i])
    return float(px) if np.isfinite(px) and px > 0 else None


def _apply_exit_slippage(fill: float, side: str, point: float) -> float:
    slip = float(EXIT_SLIPPAGE_POINTS) * float(point)
    if slip == 0.0:
        return float(fill)
    return float(fill - slip if side == "BUY" else fill + slip)


def levels(
    entry: float,
    c: Candidate,
    mode: str,
    rr: float,
    atr_value: float,
    point: float,
) -> Optional[dict]:
    """Build SL/TP levels for the legacy single-config path.

    Kept in V5 to eliminate the V4 `levels()` NameError and to guarantee the
    legacy path uses the exact same stop logic as the accelerated engine.
    """
    if not np.isfinite(entry) or entry <= 0:
        return None
    a = float(atr_value) if np.isfinite(atr_value) else np.nan
    p = float(point)
    if c.side == "BUY":
        structural = float(c.anchor_low)
        buffer = max(p * SL_BUFFER_POINTS, a * SL_BUFFER_ATR) if np.isfinite(a) and a > 0 else p * SL_BUFFER_POINTS
        sl = structural - buffer
        if mode == "STRUCTURAL_ATR_BOUNDED" and np.isfinite(a) and a > 0:
            sl = min(sl, entry - ATR_FLOOR * a)
            sl = max(sl, entry - ATR_CAP * a)
        risk = entry - sl
        if risk <= 0 or (np.isfinite(a) and a > 0 and risk > ATR_CAP * a):
            return None
        tp = entry + rr * risk
    else:
        structural = float(c.anchor_high)
        buffer = max(p * SL_BUFFER_POINTS, a * SL_BUFFER_ATR) if np.isfinite(a) and a > 0 else p * SL_BUFFER_POINTS
        sl = structural + buffer
        if mode == "STRUCTURAL_ATR_BOUNDED" and np.isfinite(a) and a > 0:
            sl = max(sl, entry + ATR_FLOOR * a)
            sl = min(sl, entry + ATR_CAP * a)
        risk = sl - entry
        if risk <= 0 or (np.isfinite(a) and a > 0 and risk > ATR_CAP * a):
            return None
        tp = entry - rr * risk
    return {"entry": float(entry), "sl": float(sl), "tp": float(tp), "risk": float(risk)}


def eval_ticks(
    t: dict,
    entry_time: pd.Timestamp,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    risk: float,
) -> tuple[str, pd.Timestamp, float, float]:
    """Evaluate gross price performance using one broker-neutral tick price.

    No spread or commission is deducted. SL/TP are filled at the exact target
    levels plus optional explicit exit slippage (default 0).
    """
    time_msc = np.asarray(t["time_msc"], dtype=np.int64)
    if time_msc.size == 0:
        raise ValueError("Tick data is empty.")
    start_idx = int(np.searchsorted(time_msc, int(pd.Timestamp(entry_time).timestamp() * 1000), side="left"))
    if start_idx >= len(time_msc):
        last_idx = len(time_msc) - 1
        return "PERIOD_END", pd.to_datetime(int(time_msc[last_idx]), unit="ms", utc=True), float(entry), 0.0

    end_target = int(pd.Timestamp(entry_time).timestamp() * 1000 + MAX_HOLD_MINUTES * 60 * 1000)
    end_idx = min(int(np.searchsorted(time_msc, end_target, side="right") - 1), len(time_msc) - 1)
    data_end = end_idx >= len(time_msc) - 1
    neutral = _neutral_price_arrays(t)
    point = float(t["metadata"].get("point", _fallback_point("XAUUSD")) or _fallback_point("XAUUSD"))

    for i in range(start_idx, end_idx + 1):
        px = float(neutral[i])
        if not np.isfinite(px) or px <= 0:
            continue
        if side == "BUY":
            if px <= sl:
                fill = _apply_exit_slippage(float(sl), side, point)
                return "SL", pd.to_datetime(int(time_msc[i]), unit="ms", utc=True), fill, float((fill - entry) / risk)
            if px >= tp:
                fill = _apply_exit_slippage(float(tp), side, point)
                return "TP", pd.to_datetime(int(time_msc[i]), unit="ms", utc=True), fill, float((fill - entry) / risk)
        else:
            if px >= sl:
                fill = _apply_exit_slippage(float(sl), side, point)
                return "SL", pd.to_datetime(int(time_msc[i]), unit="ms", utc=True), fill, float((entry - fill) / risk)
            if px <= tp:
                fill = _apply_exit_slippage(float(tp), side, point)
                return "TP", pd.to_datetime(int(time_msc[i]), unit="ms", utc=True), fill, float((entry - fill) / risk)

    px = float(neutral[end_idx])
    if not np.isfinite(px) or px <= 0:
        px = float(entry)
    gross = (px - entry) if side == "BUY" else (entry - px)
    return (
        "PERIOD_END" if data_end else "TIMEOUT",
        pd.to_datetime(int(time_msc[end_idx]), unit="ms", utc=True),
        px,
        float(gross / risk),
    )


if NUMBA_AVAILABLE:
    _njit = njit
    _prange = prange
else:
    def _njit(*args, **kwargs):
        def deco(func):
            return func
        return deco
    def _prange(n):
        return range(n)


@_njit(cache=True, nogil=True, parallel=True)
def _run_configs_numba(
    time_msc, price_quote, starts, expiry_idx, sides,
    entry_prices, anchor_low, anchor_high, atr1, points, rr_values, mode_flags,
    sl_buffer_points, sl_buffer_atr, atr_floor, atr_cap, slippage_points
):
    ncfg = len(rr_values) * len(mode_flags)
    ncand = len(starts)
    n_ticks = len(time_msc)
    result = np.full((ncfg, ncand), -1, dtype=np.int8)  # 0 SL, 1 TP, 2 TO, 3 PE
    exit_idx = np.full((ncfg, ncand), -1, dtype=np.int64)
    pnl = np.zeros((ncfg, ncand), dtype=np.float64)
    sl_out = np.full((ncfg, ncand), np.nan, dtype=np.float64)
    tp_out = np.full((ncfg, ncand), np.nan, dtype=np.float64)

    for cfg in _prange(ncfg):
        rr = rr_values[cfg // len(mode_flags)]
        mode = mode_flags[cfg % len(mode_flags)]
        free_idx = -1
        for k in range(ncand):
            start_idx = starts[k]
            if start_idx <= free_idx or start_idx >= n_ticks:
                continue
            stop_idx = expiry_idx[k]
            if stop_idx < start_idx:
                continue
            data_end = stop_idx >= n_ticks - 1
            if data_end:
                stop_idx = n_ticks - 1
            raw = entry_prices[k]
            if not np.isfinite(raw) or raw <= 0:
                continue
            side = sides[k]
            # Broker-neutral entry price; no spread and no commission.
            entry = raw
            point = points[k]
            a = atr1[k]
            if side == 1:
                structural = anchor_low[k]
                buffer = max(point * sl_buffer_points, a * sl_buffer_atr) if np.isfinite(a) and a > 0 else point * sl_buffer_points
                sl = structural - buffer
                if mode == 1 and np.isfinite(a) and a > 0:
                    sl = min(sl, entry - atr_floor * a)
                    sl = max(sl, entry - atr_cap * a)
                risk = entry - sl
                if risk <= 0 or (np.isfinite(a) and a > 0 and risk > atr_cap * a):
                    continue
                tp = entry + rr * risk
            else:
                structural = anchor_high[k]
                buffer = max(point * sl_buffer_points, a * sl_buffer_atr) if np.isfinite(a) and a > 0 else point * sl_buffer_points
                sl = structural + buffer
                if mode == 1 and np.isfinite(a) and a > 0:
                    sl = max(sl, entry + atr_floor * a)
                    sl = min(sl, entry + atr_cap * a)
                risk = sl - entry
                if risk <= 0 or (np.isfinite(a) and a > 0 and risk > atr_cap * a):
                    continue
                tp = entry - rr * risk
            sl_out[cfg, k] = sl
            tp_out[cfg, k] = tp

            hit = -1
            hit_i = -1
            i = start_idx
            while i <= stop_idx:
                q = price_quote[i]
                if np.isfinite(q) and q > 0:
                    if side == 1:
                        if q <= sl:
                            hit, hit_i = 0, i
                            break
                        if q >= tp:
                            hit, hit_i = 1, i
                            break
                    else:
                        if q >= sl:
                            hit, hit_i = 0, i
                            break
                        if q <= tp:
                            hit, hit_i = 1, i
                            break
                i += 1

            if hit_i >= 0:
                q = price_quote[hit_i]
                point_slip = slippage_points * point
                if side == 1:
                    fill = (sl if hit == 0 else tp) - point_slip
                    gross = fill - entry
                else:
                    fill = (sl if hit == 0 else tp) + point_slip
                    gross = entry - fill
                pnl[cfg, k] = gross / risk
                result[cfg, k] = hit
                exit_idx[cfg, k] = hit_i
                free_idx = hit_i
            else:
                q = price_quote[stop_idx]
                if not np.isfinite(q) or q <= 0:
                    q = entry
                gross = q - entry if side == 1 else entry - q
                pnl[cfg, k] = gross / risk
                result[cfg, k] = 3 if data_end else 2
                exit_idx[cfg, k] = stop_idx
                free_idx = stop_idx
    return result, exit_idx, pnl, sl_out, tp_out


def _quote_arrays(t: dict) -> np.ndarray:
    """Broker-neutral tick price used for both BUY and SELL paths.

    A single price series deliberately removes Bid/Ask spread from the
    strategy-performance calculation.
    """
    return _neutral_price_arrays(t)


def run_symbol_fast(
    symbol: str, t: dict, b1: pd.DataFrame, b5: pd.DataFrame, b15: pd.DataFrame,
    atr1: np.ndarray, family: str, filter_profile: str = V15_PROFILE,
    candidates: Optional[list[Candidate]] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    if candidates is None:
        candidates = build_candidates(b1, b5, b15, family, filter_profile=filter_profile, symbol=symbol)
    nc = len(candidates)
    if nc == 0:
        return pd.DataFrame(), []
    time_msc = np.ascontiguousarray(t["time_msc"].astype(np.int64, copy=False))
    price_quote = _quote_arrays(t)
    point = float(t["metadata"].get("point", _fallback_point(symbol)) or _fallback_point(symbol))

    starts = np.empty(nc, dtype=np.int64)
    expiry_idx = np.empty(nc, dtype=np.int64)
    sides = np.empty(nc, dtype=np.int8)
    entries = np.empty(nc, dtype=np.float64)
    lows = np.empty(nc, dtype=np.float64)
    highs = np.empty(nc, dtype=np.float64)
    atrvals = np.empty(nc, dtype=np.float64)
    points = np.full(nc, point, dtype=np.float64)
    for k, c in enumerate(candidates):
        c.symbol = symbol
        starts[k] = int(np.searchsorted(time_msc, int(c.entry_time.timestamp() * 1000), side="left"))
        sides[k] = 1 if c.side == "BUY" else -1
        if starts[k] >= len(time_msc):
            entries[k] = np.nan
            expiry_idx[k] = len(time_msc)
            lows[k] = c.anchor_low
            highs[k] = c.anchor_high
            atrvals[k] = atr1[c.signal_idx]
            continue
        i = starts[k]
        entries[k] = price_quote[i]
        expiry_ms = int(c.entry_time.timestamp() * 1000 + MAX_HOLD_MINUTES * 60 * 1000)
        expiry_idx[k] = int(np.searchsorted(time_msc, expiry_ms, side="right") - 1)
        lows[k] = c.anchor_low
        highs[k] = c.anchor_high
        atrvals[k] = atr1[c.signal_idx]

    rr_values = np.asarray(RR_VALUES, dtype=np.float64)
    mode_flags = np.asarray([0, 1], dtype=np.int8)
    result, exit_idx, pnl, sl_out, tp_out = _run_configs_numba(
        time_msc, price_quote, starts, expiry_idx, sides, entries,
        lows, highs, atrvals, points, rr_values, mode_flags,
        float(SL_BUFFER_POINTS), float(SL_BUFFER_ATR), float(ATR_FLOOR),
        float(ATR_CAP), float(EXIT_SLIPPAGE_POINTS)
    )
    if "sell_bid" in t and np.any(sides == -1):
        # M1 OHLC replay: SELL trades use the high-before-low path (worst case for SELL).
        # Exit bars are identical on both paths, so position overlap decisions match.
        sell_quote = _neutral_price_arrays({"bid": t["sell_bid"], "ask": t["sell_ask"], "last": t["last"]})
        s_res, s_exit, s_pnl, s_sl, s_tp = _run_configs_numba(
            time_msc, sell_quote, starts, expiry_idx, sides, entries,
            lows, highs, atrvals, points, rr_values, mode_flags,
            float(SL_BUFFER_POINTS), float(SL_BUFFER_ATR), float(ATR_FLOOR),
            float(ATR_CAP), float(EXIT_SLIPPAGE_POINTS)
        )
        sell = sides == -1
        result[:, sell] = s_res[:, sell]; exit_idx[:, sell] = s_exit[:, sell]
        pnl[:, sell] = s_pnl[:, sell]; sl_out[:, sell] = s_sl[:, sell]; tp_out[:, sell] = s_tp[:, sell]
    data_source = t.get("data_source", "TICK")

    rows, stats = [], []
    for cfg in range(result.shape[0]):
        rr = float(rr_values[cfg // 2])
        mode = "STRUCTURAL" if cfg % 2 == 0 else "STRUCTURAL_ATR_BOUNDED"
        rrows = []
        for k in range(nc):
            code = int(result[cfg, k]); ex = int(exit_idx[cfg, k])
            if code < 0 or ex < 0:
                continue
            side = "BUY" if sides[k] == 1 else "SELL"
            ex_ts = pd.to_datetime(int(time_msc[ex]), unit="ms", utc=True)
            outcome = "SL" if code == 0 else ("TP" if code == 1 else ("PERIOD_END" if code == 3 else "TIMEOUT"))
            rrows.append({
                "symbol": symbol, "family": family, "filter_profile": filter_profile, "rr": rr, "sl_mode": mode,
                "signal_time": pd.Timestamp(candidates[k].signal_time),
                "entry_time": pd.Timestamp(candidates[k].entry_time),
                "exit_time": ex_ts, "side": side, "entry": float(entries[k]),
                "sl": float(sl_out[cfg, k]), "tp": float(tp_out[cfg, k]),
                "result": outcome, "pnl_r": float(pnl[cfg, k]),
                "cost_model": "GROSS_NO_SPREAD_NO_COMMISSION",
                "data_source": data_source,
                "setup_id": candidates[k].setup_id,
                **candidates[k].features,
            })
        if not rrows:
            stats.append({"symbol":symbol,"family":family,"filter_profile":filter_profile,"rr":rr,"sl_mode":mode,"candidates":nc,"trades":0,"wins":0,"losses":0,"timeout":0,"period_end":0,"wr_pct":0.0,"pf":0.0,"total_r":0.0,"closed_r":0.0,"avg_r":0.0,"max_dd_r":0.0})
            continue
        dfx = pd.DataFrame(rrows)
        wins = int((dfx.result == "TP").sum()); losses = int((dfx.result == "SL").sum())
        timeout = int((dfx.result == "TIMEOUT").sum()); pe = int((dfx.result == "PERIOD_END").sum())
        closed = dfx[dfx.result.isin(["TP","SL"])]
        gp = float(closed.loc[closed.pnl_r > 0, "pnl_r"].sum()) if not closed.empty else 0.0
        gl = abs(float(closed.loc[closed.pnl_r < 0, "pnl_r"].sum())) if not closed.empty else 0.0
        pf = gp/gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
        eq = peak = dd = 0.0
        for x in dfx.pnl_r:
            eq += float(x); peak = max(peak, eq); dd = max(dd, peak-eq)
        closed_r = float(closed.pnl_r.sum()) if not closed.empty else 0.0
        stats.append({
            "symbol":symbol,"family":family,"filter_profile":filter_profile,"rr":rr,"sl_mode":mode,"candidates":nc,"trades":len(dfx),
            "wins":wins,"losses":losses,"timeout":timeout,"period_end":pe,
            "wr_pct":wins/(wins+losses)*100 if wins+losses else 0.0,"pf":pf,
            "total_r":float(dfx.pnl_r.sum()),"closed_r":closed_r,
            "avg_r":closed_r/len(closed) if len(closed) else 0.0,"max_dd_r":dd,
        })
        rows.extend(rrows)
    return pd.DataFrame(rows), stats


def _add_quality_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add a non-filtering 0-4 quality score using signal-close features only."""
    x = df.copy()
    for col in QUALITY_THRESHOLDS:
        if col not in x.columns:
            x[col] = np.nan

    htf = pd.to_numeric(x["htf_alignment_count"], errors="coerce")
    body_atr = pd.to_numeric(x["displacement_body_atr"], errors="coerce")
    body_ratio = pd.to_numeric(x["displacement_body_ratio"], errors="coerce")
    close_loc = pd.to_numeric(x["displacement_close_location"], errors="coerce")

    score = (
        (htf >= QUALITY_THRESHOLDS["htf_alignment_count"]).astype(int)
        + (body_atr >= QUALITY_THRESHOLDS["displacement_body_atr"]).astype(int)
        + (body_ratio >= QUALITY_THRESHOLDS["displacement_body_ratio"]).astype(int)
        + (close_loc >= QUALITY_THRESHOLDS["displacement_close_location"]).astype(int)
    )
    x["quality_score"] = score.astype(int)
    x["quality_score_max"] = 4
    x["quality_score_version"] = QUALITY_SCORE_VERSION
    x["quality_band"] = pd.cut(
        x["quality_score"],
        bins=[-1, 1, 2, 3, 4],
        labels=["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
    ).astype(str)
    return x


def _candidate_level_view(trades: pd.DataFrame) -> pd.DataFrame:
    """Reduce the RR/SL matrix to one row per candidate for confluence analysis."""
    if trades.empty:
        return trades.copy()
    cols = [
        "symbol", "filter_profile", "family", "setup_id", "signal_time", "entry_time", "side",
        "quality_score", "quality_band",
    ] + [c for c in FEATURE_NUMERIC if c in trades.columns] + (["utc_block"] if "utc_block" in trades.columns else [])
    cols = list(dict.fromkeys([c for c in cols if c in trades.columns]))
    return trades[cols].drop_duplicates(
        subset=["symbol", "filter_profile", "family", "setup_id"], keep="first"
    ).copy()


def _compute_causal_confluence(candidates: pd.DataFrame, window_minutes: int = CONFLUENCE_WINDOW_MINUTES) -> pd.DataFrame:
    """Count distinct families seen in the preceding/current causal window."""
    if candidates.empty:
        return candidates.copy()
    x = candidates.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.sort_values(["symbol", "filter_profile", "side", "signal_time", "family"]).reset_index(drop=True)
    x["confluence_count"] = 1
    x["confluence_families"] = x["family"].map(FAMILY_LABELS).fillna(x["family"])
    x["confluence_window_min"] = int(window_minutes)

    window = pd.Timedelta(minutes=int(window_minutes))
    grouped = x.groupby(["symbol", "filter_profile", "side"], sort=False).groups
    for (_, _, _), idxs in grouped.items():
        idxs = list(idxs)
        left = 0
        counts = {}
        fams = [x.at[idx, "family"] for idx in idxs]
        times = [x.at[idx, "signal_time"] for idx in idxs]
        for right, idx in enumerate(idxs):
            while left <= right and times[right] - times[left] > window:
                old = fams[left]
                counts[old] = counts.get(old, 0) - 1
                if counts[old] <= 0:
                    counts.pop(old, None)
                left += 1
            current_fams = set(counts)
            current_fams.add(fams[right])
            ordered = sorted(current_fams, key=lambda f: (FAMILY_LABELS.get(f, f), f))
            x.at[idx, "confluence_count"] = int(len(ordered))
            x.at[idx, "confluence_families"] = "+".join(FAMILY_LABELS.get(f, f) for f in ordered)
            counts[fams[right]] = counts.get(fams[right], 0) + 1
    return x


def _attach_quality_confluence(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enrich trades and return a one-row-per-candidate quality/confluence table."""
    if trades.empty:
        return trades.copy(), pd.DataFrame()
    x = _add_quality_score_frame(trades)
    c = _compute_causal_confluence(_candidate_level_view(x))
    keys = ["symbol", "filter_profile", "family", "setup_id"]
    enrich = keys + [
        "quality_score", "quality_band", "confluence_count", "confluence_families",
        "confluence_window_min",
    ]
    lookup = c[enrich].drop_duplicates(keys)
    x = x.drop(columns=[col for col in enrich if col not in keys and col in x.columns], errors="ignore")
    x = x.merge(lookup, on=keys, how="left", validate="many_to_one")
    x["confluence_confirmed"] = x["confluence_count"].ge(2)
    x["triple_confluence"] = x["confluence_count"].eq(3)
    return x, c


def _metric_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Create descriptive performance rows by quality/confluence bucket."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        closed = g[g.result.isin(["TP", "SL"])]
        wins = int((closed.result == "TP").sum())
        losses = int((closed.result == "SL").sum())
        gp = float(closed.loc[closed.pnl_r > 0, "pnl_r"].sum()) if not closed.empty else 0.0
        gl = abs(float(closed.loc[closed.pnl_r < 0, "pnl_r"].sum())) if not closed.empty else 0.0
        row = dict(zip(group_cols, key))
        row.update({
            "trades": int(len(g)), "wins": wins, "losses": losses,
            "timeout": int((g.result == "TIMEOUT").sum()),
            "period_end": int((g.result == "PERIOD_END").sum()),
            "wr_pct": wins / (wins + losses) * 100.0 if wins + losses else np.nan,
            "pf": gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0),
            "closed_r": float(closed.pnl_r.sum()) if not closed.empty else 0.0,
            "avg_r_closed": float(closed.pnl_r.mean()) if not closed.empty else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)



def _candidates_to_frame(candidates: list[Candidate], symbol: str, filter_profile: str) -> pd.DataFrame:
    """Materialize ALL independent A/B/C candidates for causal quality/confluence labeling."""
    if not candidates:
        return pd.DataFrame()
    rows = []
    for c in candidates:
        rows.append({
            "symbol": symbol,
            "family": c.family,
            "filter_profile": filter_profile,
            "setup_id": c.setup_id,
            "signal_time": pd.Timestamp(c.signal_time),
            "entry_time": pd.Timestamp(c.entry_time),
            "side": c.side,
            **c.features,
        })
    return pd.DataFrame(rows)


def _attach_from_candidate_labels(trades: pd.DataFrame, candidate_labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach labels to executed trades while confluence is calculated from all candidates."""
    if candidate_labels.empty:
        return trades.copy(), candidate_labels.copy()
    c = _add_quality_score_frame(candidate_labels)
    c = _compute_causal_confluence(_candidate_level_view(c))
    keys = ["symbol", "filter_profile", "family", "setup_id"]
    enrich = keys + ["quality_score", "quality_band", "confluence_count", "confluence_families", "confluence_window_min"]
    lookup = c[enrich].drop_duplicates(keys)
    x = trades.copy()
    x = x.drop(columns=[col for col in enrich if col not in keys and col in x.columns], errors="ignore")
    x = x.merge(lookup, on=keys, how="left", validate="many_to_one")
    x["confluence_confirmed"] = x["confluence_count"].ge(2)
    x["triple_confluence"] = x["confluence_count"].eq(3)
    return x, c




def _performance_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = _add_quality_score_frame(df)
    x["is_closed"] = x["result"].isin(["TP", "SL"])
    x["win_label"] = np.where(x["result"] == "TP", 1.0, np.where(x["result"] == "SL", 0.0, np.nan))
    return x


def _basic_stats(g: pd.DataFrame) -> dict:
    closed = g[g["result"].isin(["TP", "SL"])]
    wins = int((closed["result"] == "TP").sum())
    losses = int((closed["result"] == "SL").sum())
    total_r = float(g["pnl_r"].sum()) if not g.empty else 0.0
    closed_r = float(closed["pnl_r"].sum()) if not closed.empty else 0.0
    return {
        "trades": int(len(g)),
        "closed": int(len(closed)),
        "wins": wins,
        "losses": losses,
        "wr_pct": wins / (wins + losses) * 100.0 if wins + losses else np.nan,
        "avg_r_all": float(g["pnl_r"].mean()) if len(g) else np.nan,
        "avg_r_closed": float(closed["pnl_r"].mean()) if len(closed) else np.nan,
        "total_r": total_r,
    }


def _basic_stats(g:pd.DataFrame)->dict:
    closed=g[g.result.isin(["TP","SL"])]
    wins=int((closed.result=="TP").sum()); losses=int((closed.result=="SL").sum())
    return {"trades":len(g),"closed":len(closed),"wins":wins,"losses":losses,"wr_pct":wins/(wins+losses)*100 if wins+losses else np.nan,"avg_r_all":g.pnl_r.mean() if len(g) else np.nan,"avg_r_closed":closed.pnl_r.mean() if len(closed) else np.nan,"total_r":g.pnl_r.sum() if len(g) else 0.0,"closed_r":closed.pnl_r.sum() if len(closed) else 0.0}


def _write_feature_analysis(trades:pd.DataFrame,candidates:pd.DataFrame,out:Path)->None:
    if candidates.empty:return
    c=_compute_causal_confluence(_add_quality_score_frame(candidates.copy())); c.to_csv(out/"SMC_V15_ALL_CANDIDATES.csv",index=False,encoding="utf-8-sig")
    if trades.empty:return
    x=_add_quality_score_frame(trades.copy()); x["win_label"]=np.where(x.result=="TP",1.0,np.where(x.result=="SL",0.0,np.nan))
    base=x[(x.rr==1.5)&(x.sl_mode=="STRUCTURAL")]
    corr=[]
    for (sym,fam),g in base.groupby(["symbol","family"],sort=False):
        closed=g[g.result.isin(["TP","SL"])].copy()
        if len(closed)<15:continue
        for f in FEATURE_NUMERIC:
            if f not in closed:continue
            v=pd.to_numeric(closed[f],errors="coerce"); m=v.notna()&closed.win_label.notna()
            if int(m.sum())<15 or v[m].nunique()<3:continue
            corr.append({"symbol":sym,"family":fam,"feature":f,"n_closed":int(m.sum()),"spearman_with_win":float(v[m].corr(closed.loc[m,"win_label"],method="spearman")),"spearman_with_r":float(v[m].corr(closed.loc[m,"pnl_r"],method="spearman"))})
    pd.DataFrame(corr).to_csv(out/"SMC_V15_FEATURE_CORRELATIONS.csv",index=False,encoding="utf-8-sig")
    _metric_rows(x,["symbol","family","rr","sl_mode","quality_score","quality_band"]).to_csv(out/"SMC_V15_QUALITY_ANALYSIS.csv",index=False,encoding="utf-8-sig")
    _metric_rows(x,["symbol","family","rr","sl_mode","confluence_count","confluence_families"]).to_csv(out/"SMC_V15_CONFLUENCE_ANALYSIS.csv",index=False,encoding="utf-8-sig")
    _metric_rows(x[x.triple_confluence],["symbol","family","rr","sl_mode","side","confluence_families"]).to_csv(out/"SMC_V15_TRIPLE_CONFLUENCE.csv",index=False,encoding="utf-8-sig")
    q=c.groupby(["symbol","family","quality_score","quality_band"],sort=False).size().reset_index(name="candidate_count"); q.to_csv(out/"SMC_V15_CANDIDATE_QUALITY_COUNTS.csv",index=False,encoding="utf-8-sig")
    dictionary=("SMC V15 FEATURE DICTIONARY\n"+"="*96+"\n"+"Features are available at or before signal close; no future outcome/exit data is used.\n\nExisting SMC features preserved. Added technical/market-state features plus the V15 causal SMC-event expansion (sweep reclaim/wick, sweep-to-BOS timing/move, BOS break strength, displacement expansion/compression, sequence age, liquidity distances/asymmetry, M5/M15 trend strength/consistency, and family-specific OB/FVG/midpoint geometry). Context layer (M1 bars and clock only, no H1/H4/D1): London/New York session (DST-aware) and minutes into it, today's range/position/open move, Asian range/position, first intraday liquidity (today's/Asian extreme) in R, stop size in ATR and tick activity. Side-aware features are aligned so positive = in favour of the trade. Strategy rules remain unchanged.\n\nNumeric features:\n"+"\n".join("- "+f for f in FEATURE_NUMERIC)+"\n\nCategorical features:\n"+"\n".join("- "+f for f in FEATURE_CATEGORICAL)+"\n\nV15 candidate limits:\nA: "+str(V15_FILTERS["A_OB_RETEST_SEQUENCE"])+"\nB: "+str(V15_FILTERS["B_FVG_RETEST_SEQUENCE"])+"\nC: "+str(V15_FILTERS["C_DISPLACEMENT_MID_RETEST"])+"\n")
    (out/"SMC_V15_FEATURE_DICTIONARY.txt").write_text(dictionary,encoding="utf-8")



def _config_tag(rr: float, sl_mode: str) -> str:
    rr_tag = str(rr).replace(".", "p")
    mode_tag = str(sl_mode).lower()
    return f"rr{rr_tag}_{mode_tag}"


def _build_symbol_dataset(
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Build one row per unique V15 candidate, with all execution outcomes as columns.

    This is deliberately NOT an ML model. It preserves the candidate-level causal
    feature vector exactly once and stores the outcome of every V15 RR/SL scenario
    beside it. No candidate is duplicated merely because multiple exit scenarios
    were tested.
    """
    if candidates.empty:
        return pd.DataFrame()

    c = candidates.copy()
    c = _add_quality_score_frame(c)
    c = _compute_causal_confluence(c)

    key = ["symbol", "filter_profile", "family", "setup_id"]
    base_cols = key + [
        "signal_time", "entry_time", "side", "data_source", "quality_score", "quality_score_max",
        "quality_score_version", "quality_band", "confluence_count",
        "confluence_families", "confluence_window_min",
    ] + [f for f in FEATURE_NUMERIC if f in c.columns] + [
        f for f in FEATURE_CATEGORICAL if f in c.columns and f not in {"side", "family"}
    ]
    base_cols = list(dict.fromkeys([col for col in base_cols if col in c.columns]))
    base = c[base_cols].drop_duplicates(key, keep="first").copy()
    base["dataset_version"] = "V15_DATASET"
    if "candidate_tier" not in base.columns:
        base["candidate_tier"] = "UNKNOWN"
    base["ml_ready"] = True

    if trades.empty:
        return base.sort_values(["symbol", "family", "signal_time", "setup_id"]).reset_index(drop=True)

    t = trades.copy()
    t["signal_time"] = pd.to_datetime(t["signal_time"], utc=True, errors="coerce")
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
    t = t[t["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()

    for rr in RR_VALUES:
        for mode in SL_MODES:
            tag = _config_tag(rr, mode)
            outcome_cols = [
                f"outcome_{tag}", f"pnl_r_{tag}", f"entry_{tag}", f"sl_{tag}",
                f"tp_{tag}", f"exit_time_{tag}", f"win_label_{tag}", f"closed_label_{tag}",
            ]
            g = t[(np.isclose(pd.to_numeric(t["rr"], errors="coerce"), float(rr))) & (t["sl_mode"] == mode)].copy()
            if not g.empty:
                # A candidate can occur at most once per family/setup for a given config.
                g = g.sort_values("exit_time").drop_duplicates(key, keep="first")
                keep = key + ["result", "pnl_r", "entry", "sl", "tp", "exit_time"]
                g = g[keep].copy()
                rename = {
                    "result": f"outcome_{tag}",
                    "pnl_r": f"pnl_r_{tag}",
                    "entry": f"entry_{tag}",
                    "sl": f"sl_{tag}",
                    "tp": f"tp_{tag}",
                    "exit_time": f"exit_time_{tag}",
                }
                g = g.rename(columns=rename)
                g[f"win_label_{tag}"] = np.where(
                    g[f"outcome_{tag}"] == "TP", 1.0,
                    np.where(g[f"outcome_{tag}"] == "SL", 0.0, np.nan),
                )
                g[f"closed_label_{tag}"] = g[f"win_label_{tag}"].notna()
                base = base.merge(g, on=key, how="left", validate="one_to_one")
            else:
                # Preserve a stable schema even if one RR/SL scenario has no executed rows.
                for col in outcome_cols:
                    base[col] = np.nan

    return base.sort_values(["symbol", "family", "signal_time", "setup_id"]).reset_index(drop=True)


def _write_symbol_datasets(
    symbol: str,
    dataset: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    trades: pd.DataFrame,
    root_out: Path,
) -> None:
    """Write only V15 datasets for one symbol in an isolated folder."""
    sym_dir = root_out / "datasets" / str(symbol).upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    if not dataset.empty:
        dataset.to_csv(sym_dir / f"SMC_V15_{symbol.upper()}_DATASET.csv", index=False, encoding="utf-8-sig")
    if not candidate_frame.empty:
        c = _compute_causal_confluence(_add_quality_score_frame(candidate_frame.copy()))
        c.to_csv(sym_dir / f"SMC_V15_{symbol.upper()}_CANDIDATES.csv", index=False, encoding="utf-8-sig")
    if not trades.empty:
        trades.to_csv(sym_dir / f"SMC_V15_{symbol.upper()}_TRADES.csv", index=False, encoding="utf-8-sig")


def _write_dataset_manifest(symbols: list[str], root_out: Path) -> None:
    manifest = [
        "SMC V15 HISTORICAL DATASET MANIFEST",
        "=" * 96,
        "Purpose: candidate/feature/outcome export for future per-symbol ML.",
        "No ML training or inference is performed by V15.",
        f"Period UTC: {PERIOD_START.isoformat()} -> {PERIOD_END.isoformat()}",
        "Data source: existing Tick Data/cache/download/input workflow is unchanged.",
        "Families: A, B, C are independent; no cross-family deduplication.",
        "Each dataset row represents ONE unique strategy candidate per symbol/family/setup_id.",
        "RR/SL scenarios are stored as separate outcome columns, not duplicate candidate rows.",
        "All exported features are causal and available no later than the signal candle close.",
        "Per-symbol isolation: no candidate/outcome data from another symbol is placed into a symbol dataset.",
        "",
        "Symbols:",
        *[f"- {s}" for s in symbols],
        "",
        "Per-symbol files:",
        "- SMC_V15_<SYMBOL>_DATASET.csv : one row per unique candidate + features + outcome matrix",
        "- SMC_V15_<SYMBOL>_CANDIDATES.csv : raw candidate-level features/quality/confluence",
        "- SMC_V15_<SYMBOL>_TRADES.csv : executed V15 scenario results",
    ]
    (root_out / "SMC_V15_DATASET_MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


DATA_GAP_WARN_DAYS = 5  # longer than a weekend + holiday


def data_gap_report(symbol: str, t: dict) -> list[dict]:
    """List holes in the tick history longer than DATA_GAP_WARN_DAYS (e.g. missing years)."""
    ms = np.asarray(t["time_msc"], dtype=np.int64)
    if len(ms) < 2:
        return []
    gap_ms = np.diff(ms)
    idx = np.where(gap_ms > DATA_GAP_WARN_DAYS * 86_400_000)[0]
    rows = []
    for k in idx:
        start = pd.to_datetime(int(ms[k]), unit="ms", utc=True)
        end = pd.to_datetime(int(ms[k + 1]), unit="ms", utc=True)
        rows.append({"symbol": symbol, "gap_start": start, "gap_end": end, "gap_days": round(float(gap_ms[k]) / 86_400_000, 1)})
    first = pd.to_datetime(int(ms[0]), unit="ms", utc=True)
    last = pd.to_datetime(int(ms[-1]), unit="ms", utc=True)
    if first - pd.Timestamp(PERIOD_START) > pd.Timedelta(days=DATA_GAP_WARN_DAYS):
        rows.insert(0, {"symbol": symbol, "gap_start": pd.Timestamp(PERIOD_START), "gap_end": first,
                        "gap_days": round((first - pd.Timestamp(PERIOD_START)).total_seconds() / 86400, 1)})
    if pd.Timestamp(PERIOD_END) - last > pd.Timedelta(days=DATA_GAP_WARN_DAYS):
        rows.append({"symbol": symbol, "gap_start": last, "gap_end": pd.Timestamp(PERIOD_END),
                     "gap_days": round((pd.Timestamp(PERIOD_END) - last).total_seconds() / 86400, 1)})
    return rows


def prepare_symbol_data(t: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build M1/M5/M15 bars once per symbol and reuse them across all configs."""
    print("  Building M1/M5/M15 bars once...", flush=True)
    b1 = ticks_to_bars(t, M1_RULE)
    b5 = ticks_to_bars(t, M5_RULE)
    b15 = ticks_to_bars(t, M15_RULE)
    atr1 = atr(b1, ATR_PERIOD).to_numpy(float)
    print("  Building M1 session/range context features once...", flush=True)
    b1 = add_context_features(b1)
    print(f"  Bars ready | M1={len(b1):,} M5={len(b5):,} M15={len(b15):,}", flush=True)
    return b1, b5, b15, atr1


def run_symbol_legacy(
    symbol: str, t: dict, b1: pd.DataFrame, b5: pd.DataFrame,
    b15: pd.DataFrame, atr1: np.ndarray, family: str,
    rr: float, mode: str,
) -> tuple[pd.DataFrame, dict]:
    candidates = build_candidates(b1, b5, b15, family, filter_profile=V15_PROFILE, symbol=symbol)
    rows = []
    free = None
    total_candidates = len(candidates)
    point = float(t["metadata"].get("point", _fallback_point(symbol)) or _fallback_point(symbol))

    for c in candidates:
        c.symbol = symbol
        if free is not None and c.entry_time <= free:
            continue
        raw = tick_entry_price(t, c.entry_time, c.side)
        if raw is None:
            continue
        lv = levels(raw, c, mode, rr, atr1[c.signal_idx], point)
        if lv is None:
            continue
        result, exit_time, exit_price, pnl = eval_ticks(
            t, c.entry_time, c.side, lv["entry"], lv["sl"], lv["tp"], lv["risk"]
        )
        rows.append({
            "symbol": symbol, "family": family, "rr": rr, "sl_mode": mode,
            "signal_time": c.signal_time, "entry_time": c.entry_time,
            "exit_time": exit_time, "side": c.side,
            "entry": lv["entry"], "sl": lv["sl"], "tp": lv["tp"],
            "result": result, "pnl_r": float(pnl),
            "cost_model": "GROSS_NO_SPREAD_NO_COMMISSION",
        })
        free = exit_time

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "symbol": symbol, "family": family, "rr": rr, "sl_mode": mode,
            "candidates": total_candidates, "trades": 0, "wins": 0, "losses": 0,
            "timeout": 0, "period_end": 0, "wr_pct": 0.0, "pf": 0.0,
            "total_r": 0.0, "closed_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0,
        }

    wins = int((df.result == "TP").sum())
    losses = int((df.result == "SL").sum())
    timeout = int((df.result == "TIMEOUT").sum())
    pe = int((df.result == "PERIOD_END").sum())
    closed = df[df.result.isin(["TP", "SL"])]
    gp = float(closed.loc[closed.pnl_r > 0, "pnl_r"].sum()) if not closed.empty else 0.0
    gl = abs(float(closed.loc[closed.pnl_r < 0, "pnl_r"].sum())) if not closed.empty else 0.0
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    eq = peak = dd = 0.0
    for x in df.pnl_r:
        eq += float(x); peak = max(peak, eq); dd = max(dd, peak - eq)
    closed_r = float(closed.pnl_r.sum()) if not closed.empty else 0.0
    return df, {
        "symbol": symbol, "family": family, "rr": rr, "sl_mode": mode,
        "candidates": total_candidates, "trades": len(df),
        "wins": wins, "losses": losses, "timeout": timeout, "period_end": pe,
        "wr_pct": wins / (wins + losses) * 100 if wins + losses else 0.0,
        "pf": pf, "total_r": float(df.pnl_r.sum()), "closed_r": closed_r,
        "avg_r": closed_r / len(closed) if len(closed) else 0.0, "max_dd_r": dd,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "results" / "SMC_V15"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SMC MTF RESEARCH V15 | CANDIDATE + DATASET GENERATOR | TICK DATA | GROSS / NO BROKER COSTS")
    print("Period (UTC): 2020-01-01 -> 2026-09-30")
    print("=" * 100)
    print("V15 only. No previous-version results are generated, loaded, compared, or written.")
    print("A/B/C remain independent; cross-family deduplication is disabled.")
    print("ML training/inference is NOT included. Each symbol gets an isolated candidate dataset.")

    try:
        requested = sys.argv[1].strip().upper() if len(sys.argv) > 1 else input("> ").strip().upper()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return

    symbols = [requested] if requested else list(SYMBOLS)
    print(f"Symbols: {', '.join(symbols)}")
    print("Data layer: unchanged | Tick cache preferred, CSV fallback and symbol download/input workflow unchanged.")
    print(f"V15 filters: {V15_FILTERS}")

    summaries: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    errors: list[dict] = []
    data_gaps: list[dict] = []

    # Keep isolated per-symbol collections so future ML datasets cannot mix symbols.
    per_symbol_candidates: dict[str, list[pd.DataFrame]] = {s: [] for s in symbols}
    per_symbol_trades: dict[str, list[pd.DataFrame]] = {s: [] for s in symbols}

    for symbol in symbols:
        print(f"\n### {symbol} | V15 CANDIDATES / DATASET")
        try:
            t = load_tick_cache(symbol)
            print(f"Data source: {t.get('data_source', 'TICK')} | {t.get('data_path', data_path(symbol))} | price points={len(t['time_msc']):,}", flush=True)
            gaps = data_gap_report(symbol, t)
            data_gaps.extend(gaps)
            for gp in gaps:
                print(f"  [DATA GAP] {gp['gap_start']} -> {gp['gap_end']} ({gp['gap_days']} days without ticks)", flush=True)
            b1, b5, b15, atr1 = prepare_symbol_data(t)

            for family in FAMILIES:
                try:
                    candidates, diag = build_candidates(
                        b1, b5, b15, family,
                        filter_profile=V15_PROFILE,
                        diagnostics=True,
                        symbol=symbol,
                    )
                    diag.update({"symbol": symbol, "version": "V15"})
                    diagnostics.append(diag)

                    cf = _candidates_to_frame(candidates, symbol, V15_PROFILE)
                    if not cf.empty:
                        cf["data_source"] = t.get("data_source", "TICK")
                        candidate_frames.append(cf)
                        per_symbol_candidates[symbol].append(cf)

                    print(
                        f"[{symbol}] {family} | sequence_ready={diag['sequence_ready']} "
                        f"| f1={diag['filter_1_pass']} | f2={diag['filter_2_pass']} "
                        f"| candidates={diag['final_unique']}", flush=True
                    )

                    df, stats = run_symbol_fast(
                        symbol, t, b1, b5, b15, atr1, family,
                        filter_profile=V15_PROFILE,
                        candidates=candidates,
                    )
                    summaries.extend(stats)
                    if not df.empty:
                        trade_frames.append(df)
                        per_symbol_trades[symbol].append(df)

                    for st in stats:
                        print(
                            f"  RR={st['rr']:g} | {st['sl_mode']} | trades={st['trades']} "
                            f"| TP={st['wins']} | SL={st['losses']} | TO={st['timeout']} "
                            f"| WR={st['wr_pct']:.2f}% | PF={st['pf']:.3f} "
                            f"| R={st['total_r']:.3f} | DD={st['max_dd_r']:.3f}", flush=True
                        )
                except Exception as exc:
                    errors.append({
                        "symbol": symbol,
                        "family": family,
                        "filter_profile": V15_PROFILE,
                        "error": str(exc),
                    })
                    print(f"[ERROR] {symbol} {family}: {exc}", flush=True)
        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "filter_profile": V15_PROFILE,
                "error": str(exc),
            })
            print(f"[ERROR] {symbol}: {exc}", flush=True)

    s = pd.DataFrame(summaries)
    tr = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    candidates_all = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()

    # Attach causal labels to the combined V15 audit files.
    tr_v15, c_labeled = _attach_from_candidate_labels(tr, candidates_all)
    if not tr_v15.empty:
        tr_v15["model_version"] = "V15"
    if not s.empty:
        s["model_version"] = "V15"

    # Combined V15-only audit outputs.
    s.to_csv(out / "SMC_V15_SUMMARY.csv", index=False, encoding="utf-8-sig")
    tr_v15.to_csv(out / "SMC_V15_ALL_TRADES.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(out / "SMC_V15_FILTER_DIAGNOSTICS.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(out / "SMC_V15_ALL_ERRORS.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(data_gaps, columns=["symbol", "gap_start", "gap_end", "gap_days"]).to_csv(
        out / "SMC_V15_DATA_GAPS.csv", index=False, encoding="utf-8-sig")
    if not c_labeled.empty:
        c_labeled.to_csv(out / "SMC_V15_CANDIDATE_FEATURES.csv", index=False, encoding="utf-8-sig")
    _write_feature_analysis(tr_v15, c_labeled, out)

    # Isolated per-symbol dataset generation.
    for symbol in symbols:
        cf = pd.concat(per_symbol_candidates[symbol], ignore_index=True) if per_symbol_candidates[symbol] else pd.DataFrame()
        tf = pd.concat(per_symbol_trades[symbol], ignore_index=True) if per_symbol_trades[symbol] else pd.DataFrame()
        dataset = _build_symbol_dataset(cf, tf, symbol)
        _write_symbol_datasets(symbol, dataset, cf, tf, out)
        print(f"[{symbol}] Dataset rows={len(dataset):,} | candidates={len(cf):,} | trade-scenario rows={len(tf):,}")

    _write_dataset_manifest(symbols, out)

    report = (
        "SMC MTF RESEARCH V15\n" + "=" * 100 + "\n"
        + f"Period: {PERIOD_START.isoformat()} -> {PERIOD_END.isoformat()}\n"
        + f"Symbols requested: {', '.join(symbols)}\n"
        + f"Families: {', '.join(FAMILIES)}\n"
        + f"Version: V15 ({V15_PROFILE})\n"
        + f"Feature layer: {FEATURE_LAYER_VERSION}\n"
        + f"RR: {RR_VALUES}\n"
        + f"SL modes: {', '.join(SL_MODES)}\n"
        + f"Max hold: {MAX_HOLD_MINUTES} min\n"
        + "Data layer unchanged; Tick cache preferred, CSV fallback and symbol-input/download workflow unchanged.\n"
        + "V15 candidate expansion: sweep 0.45 ATR; A retest age <=4; B BOS <=0.22 ATR; C BOS <=0.40 ATR.\n"
        + "Structural safeguards unchanged: displacement, confirmation, sweep/retest recency, HTF alignment and M15 veto.\n"
        + "No ML training/inference. Feature vectors are causal and exported per symbol.\n"
        + "Each per-symbol dataset has one row per unique candidate; RR/SL outcomes are columns, not duplicated candidates.\n"
        + "No previous-version result matrix is loaded, compared, or written.\n"
    )
    (out / "SMC_V15_RUN_REPORT.txt").write_text(report, encoding="utf-8")

    print(f"\nV15 BACKTEST + DATASET EXPORT COMPLETE. Results: {out}")
    print(f"Per-symbol datasets: {out / 'datasets'}")
    if errors:
        print(f"Errors: {len(errors)} (see SMC_V15_ALL_ERRORS.csv)")
    if data_gaps:
        print(f"Data gaps > {DATA_GAP_WARN_DAYS} days: {len(data_gaps)} (see SMC_V15_DATA_GAPS.csv)")


if __name__ == "__main__":
    main()
