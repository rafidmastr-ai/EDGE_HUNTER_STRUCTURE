"""
SMC_V15 trade filter — Logistic Regression (L2).

Learns, per symbol, which SMC_V15 trades tend to win and which tend to lose,
then keeps only trades whose predicted win probability is above a threshold.

Each symbol is trained and saved on its own: a model trained on XAUUSD is only
ever used to predict XAUUSD trades. New symbols are picked up automatically as
soon as results/SMC_V15/datasets/<SYMBOL>/SMC_V15_<SYMBOL>_TRADES.csv exists.

Usage:
    python ml_logreg_l2.py train                     # every symbol found
    python ml_logreg_l2.py train --symbol XAUUSD     # one symbol
    python ml_logreg_l2.py predict --symbol XAUUSD --input new_trades.csv [--output scored.csv]

Split: chronological 80% train / 20% validation (no shuffling, the validation
period is strictly after the training period, with an embargo so no training
trade is still open when validation starts).
The decision threshold is chosen on out-of-fold predictions inside the 80%
training data only; the 20% validation set is touched once, for the report.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_NAME = "logreg_l2"
HERE = Path(__file__).resolve().parent
DATASETS_DIR = HERE / "results" / "SMC_V15" / "datasets"
ML_DIR = HERE / "results" / "ml_data"

TRAIN_FRACTION = 0.80
CV_SPLITS = 5
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
MIN_WINNERS_KEPT = 0.50  # threshold may not throw away more than half of the winners
MIN_TRADES_KEPT = 0.10
# Safety guard: the filter is only switched on when the training data shows a real edge
# (out-of-fold AUC and out-of-fold net-R improvement). Otherwise every trade is kept.
MIN_EDGE_AUC = 0.55
RANDOM_STATE = 42

# Columns that are outcomes, identifiers, timestamps or raw price levels.
EXCLUDED = {
    "symbol", "filter_profile", "signal_time", "entry_time", "exit_time", "setup_id",
    "result", "pnl_r", "cost_model", "entry", "sl", "tp",
    "utc_hour", "day_of_week",                      # cyclic sin/cos versions are used instead
    "macd_line", "macd_signal", "macd_hist",       # raw price units; side-adjusted/ATR versions kept
}
CATEGORICAL = ["side", "family", "utc_block", "candidate_tier", "candidate_gate_reason", "sl_mode"]

# Direction-dependent features: for a SELL the "good" sign is reversed, so a linear
# model needs them multiplied by +1 (BUY) / -1 (SELL). Raw versions are dropped.
DIRECTIONAL = [
    "ret_1", "ret_3", "ret_30", "ret_5_atr", "ret_15_atr", "dist_ema20_atr", "dist_ema50_atr",
    "dist_ema200_atr", "ema20_50_gap_atr", "ema50_200_gap_atr", "ema20_slope_5_atr",
    "ema50_slope_10_atr", "dist_vwap60_atr", "rsi14_slope_3", "macd_hist_slope_3",
    "m5_dir", "m15_dir", "plus_di14", "minus_di14",
]
CENTERED_DIRECTIONAL = {"rsi14": 50.0, "stoch_k14": 50.0, "bb_percent_b": 0.5, "close_location_3": 0.5}


# ----------------------------------------------------------------------------- data
def available_symbols() -> list[str]:
    return sorted(p.parent.name for p in DATASETS_DIR.glob("*/SMC_V15_*_TRADES.csv"))


def load_trades(symbol: str) -> pd.DataFrame:
    base = DATASETS_DIR / symbol / f"SMC_V15_{symbol}_"
    df = pd.read_csv(f"{base}TRADES.csv", encoding="utf-8-sig")
    dataset = Path(f"{base}DATASET.csv")
    if dataset.exists():  # causal confluence count lives only in the candidate dataset
        conf = pd.read_csv(dataset, encoding="utf-8-sig", usecols=["setup_id", "confluence_count"])
        df = df.merge(conf.drop_duplicates("setup_id"), on="setup_id", how="left")
    for col in ("signal_time", "exit_time"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "confluence_count" not in x:
        x["confluence_count"] = np.nan
    sign = np.where(x["side"].astype(str).str.upper() == "BUY", 1.0, -1.0)
    for col in DIRECTIONAL:
        if col in x:
            x[f"sa_{col}"] = pd.to_numeric(x[col], errors="coerce") * sign
    for col, center in CENTERED_DIRECTIONAL.items():
        if col in x:
            x[f"sa_{col}"] = (pd.to_numeric(x[col], errors="coerce") - center) * sign
    entry = pd.to_numeric(x["entry"], errors="coerce")
    x["risk_pct"] = (entry - pd.to_numeric(x["sl"], errors="coerce")).abs() / entry * 100.0
    x["rr"] = pd.to_numeric(x["rr"], errors="coerce")
    for col in CATEGORICAL:
        x[col] = x[col].astype(str) if col in x else "NA"
    return x


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    dropped = EXCLUDED | set(DIRECTIONAL) | set(CENTERED_DIRECTIONAL) | set(CATEGORICAL)
    numeric = [c for c in df.columns if c not in dropped and pd.api.types.is_numeric_dtype(df[c])]
    return numeric, list(CATEGORICAL)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["signal_time", "setup_id", "rr", "sl_mode"]).reset_index(drop=True)
    times = np.sort(df["signal_time"].unique())
    cut = times[int(len(times) * TRAIN_FRACTION)]
    train = df[df["signal_time"] < cut]
    valid = df[df["signal_time"] >= cut]
    train = train[train["exit_time"] < cut]  # embargo: no training trade overlaps validation
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def time_folds(times: pd.Series, n_splits: int):
    """Expanding-window folds; all configs of one setup stay in the same fold."""
    uniq = np.sort(times.unique())
    rank = np.searchsorted(uniq, times.to_numpy())
    for tr_u, va_u in TimeSeriesSplit(n_splits=n_splits).split(uniq):
        tr = np.where(rank <= tr_u.max())[0]
        va = np.where((rank >= va_u.min()) & (rank <= va_u.max()))[0]
        cut = uniq[va_u.min()]
        tr = tr[times.iloc[tr].to_numpy() < cut]
        yield tr, va


# ---------------------------------------------------------------------------- model
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its training 1st/99th percentile (LR is sensitive to outliers)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_pipeline(numeric: list[str], categorical: list[str], C: float) -> Pipeline:
    num = Pipeline([
        ("clip", Winsorizer()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    pre = ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])
    # L2 is sklearn's default penalty; C is the inverse regularisation strength.
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                             solver="lbfgs", random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


def cross_validate(train: pd.DataFrame, numeric, categorical, y) -> tuple[float, np.ndarray]:
    """Pick C by out-of-fold AUC (ranking quality is what the filter needs); return C and OOF probabilities."""
    folds = list(time_folds(train["signal_time"], CV_SPLITS))
    best = (-np.inf, None, None)
    for C in C_GRID:
        oof = np.full(len(train), np.nan)
        for tr, va in folds:
            model = build_pipeline(numeric, categorical, C).fit(train.iloc[tr], y[tr])
            oof[va] = model.predict_proba(train.iloc[va])[:, 1]
        mask = ~np.isnan(oof)
        auc = roc_auc_score(y[mask], oof[mask])
        print(f"    C={C:<6} oof auc={auc:.4f}")
        if auc > best[0]:
            best = (auc, C, oof)
    return best[1], best[2]


# ------------------------------------------------------------------------- evaluation
def trade_stats(pnl: np.ndarray) -> dict:
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "trades": int(len(pnl)), "winners": int(len(wins)), "losers": int(len(losses)),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "net_r": float(pnl.sum()), "avg_r": float(pnl.mean()) if len(pnl) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
    }


def threshold_curve(prob: np.ndarray, pnl: np.ndarray) -> pd.DataFrame:
    total_w, total_l = (pnl > 0).sum(), (pnl <= 0).sum()
    rows = []
    for thr in np.round(np.arange(0.20, 0.80, 0.01), 2):
        keep = prob >= thr
        s = trade_stats(pnl[keep])
        s.update(threshold=thr,
                 winners_kept_pct=s["winners"] / total_w if total_w else 0.0,
                 losers_removed_pct=1 - s["losers"] / total_l if total_l else 0.0,
                 trades_kept_pct=keep.mean())
        rows.append(s)
    return pd.DataFrame(rows)


def choose_threshold(prob: np.ndarray, pnl: np.ndarray) -> float:
    """Max net R on out-of-fold data while keeping enough winners and trades."""
    curve = threshold_curve(prob, pnl)
    ok = curve[(curve.winners_kept_pct >= MIN_WINNERS_KEPT) & (curve.trades_kept_pct >= MIN_TRADES_KEPT)]
    if ok.empty:
        return float(curve.threshold.min())
    return float(ok.loc[ok.net_r.idxmax(), "threshold"])


def report(prob: np.ndarray, valid: pd.DataFrame, y: np.ndarray, threshold: float) -> dict:
    pnl = valid["pnl_r"].to_numpy(float)
    keep = prob >= threshold
    base, filt = trade_stats(pnl), trade_stats(pnl[keep])
    return {
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
        "threshold": threshold,
        "baseline": base,
        "filtered": filt,
        "winners_kept_pct": filt["winners"] / base["winners"] if base["winners"] else 0.0,
        "losers_removed_pct": 1 - filt["losers"] / base["losers"] if base["losers"] else 0.0,
    }


# ------------------------------------------------------------------------------ train
def train_symbol(symbol: str) -> dict:
    print(f"\n=== {symbol} — {MODEL_NAME}")
    df = add_features(load_trades(symbol))
    train, valid = chronological_split(df)
    numeric, categorical = feature_columns(df)
    y_tr = (train["pnl_r"] > 0).astype(int).to_numpy()
    y_va = (valid["pnl_r"] > 0).astype(int).to_numpy()
    print(f"  train {len(train)} trades ({train.signal_time.min()} → {train.signal_time.max()})")
    print(f"  valid {len(valid)} trades ({valid.signal_time.min()} → {valid.signal_time.max()})")

    C, oof = cross_validate(train, numeric, categorical, y_tr)
    mask = ~np.isnan(oof)
    oof_pnl = train["pnl_r"].to_numpy(float)[mask]
    threshold = choose_threshold(oof[mask], oof_pnl)
    oof_auc = float(roc_auc_score(y_tr[mask], oof[mask]))
    oof_gain = float(oof_pnl[oof[mask] >= threshold].sum() - oof_pnl.sum())
    edge = oof_auc >= MIN_EDGE_AUC and oof_gain > 0
    deployed_threshold = threshold if edge else 0.0
    model = build_pipeline(numeric, categorical, C).fit(train, y_tr)
    prob = model.predict_proba(valid)[:, 1]
    metrics = report(prob, valid, y_va, threshold)

    out = ML_DIR / symbol / MODEL_NAME
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    names = model.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": model.named_steps["clf"].coef_[0]})
    coefs.reindex(coefs.coef.abs().sort_values(ascending=False).index).to_csv(out / "feature_importance.csv", index=False)
    threshold_curve(prob, valid["pnl_r"].to_numpy(float)).to_csv(out / "validation_threshold_curve.csv", index=False)
    valid.assign(prob_win=prob, take_trade=prob >= deployed_threshold)[
        ["setup_id", "signal_time", "side", "family", "rr", "sl_mode", "result", "pnl_r", "prob_win", "take_trade"]
    ].to_csv(out / "validation_predictions.csv", index=False)
    meta = {
        "model": MODEL_NAME, "symbol": symbol, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "pnl_r > 0", "split": "chronological 80/20 with embargo",
        "train_period": [train.signal_time.min(), train.signal_time.max()],
        "valid_period": [valid.signal_time.min(), valid.signal_time.max()],
        "train_trades": len(train), "valid_trades": len(valid), "C": C,
        "numeric_features": numeric, "categorical_features": categorical,
        "threshold_rule": f"max out-of-fold net R, winners kept >= {MIN_WINNERS_KEPT:.0%}",
        "oof_auc": oof_auc, "oof_net_r_gain": oof_gain,
        "edge_detected": edge,
        "deployed_threshold": deployed_threshold,
        "edge_rule": f"oof AUC >= {MIN_EDGE_AUC} and positive oof net-R gain; otherwise keep all trades",
        "validation": metrics,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    b, f = metrics["baseline"], metrics["filtered"]
    print(f"  C={C} threshold={threshold:.2f} AUC={metrics['auc']:.3f}")
    print(f"  validation baseline: {b['trades']} trades, WR {b['win_rate']:.1%}, net {b['net_r']:+.1f}R, PF {b['profit_factor']:.2f}")
    print(f"  validation filtered: {f['trades']} trades, WR {f['win_rate']:.1%}, net {f['net_r']:+.1f}R, PF {f['profit_factor']:.2f}")
    print(f"  winners kept {metrics['winners_kept_pct']:.1%}, losers removed {metrics['losers_removed_pct']:.1%}")
    print(f"  edge detected: {edge} (oof AUC {oof_auc:.3f}, oof gain {oof_gain:+.1f}R) → deployed threshold {deployed_threshold:.2f}")
    return meta


def summary_row(meta: dict) -> dict:
    v = meta["validation"]
    b, f = v["baseline"], v["filtered"]
    return {
        "symbol": meta["symbol"], "model": meta["model"], "edge_detected": meta["edge_detected"],
        "oof_auc": round(meta["oof_auc"], 4), "valid_auc": round(v["auc"], 4),
        "candidate_threshold": v["threshold"], "deployed_threshold": meta["deployed_threshold"],
        "base_trades": b["trades"], "base_win_rate": round(b["win_rate"], 4), "base_net_r": round(b["net_r"], 2),
        "base_pf": round(b["profit_factor"], 3), "filt_trades": f["trades"], "filt_win_rate": round(f["win_rate"], 4),
        "filt_net_r": round(f["net_r"], 2), "filt_pf": round(f["profit_factor"], 3),
        "winners_kept_pct": round(v["winners_kept_pct"], 4), "losers_removed_pct": round(v["losers_removed_pct"], 4),
    }


# ---------------------------------------------------------------------------- predict
def predict(symbol: str, input_csv: Path, output_csv: Path | None) -> pd.DataFrame:
    out = ML_DIR / symbol / MODEL_NAME
    if not (out / "model.joblib").exists():
        raise SystemExit(f"No {MODEL_NAME} model for {symbol}. Train it first: python {Path(__file__).name} train --symbol {symbol}")
    meta = json.loads((out / "meta.json").read_text())
    model = joblib.load(out / "model.joblib")
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if "symbol" in df and not (df["symbol"].astype(str) == symbol).all():
        raise SystemExit(f"Input contains symbols other than {symbol}; each model only scores its own symbol.")
    x = add_features(df)
    for col in meta["numeric_features"]:
        if col not in x:
            x[col] = np.nan
    df["prob_win"] = model.predict_proba(x)[:, 1]
    df["take_trade"] = df["prob_win"] >= meta["deployed_threshold"]
    output_csv = output_csv or input_csv.with_name(f"{input_csv.stem}_{MODEL_NAME}_scored.csv")
    df.to_csv(output_csv, index=False)
    print(f"Scored {len(df)} rows → {output_csv} (kept {int(df.take_trade.sum())})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--symbol", action="append", help="symbol to train (repeatable); default: all")
    p = sub.add_parser("predict")
    p.add_argument("--symbol", required=True)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = ap.parse_args()

    if args.cmd == "train":
        symbols = [s.upper() for s in args.symbol] if args.symbol else available_symbols()
        rows = [summary_row(train_symbol(s)) for s in symbols]
        ML_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = ML_DIR / f"SUMMARY_{MODEL_NAME}.csv"
        if summary_path.exists():  # keep rows of symbols that were not retrained this run
            old = pd.read_csv(summary_path)
            rows = old[~old.symbol.isin(symbols)].to_dict("records") + rows
        pd.DataFrame(rows).sort_values("symbol").to_csv(summary_path, index=False)
        print(f"\nSummary → {summary_path}")
    else:
        predict(args.symbol.upper(), args.input, args.output)


if __name__ == "__main__":
    main()
