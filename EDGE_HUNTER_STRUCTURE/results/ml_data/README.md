# SMC_V15 ML trade filters

Two independent per-symbol models that score each SMC_V15 trade with a win probability:

| Script | Model |
|---|---|
| `ml_logreg_l2.py` | Logistic Regression, L2 penalty |
| `ml_catboost.py` | CatBoost Classifier |

```bash
pip install -r requirements-ml.txt
python ml_logreg_l2.py train                       # all symbols in results/SMC_V15/datasets
python ml_catboost.py train --symbol XAUUSD        # one symbol
python ml_catboost.py predict --symbol XAUUSD --input new_trades.csv
```

## Method
- **Label:** `pnl_r > 0` (winner) vs `pnl_r <= 0` (loser), from `SMC_V15_<SYMBOL>_TRADES.csv`.
- **Per symbol:** each symbol is trained and saved separately. A model only scores its own symbol.
- **Split:** chronological, with the first 80% of setups for training and the last 20% for validation. There is no shuffling, because every setup appears 6 times (3 RR × 2 SL modes) and a random split would leak. An embargo drops any training trade still open when validation starts.
- **Features:** all causal setup and market features, `rr`, `sl_mode`, `risk_pct` and `confluence_count`, plus BUY/SELL-aligned versions of the directional features. Outcomes, timestamps, IDs and raw price levels are excluded.
- **Tuning:** the hyper-parameters, tree count and probability threshold are chosen with time-ordered CV inside the 80% training data only. The threshold maximises out-of-fold net R while keeping at least 50% of winners.
- **Safety guard:** the filter is only switched on (`edge_detected: true`) when the out-of-fold AUC is at least 0.55 and filtering improved out-of-fold net R. Otherwise `deployed_threshold = 0`, which keeps every trade.

## Output per symbol: `<SYMBOL>/<model>/`
- `model.joblib` / `model.cbm`: the trained model
- `meta.json`: features, parameters, periods, validation metrics, deployed threshold
- `feature_importance.csv`: LR coefficients (standardised) or CatBoost importances
- `validation_predictions.csv`: prediction for every validation trade
- `validation_threshold_curve.csv`: trades, winners kept and losers removed at every threshold

`SUMMARY_logreg_l2.csv` and `SUMMARY_catboost.csv` compare the baseline with the filtered result on the validation 20%.
