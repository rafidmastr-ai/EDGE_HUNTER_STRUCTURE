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
- **Features:** all causal setup and market features, `rr`, `sl_mode` and `confluence_count`, plus BUY/SELL-aligned versions of the directional features. Outcomes, timestamps, IDs, raw price levels and price-scaled features (MACD values, `macd_hist_slope_3`) are excluded, because their size changes with the price level.
- **Logistic Regression:** each continuous feature is cut into 5 quantile bins, so the linear model can learn non-linear effects. It also gets missing-value indicators, and each setup gets weight 1/n so its 6 RR/SL copies count once. C is tuned by CV.
- **CatBoost:** the tree count is fixed and chosen by CV (depth 4/6, 300–800 trees, `l2_leaf_reg=30`). There is no early stopping: on this noisy data it halted after 2–5 trees, which gives an almost constant model.
- **Threshold:** chosen inside the 80% training data to maximise out-of-fold net R while keeping at least 50% of winners.
- **Filter switch (`filter_deployed`):** the filter is on only when all of the following hold. Otherwise `deployed_threshold = 0`, which keeps every trade.
  1. Out-of-fold AUC is at least 0.53.
  2. Out-of-fold net R improves overall and in at least 3 of the 5 CV periods.
  3. Net R also improves on the untouched 20% validation period. Validation is used only for this yes/no gate and never for tuning.

## Diagnosis (October 2026 data)
- The 6 RR/SL rows of a setup have almost the same label (correlation about 0.9). The real sample is the number of setups (814–3,558), not the number of rows.
- After removing the duplicate rows, per-feature drift between train and validation is modest. The exception is the price-scaled features, which were removed.
- XAUUSD has no data for 2021–2024. Its relationships flip between 2020 and 2025–2026, and the linear model's out-of-fold AUC is below 0.5.
- Both models reach an out-of-fold AUC of about 0.53–0.56 inside training, but this does not reliably carry over to the most recent 20%. With the current features, signal is weak.

## Output per symbol: `<SYMBOL>/<model>/`
- `model.joblib` / `model.cbm`: the trained model
- `meta.json`: features, parameters, periods, validation metrics, deployed threshold
- `feature_importance.csv`: LR coefficients (standardised) or CatBoost importances
- `validation_predictions.csv`: prediction for every validation trade
- `validation_threshold_curve.csv`: trades, winners kept and losers removed at every threshold

`SUMMARY_logreg_l2.csv` and `SUMMARY_catboost.csv` compare the baseline with the filtered result on the validation 20%.
