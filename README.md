# 24-Hour PM2.5 Forecasting with LightGBM

Forecasting hourly PM2.5 concentrations for the next 24 hours in a single city, using about 3.4 years of hourly weather and air-quality data. The project uses a **direct multi-horizon** approach with gradient boosting and evaluates it against naive baselines on a held-out, time-ordered test set.

## Overview

Air pollution builds up and clears over hours to days, depending on recent pollution levels, weather, and seasonal events such as crop burning and festivals. This project builds one LightGBM model for each forecast horizon (t+1 to t+24 hours) and measures how much each model improves on simple baselines like "same hour yesterday".

**Highlights**
- 29,040 hourly rows (Aug 2022 to Nov 2025), with no missing hours
- 24 direct models, one per horizon, avoiding the error build-up of recursive forecasting
- Leakage-aware design: only information available at forecast time is used, plus calendar flags for the target hour
- Chronological train / validation / test split with a 24-hour gap, and the test set covers winter
- Evaluation against persistence and same-hour-yesterday baselines, with separate metrics for high-pollution hours and AQI-category accuracy

## Data

Hourly records with 25 columns:

| Group | Columns |
|---|---|
| Time | `datetime`, `year`, `month`, `hour`, `day_of_week` |
| Weather | temperature, humidity, wind speed / direction / gusts, `wind_stagnation`, precipitation, pressure, solar radiation, cloud cover |
| Pollutants | `pm2_5`, `pm10`, `co`, `no2`, `so2`, `o3`, `dust`, `aod` |
| Event flags | `festival_period`, `crop_burning_season` |

Data source: _add your source and city here_.

## Approach

**Target.** PM2.5 at t+h for h = 1..24, modelled on a `log1p` scale to handle the right-skewed distribution (median about 28 µg/m³, max about 581 µg/m³). Metrics are computed back on the original scale.

**Features (all known at forecast time t)**
- Lags of PM2.5 and PM10 (1 to 168 hours) and lags of NO2, CO, SO2, O3
- Rolling mean, max, std and trend of PM2.5 over 6 to 168 hours
- Wind direction converted to u/v components; rolling wind, stagnation, humidity, temperature, solar and pressure-change features
- Accumulated rainfall over 6, 24 and 72 hours as a washout signal
- Cyclical hour and day-of-year encodings
- Calendar information for the **target** hour (hour, weekday, month, festival and burning flags), which is legitimately known in advance

**Model.** `LGBMRegressor` with a low learning rate (0.03), subsampling, and early stopping on the validation set. A final refit on train + validation is saved for inference.

**Validation.** Chronological split (no shuffling):

| Split | Period |
|---|---|
| Train | Aug 2022 to May 2024 |
| Validation | Jun to Dec 2024 |
| Test | Jan to Nov 2025 |

A 24-hour gap at each boundary prevents targets from leaking across splits.

## Results

Test-set performance (MAE in µg/m³). "Best baseline" is the better of persistence and same-hour-yesterday for that horizon.

| Horizon | Model MAE | Model RMSE | Best baseline MAE | AQI-category accuracy |
|---|---|---|---|---|
| +1 h | 1.88 | 3.21 | 2.04 | 0.94 |
| +6 h | 5.52 | 8.32 | 7.63 | 0.83 |
| +12 h | 6.66 | 10.22 | 7.82 | 0.80 |
| +18 h | 7.24 | 11.05 | 7.82 | 0.78 |
| +24 h | 7.54 | 11.51 | 7.81 | 0.77 |

Averaged over all 24 horizons, the model reaches an MAE of **6.20** against **7.82** for same-hour-yesterday.

**Takeaways**
- Gains are largest at short and medium horizons (about 15 to 28% better than the best baseline at 6 to 12 hours).
- At +24 h the improvement is modest (about 3.5%): without forecast weather for the target hour, the model has little information beyond the current level and the season.
- The current PM2.5 value and its recent lags dominate feature importance at every horizon.

## Limitations

- **Spikes are underpredicted.** On hours where PM2.5 exceeds 90 µg/m³, the model's MAE (26.2) is slightly worse than the yesterday baseline (24.0), with a strong negative bias. Tree ensembles and a log-scale loss both shrink extreme values.
- **No future weather.** Only observed history is used, so next-day wind, rain and mixing conditions are not captured.
- **Single station, one winter in the test set**, so peak-hour metrics are noisy.
- Per-horizon models can give slightly jagged hour-to-hour forecasts.

## Planned improvements

- Add historical weather *forecasts* for the target hour as features
- Reduce peak bias (raw-scale or Huber loss, change-based targets, sample weighting, quantile models)
- Single stacked model with horizon as a feature for smoother forecasts
- Prediction intervals via quantile regression
- Alert-style evaluation (recall and precision for exceeding a threshold)
- Comparison with a small GRU / TCN and an ensemble

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo>
pip install numpy pandas scikit-learn lightgbm joblib
```

1. Place your data file in the project folder and set `CSV_PATH` in the script.
2. Run the training script. It trains all 24 models, prints the per-horizon metrics, and saves `pm25_24h_lgbm.joblib`.
3. Use `forecast_next_24h(history_df)` to produce a 24-hour forecast from at least ~200 hours of recent data.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, joblib

## Author

Harsh Panindre
