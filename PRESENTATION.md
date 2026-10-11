# Presentation: Inventory Anomaly Detection

## Slide 1 — Project title
- Machine Learning-Based Inventory Anomaly Detection System
- Flask + unsupervised machine learning

**Say:** This project helps inventory managers prioritize records that look unusual.

## Slide 2 — Business problem
- Overstock, low coverage, unusual orders, demand changes
- Manual review does not scale

**Say:** The goal is not to accuse anyone; it is to direct attention to the right records.

## Slide 3 — Objective
- Learn normal multivariate behavior
- Produce a 0–1 anomaly score
- Recommend an investigation action

**Say:** The workflow is observe, detect, investigate, decide, act, measure.

## Slide 4 — Why unsupervised?
- No trusted Normal/Anomaly target in the source
- Model learns patterns instead of fabricated labels

**Say:** This is not ordinary classification or regression.

## Slide 5 — Dataset
- Kaggle Retail Store Inventory Forecasting Dataset
- 73,100 observations, CC0, synthetic-but-realistic
- Store, product, region, stock, sales, orders, forecast, price

**Say:** Synthetic means it is useful for education but not a claim about real company performance.

## Slide 6 — Preprocessing
- Validate required columns and types
- Drop duplicates; median-impute missing numeric values
- Robust scaling
- Preserve valid unusual observations

**Say:** Removing outliers before anomaly detection would remove the target signal.

## Slide 7 — Feature engineering
- Stock-to-sales ratio and days of inventory
- Order-to-sales ratio
- Demand change and rolling sales average
- Forecast gap, price gap, inventory value, turnover

**Say:** Ratios make business meaning visible and use safe zero-division handling.

## Slide 8 — Algorithms
- IQR baseline
- KDE density
- GMM likelihood
- K-Means distance
- PCA visualization

**Say:** Each method sees unusualness differently; the combined score is more useful than one rule.

## Slide 9 — Score and threshold
- Standardized score from 0 to 1
- 95th percentile threshold
- Current result: 5.0% flagged

**Say:** A lower threshold raises coverage and workload; a higher threshold creates fewer alerts.

## Slide 10 — Flask dashboard
- KPI cards, charts, table filters
- Detail page, CSV upload, downloadable results

**Say:** Training and inference are separate; the app loads saved Joblib artifacts.

## Slide 11 — Decision support
- High coverage → review overstock
- Low coverage → review replenishment
- Large order ratio → review purchase plan
- Demand shift → validate demand signal

**Say:** Recommendations are suggestions for human investigation.

## Slide 12 — Limitations and conclusion
- Synthetic data and false positives
- Product-specific patterns and threshold sensitivity
- Next: real data, alerts, monitoring, access control

**Say:** The project demonstrates a technically correct, responsible end-to-end anomaly workflow.
