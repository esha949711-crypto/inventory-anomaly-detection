# Machine Learning-Based Inventory Anomaly Detection System

## 1. Introduction
Inventory teams need an early-warning view of records that do not look like the normal operating pattern. This project combines operational data, unsupervised machine learning, and Flask decision support.

## 2. Business problem
Managers may miss overstock, stockout risk, sudden demand changes, unusual orders, or slow-moving inventory when reviewing thousands of daily rows.

## 3. Why anomaly detection?
The Kaggle source has no trusted ground-truth anomaly label. The system therefore learns structure from the observations and ranks unusual records. It does not turn the task into classification.

## 4. Dataset
The original Kaggle Retail Store Inventory Forecasting Dataset contains 73,100 synthetic-but-realistic daily store/product observations and is CC0. See the README data dictionary for every original and derived field.

## 5. Preprocessing and feature engineering
Invalid numeric values are converted safely and median-imputed; duplicates are removed; potential anomalies are not filtered out before modeling. Robust scaling limits the influence of extremes. Ratios use safe division. Derived signals include stock-to-sales ratio, days of inventory, order-to-sales ratio, demand change, rolling sales average, forecast gap, price gap, inventory value, and turnover.

## 6. Algorithms
The IQR method is a simple baseline. KDE estimates local density. GMM estimates likelihood under a mixture of normal patterns. K-Means supplies cluster-distance evidence. PCA compresses the feature space for visual analysis. The combined score is standardized from 0 to 1.

## 7. Threshold and decision support
The default is the 95th percentile of training scores. A higher threshold reduces alerts but may miss unusual records; a lower threshold catches more records but increases investigation workload. Reasons are phrased as indicator evidence, not exact causal explanations.

## 8. Flask application
Training is separate from inference. Joblib artifacts are loaded once per request, while upload validation protects the workflow from missing columns, empty data, invalid types, and oversized files. The dashboard exposes KPIs, charts, filters, detail pages, and CSV download.

## 9. Limitations
Synthetic data does not represent a real company; unsupervised models can produce false positives; products have different normal patterns; and anomalies require human validation. An anomaly is not automatically fraud.

## 10. Future improvements
Add real company data, time-aware retraining, automated alerts, supplier risk, demand forecasting, stockout prediction, database integration, role-based access, and production monitoring.
