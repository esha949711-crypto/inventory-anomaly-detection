# Terminology Sheet

| Term | Simple definition | Project meaning | Example |
|---|---|---|---|
| Machine Learning | Learning patterns from data | Learns inventory behavior | Score unusual rows |
| Unsupervised Learning | Learning without target labels | Main problem type | KDE learns density |
| Anomaly Detection | Finding unusual observations | Flags inventory patterns | Unusual stock/sales |
| Inventory | Goods held by a business | Stock level field | 300 units |
| SKU | Product identifier | Product ID | P0001 |
| Feature | Input variable | Sales, stock, ratios | Units Sold |
| Observation | One data row | Store-product-date record | One daily row |
| Dataset | Collection of observations | Kaggle CSV | 73,100 rows |
| Preprocessing | Preparing data | Imputation and scaling | RobustScaler |
| Feature Engineering | Creating useful variables | Ratios and change | Days of inventory |
| Outlier | Extreme value | Candidate unusual point | Very high stock |
| Anomaly Score | Unusualness measure | 0–1 combined score | 0.91 |
| Threshold | Decision cutoff | 95th percentile | Score ≥ cutoff |
| KDE | Kernel density estimate | Low density = unusual | Sparse pattern |
| GMM | Gaussian mixture model | Low likelihood = unusual | Rare combination |
| PCA | Dimensionality reduction | 2D visual support | PC1/PC2 |
| K-Means | Clustering algorithm | Cluster distance evidence | Far from center |
| Clustering | Grouping similar rows | K-Means groups behavior | Five clusters |
| Density Estimation | Estimating concentration | KDE probability density | Dense/rare regions |
| Model Training | Fitting a model | Offline artifact creation | train_model.py |
| Model Inference | Applying a saved model | Upload scoring | app.py |
| Flask | Python web framework | Serves dashboard | localhost:5000 |
| API | Program interface | Future extension point | JSON scoring endpoint |
| Dashboard | Visual control screen | KPIs and charts | Anomalies page |
| KPI | Key performance indicator | Stock, sales, turnover | Average turnover |
| Data Leakage | Improper future/target use | Avoided with history-only features | No future sales |
| Synthetic Data | Artificial but realistic data | Kaggle source description | Educational dataset |
