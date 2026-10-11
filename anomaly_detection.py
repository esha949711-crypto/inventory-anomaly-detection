"""Unsupervised inventory anomaly models and business explanations."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.metrics import silhouette_score


def robust_minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def baseline_iqr_score(X):
    q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
    iqr = q3 - q1 + 1e-9
    return np.max(np.maximum((q1 - X) / iqr, (X - q3) / iqr), axis=1)


def fit_models(X, random_state=42):
    # KDE is fit on a capped stratified-ish sample for practical inference speed.
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X), size=min(len(X), 6000), replace=False)
    kde = KernelDensity(bandwidth=0.65, kernel="gaussian").fit(X[sample_idx])
    gmm = GaussianMixture(n_components=4, covariance_type="diag", random_state=random_state, reg_covar=1e-4, max_iter=150).fit(X)
    pca = PCA(n_components=2, random_state=random_state).fit(X)
    kmeans = KMeans(n_clusters=5, random_state=random_state, n_init=10).fit(X)
    return {"kde": kde, "gmm": gmm, "pca": pca, "kmeans": kmeans}


def score_models(models, X):
    kde_raw = -models["kde"].score_samples(X)
    gmm_raw = -models["gmm"].score_samples(X)
    cluster_dist = models["kmeans"].transform(X).min(axis=1)
    baseline_raw = baseline_iqr_score(X)
    score = (0.30 * robust_minmax(kde_raw) + 0.30 * robust_minmax(gmm_raw) + 0.20 * robust_minmax(cluster_dist) + 0.20 * robust_minmax(baseline_raw))
    pca_xy = models["pca"].transform(X)
    labels = models["kmeans"].predict(X)
    return {"anomaly_score": np.clip(score, 0, 1), "kde_score": robust_minmax(kde_raw), "gmm_score": robust_minmax(gmm_raw), "kmeans_score": robust_minmax(cluster_dist), "baseline_score": robust_minmax(baseline_raw), "pca_1": pca_xy[:, 0], "pca_2": pca_xy[:, 1], "cluster": labels}


def explain_record(row):
    reasons = []
    if row.get("Stock_to_Sales_Ratio", 0) >= 8: reasons.append("stock is high relative to daily sales")
    if row.get("Stock_to_Sales_Ratio", 0) <= 1.5 and row.get("Units Sold", 0) > 0: reasons.append("stock coverage is low relative to sales")
    if row.get("Units Sold", 0) > row.get("Demand Forecast", 0) * 1.45: reasons.append("sales are much higher than forecast")
    if row.get("Units Sold", 0) < row.get("Demand Forecast", 0) * 0.45: reasons.append("sales are much lower than forecast")
    if row.get("Order_to_Sales_Ratio", 0) >= 3: reasons.append("order quantity is large relative to sales")
    if abs(row.get("Demand_Change", 0)) > max(20, row.get("Rolling_Sales_Average", 0) * 0.75): reasons.append("demand changed sharply versus the previous observation")
    if row.get("Days_of_Inventory", 0) > 30: reasons.append("inventory has unusually long coverage")
    if row.get("Price_Gap", 0) > 20: reasons.append("price is notably above competitor pricing")
    if not reasons: reasons.append("the combined inventory pattern is rare compared with learned normal behavior")
    return "; ".join(reasons[:3])


def recommend_action(row):
    if row.get("Stock_to_Sales_Ratio", 0) >= 8: return "Investigate overstock and review replenishment plan."
    if row.get("Stock_to_Sales_Ratio", 0) <= 1.5 and row.get("Units Sold", 0) > 0: return "Review replenishment requirement and stockout risk."
    if row.get("Order_to_Sales_Ratio", 0) >= 3: return "Review purchase/order quantity against demand."
    if row.get("Days_of_Inventory", 0) > 30: return "Investigate slow-moving or aging inventory."
    if abs(row.get("Demand_Change", 0)) > max(20, row.get("Rolling_Sales_Average", 0) * 0.75): return "Validate demand change and adjust planning assumptions."
    return "Review record with the warehouse or supply-chain owner."


def model_quality(models, X):
    sample = X[np.random.default_rng(42).choice(len(X), min(len(X), 5000), replace=False)]
    return {"silhouette_score": float(silhouette_score(sample, models["kmeans"].predict(sample))) if len(np.unique(models["kmeans"].labels_)) > 1 else 0.0}
