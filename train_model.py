"""Train the unsupervised inventory anomaly pipeline once, then serve saved artifacts."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from feature_engineering import engineer_features
from preprocessing import build_preprocessor, transform_features
from anomaly_detection import fit_models, score_models, explain_record, recommend_action
from evaluation import evaluate, make_eda_plots

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'retail_store_inventory.csv'
MODEL_DIR = ROOT / 'models'


def train(data_path=DATA_PATH):
    MODEL_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(data_path)
    df = engineer_features(raw)
    preprocessor = build_preprocessor()
    preprocessor.fit(df[__import__('feature_engineering').FEATURE_COLUMNS])
    X = transform_features(df, preprocessor)
    models = fit_models(X)
    scores = score_models(models, X)
    # A percentile threshold is configurable in the app; 95th percentile is the
    # documented default and avoids a random hard-coded cutoff.
    threshold = float(np.quantile(scores['anomaly_score'], 0.95))
    result = df.copy()
    for key, val in scores.items(): result[key] = val
    result['Anomaly Score'] = result['anomaly_score'].round(6)
    result['Status'] = np.where(result['anomaly_score'] >= threshold, 'ANOMALY', 'NORMAL')
    result['Key Indicators'] = result.apply(explain_record, axis=1)
    result['Recommended Action'] = result.apply(recommend_action, axis=1)
    result['Inventory_Turnover'] = (result['Units Sold'] / result['Inventory Level'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    result.to_csv(ROOT / 'outputs' / 'anomaly_results.csv', index=False)
    joblib.dump(preprocessor, MODEL_DIR / 'preprocessing.pkl')
    joblib.dump(models, MODEL_DIR / 'models.pkl')
    metadata = {
        'data_source': 'Kaggle: Retail Store Inventory Forecasting Dataset',
        'source_url': 'https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset',
        'license': 'CC0 Public Domain', 'rows': int(len(df)), 'features': list(X.shape),
        'threshold': threshold, 'threshold_percentile': 95,
        'feature_columns': list(df.columns), 'model_weights': {'kde': .30, 'gmm': .30, 'kmeans': .20, 'baseline_iqr': .20},
    }
    (MODEL_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    report = evaluate(df, models, X, scores, threshold)
    plots = make_eda_plots(raw.assign(Date=pd.to_datetime(raw['Date'])))
    metadata['report'] = report; metadata['plots'] = plots
    (MODEL_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'rows': len(df), 'threshold': threshold, 'anomalies': int((result['Status']=='ANOMALY').sum()), 'report': report}, indent=2))
    return result

if __name__ == '__main__':
    train()
