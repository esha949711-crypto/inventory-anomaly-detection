"""Evaluation/reporting helpers. No classification accuracy is used."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anomaly_detection import model_quality


def evaluate(df, models, X, scores, threshold):
    anomaly = scores['anomaly_score'] >= threshold
    report = {
        'records': int(len(df)), 'threshold': float(threshold),
        'anomalies': int(anomaly.sum()), 'anomaly_percentage': float(anomaly.mean() * 100),
        'mean_score': float(scores['anomaly_score'].mean()),
        'model_comparison': {k: float((scores[k] >= np.quantile(scores[k], 0.95)).mean() * 100) for k in ['kde_score','gmm_score','kmeans_score','baseline_score']},
        **model_quality(models, X),
    }
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/evaluation.json').write_text(json.dumps(report, indent=2))
    return report


def make_eda_plots(df, output_dir='static/plots'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid')
    plots = {}
    def save(name):
        path = str(Path(output_dir) / name)
        plt.tight_layout(); plt.savefig(path, dpi=130, bbox_inches='tight'); plt.close(); return '/' + path
    plt.figure(figsize=(7,4)); sns.histplot(df['Units Sold'], bins=40, color='#2dd4bf'); plt.title('Daily sales distribution'); plots['sales_dist']=save('sales_distribution.png')
    plt.figure(figsize=(7,4)); sns.scatterplot(data=df.sample(min(5000,len(df)), random_state=42), x='Units Sold', y='Inventory Level', hue='Category', alpha=.45, legend=False); plt.title('Sales vs inventory'); plots['sales_stock']=save('sales_vs_stock.png')
    nums=['Inventory Level','Units Sold','Units Ordered','Demand Forecast','Price','Discount']
    plt.figure(figsize=(8,6)); sns.heatmap(df[nums].corr(), cmap='vlag', center=0, annot=True, fmt='.1f'); plt.title('Operational feature correlation'); plots['correlation']=save('correlation_heatmap.png')
    trend=df.groupby('Date', as_index=False).agg(Units_Sold=('Units Sold','sum'), Inventory=('Inventory Level','mean'))
    plt.figure(figsize=(9,4)); plt.plot(trend['Date'], trend['Units_Sold'], label='Units sold', color='#14b8a6'); plt.twinx().plot(trend['Date'], trend['Inventory'], label='Average inventory', color='#f59e0b'); plt.title('Sales and inventory trend'); plots['trend']=save('time_trend.png')
    return plots
