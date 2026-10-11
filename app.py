from pathlib import Path
import json
import io
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from feature_engineering import engineer_features, validate_columns, FEATURE_COLUMNS
from anomaly_detection import score_models, explain_record, recommend_action

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = 'inventory-anomaly-demo-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
MODEL_DIR = ROOT / 'models'; OUTPUT_DIR = ROOT / 'outputs'; UPLOAD_DIR = ROOT / 'uploads'
for p in [OUTPUT_DIR, UPLOAD_DIR]: p.mkdir(exist_ok=True)


def artifacts():
    required = [MODEL_DIR/'preprocessing.pkl', MODEL_DIR/'models.pkl', MODEL_DIR/'metadata.json']
    if not all(p.exists() for p in required): raise FileNotFoundError('Model artifacts are missing. Run python train_model.py first.')
    return joblib.load(required[0]), joblib.load(required[1]), json.loads(required[2].read_text())


def current_results():
    path = OUTPUT_DIR / 'current_results.csv'
    if not path.exists(): path = OUTPUT_DIR / 'anomaly_results.csv'
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path)


def view_rows(df, limit=200):
    if df.empty: return []
    out = df.copy(); out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return out.head(limit).replace({np.nan: None}).to_dict('records')


def summary(df):
    if df.empty: return {'total':0,'normal':0,'anomalies':0,'pct':0,'avg_stock':0,'avg_sales':0,'avg_turnover':0}
    anomaly = df.get('Status', pd.Series(dtype=str)).eq('ANOMALY')
    return {'total':len(df),'normal':int((~anomaly).sum()),'anomalies':int(anomaly.sum()),'pct':round(float(anomaly.mean()*100),2),'avg_stock':round(float(df['Inventory Level'].mean()),1),'avg_sales':round(float(df['Units Sold'].mean()),1),'avg_turnover':round(float(df['Inventory_Turnover'].mean()),2)}


def chart_data(df):
    if df.empty: return {}
    anomaly = df[df['Status']=='ANOMALY']
    product = anomaly.groupby('Product ID').size().sort_values(ascending=False).head(8)
    store = anomaly.groupby('Store ID').size().sort_values(ascending=False).head(8)
    date = anomaly.assign(Date=pd.to_datetime(anomaly['Date'])).groupby(anomaly['Date'].astype(str)).size().tail(30)
    return {'status':[int((df['Status']=='NORMAL').sum()), int((df['Status']=='ANOMALY').sum())], 'scores':df['Anomaly Score'].round(3).tolist()[:10000], 'product_labels':product.index.tolist(),'product_values':product.tolist(),'store_labels':store.index.tolist(),'store_values':store.tolist(),'trend_labels':date.index.tolist(),'trend_values':date.tolist(),'pca':{'x':df.get('pca_1',pd.Series()).tolist()[:3000],'y':df.get('pca_2',pd.Series()).tolist()[:3000],'status':df.get('Status',pd.Series()).tolist()[:3000]}}


def run_inference(raw, threshold=None):
    pre, models, metadata = artifacts()
    df = engineer_features(raw)
    X = pre.transform(df[FEATURE_COLUMNS])
    scores = score_models(models, X)
    threshold = float(threshold if threshold is not None else metadata['threshold'])
    for k,v in scores.items(): df[k] = v
    df['Anomaly Score'] = df['anomaly_score'].round(6)
    df['Status'] = np.where(df['anomaly_score'] >= threshold, 'ANOMALY', 'NORMAL')
    df['Key Indicators'] = df.apply(explain_record, axis=1)
    df['Recommended Action'] = df.apply(recommend_action, axis=1)
    # Derived turnover is a clear, project-specific KPI.
    df['Inventory_Turnover'] = (df['Units Sold'] / df['Inventory Level'].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    df.to_csv(OUTPUT_DIR/'current_results.csv', index=False)
    return df, threshold

@app.context_processor
def inject(): return {'project_name':'Inventory Intelligence','source_url':'https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset'}

@app.route('/')
def index():
    try: df=current_results(); meta=artifacts()[2]
    except FileNotFoundError: df=pd.DataFrame(); meta={}
    return render_template('index.html', summary=summary(df), meta=meta)

@app.route('/dashboard')
def dashboard():
    try: df=current_results(); meta=artifacts()[2]
    except FileNotFoundError as e: flash(str(e),'danger'); return redirect(url_for('index'))
    return render_template('dashboard.html', summary=summary(df), charts=chart_data(df), meta=meta)

@app.route('/anomalies')
def anomalies():
    df=current_results()
    if not df.empty:
        for col in ['Product ID','Store ID','Category','Status']:
            val=request.args.get(col.lower().replace(' ','_'))
            if val: df=df[df[col].astype(str).eq(val)]
        try:
            minimum=float(request.args.get('min_score','0')); df=df[df['Anomaly Score']>=minimum]
        except ValueError: pass
    return render_template('anomalies.html', rows=view_rows(df), total=len(df), filters={'products':sorted(df['Product ID'].dropna().astype(str).unique()) if not df.empty else [],'stores':sorted(df['Store ID'].dropna().astype(str).unique()) if not df.empty else []})

@app.route('/details/<int:row_id>')
def details(row_id):
    df=current_results()
    if row_id<0 or row_id>=len(df): flash('That inventory record was not found.','warning'); return redirect(url_for('anomalies'))
    return render_template('details.html', row=df.iloc[row_id].to_dict(), row_id=row_id)

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        file=request.files.get('file')
        if not file or not file.filename.lower().endswith('.csv'): flash('Please choose a valid CSV file.','danger'); return redirect(url_for('upload'))
        name=secure_filename(file.filename)
        path=UPLOAD_DIR/name; file.save(path)
        try:
            raw=pd.read_csv(path); missing=validate_columns(raw)
            if missing: flash('Missing required columns: ' + ', '.join(missing),'danger'); return redirect(url_for('upload'))
            df, threshold=run_inference(raw)
            flash(f'Analysis complete: {len(df):,} records scored at the saved 95th-percentile threshold ({threshold:.3f}).','success'); return redirect(url_for('results'))
        except Exception as exc: flash('The CSV could not be analyzed safely: ' + str(exc),'danger'); return redirect(url_for('upload'))
    return render_template('upload.html')

@app.route('/analyze')
def analyze():
    try: df, threshold=run_inference(pd.read_csv(ROOT/'data'/'retail_store_inventory.csv'))
    except Exception as exc: flash('Analysis failed: '+str(exc),'danger'); return redirect(url_for('index'))
    flash(f'Refreshed analysis for {len(df):,} observations using threshold {threshold:.3f}.','success'); return redirect(url_for('dashboard'))

@app.route('/results')
def results(): return render_template('results.html', rows=view_rows(current_results(), 100), summary=summary(current_results()))

@app.route('/blog')
def blog(): return render_template('blog.html')

@app.route('/download')
def download():
    path=OUTPUT_DIR/'current_results.csv'
    if not path.exists(): path=OUTPUT_DIR/'anomaly_results.csv'
    return send_file(path, as_attachment=True, download_name='inventory_anomaly_results.csv', mimetype='text/csv')

@app.errorhandler(413)
def too_large(_): flash('File is too large. Please upload a CSV under 16 MB.','danger'); return redirect(url_for('upload'))

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, debug=False)
