import pandas as pd
from app import app, current_results
from feature_engineering import engineer_features, validate_columns

def test_routes():
    client=app.test_client()
    for path in ['/', '/dashboard', '/anomalies', '/results', '/blog', '/upload']:
        r=client.get(path); assert r.status_code==200, (path,r.status_code)

def test_schema_and_results():
    raw=pd.read_csv('data/retail_store_inventory.csv').head(20)
    assert validate_columns(raw)==[]
    df=engineer_features(raw)
    assert {'Stock_to_Sales_Ratio','Days_of_Inventory','Demand_Change'}.issubset(df.columns)
    result=current_results(); assert len(result)>0 and {'Anomaly Score','Status','Key Indicators'}.issubset(result.columns)

if __name__=='__main__':
    test_routes(); test_schema_and_results(); print('smoke tests passed')
