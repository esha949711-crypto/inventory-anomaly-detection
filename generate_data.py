"""Optional helper for creating a small clearly-labelled demo CSV.
The shipped application uses the original Kaggle CSV in data/retail_store_inventory.csv.
"""
from pathlib import Path
import pandas as pd

SOURCE = "https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset"

def make_demo(source_csv='data/retail_store_inventory.csv', output='data/demo_inventory.csv', rows=1000):
    df = pd.read_csv(source_csv).head(rows).copy()
    df.to_csv(output, index=False)
    Path(str(output) + '.NOTICE').write_text('Synthetic/demo subset for educational use; original source is Kaggle dataset: ' + SOURCE + '\n')
    return output

if __name__ == '__main__':
    print(make_demo())
