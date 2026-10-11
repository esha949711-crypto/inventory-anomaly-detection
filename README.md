# Inventory Anomaly Detection System Using Machine Learning

An intelligent inventory analysis web application built with Python, Flask, and Scikit-learn that uses **Isolation Forest** to identify unusual inventory patterns, unexpected sales behavior, and potential stock discrepancies. The system provides an interactive dashboard to help users analyze inventory records and make more informed inventory management decisions.

## Project Overview

Inventory management is essential for retail businesses, supermarkets, warehouses, and supply chain operations. Unusual changes in stock levels, sales, demand, or replenishment can lead to operational inefficiencies and require further investigation.

The **Inventory Anomaly Detection System** applies unsupervised machine learning to identify observations that differ from typical inventory patterns. Using the Isolation Forest algorithm, the application flags potentially unusual records and presents the results through a Flask-based web interface.

The project is designed to support inventory monitoring, anomaly investigation, and data-driven decision-making.

> **Important:** An anomaly is an unusual data pattern, not proof of fraud, theft, or an incorrect inventory record.

## Project Objectives

- Detect unusual inventory patterns using Isolation Forest.
- Identify potentially abnormal stock levels and sales behavior.
- Analyze inventory data to highlight records requiring investigation.
- Present anomaly results through an interactive web dashboard.
- Support inventory monitoring with charts and downloadable reports.
- Demonstrate the practical application of unsupervised machine learning.

## Key Features

- **Isolation Forest anomaly detection:** Flags observations that differ from typical data patterns.
- **Interactive Flask dashboard:** Presents inventory summaries and visualizations.
- **Inventory analysis:** Helps investigate unusual stock, sales, and demand records.
- **CSV data processing:** Supports analysis of datasets that follow the required input schema.
- **Anomaly results:** Displays normal and anomalous observations.
- **Data visualization:** Helps users explore inventory distributions and relationships.
- **Downloadable reports:** Supports exporting results for further analysis.
- **Saved model artifacts:** Allows the trained model to be loaded when the application starts.

## Dataset

The project uses the **Retail Store Inventory Forecasting Dataset** from Kaggle.

- **Dataset:** [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
- **Dataset size:** 73,100 observations, according to the dataset description.
- **Data type:** Synthetic retail inventory data designed to represent realistic scenarios.
- **License:** The source documentation identifies the dataset as CC0 Public Domain.

The dataset supports educational experimentation with inventory analytics. It should not be interpreted as verified operational data from a specific retail company.

### Main Variables

| Variable | Description |
|---|---|
| Date | Date of the inventory observation |
| Store ID | Store identifier |
| Product ID | Product identifier |
| Category | Product category |
| Region | Store region |
| Inventory Level | Recorded stock quantity |
| Units Sold | Number of units sold |
| Units Ordered | Number of units ordered |
| Demand Forecast | Forecast demand |
| Price | Product price |
| Discount | Discount value or indicator |
| Weather Condition | Recorded weather category |
| Holiday/Promotion | Holiday or promotion indicator |
| Competitor Pricing | Competitor product price |
| Seasonality | Seasonal information |

The features used by the trained model depend on the preprocessing and feature-selection steps implemented in the project.

## Machine Learning Algorithm: Isolation Forest

### What Is Isolation Forest?

Isolation Forest is an **unsupervised machine learning algorithm** designed to identify anomalies in datasets. Unlike supervised classification, it does not require pre-labelled examples of normal and anomalous inventory records.

The algorithm isolates observations through random feature selection and random split points. Unusual observations are often isolated using fewer splits than observations following common patterns.

### How It Works

1. **Data preparation:** Inventory records are cleaned and prepared for analysis.
2. **Feature selection:** Relevant inventory and sales features are selected.
3. **Model training:** Isolation Forest learns patterns from the prepared observations.
4. **Anomaly detection:** The model identifies observations that appear unusual relative to the learned patterns.
5. **Result classification:** Observations are labelled as normal or anomalous according to the model's decision threshold.
6. **Visualization:** The results are displayed in the web application for investigation.

### Why Isolation Forest?

- It is suitable for unsupervised anomaly detection.
- It does not require pre-labelled anomaly examples.
- It can identify unusual combinations of inventory features.
- It is suitable for analyzing multiple numerical features together.
- It can support anomaly investigation in retail and warehouse datasets.

### Model Output

Scikit-learn's Isolation Forest typically returns:

| Output | Meaning |
|---|---|
| `1` | Observation classified as normal |
| `-1` | Observation flagged as an anomaly |

The model's decision function and anomaly scores can also help distinguish observations that are more unusual from those that are less unusual. The precise score interpretation depends on the scoring method used in the application.

### Example Model Implementation

The following example illustrates how Isolation Forest can be trained. Its settings should match the actual model configuration used in your project.

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

model.fit(X_train)

predictions = model.predict(X_test)

# 1  = Normal observation
# -1 = Anomalous observation
```

The `contamination` parameter controls the thresholding behavior used to identify anomalies. When set to `"auto"`, Scikit-learn uses its default thresholding approach.

## System Workflow

The application follows this general workflow:

1. Load the inventory dataset.
2. Validate the data and required columns.
3. Clean and preprocess the records.
4. Select the features used for anomaly detection.
5. Load or train the Isolation Forest model.
6. Generate anomaly predictions and scores.
7. Identify records flagged as anomalous.
8. Display results through the Flask dashboard.
9. Export the results for further investigation.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Web application and backend |
| Scikit-learn | Isolation Forest implementation |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| HTML and CSS | Web interface |
| JavaScript | Interactive frontend elements, where implemented |
| Joblib | Saving and loading trained models |

## Project Structure

```text
inventory-anomaly-detection/
├── app.py
├── train_model.py
├── feature_engineering.py
├── preprocessing.py
├── anomaly_detection.py
├── evaluation.py
├── requirements.txt
├── README.md
├── data/
│   └── retail_store_inventory.csv
├── models/
│   └── isolation_forest_model.joblib
├── outputs/
│   ├── anomaly_results.csv
│   └── evaluation.json
├── notebooks/
│   └── EDA.ipynb
├── templates/
├── static/
└── uploads/
```

*This is a suggested structure. Keep the filenames aligned with your actual repository.*

## Installation and Setup

### Prerequisites

- Python 3.10 or another version compatible with your dependencies.
- Git.
- Visual Studio Code or another Python development environment.

### 1. Clone the Repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd inventory-anomaly-detection
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the Model

If your repository contains `train_model.py`, run:

```bash
python train_model.py
```

This step should preprocess the dataset, train Isolation Forest, and save the model artifact according to your implementation.

### 5. Run the Flask Application

```bash
python app.py
```

### 6. Open the Application

Navigate to:

http://127.0.0.1:5000

Keep the terminal running while using the web application.

## How to Use the Application

1. Launch the Flask application.
2. Open the dashboard.
3. Review the available inventory summaries and charts.
4. Navigate to the anomaly analysis section.
5. Upload a compatible inventory CSV file if supported by your implementation.
6. Review records flagged as anomalous.
7. Investigate unusual records using the available inventory information.
8. Export results if the download feature is enabled.

## Model Evaluation

Isolation Forest is an unsupervised algorithm, so its output does not automatically establish whether each flagged observation is genuinely incorrect.

Without trusted anomaly labels, accuracy, precision, recall, and F1-score cannot independently establish real-world detection performance.

Useful evaluation approaches include:

- Reviewing flagged records with inventory experts.
- Inspecting anomaly score distributions.
- Comparing results with verified historical incidents, when available.
- Assessing whether identified patterns are operationally meaningful.
- Testing the model on separate data to examine its stability.

Any reported performance metrics should be based on actual evaluation results rather than assumed values.

## Business Applications

The system can support inventory analysis in:

- Retail stores and supermarkets.
- Warehousing and distribution.
- Supply chain management.
- Inventory planning and replenishment.
- Sales and demand monitoring.
- Stock discrepancy investigation.

### Potential Benefits

- Highlights unusual inventory records for review.
- Helps identify unexpected stock and sales patterns.
- Supports more efficient inventory monitoring.
- Improves visibility into potential discrepancies.
- Provides visual summaries to support operational decisions.

These are potential benefits; their actual impact should be validated using real business data.

## Limitations

- The dataset is synthetic and may not represent every real retail environment.
- Isolation Forest can flag normal observations as anomalies or miss genuine anomalies.
- Results depend on the selected features and model configuration.
- Different products and stores may have different normal patterns.
- Anomaly detection alone does not predict future stockouts.
- Human review is required before taking operational action.

## Future Improvements

- Validate the system using real-world inventory data.
- Customize anomaly thresholds for different stores and product categories.
- Add model monitoring and periodic retraining.
- Integrate notifications for important anomalies.
- Connect the application to inventory databases.
- Add a separate stockout prediction model if future demand prediction is required.
- Improve deployment security and access controls.

## Author

**Machine Learning Project**

This project demonstrates the application of unsupervised machine learning and web development to inventory analysis using Python, Flask, and Isolation Forest.

## License

The dataset's license and the project's source-code license are separate. Refer to the dataset source for its terms and add a suitable license file to this repository if you wish to license your code.

## Disclaimer

This application is intended for educational and decision-support purposes. Anomaly detection results should be reviewed in context and should not be treated as proof of fraud, theft, or incorrect inventory records.
