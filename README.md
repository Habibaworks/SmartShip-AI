
# 📦 SmartShip AI
### Late-Delivery Risk Prediction for E-Commerce

SmartShip AI is a machine learning application that predicts the risk of late delivery for Brazilian e-commerce orders and supports historical seller performance monitoring.

Built as **Machine Learning Final Project — Project 01** using the **Olist Brazilian E-Commerce Dataset**.

## Live Application

**Streamlit App**: [Try SmartShip AI](https://smartship-ai-delivery-risk.streamlit.app/)  

## Project Overview

Late deliveries can negatively affect customer experience and create operational challenges for e-commerce marketplaces.

SmartShip AI helps logistics teams:
- Identify orders with elevated delivery-delay risk.
- Prioritize monitoring and early customer communication.
- Explore historical seller-associated delivery performance.

## Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The dataset contains approximately 100,000 historical orders from 2016–2018 across multiple related tables.

The project integrates orders, order items, sellers, customers, products and geolocation data.

**Target variable:** `is_late = 1` when an order is delivered after its estimated delivery date.

## Machine Learning Workflow

1. Data understanding and cleaning
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Preprocessing using scikit-learn pipelines
5. Baseline and model comparison
6. Cross-validation and hyperparameter tuning
7. Validation-based threshold selection
8. Final test-set evaluation
9. Feature importance and business recommendations
10. Streamlit application development

## Final Model Results

The final model is a **HistGradientBoostingClassifier** integrated into a scikit-learn Pipeline.

| Metric | Test Result |
|---|---:|
| Accuracy | 73.02% |
| Precision | 18.52% |
| Recall | 68.43% |
| F1-Score | 29.15% |
| ROC-AUC | 78.34% |
| PR-AUC | 28.28% |

**Decision threshold:** 0.4801, selected on validation data.

The model prioritizes identifying potentially late orders, while its relatively low precision means false alerts are an important limitation.

Model scores are **not calibrated probabilities**.

## Key Predictive Features

Permutation feature importance identified:

1. Purchase month
2. Estimated delivery days
3. Customer state
4. Shipping distance
5. Same-state shipping indicator

These features support prediction but do not establish causation.

## Streamlit Application

The application includes two main pages.

### Order Risk Predictor
- Product and shipping information
- Seller and customer locations
- ZIP-prefix-based distance estimation
- Late-delivery risk score and classification
- Operational recommendations

### Seller Performance Dashboard
- Seller late-delivery rates
- State and minimum-order filters
- Comparison chart and seller detail table
- Downloadable CSV reports

## Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal.

## Project Structure

```text
SmartShip-AI/
├── app.py
├── smartship.ipynb
├── smartship_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── customer_state_centroids.csv
    ├── seller_state_centroids.csv
    ├── seller_stats.csv
    ├── zip_centroids.csv
    ├── zip_states.csv
    └── product_category_name_translation.csv
```

The original large Olist datasets are excluded from GitHub. Download them from the official dataset source to reproduce the notebook.

## Limitations

- The dataset covers historical orders from 2016–2018.
- Only delivered orders were included in model training.
- Predictions may not generalize to current logistics conditions without additional validation.
- Shipping distance uses ZIP-prefix centroids and the Haversine formula, not road distance.
- Seller-associated delay rates do not prove seller fault.
- The model has a substantial false-alert rate and should support human review rather than automatic operational decisions.

## Technology Stack

Python · pandas · NumPy · scikit-learn · Streamlit · joblib · Matplotlib

## Project Status

- [x] Data preprocessing and EDA
- [x] Model training and evaluation
- [x] Streamlit application
- [x] Local application testing
- [ ] Public deployment
- [ ] Live application link and screenshots

## Acknowledgment

Dataset provided by Olist through Kaggle.

Developed as part of a Machine Learning Final Project.
