# 🚨 Auto Insurance Fraud Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Champion_Model-orange.svg)](https://lightgbm.readthedocs.io/)

A Machine Learning project focused on detecting fraudulent auto insurance claims. This repository covers the entire Data Science pipeline from Exploratory Data Analysis (EDA) and Feature Engineering to Model Tuning and Deployment via a Web Application.

## 📌 Project Overview

Insurance fraud is a significant issue that costs companies millions each year. The goal of this project is to build a highly interpretable and effective Machine Learning model that minimizes false positives while capturing as many fraudulent claims as possible (Target Recall >= 0.65).

To make the model practical for real-world operations, a **"Lite Model"** was developed using only the **Top 10 most important features**, which is seamlessly deployed via a Streamlit web application.

## 📂 Repository Structure

```text
📦 Insurance-Fraud-Detection
 ┣ 📂 Data
 ┃ ┣ 📜 fraud_oracle.csv          # Raw dataset
 ┃ ┗ 📜 fraud_oracle_fe.csv       # Preprocessed dataset with engineered features
 ┣ 📂 NoteBooks
 ┃ ┣ 📜 exploratory-data-analysis.ipynb       # EDA and Insight generation
 ┃ ┣ 📜 feature_engineering_preparation.ipynb # Data cleaning and feature creation
 ┃ ┗ 📜 modeling.ipynb                        # Model training, SMOTE, and Threshold Tuning
 ┣ 📂 Steamlit
 ┃ ┣ 📜 app.py                    # Streamlit web application script
 ┃ ┗ 📜 fraud_lite_model.pkl      # Saved LightGBM Lite Model and Threshold
 ┣ 📜 requirements.txt            # Python dependencies
 ┗ 📜 README.md                   # Project documentation
```

## 🚀 How to Run the App (Local Setup)

**1. Clone the repository:**

```bash
git clone https://github.com/puttibenz/Insurance-Fraud-Detection.git
cd Insurance-Fraud-Detection
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app:**

```bash
cd Steamlit
streamlit run app.py
```

The application will open automatically in your web browser.

## 🧠 Data Science Methodology

### 1. Exploratory Data Analysis (EDA) & Feature Engineering

- **Handled Missing/Anomalous Data:** Detected and handled anomalous age data (Age = 0) by replacing with the median and creating an anomaly flag.
- **Feature Creation:** Extracted new variables (Red Flags) from high-risk behaviors, such as:
  - `HighRisk_Utility_Young`: Young drivers operating utility vehicles (fraud rate as high as 19.4%)
  - `Is_Cross_Month_Claim`: Claims filed across month boundaries before reporting
  - `Is_Weekend_Accident`: Accidents occurring on weekends

### 2. Modeling & Handling Imbalanced Data

- The dataset is significantly imbalanced (fraud accounts for only ~6% of all records).
- Used **Class Weights** (`scale_pos_weight`) to prioritize the minority class (Fraud), avoiding synthetic oversampling (SMOTE) to preserve the natural distribution of the data.
- Evaluated and compared tree-based models: **Random Forest**, **XGBoost**, and **LightGBM**.

### 3. Threshold Tuning for Business Value

- The standard threshold of 0.5 produced Precision too low for practical enterprise use.
- An **Optimal Threshold** was found to maximize Precision while maintaining a safety constraint of Recall ≥ 0.65.

## 🏆 Results & Champion Model

**LightGBM** was the best-performing model and was selected as the Champion Model.

| Metric | Score |
|---|---|
| ROC-AUC | 0.8358 |
| Recall | 0.78 |

The model detects 78% of fraudulent claims while controlling false alerts to an acceptable level for further investigation.

**Top 5 Most Important Features:**

1. `BasePolicy_Liability` — Low-risk indicator (third-party liability coverage)
2. `Fault_Third Party` — Low-risk indicator (counterparty at fault)
3. `AddressChange_Claim` — High-risk indicator (address changed shortly before claim)
4. `Deductible` — Deductible amount
5. `Year / MonthClaimed` — Temporal behavior patterns

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Libraries | Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, Matplotlib, Seaborn |
| Deployment | Streamlit |
