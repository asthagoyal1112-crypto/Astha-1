# Marketing Campaign Response Prediction Using Machine Learning

**Predicting Customer Response to Marketing Campaigns for Targeted Customer Engagement**

A single-file Streamlit analytics dashboard that cleans the Kaggle *Customer Personality Analysis*
data, explores campaign response, trains three classifiers, explains them in business language,
scores new customers and simulates campaign ROI.

> The model outputs a **probability of response**, not a guarantee. Recommendations built on it
> are business decisions that a marketing team should validate.

---

## 1. Folder structure

```text
MarketingCampaignProject/
│
├── marketing_campaign_app.py      <- the entire application (one file)
├── requirements.txt
├── README.md
│
└── data/
    └── marketing_campaign.csv     <- Kaggle dataset (tab-separated)
```

Everything (data loading, cleaning, feature engineering, ML pipeline, plots, UI) lives in
`marketing_campaign_app.py`. No pickle/joblib files are written.

## 2. Installation

Python 3.10 or newer is recommended (developed and tested on Python 3.12).

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly xgboost
```

or, equivalently:

```bash
pip install -r requirements.txt
```

Optional (adds a SHAP tab on the Model Explainability page; the app works without it):

```bash
pip install shap
```

Optional packages are detected at start-up. If Plotly is missing the charts fall back to
Matplotlib/Seaborn; if XGBoost is missing `HistGradientBoostingClassifier` is used instead; if SHAP
is missing the page shows feature-importance/coefficient explanations.

## 3. Run

```bash
streamlit run marketing_campaign_app.py
```

The app opens in your browser (default <http://localhost:8501>).

## 4. Providing the dataset

| Option | How |
|---|---|
| **A. Upload** | Sidebar → *Upload marketing_campaign.csv* |
| **B. Default file** | Place the file at `./data/marketing_campaign.csv` (already included here) |

The Kaggle file is **tab-separated**; the loader also accepts comma, semicolon and pipe
separators. If no dataset is found the app shows instructions – it never creates fake data.

Source: <https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis>

## 5. Pages

| Page | What it does |
|---|---|
| Dashboard | KPI cards (customers, response rate, avg income/spending, best model, ROC-AUC), response distribution, top factors, campaign insights |
| Data Overview | KPIs, preview, data types, missing values, duplicates, numeric/categorical summaries, data dictionary, cleaning log, cleaned-data download |
| EDA & Insights | Demographics, behaviour, response analysis, correlation heatmap |
| Campaign Analysis | Response by campaign history, income, education, age, spending, channel with interpretation |
| Model Training | *Train Models* button, progress, comparison table, documented selection criterion, confusion matrix, ROC and PR curves |
| Prediction | Interactive single-customer form → probability, gauge, marketing tier |
| Customer Targeting | Upload a customer CSV → scored targeting table → `campaign_targeting_predictions.csv`; budget/ROI scenario simulation |
| Model Explainability | Feature importance / coefficients, business interpretation, optional SHAP |
| Business Recommendations | Data-driven recommendations (targeting, efficiency, engagement, channel, campaign history) |
| About Project | Objective, dataset, methodology, assumptions, limitations, package availability |

Sidebar filters (Education, Marital Status, Country, Age, Income, Recency) update the KPIs and
charts. Models are trained once and kept in `st.session_state`, so changing a filter never retrains.

## 6. Methodology in brief

1. **Cleaning** – duplicates (ignoring ID) removed; income ≤ 0 → missing; negative counts → missing;
   ages outside 18–100 removed; every step is written to an on-screen cleaning log.
2. **Feature engineering** – Age, Customer_Tenure_Days, Total_Spending, Total_Purchases,
   Total_Children, Is_Parent, Previous_Campaign_Acceptance, Avg_Spend_Per_Purchase,
   Deal_Purchase_Ratio, channel shares, Web_Visits_Per_Web_Purchase.
3. **Split** – stratified 80/20, `random_state=42`.
4. **Pipeline** – `ColumnTransformer` (numeric: median impute + scale; categorical: mode impute +
   one-hot) inside a `Pipeline`, fitted on the training split only.
5. **Models** – Logistic Regression, Random Forest, XGBoost (fallback: HistGradientBoosting).
6. **Threshold** – the decision threshold is tuned on 5-fold cross-validated *training* predictions
   (maximum F1), never on the test set.
7. **Model selection** – highest cross-validated PR-AUC on the training split (ROC-AUC breaks ties).
   The test set is used only for the final report.

### Data-leakage handling

* `Response` (outcome of the latest campaign) is never in `X`.
* `AcceptedCmp1–5` are treated as **earlier** campaigns, known before the latest one launches, so
  they are allowed as predictors. This is a documented assumption; the Model Training page has a
  checkbox to remove them to see how much the model depends on them.
* `ID`, raw dates, `Year_Birth` and constant columns (`Z_CostContact`, `Z_Revenue`) are excluded.

## 7. Results on the Kaggle dataset (default settings)

After cleaning: 2,055 customers, 15.2 % responders (313), 1,644 train / 411 test.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.842 | 0.489 | 0.730 | 0.586 | 0.893 | 0.629 |
| Random Forest | 0.849 | 0.507 | 0.603 | 0.551 | 0.859 | 0.514 |
| XGBoost | 0.827 | 0.458 | 0.698 | 0.553 | 0.878 | 0.581 |

These figures are produced by the app itself at run time (nothing is hard-coded), so they can
differ slightly with different library versions or a different CSV.

## 8. Limitations

* One campaign, one company, ~2,000 customers; the test set holds only 63 responders, so metrics
  carry noticeable uncertainty.
* Response is associated with, not caused by, customer attributes.
* No cost or revenue data are in the dataset; the ROI simulator uses assumptions you enter.
* Whether `AcceptedCmp1–5` are strictly earlier campaigns is an assumption of this project.
* Models need re-checking if customer behaviour changes over time (drift).

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| "Dataset not found" | Put `marketing_campaign.csv` in `./data/` or upload it in the sidebar |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Run from another folder and dataset not found | Run the command from inside `MarketingCampaignProject/` |
| SHAP tab says it is unavailable | `pip install shap` (optional) |
