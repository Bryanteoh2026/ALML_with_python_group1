# ET0737 Telco Customer Churn Prediction and Segmentation

## 1. Project overview

This project applies Python machine learning to a **customers and marketing** scenario. Its main objective is to predict whether a telecommunications customer is likely to discontinue the service. It also uses K-Means clustering to group customers into marketing segments.

The core prediction problem is **binary classification**:

- `Churn = No` means the customer continues the service.
- `Churn = Yes` means the customer leaves/discontinues the service.

The clustering section is a supporting analysis that helps marketing teams understand groups of customers with similar tenure, charges and optional-service usage.

## 2. Dataset

Dataset file supplied for this project:

- `data/telco_customer.csv`
- 7,043 customer records
- 21 original columns
- Target column: `Churn`

The data describes a fictional telecommunications company and includes customer demographics, account information, subscribed services, contract type, payment method and charges.

Dataset references for the presentation slide:

- IBM sample description: https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn
- Common Kaggle copy of the 7,043-row, 21-column file: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Important: The project uses the real CSV supplied by the user; it is not AI-generated.

## 3. Project folder structure

```text
ET0737_Telco_Churn_Project/
├── 01_data_cleaning.py
├── 02_model_training_analysis.py
├── 03_customer_segmentation.py
├── app.py
├── project_utils.py
├── run_all.py
├── run_website.bat
├── run_website.sh
├── requirements.txt
├── WEBSITE_LINK.txt
├── AI_USAGE_AND_REFLECTION.md
├── README.md
├── data/
│   ├── telco_customer.csv
│   └── telco_customer_cleaned.csv
├── models/
│   ├── best_churn_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── Telco_Churn_Project.ipynb
└── outputs/
    ├── figures/
    ├── model_comparison.csv
    ├── final_test_metrics.csv
    ├── feature_importance.csv
    ├── customer_segment_summary.csv
    └── other reports
```

## 4. Data cleaning and preparation

Run:

```bash
python 01_data_cleaning.py
```

The script performs these steps:

1. Removes extra spaces from text values.
2. Converts blank strings into missing values.
3. Converts `TotalCharges` from text into a numeric column.
4. Sets `TotalCharges` to zero for the 11 new customers with `tenure = 0`.
5. Removes duplicate records and repeated customer IDs if present.
6. Keeps valid target values (`Yes` and `No`).
7. Adds two simple engineered features:
   - `NumOptionalServices`: number of optional services subscribed to.
   - `TenureGroup`: customer lifecycle band based on tenure.

### Data-leakage prevention

The cleaning file does **not** fit scaling or one-hot encoding on the full dataset. Statistical transformations are placed inside a Scikit-learn `Pipeline`, so they are fitted using training data only.

## 5. Training, validation and test design

The data is stratified so each split has approximately the same churn percentage:

- Training: 60%
- Validation: 20%
- Testing: 20%

The test set is kept untouched during model selection. Models are compared using the validation set. The selected model is then trained on the combined training and validation data and evaluated once on the test set.

## 6. Preprocessing pipeline

### Numeric columns

- Missing values: median imputation
- Scaling: `StandardScaler`

### Categorical columns

- Missing values: most-frequent imputation
- Encoding: `OneHotEncoder(handle_unknown="ignore")`

The preprocessor and model are combined in one pipeline to avoid leakage and make website predictions consistent with training.

## 7. Models trained

The following models are used because they match the ET0737 scope and are suitable for classification:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Neural Network (`MLPClassifier`)
5. Tuned Random Forest

### Hyperparameter tuning

`GridSearchCV` tunes the Random Forest using training data only and 3-fold stratified cross-validation. The grid compares:

- Number of trees
- Maximum tree depth
- Minimum samples per leaf
- Number of features considered at each split

The model with the highest **validation F1-score** is selected. F1 is useful because the dataset is imbalanced: churn customers are the minority class.

## 8. Evaluation metrics

The project does not rely on one metric. It reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Training and validation F1 are compared to identify possible overfitting. A large positive `F1_Gap` means training performance is much better than validation performance.

## 9. Visualisations

Matplotlib and Seaborn create the following plots:

1. Churn class distribution
2. Churn percentage by contract type
3. Tenure distribution by churn status
4. Monthly-charge boxplot
5. Correlation heatmap
6. Model comparison chart
7. Training vs validation F1 overfitting check
8. Validation confusion matrices
9. ROC curves
10. Precision-recall curves
11. Neural-network loss curve
12. Neural-network validation curve
13. Final test confusion matrix
14. Permutation feature importance
15. K-Means elbow curve
16. PCA customer-segment chart
17. Segment churn-rate chart

## 10. Customer clustering

Run:

```bash
python 03_customer_segmentation.py
```

K-Means uses these features without using the target label:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `NumOptionalServices`

The elbow curve compares `k = 2` to `k = 8`. Four clusters are used for an understandable marketing segmentation. `Churn` is used only after clustering to profile each segment's churn rate; it is not used to create the clusters.

## 11. Interactive website (wow factor)

Install requirements:

```bash
python -m pip install -r requirements.txt
```

Run the full training pipeline once:

```bash
python run_all.py
```

Start the website:

```bash
python -m streamlit run app.py
```

Open this local link:

```text
http://localhost:8501
```

The website allows a user to select customer profile, contract, payment, service and charge conditions. It displays:

- Churn probability
- Low/medium/high churn-risk category
- Continue/discontinue prediction
- Simple retention recommendations

The application is a decision-support demonstration and should not be used as the only reason to treat a customer differently.

## 12. Business insights to discuss

Use the generated figures and feature-importance CSV to support final claims. Typical insights to check in the actual outputs include:

- Month-to-month contracts may have a higher churn rate.
- Newer customers may be more likely to churn than long-tenure customers.
- Higher monthly charges may be associated with higher churn risk.
- Customers without online security or technical support may be useful targets for retention bundles.
- Electronic-check users may require payment-experience improvements.

Do not present these as facts until your group verifies them using the generated plots and tables.

## 13. Responsible AI and project limitations

- The dataset describes a fictional telco sample and may not represent Singapore customers.
- The model identifies association, not causation.
- Some customer groups may be underrepresented.
- Predictions can be wrong, especially around the decision threshold.
- Marketing action should include human review and fair-treatment checks.
- Do not use protected personal characteristics alone to make harmful decisions.
- Record how AI suggestions were tested, accepted or rejected.

## 14. Suggested presentation flow

1. Problem statement and business value
2. Dataset and target
3. Cleaning decisions and feature engineering
4. Train/validation/test split and leakage prevention
5. Four baseline models and Random Forest tuning
6. Model comparison and overfitting check
7. Final test metrics and confusion matrix
8. Important features and marketing recommendations
9. Customer clustering and segment profiles
10. Website demonstration
11. Limitations, responsible AI and GenAI reflection

## 15. Commands summary

```bash
python -m pip install -r requirements.txt
python run_all.py
python -m streamlit run app.py
```

## 16. Actual generated result in this project copy

The completed run selected **Tuned Random Forest**. On the untouched test set it achieved:

- Accuracy: 0.7608
- Precision: 0.5341
- Recall: 0.7754
- F1-score: 0.6325
- ROC-AUC: 0.8420

The website was startup-tested successfully at `http://localhost:8501` using the included saved model.
See `RESULTS_SUMMARY.md` for an easy-to-present interpretation.
