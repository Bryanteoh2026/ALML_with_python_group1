# Telco Customer Churn Prediction Project

## 1. Project Scenario

This project helps a telecommunications company predict whether a customer is likely to leave the service. This is called **customer churn prediction**.

The company can use the prediction before a contract renewal or during a retention call. Customers with a higher churn risk can receive suitable support, a better mobile plan, a handset upgrade, or a loyalty offer.

## 2. Dataset

File: `data/cell2celltrain.csv`

The dataset contains 51,047 customer records and 58 original columns. The target column is `Churn`:

- `No`: customer stays
- `Yes`: customer leaves

The dataset contains missing values. The supplied data has no exact duplicate rows, but the cleaning code checks for and removes duplicates automatically.

## 3. Project File Structure

```text
ALML_with_python_group1_completed/
│
├── main.py                    Runs the complete machine learning project
├── cleaning.py                Loads, checks and cleans the dataset
├── preprocessing.py           Selects features and prepares model data
├── training.py                Creates and trains four models
├── evaluation.py              Calculates evaluation measurements
├── visualization.py           Creates graphs and confusion matrices
├── app.py                     Flask churn prediction website
├── requirements.txt           Required Python libraries
├── run_project.bat            Easy Windows runner
├── run_project.sh             Easy macOS/Linux runner
│
├── data/
│   ├── cell2celltrain.csv
│   └── cleaned_cell2cell.csv
│
├── models/
│   ├── best_model.joblib
│   └── model_information.json
│
├── outputs/
│   ├── validation_model_results.csv
│   ├── best_model_test_results.csv
│   ├── classification_report.txt
│   └── nine PNG visualisations
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

## 4. Data Preparation and Cleaning

The project performs these steps:

1. Loads the CSV file with pandas.
2. Prints the number of rows and columns.
3. Checks missing values.
4. Checks exact duplicate rows.
5. Reports possible numeric outliers using the IQR method.
6. Removes spaces from column names and text values.
7. Changes `HandsetPrice` from text into numeric data.
8. Converts `Unknown` handset prices into missing values.
9. Removes exact duplicates.
10. Keeps rows with valid churn targets.
11. Saves `cleaned_cell2cell.csv`.

Missing feature values are not deleted. During model preprocessing:

- numeric missing values are filled using the median;
- categorical missing values are filled using the most common value;
- numeric columns are standardised;
- categorical columns are one-hot encoded.

## 5. Model Selection and Training

The data is split using stratified sampling:

- 70% training data
- 15% validation data
- 15% test data

Four models are trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Neural Network using `MLPClassifier`

The validation set is used to compare the models. The best model is selected using the highest F1-score. A probability threshold is also selected using the validation data.

## 6. Performance Analysis and Validation

The project calculates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrices

Why not use accuracy alone? The dataset has more non-churn customers than churn customers. A model can have reasonable accuracy while missing many customers who actually churn. Recall and F1-score give a more balanced view.

The final selected model is evaluated one time on the unseen test set.

## 7. Data Visualisations

The project creates:

1. Customer churn distribution
2. Missing-value bar chart
3. Numeric feature distributions
4. Correlation heatmap
5. Model performance comparison
6. Four confusion matrices
7. ROC curves
8. Random Forest feature importance
9. Neural Network loss curve

The graphs use Matplotlib and Seaborn and are saved in the `outputs` folder.

## 8. Website Wow Factor

The Flask website opens with a completely blank form. There are no automatic number values and no automatically selected dropdown answers. The user must build the customer profile before making a prediction.

The website includes:

- three clear sections for usage, service experience, and customer profile;
- a live completion meter for all 19 fields;
- green check marks when a field is completed;
- a live customer snapshot that updates while the user types;
- a sample-customer button for a quick demonstration;
- a clear-all button;
- friendly validation for incomplete or incorrect entries;
- an animated churn-probability gauge;
- a fun result card with retention missions;
- a celebration effect when the customer is predicted to stay;
- a responsive layout for laptop and phone screens.

The user can enter conditions such as monthly revenue, monthly minutes, overage minutes, dropped calls, customer-care calls, months in service, phone age, retention activity, credit rating, area type, and handset condition.

The website displays whether the customer is likely to stay or churn, the estimated churn probability, and simple retention suggestions. The suggestions are rule-based business ideas. The machine learning prediction does not prove that one factor caused the customer to churn.

## 9. How to Run on Windows

### Easy method

To run the full machine learning project, double-click:

```text
run_project.bat
```

To open only the already-trained website, double-click:

```text
run_website.bat
```

### Manual method

Open Command Prompt or the VS Code terminal inside the project folder:

```bash
python -m pip install -r requirements.txt
python main.py
python app.py
```

Open this local website link in your browser:

```text
http://127.0.0.1:5000
```

Press `Ctrl + C` in the terminal to stop the website.

## 10. How to Run on macOS or Linux

```bash
chmod +x run_project.sh
./run_project.sh
```

Then open:

```text
http://127.0.0.1:5000
```

## 11. Important Notes

- Run `main.py` before `app.py` when the model files do not exist.
- The website link is local to the computer running Flask. It is not a public internet link.
- Model performance may not be extremely high because customer churn is affected by many real-world factors that are not available in the dataset.
- The project is written in a simple modular style so students can explain each file separately.

## 12. Results Generated in This Package

The included run selected **Random Forest** as the best model using validation F1-score.

Final test-set results using the selected business threshold of 0.44:

| Measurement | Result |
|---|---:|
| Accuracy | 0.5325 |
| Precision | 0.3587 |
| Recall | 0.7898 |
| F1-score | 0.4933 |
| ROC-AUC | 0.6683 |

The lower threshold is intentionally more sensitive to possible churners. It correctly detects about 79% of customers who churn, but it also creates more false warnings. A telco can change the threshold depending on whether it prefers stronger churn detection or fewer unnecessary retention offers.
