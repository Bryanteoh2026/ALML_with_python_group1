# GenAI Usage and Reflection Record

This file is a starting template. Replace names and add screenshots or exact outputs required by your lecturer.

## AI tool

ChatGPT

## Date

16 July 2026

## Main prompt used

“Help us finish a Python customer churn prediction project, including data preparation and cleaning, model selection and training, performance analysis and validation, data visualisation using Matplotlib and Seaborn, feature importance, a loss curve, confusion matrices, and an interactive website. Keep the code simple and clean.”

## Recommendations adopted

- Use classification as the main task because `Churn` has two known labels.
- Add K-Means clustering as a supporting customer-segmentation analysis.
- Use a strict train/validation/test split.
- Put imputation, scaling and one-hot encoding inside a pipeline to prevent leakage.
- Compare Logistic Regression, Decision Tree, Random Forest and Neural Network.
- Tune Random Forest with GridSearchCV.
- Select the deployment model using validation F1-score.
- Add a Streamlit interface for single-customer prediction.

## Recommendations that should be critically checked

- Four clusters were selected for usability, but the elbow plot must be reviewed by the group.
- F1-score was selected as the main model-selection metric, but the business may prefer higher recall if missing churners is expensive.
- Retention recommendations in the website are rule-based suggestions and are not causal conclusions.

## How the output was validated

- The supplied CSV was inspected for shape, column types, duplicates and missing values.
- `TotalCharges` conversion was checked.
- Preprocessing was placed inside a Scikit-learn pipeline.
- Training, validation and testing data were separated before model selection.
- Multiple metrics and confusion matrices were generated.
- Training-versus-validation F1 was used as an overfitting check.
- The final website loads the exact saved preprocessing-and-model pipeline.

## Limitations of the AI-generated assistance

- AI can produce code that runs but may still contain unsuitable business assumptions.
- Model performance can vary across software versions and random seeds.
- AI cannot confirm that a marketing action causes lower churn.
- The group must understand every code section and be able to explain it during Q&A.

## Impact on submission

The AI output supported code structure, preprocessing design, model comparison, visualisation generation, documentation and the website prototype. Group members must test the files, review the results, make their own conclusions and acknowledge the use of AI in the official declaration form.
