# Results Summary and Business Interpretation

## Selected classification model

The **Tuned Random Forest** had the highest validation F1-score and was selected for deployment.

### Final untouched test-set performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7608 |
| Precision | 0.5341 |
| Recall | 0.7754 |
| F1-score | 0.6325 |
| ROC-AUC | 0.8420 |

### How to explain the results

- **Accuracy 76.1%:** about three quarters of test customers were classified correctly.
- **Recall 77.5%:** the model found about 78% of customers who actually churned.
- **Precision 53.4%:** among customers flagged as churn risks, about 53% actually churned.
- **F1-score 63.3%:** balances churn precision and churn recall.
- **ROC-AUC 0.842:** the model separates churn and non-churn customers well across different thresholds.

The model is designed to prioritise recall because missing a real churn customer may mean losing the chance to offer a retention action. The trade-off is more false alarms.

## Final confusion matrix

From 1,409 test customers:

- Correctly predicted no churn: 782
- Incorrectly flagged as churn: 253
- Missed churn customers: 84
- Correctly identified churn customers: 290

## Most important model features

Permutation importance identified these as the strongest features:

1. Contract
2. Internet service
3. Tenure group
4. Tenure
5. Total charges

Marketing interpretation:

- Contract design is strongly connected to churn risk.
- Customer lifecycle stage matters; new customers require more onboarding support.
- Internet plan and accumulated account value help separate customer risk levels.

These are associations in this dataset, not proof that changing one factor will directly prevent churn.

## Customer segments

| Segment | Customers | Average tenure | Average monthly charge | Churn rate | Suggested action |
|---|---:|---:|---:|---:|---|
| 1 | 1,982 | 59.8 months | 91.26 | 14.4% | Reward high-value long-term customers and prevent price dissatisfaction. |
| 2 | 1,884 | 8.8 months | 37.81 | 32.2% | Improve onboarding and early-service engagement. |
| 3 | 2,185 | 18.3 months | 79.78 | 42.7% | Highest-priority retention group; review plan value, support and contract offers. |
| 4 | 992 | 53.3 months | 29.94 | 4.3% | Low-risk loyal customers; use renewal and suitable upsell campaigns. |

## Recommended business strategy

1. Contact high-risk customers before contract renewal.
2. Offer longer-contract incentives to suitable month-to-month customers.
3. Improve onboarding during the first year.
4. Review high-charge plans and show customers the value of included services.
5. Offer technical support or online-security bundles where appropriate.
6. Use the model as decision support, with human review and fair-treatment checks.
