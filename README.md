# Credit Score Prediction

## Overview
This project predicts customer credit scores using Machine Learning algorithms based on financial and credit-related features.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle

---

## Algorithms Used
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Features
- Annual Income
- Outstanding Debt
- Credit Utilization Ratio
- Number of Loans
- EMI per Month
- Payment Behaviour
- Credit History Age

---

## Workflow
1. Data Cleaning  
2. Data Preprocessing  
3. Feature Engineering  
4. Model Training  
5. Model Evaluation  
6. Hyperparameter Tuning  
7. Model Saving  

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

---

## Best Accuracy
| Model | Accuracy |
|-------|-----------|
| Random Forest | 99% |
| XGBoost | 99% |

---

## Model Saving
```python
import pickle
pickle.dump(model, open('credit_score_model.pkl', 'wb'))
