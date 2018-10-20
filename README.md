# Kaggle/Santander Value Prediction

<div align="center">
  <img src="rsc/logo.png"><br><br>
</div>

## Abstract
[Kaggle Santander Value Prediction Competition](https://www.kaggle.com/c/santander-value-prediction-challenge)

- Host : **Santander**, British bank, wholly owned by the Spanish Santander Group.
- Prize : $ 60,000
- Problem : Regression
- Evaluation : Root Mean Squared Log Error
- Period : June 19 2018 ~ Aug 21 2018 (63 days)

**Santander Bank** aims to predict the value of the transactions for each potential customers.

Competition data is completely anonymized, and size of the train set is quite small (~4k rows). Given the task, anonymized data must be a time-series data encrypted in specific method. Kagglers have identified a data leakage (or specifically how the data has been encrypted) and utilized the lag data which is often a strong predictor in time-series. Top scoring methods must include data leakage information, otherwise the score is too low to compete. 

I share a baseline method, with no Feature Engineering and simple RandomForest regressor.
I conduct simple feature engineering ideas and use LightGBM model for next version.
Additional feature engineering ideas and using XGBoost and CatBoost further pushes the score to around Private LB 1.37
Next, we use leakage data to obtain better Private LB scores.

I have decided to accept the nature of data leakage in Kaggle competition. Instead of avoiding competitions that include leakage, I would like to learn how kagglers have found the leakage and explored the leakage, as they are the product of extensive data exploration, which I admire in terms of skill-set.


## Result
| Submission | CV LogLoss | Public LB | Rank | Private LB | Rank |
|:----------:|:----------:|:---------:|:----:|:----------:|:----:|
| baseline | - | 1.93257 | - | 1.87086 | 
| [Exp 01] Feature Selection & Feature Interaction + LightGBM | - | 1.57676 | - | 1.53769 | 
| [Exp 02] Feature Selection & PCA & Statistical features + CatBoost/XGBoost/LightGBM | - | 1.41484 | - | 1.37273

## How to Run

- for `baseline`, see `code/baseline.py`
- for `Exp 01`, see `code/[LB 1.53769] [FE] feature selection, feature interaction [Model] LightGBM.ipynb`
- for `Exp 02`, see `code/[LB 1.37246] [FE] feature selection, pca, statistical features [Model] Catboost, XGBoost, LightGBM.ipynb`

## TODO
- [ ] reproduce Jack's code in python
  - [code LB 0.52785](https://www.kaggle.com/rsakata/21st-place-solution-bug-fixed-private-0-52785)
- [ ] reproduce Giba's code (only non-leakage part) in python
  - [code LB 0.5178](https://www.kaggle.com/titericz/winner-model-giba-single-xgb-lb0-5178)
