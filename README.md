# Used-Car Price Analysis — Capstone Practice

This repository contains a complete practice project for the Emeritus/PCC ML capstone assignment **What Drives the Price of a Car?**.

## Project objective

Analyze used-car listings to understand which vehicle characteristics are associated with price and build regression models that can support used-car dealership inventory and pricing decisions.

## Dataset

This practice project uses a synthetic used-car dataset created for educational purposes. It contains 100,250 generated listings with vehicle, condition, mileage, engine, drivetrain, location, and pricing attributes. The synthetic data should not be interpreted as real-market evidence.

## Analysis

The notebook follows a CRISP-DM-style workflow:

- Business understanding
- Data understanding and cleaning
- Exploratory data analysis
- Feature engineering
- Linear Regression
- Ridge Regression
- Lasso Regression
- 5-fold cross-validation
- GridSearchCV hyperparameter tuning
- RMSE, MAE, and R² evaluation
- Coefficient interpretation
- Business findings and recommendations

## Files

- `UsedCarPriceAnalysis.ipynb` — complete analysis notebook
- `data/synthetic_used_cars.csv` — synthetic dataset used by the notebook

## Business focus

The final analysis translates model results into practical recommendations for used-car inventory acquisition, valuation, and pricing.

## Note

For formal course submission, replace the synthetic dataset with the official course dataset if the program requires the supplied dataset.
