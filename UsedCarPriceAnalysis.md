# Used-Car Price Analysis — Capstone Practice

This notebook follows a CRISP-DM-style workflow for the Emeritus/PCC ML practical application **What Drives the Price of a Car?**

**Important:** This project uses a reproducible synthetic dataset because the official course dataset was not supplied. Numerical findings are illustrative.

## 1. Business Understanding

**Business question:** Which vehicle characteristics most strongly influence used-car prices, and how accurately can regression models predict the price of a used vehicle?

The dealership can use the analysis to support inventory acquisition, valuation, and pricing decisions.

## 2. Data Generation and Understanding

The next cell creates a reproducible synthetic marketplace dataset with 100,000 listings.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)
n = 100000

manufacturers = np.array(["ford","chevrolet","toyota","honda","nissan","bmw","mercedes-benz","audi","volkswagen","hyundai","kia","subaru","mazda","jeep","ram"])
regions = np.array(["atlanta","boston","chicago","dallas","denver","detroit","houston","las_vegas","los_angeles","miami","new_york","phoenix","portland","seattle"])
states = np.array(["CA","TX","FL","NY","WA","CO","GA","IL","MI","AZ","NV","MA","OR","NC"])
fuel_types = np.array(["gas","diesel","hybrid","electric"])
transmissions = np.array(["automatic","manual"])
conditions = np.array(["excellent","good","fair","salvage"])
drives = np.array(["4wd","fwd","rwd"])
sizes = np.array(["compact","mid-size","full-size"])
vehicle_types = np.array(["sedan","SUV","truck","pickup","coupe","hatchback","wagon","van"])
colors = np.array(["black","white","silver","gray","blue","red","green","brown"])

manufacturer_base = {"ford":22000,"chevrolet":21500,"toyota":24500,"honda":24000,"nissan":20500,"bmw":36000,"mercedes-benz":39000,"audi":37000,"volkswagen":25500,"hyundai":19000,"kia":18500,"subaru":26500,"mazda":22500,"jeep":29000,"ram":31500}
type_adj = {"sedan":0,"SUV":6500,"truck":7500,"pickup":7000,"coupe":3500,"hatchback":1000,"wagon":2500,"van":4500}
condition_adj = {"excellent":6500,"good":2500,"fair":-3500,"salvage":-8500}
fuel_adj = {"gas":0,"diesel":3500,"hybrid":5000,"electric":11000}
trans_adj = {"automatic":1800,"manual":0}
drive_adj = {"4wd":4500,"fwd":0,"rwd":1800}
size_adj = {"compact":0,"mid-size":1800,"full-size":4200}

manufacturer = rng.choice(manufacturers,n)
region = rng.choice(regions,n)
state = rng.choice(states,n)
fuel = rng.choice(fuel_types,n,p=[.72,.10,.10,.08])
transmission = rng.choice(transmissions,n,p=[.88,.12])
condition = rng.choice(conditions,n,p=[.18,.64,.15,.03])
drive = rng.choice(drives,n,p=[.28,.57,.15])
size = rng.choice(sizes,n,p=[.32,.48,.20])
vehicle_type = rng.choice(vehicle_types,n)
paint_color = rng.choice(colors,n)
year = rng.integers(2005,2025,n)
age = 2025-year
odometer = np.clip(age*rng.normal(11500,2800,n)+rng.normal(0,18000,n),3000,280000).astype(int)
horsepower = np.clip(120+rng.normal(0,25,n)+np.where(np.isin(vehicle_type,["SUV","truck","pickup"]),45,0),70,500).round().astype(int)
engine_liters = np.clip(horsepower/45+rng.normal(0,.35,n),1,7).round(1)

price = (
    np.array([manufacturer_base[m] for m in manufacturer])
    + np.array([type_adj[t] for t in vehicle_type])
    + np.array([condition_adj[c] for c in condition])
    + np.array([fuel_adj[f] for f in fuel])
    + np.array([trans_adj[t] for t in transmission])
    + np.array([drive_adj[d] for d in drive])
    + np.array([size_adj[s] for s in size])
    + (year-2015)*1250
    - odometer*.075
    + horsepower*55
    + engine_liters*450
    + rng.normal(0,4500,n)
)
price = np.clip(price,2500,95000).round().astype(int)

cars = pd.DataFrame({
    "price":price, "year":year, "manufacturer":manufacturer, "condition":condition,
    "fuel":fuel, "odometer":odometer, "transmission":transmission, "drive":drive,
    "size":size, "type":vehicle_type, "paint_color":paint_color, "state":state,
    "region":region, "horsepower":horsepower, "engine_liters":engine_liters
})

# Introduce realistic missing values for the cleaning exercise.
for col, rate in [("condition",.02),("fuel",.01),("transmission",.01),("drive",.02),("paint_color",.02)]:
    mask = rng.random(n) < rate
    cars.loc[mask,col] = np.nan

print(cars.shape)
cars.head()
```

## 3. Data Cleaning and Feature Engineering

```python
print(cars.info())
print(cars.isna().sum().sort_values(ascending=False))

cars = cars.drop_duplicates().copy()
cars["vehicle_age"] = 2025 - cars["year"]

cars.describe(include="all").T
```

## 4. Exploratory Data Analysis

### Price distribution

```python
plt.figure(figsize=(10,5))
plt.hist(cars["price"], bins=60)
plt.title("Distribution of Used-Car Prices")
plt.xlabel("Price ($)")
plt.ylabel("Number of Listings")
plt.tight_layout()
plt.show()
```

### Price versus mileage and age

```python
sample = cars.sample(12000, random_state=42)
fig, ax = plt.subplots(1,2,figsize=(14,5))
sns.scatterplot(data=sample,x="odometer",y="price",alpha=.25,ax=ax[0])
ax[0].set_title("Price vs. Odometer")
ax[0].set_xlabel("Odometer (miles)")
ax[0].set_ylabel("Price ($)")
sns.scatterplot(data=sample,x="vehicle_age",y="price",alpha=.25,ax=ax[1])
ax[1].set_title("Price vs. Vehicle Age")
ax[1].set_xlabel("Vehicle Age (years)")
ax[1].set_ylabel("Price ($)")
plt.tight_layout()
plt.show()
```

### Manufacturer

```python
order = cars.groupby("manufacturer")["price"].median().sort_values(ascending=False).index
plt.figure(figsize=(13,6))
sns.boxplot(data=cars,x="manufacturer",y="price",order=order)
plt.title("Used-Car Price by Manufacturer")
plt.xlabel("Manufacturer")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Condition

```python
plt.figure(figsize=(9,5))
sns.boxplot(data=cars,x="condition",y="price",order=["salvage","fair","good","excellent"])
plt.title("Used-Car Price by Condition")
plt.xlabel("Condition")
plt.ylabel("Price ($)")
plt.tight_layout()
plt.show()
```

### Correlation

```python
num_cols = ["price","year","odometer","horsepower","engine_liters","vehicle_age"]
plt.figure(figsize=(9,7))
sns.heatmap(cars[num_cols].corr(),annot=True,fmt=".2f",cmap="coolwarm",center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
```

## 5. Modeling

We compare Linear Regression, Ridge, and Lasso. RMSE is the primary metric because it is expressed in dollars and penalizes large errors. MAE is also reported for interpretability.

```python
# Sample 30,000 rows to keep CV/GridSearch practical on a normal laptop.
model_data = cars.sample(n=min(30000,len(cars)),random_state=42)

X = model_data.drop(columns=["price"])
y = model_data["price"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num",numeric_transformer,numeric_features),
    ("cat",categorical_transformer,categorical_features)
])

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=.20,random_state=42
)

models = {
    "Linear Regression":LinearRegression(),
    "Ridge":Ridge(alpha=10),
    "Lasso":Lasso(alpha=.01,max_iter=10000,random_state=42)
}

model_results = []
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor",preprocessor),("model",estimator)])
    pipe.fit(X_train,y_train)
    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    model_results.append({
        "Model":name,
        "Train RMSE":np.sqrt(mean_squared_error(y_train,train_pred)),
        "Test RMSE":np.sqrt(mean_squared_error(y_test,test_pred)),
        "Test MAE":mean_absolute_error(y_test,test_pred),
        "Test R2":r2_score(y_test,test_pred)
    })

results = pd.DataFrame(model_results).sort_values("Test RMSE")
results
```

## 6. Cross-Validation

```python
cv = KFold(n_splits=5,shuffle=True,random_state=42)
cv_rows = []

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor",preprocessor),("model",estimator)])
    scores = -cross_val_score(
        pipe,X_train,y_train,cv=cv,
        scoring="neg_root_mean_squared_error",n_jobs=-1
    )
    cv_rows.append({
        "Model":name,
        "CV RMSE Mean":scores.mean(),
        "CV RMSE Std":scores.std()
    })

pd.DataFrame(cv_rows).sort_values("CV RMSE Mean")
```

## 7. GridSearchCV

```python
ridge_pipe = Pipeline([
    ("preprocessor",preprocessor),
    ("model",Ridge())
])

param_grid = {"model__alpha":[.1,1,10,50,100]}

grid = GridSearchCV(
    ridge_pipe,param_grid=param_grid,cv=5,
    scoring="neg_root_mean_squared_error",n_jobs=-1
)
grid.fit(X_train,y_train)

print("Best parameters:",grid.best_params_)
print("Best CV RMSE:",-grid.best_score_)

best_model = grid.best_estimator_
pred = best_model.predict(X_test)

print("Test RMSE:",np.sqrt(mean_squared_error(y_test,pred)))
print("Test MAE:",mean_absolute_error(y_test,pred))
print("Test R2:",r2_score(y_test,pred))
```

## 8. Coefficient Interpretation

```python
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
coefs = best_model.named_steps["model"].coef_

coef_df = pd.DataFrame({
    "feature":feature_names,
    "coefficient":coefs,
    "absolute_coefficient":np.abs(coefs)
}).sort_values("absolute_coefficient",ascending=False)

coef_df.head(20)
```

Coefficients represent model associations after preprocessing; they should not be interpreted as causal effects.

## 9. Actual vs Predicted

```python
plot_df = pd.DataFrame({"Actual":y_test,"Predicted":pred})
plt.figure(figsize=(8,7))
sns.scatterplot(
    data=plot_df.sample(min(10000,len(plot_df)),random_state=42),
    x="Actual",y="Predicted",alpha=.35
)
lims=[min(plot_df.min()),max(plot_df.max())]
plt.plot(lims,lims,"--")
plt.title("Actual vs. Predicted Used-Car Price")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.tight_layout()
plt.show()
```

## 10. Business Findings and Recommendations

### Findings

- Vehicle age and mileage are important price predictors.
- Manufacturer and vehicle type create meaningful price differences.
- Vehicle condition has a strong relationship with expected price.
- Engine and performance characteristics add predictive information.
- Regularization helps manage the large number of features produced by categorical encoding.

### Recommendations

1. Use age and mileage as core inputs for inventory valuation.
2. Segment acquisition and pricing rules by manufacturer and vehicle type.
3. Standardize condition assessment during acquisition.
4. Use the model as decision support alongside local market supply, competition, inspection results, and current inventory.
5. Retrain the model periodically as market conditions change.

## 11. Limitations

This is a synthetic educational dataset. The numerical results must not be presented as evidence about the real used-car market. For formal submission, replace the generated dataset with the official course dataset if required by the program.

## Conclusion

The project demonstrates business framing, data cleaning, EDA, feature engineering, multiple regression models, cross-validation, hyperparameter tuning, evaluation, coefficient interpretation, and business recommendations.
