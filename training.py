import pandas as pd
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ===================================
# LOAD DATASET
# ===================================

housing = fetch_california_housing(as_frame=True)

df = housing.frame

# save raw data
df.to_csv(
    "data/housing.csv",
    index=False
)

# basic cleaning
df = df.drop_duplicates()

# save cleaned data
df.to_csv(
    "data/cleaned_housing.csv",
    index=False
)

# ===================================
# FEATURES AND TARGET
# ===================================

X = df.drop(
    "MedHouseVal",
    axis=1
)

y = df["MedHouseVal"]

# ===================================
# TRAIN TEST SPLIT
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===================================
# SCALING
# ===================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================================
# HYPERPARAMETER TUNING
# ===================================

param_grid = {
    "n_neighbors": [3,5,7,9,11,15,21],
    "weights": ["uniform","distance"],
    "metric": ["euclidean","manhattan"]
}

grid = GridSearchCV(
    estimator=KNeighborsRegressor(),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(
    X_train_scaled,
    y_train
)

model = grid.best_estimator_

print("\nBest Parameters")
print(grid.best_params_)

# ===================================
# PREDICTION
# ===================================

pred = model.predict(X_test_scaled)

print("\nModel Performance")

print(
    "R2 Score:",
    r2_score(y_test,pred)
)

print(
    "MAE:",
    mean_absolute_error(y_test,pred)
)

print(
    "RMSE:",
    mean_squared_error(
        y_test,
        pred
    ) ** 0.5
)

# ===================================
# SAVE MODEL
# ===================================

joblib.dump(
    model,
    "models/knn_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel Saved Successfully")