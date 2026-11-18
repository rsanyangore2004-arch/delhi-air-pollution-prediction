
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ==========================================
# 1. LOAD FINAL DATASET
# ==========================================

data = pd.read_csv("final_merged_dataset.csv")

# Convert datetime column
data['datetime'] = pd.to_datetime(data['datetime'])

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. SELECT INPUT FEATURES AND TARGET
# ==========================================

X = data[['industrial_zone',
          'hour',
          'rush_hour',
          'temperature',
          'humidity',
          'wind_speed']]

y = data['pm25']


# ==========================================
# 3. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

# Sort records by date and time
data = data.sort_values('datetime').reset_index(drop=True)

# Recreate X and y after sorting
X = data[['industrial_zone',
          'hour',
          'rush_hour',
          'temperature',
          'humidity',
          'wind_speed']]

y = data['pm25']

# Find the timestamp separating training and testing data
unique_times = (
    data['datetime']
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

split_position = int(len(unique_times) * 0.8)
split_date = unique_times.iloc[split_position]

# Keep records from the same timestamp together
train_mask = data['datetime'] < split_date
test_mask = data['datetime'] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTRAIN-TEST SPLIT")
print("---------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training ends:", data.loc[train_mask, 'datetime'].max())
print("Testing starts:", data.loc[test_mask, 'datetime'].min())


# ==========================================
# 4. LINEAR REGRESSION MODEL
# ==========================================

lr_model = LinearRegression()

# Train model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate model
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLINEAR REGRESSION RESULTS")
print("---------------------------")
print("MSE:", mse_lr)
print("MAE:", mae_lr)
print("RMSE:", rmse_lr)
print("R2:", r2_lr)


# ==========================================
# 5. DECISION TREE MODEL
# ==========================================

dt_model = DecisionTreeRegressor(random_state=42)

# Train model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate model
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDECISION TREE RESULTS")
print("---------------------------")
print("MSE:", mse_dt)
print("MAE:", mae_dt)
print("RMSE:", rmse_dt)
print("R2:", r2_dt)


# ==========================================
# 6. RANDOM FOREST MODEL
# ==========================================

# Random Forest combines multiple decision trees
# to capture complex relationships in the data.

rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate model
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRANDOM FOREST RESULTS")
print("---------------------------")
print("MSE:", mse_rf)
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2:", r2_rf)

import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, "random_forest_pm25_model.pkl")

# Save the feature names
joblib.dump(list(X.columns), "model_features.pkl")

print("\nRandom Forest model saved successfully!")

# ==========================================
# 7. FEATURE IMPORTANCE
# ==========================================

# Identify the relative importance of each input feature
importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("\nFEATURE IMPORTANCE")
print("---------------------------")
print(importance)

# Visualize feature importance
importance.sort_values().plot(kind='barh')

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Decision Tree',
        'Random Forest'
    ],
    'MSE': [mse_lr, mse_dt, mse_rf],
    'MAE': [mae_lr, mae_dt, mae_rf],
    'RMSE': [rmse_lr, rmse_dt, rmse_rf],
    'R2': [r2_lr, r2_dt, r2_rf]
})

print("\nMODEL COMPARISON")
print("---------------------------")
print(results.to_string(index=False))

# Save comparison results
results.to_csv("model_comparison_results.csv", index=False)

print("\nModel comparison saved successfully!")
print("Feature importance chart saved successfully!")
