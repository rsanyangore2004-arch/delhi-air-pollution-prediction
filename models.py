import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load final dataset
data = pd.read_csv("final_merged_dataset.csv")

# Convert datetime again 
data['datetime'] = pd.to_datetime(data['datetime'])



# SELECT FEATURES AND TARGET
X = data[['avg_traffic', 'industrial_zone', 'hour', 'rush_hour',
                'temperature', 'humidity', 'wind_speed']]
y = data['pm25']

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print("LR MSE:", mse)
print("LR R2:", r2)


#DECISION TREE
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=42)

# Train
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print("Decision Tree MSE:", mse_dt)
print("Decision Tree R2:", r2_dt)


#RANDOM FOREST
#Random Forest was used to capture non-linear relationships and identify the most important factors influencing air pollution.

#train random forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#predict
y_pred_rf = rf_model.predict(X_test)

#evaluate model
from sklearn.metrics import mean_squared_error, r2_score
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest MSE:", mse_rf)
print("Random Forest R2:", r2_rf)


#FEATURE IMPORTANCE
#which factor affects pollution more
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))

#visualize feature importance
import matplotlib.pyplot as plt
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()


#Compare Models
print("\nMODEL COMPARISON")
print("---------------------------")
print(f"Linear Regression  -> MSE: {mse:.2f}, R2: {r2:.4f}")
print(f"Decision Tree      -> MSE: {mse_dt:.2f}, R2: {r2_dt:.4f}")
print(f"Random Forest      -> MSE: {mse_rf:.2f}, R2: {r2_rf:.4f}")






