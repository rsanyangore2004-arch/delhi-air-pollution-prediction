
import joblib
import pandas as pd

# Load the trained model and feature names
model = joblib.load("random_forest_pm25_model.pkl")
features = joblib.load("model_features.pkl")

print("\n===================================")
print("     DELHI PM2.5 PREDICTION SYSTEM")
print("===================================")

try:
    industrial_zone = int(input(
        "\nIs the monitoring station in an industrial zone? (1 = Yes, 0 = No): "
    ))

    hour = int(input("Enter the hour (6-23): "))
    temperature = float(input("Enter temperature (6-44.8 °C): "))
    humidity = float(input("Enter humidity (20-100%): "))
    wind_speed = float(input("Enter wind speed (2.1-31.5): "))

    # Validate inputs
    if industrial_zone not in [0, 1]:
        raise ValueError("Industrial zone must be 0 or 1.")

    if not 6 <= hour <= 23:
        raise ValueError("Hour must be between 6 and 23.")

    if not 6 <= temperature <= 44.8:
        raise ValueError("Temperature must be between 6 and 44.8 °C.")

    if not 20 <= humidity <= 100:
        raise ValueError("Humidity must be between 20 and 100%.")

    if not 2.1 <= wind_speed <= 31.5:
        raise ValueError("Wind speed must be between 2.1 and 31.5.")

    rush_hour = 1 if hour in [8, 9, 17, 18, 19, 20] else 0

    # Prepare input in the same order used during training
    input_data = pd.DataFrame([{
        "industrial_zone": industrial_zone,
        "hour": hour,
        "rush_hour": rush_hour,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed
    }])[features]

    # Make prediction
    prediction = model.predict(input_data)[0]

    print("\n===================================")
    print("         PREDICTION RESULT")
    print("===================================")
    print(f"Predicted PM2.5 concentration: {prediction:.2f}")
    print("\nNote: This is a model estimate,")
    print("not a live air-quality measurement.")
    print("===================================")

except ValueError as error:
    print("\nInvalid input:", error)
