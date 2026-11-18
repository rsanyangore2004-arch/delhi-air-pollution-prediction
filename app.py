
from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("random_forest_pm25_model.pkl")
features = joblib.load("model_features.pkl")


# Categorise predicted PM2.5 concentration
def get_pollution_category(value):

    if value <= 30:
        return "Good", "#198754"

    elif value <= 60:
        return "Satisfactory", "#8BC34A"

    elif value <= 90:
        return "Moderate", "#FFC107"

    elif value <= 120:
        return "Poor", "#FD7E14"

    elif value <= 250:
        return "Very Poor", "#DC3545"

    else:
        return "Severe", "#800000"


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    category = None
    category_color = None
    error = None

    if request.method == "POST":
        try:
            industrial_zone = int(request.form["industrial_zone"])
            hour = int(request.form["hour"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            wind_speed = float(request.form["wind_speed"])

            if industrial_zone not in [0, 1]:
                raise ValueError("Select a valid industrial zone.")

            if not 6 <= hour <= 23:
                raise ValueError("Hour must be between 6 and 23.")

            if not 6 <= temperature <= 44.8:
                raise ValueError("Temperature must be between 6 and 44.8.")

            if not 20 <= humidity <= 100:
                raise ValueError("Humidity must be between 20 and 100.")

            if not 2.1 <= wind_speed <= 31.5:
                raise ValueError("Wind speed must be between 2.1 and 31.5.")

            rush_hour = 1 if hour in [8, 9, 17, 18, 19, 20] else 0

            input_data = pd.DataFrame([{
                "industrial_zone": industrial_zone,
                "hour": hour,
                "rush_hour": rush_hour,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed
            }])[features]

            prediction = round(model.predict(input_data)[0], 2)

            # Get pollution category and colour
            category, category_color = get_pollution_category(prediction)

        except (ValueError, TypeError) as e:
            error = str(e)

        except Exception:
            error = "Unable to generate prediction. Please check your inputs."

    return render_template(
        "index.html",
        prediction=prediction,
        category=category,
        category_color=category_color,
        error=error
    )


@app.route("/dashboard")
def dashboard():

    data = pd.read_csv("final_merged_dataset.csv")

    summary = {
        "records": f"{len(data):,}",
        "columns": len(data.columns),
        "average_pm25": round(data["pm25"].mean(), 2),
        "maximum_pm25": round(data["pm25"].max(), 2)
    }

    models = [
        {"name": "Linear Regression", "r2": 0.2737, "mae": 119.81},
        {"name": "Decision Tree", "r2": 0.6496, "mae": 68.54},
        {"name": "Random Forest", "r2": 0.7587, "mae": 56.46}
    ]

    importance = [
        {"name": "Temperature", "value": 50.12},
        {"name": "Hour", "value": 16.78},
        {"name": "Humidity", "value": 16.66},
        {"name": "Wind Speed", "value": 16.08},
        {"name": "Industrial Zone", "value": 0.21},
        {"name": "Rush Hour", "value": 0.15}
    ]

    return render_template(
        "dashboard.html",
        summary=summary,
        models=models,
        importance=importance
    )


@app.route("/visualizations")
def visualizations():

    data = pd.read_csv("final_merged_dataset.csv")
    data["datetime"] = pd.to_datetime(data["datetime"])

    image_folder = os.path.join("static", "images")
    os.makedirs(image_folder, exist_ok=True)

    # 1. Monthly average PM2.5
    monthly_pm25 = (
        data.groupby(data["datetime"].dt.to_period("M"))["pm25"]
        .mean()
    )

    plt.figure(figsize=(10, 5))
    monthly_pm25.plot(marker="o")
    plt.title("Monthly Average PM2.5 Concentration")
    plt.xlabel("Month")
    plt.ylabel("Average PM2.5")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, "monthly_pm25.png"))
    plt.close()

    # 2. Average PM2.5 by hour
    hourly_pm25 = data.groupby("hour")["pm25"].mean()

    plt.figure(figsize=(10, 5))
    hourly_pm25.plot(kind="bar")
    plt.title("Average PM2.5 by Hour of the Day")
    plt.xlabel("Hour")
    plt.ylabel("Average PM2.5")
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, "hourly_pm25.png"))
    plt.close()

    # 3. Temperature versus PM2.5
    sample_data = data.sample(
        n=min(3000, len(data)),
        random_state=42
    )

    plt.figure(figsize=(10, 5))
    plt.scatter(
        sample_data["temperature"],
        sample_data["pm25"],
        alpha=0.3
    )
    plt.title("Temperature vs PM2.5 Concentration")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("PM2.5")
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, "temperature_pm25.png"))
    plt.close()

    # 4. Ten stations with highest average PM2.5
    station_pm25 = (
        data.groupby("station_clean")["pm25"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 6))
    station_pm25.plot(kind="barh")
    plt.title("Top 10 Stations by Average PM2.5")
    plt.xlabel("Average PM2.5")
    plt.ylabel("Monitoring Station")
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, "station_pm25.png"))
    plt.close()

    return render_template("visualizations.html")


if __name__ == "__main__":
    app.run(debug=True)
