# Delhi Air Pollution Prediction System

A machine learning web application that predicts PM2.5 concentration in Delhi using environmental and time-related features.

## Project Overview

Air pollution is a major environmental challenge in Delhi, India. This project uses historical air-quality data and machine learning techniques to estimate PM2.5 concentrations.

The system allows users to enter environmental conditions and receive a predicted PM2.5 concentration together with an indicative pollution category.

## Project Objectives

- Analyse historical PM2.5 air-quality data.
- Explore relationships between environmental variables and PM2.5 concentration.
- Train and compare machine learning regression models.
- Identify important features influencing model predictions.
- Develop a Flask web application for interactive PM2.5 prediction.
- Present findings through charts and a dashboard.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualisation:** Matplotlib
- **Model Storage:** Joblib
- **Frontend:** HTML and CSS
- **Development Environment:** Visual Studio Code

## Dataset Description

The project uses the following datasets:

| Dataset | Description |
|---|---|
| Delhi Air Quality | Historical air-quality and PM2.5 measurements |
| Delhi Traffic Features | Traffic-related dataset explored during data preparation |
| Delhi Industrial Areas | Manually compiled industrial-area information |
| Final Merged Dataset | Prepared dataset used for model training |

### Data Sources

- [Delhi Pollution AQI Dataset – Kaggle](https://www.kaggle.com/datasets/vishardmehta/delhi-pollution-aqi-dataset)
- [Delhi Traffic Travel Time Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/vishardmehta/delhi-traffic-travel-time-prediction-dataset)

**Important data note:** The traffic dataset was not directly merged into the final air-quality dataset because a reliable shared location and time key was unavailable. The rush-hour variable is a time-based indicator, not a direct traffic measurement. Industrial-zone mapping is approximate.

## Features

### 1. PM2.5 Prediction
Users can enter:
- Industrial zone status
- Hour of the day
- Temperature
- Humidity
- Wind speed

The application generates a predicted PM2.5 concentration.

### 2. Pollution Category Indicator
The prediction is accompanied by an indicative pollution category:
- Good
- Satisfactory
- Moderate
- Poor
- Very Poor
- Severe

These categories are indicative and should not be interpreted as an official AQI measurement.

### 3. Dashboard
Displays:
- Dataset overview
- Average and maximum PM2.5 concentration
- Model comparison results
- Feature importance

### 4. Data Visualisations
The application presents:
- Monthly average PM2.5 concentration
- Hourly PM2.5 patterns
- Temperature versus PM2.5
- Monitoring stations with the highest average PM2.5

## Machine Learning Models

Three regression algorithms were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Random Forest achieved the best performance among the three models in the recorded evaluation.

### Model Evaluation Results

| Model | MSE | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 34,774.46 | 119.81 | 186.48 | 0.2737 |
| Decision Tree | 16,774.92 | 68.54 | 129.52 | 0.6496 |
| Random Forest | 11,552.88 | 56.46 | 107.48 | 0.7587 |

*These are results from a recorded chronological train-test evaluation. Results may change when the model is retrained.*

## Feature Importance

The recorded Random Forest feature-importance analysis identified:

| Feature | Importance |
|---|---:|
| Temperature | 50.12% |
| Hour | 16.78% |
| Humidity | 16.66% |
| Wind Speed | 16.08% |
| Industrial Zone | 0.21% |
| Rush Hour | 0.15% |

Temperature had the highest feature importance in this model. Feature importance indicates how the model uses variables; it does not establish causation.

## Project Structure

```text
air_pollution_project/
│
├── analysis.py
├── app.py
├── models.py
├── predict_pm25.py
├── visualizations.py
│
├── delhi_air_quality.csv
├── delhi_traffic_features.csv
├── delhi_industrial_areas.csv
├── final_merged_dataset.csv
│
├── model_comparison_results.csv
├── feature_importance.png
│
├── static/
│   ├── style.css
│   └── images/
│       ├── hourly_pm25.png
│       ├── monthly_pm25.png
│       └── temperature_pm25.png
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── visualizations.html
│
└── README.md
```

The trained `.pkl` model is excluded from GitHub because of its large file size. It must be generated locally before running the web application.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rsanyangore2004-arch/delhi-air-pollution-prediction.git
```

### 2. Open the project folder

```bash
cd delhi-air-pollution-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn joblib
```

### 6. Train the machine learning models

```bash
python models.py
```

This generates the trained Random Forest model and feature configuration required by the application.

### 7. Run the Flask application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Limitations

- Predictions depend on the historical data used to train the model.
- The model does not provide live air-quality readings.
- The industrial-zone indicator is based on approximate area mapping.
- The traffic dataset was not directly integrated into the final model.
- Model performance may vary with new data and different environmental conditions.

## Future Improvements

- Integrate reliable real-time air-quality data.
- Incorporate traffic data with matching location and time information.
- Expand the dataset with additional environmental variables.
- Evaluate further machine learning and time-series forecasting approaches.
- Deploy the application online.

## Author

**Rhoda Sanyangore**

GitHub: [rsanyangore2004-arch](https://github.com/rsanyangore2004-arch)

LinkedIn: [Rhoda Sanyangore](https://www.linkedin.com/in/rhoda-sanyangore-9073693a6/)

---

*This project was developed for academic and learning purposes as an application of data analysis, machine learning and web development.*