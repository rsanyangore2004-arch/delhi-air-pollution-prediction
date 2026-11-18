import pandas as pd
import numpy as np

# LOAD DATASETS
#load all datasets into pandas DataFrames

aqi = pd.read_csv("delhi_air_quality.csv")
traffic = pd.read_csv("delhi_traffic_features.csv")
industrial = pd.read_csv("delhi_industrial_areas.csv")


#UNDERSTAND DATA 

# Preview first few rows
print("AQI DATA:\n", aqi.head())
print("\nTRAFFIC DATA:\n", traffic.head())
print("\nINDUSTRIAL DATA:\n", industrial.head())

# Check column names
print("\nAQI Columns:", aqi.columns)
print("\nTraffic Columns:", traffic.columns)
print("\nIndustrial Columns:", industrial.columns)


# CLEAN airquality DATA
# Convert datetime 
aqi['datetime'] = pd.to_datetime(aqi['datetime'], dayfirst=True, errors='coerce')

# Check missing values
print("\nMissing values in AQI:\n", aqi.isnull().sum())


# CREATE MAIN DATASET
# Start with airquality dataset as base
data = aqi.copy()

# CREATE RUSH HOUR FEATURE
#Due to lack of synchronized traffic data, a time-based proxy (rush hour) was used to capture traffic variation.
# Extract hour if not already present
data['hour'] = data['datetime'].dt.hour

# (1 = high traffic, 0 = normal)
data['rush_hour'] = data['hour'].apply(
    lambda x: 1 if x in [8, 9, 17, 18, 19, 20] else 0
)
# TRAFFIC FEATURE ENGINEERING
# Problem: Traffic dataset has no datetime → cannot merge directly
# Solution: Convert traffic levels into numeric feature because machine learning models doesn't undersatnd texts

traffic_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}
traffic['traffic_score'] = traffic['traffic_density_level'].map(traffic_map)

# Create average traffic per area
#Traffic data was aggregated at the area level using mean traffic intensity to create a stable and representative feature
traffic_area = traffic.groupby('start_area')['traffic_score'].mean().reset_index()
traffic_area.columns = ['area', 'avg_traffic']

# CLEAN LOCATION NAMES 
# Remove ", Delhi" to standardize names
data['station_clean'] = data['station'].str.replace(", Delhi", "").str.strip()

# INDUSTRIAL FEATURE
# Create list of industrial areas
industrial_areas = industrial['Area_Name'].tolist()

# Match using keywords since names are slightly different
data['industrial_zone'] = data['station_clean'].apply(
    lambda x: 1 if any(ind.split()[0] in x for ind in industrial_areas) else 0
)

# FINAL CLEAN DATASET
# Keep only important columns for modeling

# FINAL DATASET
# Keep relevant features for pollution prediction.
# Traffic is excluded because the traffic records cannot
# currently be accurately matched to AQI records.

final_data = data[['datetime', 'station_clean', 'pm25',
                   'industrial_zone', 'hour', 'rush_hour',
                   'temperature', 'humidity', 'wind_speed']]

# CHECK MERGED DATASET (VERY IMPORTANT)
print("\nFINAL DATASET PREVIEW:")
print(final_data.head(10))
print("\nDATASET INFO:")
print(final_data.info())
print("\nRANDOM SAMPLE:")
print(final_data.sample(10))
print("\nSHAPE (rows, columns):", final_data.shape)


# SAVE FINAL DATASET 
final_data.to_csv("final_merged_dataset.csv", index=False)



