import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load final dataset
data = pd.read_csv("final_merged_dataset.csv")

# Convert datetime again 
data['datetime'] = pd.to_datetime(data['datetime'])


#Rush-Hour vs Pollution
#peak traffic periods increase pollution level
#s: reduce private vehicle usage during peak hours
sns.boxplot(x='rush_hour', y='pm25', data=data)
plt.title("Pollution During Rush Hours")
plt.show()

#Industrial vs Area
#industrial areas contributes significantly to air pollution
#implement stricter emission controls in industrial zones
sns.boxplot(x='industrial_zone', y='pm25', data=data)
plt.title("Industrial vs Non-Industrial Pollution")
plt.show()

#windspeed vs pollution
#higher windspeed, lower pollution since wind helps disperse pollutants, reducing concentration
sns.scatterplot(x='wind_speed', y='pm25', data=data)
plt.title("Wind Speed vs Pollution")
plt.show()

#temperature vs pollution
#temperature influence chemical reactions and pollutant buildup
sns.scatterplot(x='temperature', y='pm25', data=data)
plt.title("Temperature vs Pollution")
plt.show()

#pollution over time
plt.figure(figsize=(10,5))
plt.plot(data['datetime'], data['pm25'])
plt.title("Pollution Over Time")
plt.xticks(rotation=45)
plt.show()