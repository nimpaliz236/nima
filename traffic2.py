import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("traffic_minute_v2.csv", names=["datetime", "visits"])
df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df = df.dropna(subset=["datetime"])
df['visits'] = pd.to_numeric(df['visits'], errors='coerce')
df = df.dropna(subset=['visits'])

start_date = df['datetime'].min()
end_date = start_date + pd.Timedelta(days=5)
df_3days = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]


mean_visits = df_3days['visits'].mean()
std_visits = df_3days['visits'].std()


threshold = mean_visits + 2 * std_visits
peaks = df_3days[df_3days['visits'] > threshold]

plt.figure(figsize=(12,6))
plt.plot(df_3days['datetime'], df_3days['visits'], label='traffic')
plt.scatter(peaks['datetime'], peaks['visits'], color='red', label='piks', zorder=5)
plt.xlabel("time")
plt.ylabel("mizan traffic")
plt.title("traffic piks for astane poya")
plt.legend()
plt.tight_layout()
plt.show()



