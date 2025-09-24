import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "traffic_minute_v2.csv",
    names=["datetime", "visits"],
    parse_dates=["datetime"],
    date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S", errors='coerce')
)

df = df.dropna(subset=["datetime"])
df = df.sort_values("datetime")
df.set_index("datetime", inplace=True)
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="min")
df = df.reindex(full_index, fill_value=0)
print(df.head(10))


df = pd.read_csv("traffic_minute_v2.csv", names=["datetime", "visits"])
df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df = df.dropna(subset=["datetime"])
visits_array = pd.to_numeric(df['visits'], errors='coerce').dropna().to_numpy()
fft_result = np.fft.rfft(visits_array)
magnitude = np.abs(fft_result)
freqs = np.fft.rfftfreq(len(visits_array), d=1)
magnitude_no_dc = magnitude.copy()
magnitude_no_dc[0] = 0
dominant_index = np.argmax(magnitude_no_dc)
dominant_freq = freqs[dominant_index]
period_minutes = 1 / dominant_freq if dominant_freq != 0 else np.inf
print("فرکانس غالب واقعی:", dominant_freq)
print("دوره زمانی چرخه غالب واقعی (دقیقه):", period_minutes)


df['visits_smooth_30'] = df['visits'].rolling(window=30).mean()
plt.figure(figsize=(12,6))
plt.plot(df['datetime'], df['visits'], label='originall data', alpha=0.5)
plt.plot(df['datetime'], df['visits_smooth_30'], label='miangin motaharek30min', color='red')
plt.xlabel("time")
plt.ylabel("traffic visits")
plt.title("trrafic ghabl o baad az noiz")
plt.legend()
plt.tight_layout()
plt.show()









