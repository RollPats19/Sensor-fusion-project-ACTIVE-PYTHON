import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv("sensor_fusion_analysis.csv")

# Select features
features = df[
    [
        "audio_freq_hz",
        "vibration_g",
        "thermal_temp_c",
    
    ]
]

# Train model
model = IsolationForest(contamination=0.05, random_state=42)
df["model_anomaly"] = model.fit_predict(features)

# Convert (-1 = anomaly, 1 = normal) → (1 = anomaly, 0 = normal)
df["model_anomaly"] = df["model_anomaly"].apply(lambda x: 1 if x == -1 else 0)

# Plot
plt.figure(figsize=(12, 6))

# Normal signal
plt.plot(df["audio_freq_hz"], label="Audio Signal")

# Model-detected anomalies
plt.scatter(
    df.index[df["model_anomaly"] == 1],
    df["audio_freq_hz"][df["model_anomaly"] == 1],
    label="Model Anomaly"
)

plt.legend()
plt.title("Model Detected Anomalies Over Time")
plt.xlabel("Time Index")
plt.ylabel("Frequency (Hz)")
plt.show()





