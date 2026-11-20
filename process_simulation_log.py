import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# 1. Create a mock simulation CSV export (simulating Proteus/MATLAB export)
fs = 2000  # 2 kHz sampling rate
t = np.linspace(0, 0.25, int(fs * 0.25), endpoint=False)
np.random.seed(42)

v_clean = 5.0 * np.sin(2 * np.pi * 50 * t)
v_raw = v_clean + 1.2 * np.sin(2 * np.pi * 180 * t) + np.random.normal(0, 0.5, size=t.shape)
i_raw = 0.8 * np.sin(2 * np.pi * 50 * t - 0.3) + np.random.normal(0, 0.08, size=t.shape)

raw_df = pd.DataFrame({
    "time_s": t,
    "voltage_V": v_raw,
    "current_A": i_raw
})
raw_df.to_csv("sensor_export.csv", index=False)
print("Exported mock simulation log: sensor_export.csv")

# 2. Ingest CSV with Pandas and compute summary metrics
df = pd.read_csv("sensor_export.csv")
df["instant_power_W"] = df["voltage_V"] * df["current_A"]

# 3. Design & apply a 4th-order Butterworth low-pass filter (80 Hz cutoff)
cutoff_hz = 80.0
sos = butter(N=4, Wn=cutoff_hz, btype="lowpass", fs=fs, output="sos")
df["voltage_filtered_V"] = sosfiltfilt(sos, df["voltage_V"])

# 4. Print automated engineering summary
v_rms_raw = np.sqrt(np.mean(df["voltage_V"] ** 2))
v_rms_filt = np.sqrt(np.mean(df["voltage_filtered_V"] ** 2))
avg_power = df["instant_power_W"].mean()

print(f"Raw RMS Voltage:      {v_rms_raw:.3f} V")
print(f"Filtered RMS Voltage: {v_rms_filt:.3f} V")
print(f"Average Real Power:   {avg_power:.3f} W")

# 5. Save before-and-after filter comparison plot
plt.figure(figsize=(10, 4.5))
plt.plot(df["time_s"] * 1000, df["voltage_V"], label="Raw CSV Voltage (50 Hz + 180 Hz Noise)", alpha=0.55)
plt.plot(df["time_s"] * 1000, df["voltage_filtered_V"], label="Butterworth Filtered (80 Hz Cutoff)", linewidth=2.2, color="tab:green")
plt.title("Automated CSV Simulation Log Processing & Digital Low-Pass Filtering")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("filtered_log_report.png", dpi=150)
print("Saved filtered_log_report.png!")
