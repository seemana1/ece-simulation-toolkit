import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate a 50 Hz AC / sensor signal sampled at 1000 Hz for 0.2 seconds
fs = 1000
t = np.linspace(0, 0.2, int(fs * 0.2), endpoint=False)
clean_signal = 5.0 * np.sin(2 * np.pi * 50 * t)

# 2. Add Gaussian noise (simulating real hardware measurement noise)
np.random.seed(42)
noise = np.random.normal(0, 0.8, size=t.shape)
noisy_signal = clean_signal + noise

# 3. Plot and save high-resolution output
plt.figure(figsize=(10, 4))
plt.plot(t * 1000, noisy_signal, label="Noisy Sensor Reading", alpha=0.75)
plt.plot(t * 1000, clean_signal, label="Clean 50 Hz Reference", linewidth=2)
plt.title("Simulated 50 Hz Sensor Signal with Gaussian Noise")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("signal_preview.png", dpi=150)
print("Saved signal_preview.png successfully!")