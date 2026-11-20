import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# 1. Simulate a 50 Hz sensor signal with a 180 Hz harmonic and noise
fs = 2000  # Sampling frequency (Hz)
duration = 0.5
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

fundamental = 5.0 * np.sin(2 * np.pi * 50 * t)
harmonic = 1.2 * np.sin(2 * np.pi * 180 * t)
np.random.seed(42)
noise = np.random.normal(0, 0.6, size=t.shape)
signal = fundamental + harmonic + noise

# 2. Compute Single-Sided FFT Spectrum
N = len(signal)
yf = np.abs(rfft(signal)) * (2.0 / N)
xf = rfftfreq(N, 1 / fs)

# 3. Automatically detect dominant frequencies (> 0.5 V) and estimate SNR
peaks = xf[yf > 0.5]
signal_power = np.mean(fundamental ** 2)
noise_power = np.mean((signal - fundamental) ** 2)
snr_db = 10 * np.log10(signal_power / noise_power)

print(f"Detected Frequency Peaks (Hz): {np.round(peaks, 1)}")
print(f"Estimated SNR (dB): {snr_db:.2f} dB")

# 4. Generate Time-Domain vs. Frequency-Domain Diagnostic Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.plot(t[:200] * 1000, signal[:200], color="tab:blue")
ax1.set_title("Time-Domain Waveform (First 100 ms)")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Amplitude (V)")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(xf, yf, color="tab:red")
ax2.set_xlim(0, 400)
ax2.set_title(f"Single-Sided FFT Spectrum (SNR: {snr_db:.2f} dB)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude (V)")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("fft_diagnostic.png", dpi=150)
print("Saved fft_diagnostic.png successfully!")