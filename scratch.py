
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100
t = np.arange(N)

# Low-frequency wave
low_freq_wave = np.sin(2 * np.pi * 0.05 * t)
low_freq_shifted = np.roll(low_freq_wave, 1)

# High-frequency wave
high_freq_wave = np.sin(2 * np.pi * 0.5 * t)
high_freq_shifted = np.roll(high_freq_wave, 1)

# Plotting
plt.figure(figsize=(12, 5))

# Low-frequency plot
plt.subplot(1, 2, 1)
plt.plot(t, low_freq_wave, label="$x_{t}$")
plt.plot(t, low_freq_shifted, label="$x_{t-1}$")
plt.title("Low-Frequency Signal and Shifted Version")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# High-frequency plot
plt.subplot(1, 2, 2)
plt.plot(t, high_freq_wave, label="$x_{t}$")
plt.plot(t, high_freq_shifted, label="$x_{t-1}$")
plt.title("High-Frequency Signal and Shifted Version")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()