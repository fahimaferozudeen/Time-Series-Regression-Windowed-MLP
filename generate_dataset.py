
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate time values
time = np.linspace(0, 100, 1000)

# Generate sine wave
sine_wave = np.sin(time)

# Add random noise
np.random.seed(42)
noise = np.random.normal(0, 0.1, 1000)

# Create noisy sine wave
data = sine_wave + noise

# Create dataset
df = pd.DataFrame({
    "Time": time,
    "Sine_Value": sine_wave,
    "Noisy_Value": data
})

# Save dataset as CSV
df.to_csv("sine_noise_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())

# Plot the noisy sine wave
plt.figure(figsize=(10, 5))
plt.plot(time, data, label="Sine + Noise")
plt.plot(time, sine_wave, label="Original Sine Wave")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Sine Wave with Noise")
plt.legend()
plt.show()
