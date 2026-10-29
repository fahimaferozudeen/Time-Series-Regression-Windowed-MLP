
import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("sine_noise_dataset.csv")

# Step 2: Extract noisy sine wave values
data = df["Noisy_Value"].values

# Step 3: Define window size
window_size = 20

# Step 4: Create sliding windows
X = []
y = []

for i in range(len(data) - window_size):

    # Past 20 values as input
    X.append(data[i:i + window_size])

    # Next value as target
    y.append(data[i + window_size])

# Step 5: Convert lists into NumPy arrays
X = np.array(X)
y = np.array(y)

# Step 6: Display dataset shape
print("Sliding Window Dataset Created!")
print("Input shape:", X.shape)
print("Target shape:", y.shape)

# Step 7: Display first sample
print("\nFirst input window:")
print(X[0])

print("\nFirst target value:")
print(y[0])

# Step 8: Save the windowed dataset
window_df = pd.DataFrame(X)

window_df["Target"] = y

window_df.to_csv("windowed_dataset.csv", index=False)

print("\nWindowed dataset saved successfully!")
