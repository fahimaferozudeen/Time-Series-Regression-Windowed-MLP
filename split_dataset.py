
import pandas as pd

# Step 1: Load the windowed dataset
df = pd.read_csv("windowed_dataset.csv")

# Step 2: Calculate split sizes
total_samples = len(df)

train_size = int(total_samples * 0.70)
val_size = int(total_samples * 0.15)

# Step 3: Split data chronologically
train_data = df.iloc[:train_size]

val_data = df.iloc[
    train_size:train_size + val_size
]

test_data = df.iloc[
    train_size + val_size:
]

# Step 4: Separate input features and target
X_train = train_data.drop("Target", axis=1)
y_train = train_data["Target"]

X_val = val_data.drop("Target", axis=1)
y_val = val_data["Target"]

X_test = test_data.drop("Target", axis=1)
y_test = test_data["Target"]

# Step 5: Display dataset sizes
print("Dataset Split Completed!")

print("\nTraining Samples:", len(X_train))
print("Validation Samples:", len(X_val))
print("Testing Samples:", len(X_test))

print("\nInput shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# Step 6: Save the split datasets
train_data.to_csv("train_dataset.csv", index=False)
val_data.to_csv("validation_dataset.csv", index=False)
test_data.to_csv("test_dataset.csv", index=False)

print("\nAll datasets saved successfully!")
