import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define model
class WindowedMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Load test dataset
test_df = pd.read_csv("test_dataset.csv")

X_test = test_df.drop("Target", axis=1).values
y_test = test_df["Target"].values

X_test = torch.tensor(X_test, dtype=torch.float32)

# Load trained model
model = WindowedMLP()

model.load_state_dict(
    torch.load(
        "windowed_mlp_model.pth",
        map_location="cpu"
    )
)

model.eval()

# Predict
with torch.no_grad():
    predictions = model(X_test).numpy().flatten()

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

# Display results
print("FINAL MODEL PERFORMANCE")
print("---------------------")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")

if rmse <= 0.15:
    print("Result: Target Achieved")

# Save results
results = pd.DataFrame({
    "Actual_Value": y_test,
    "Predicted_Value": predictions
})

results.to_csv("model_predictions.csv", index=False)

print("\nPredictions saved to model_predictions.csv")