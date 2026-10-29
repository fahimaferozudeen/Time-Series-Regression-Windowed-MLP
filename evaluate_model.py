import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Define the MLP model
class WindowedMLP(nn.Module):
    def __init__(self):
        super(WindowedMLP, self).__init__()

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

# Load the trained model
model = WindowedMLP()

model.load_state_dict(
    torch.load(
        "windowed_mlp_model.pth",
        map_location="cpu"
    )
)

model.eval()

# Predict values
with torch.no_grad():
    predictions = model(X_test).numpy().flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Model Evaluation Completed!")
print("-------------------------")
print("Test RMSE:", rmse)

if rmse <= 0.15:
    print("Target Achieved! RMSE is <= 0.15")
else:
    print("Target Not Achieved. RMSE is > 0.15")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))

plt.plot(
    y_test,
    label="Actual Values"
)

plt.plot(
    predictions,
    label="Predicted Values"
)

plt.xlabel("Test Sample")
plt.ylabel("Value")
plt.title("Actual vs Predicted Values")

plt.legend()
plt.grid(True)

plt.savefig("actual_vs_predicted.png")

plt.show()

print("Graph saved as actual_vs_predicted.png")