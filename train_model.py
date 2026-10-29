
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------------
# Step 1: Set random seed
# -----------------------------------
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------
# Step 2: Load train and validation data
# -----------------------------------
train_df = pd.read_csv("train_dataset.csv")
val_df = pd.read_csv("validation_dataset.csv")

# Separate input features and target
X_train = train_df.drop("Target", axis=1).values
y_train = train_df["Target"].values

X_val = val_df.drop("Target", axis=1).values
y_val = val_df["Target"].values

# Convert data into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

# -----------------------------------
# Step 3: Build Windowed MLP model
# -----------------------------------
class WindowedMLP(nn.Module):

    def __init__(self):
        super(WindowedMLP, self).__init__()

        self.network = nn.Sequential(

            # Input: 20 previous values
            nn.Linear(20, 64),
            nn.Tanh(),

            # Hidden layer
            nn.Linear(64, 32),
            nn.Tanh(),

            # Output: Next predicted value
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Create model
model = WindowedMLP()

print("\nMLP Model:")
print(model)

# -----------------------------------
# Step 4: Define loss and optimizer
# -----------------------------------
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# -----------------------------------
# Step 5: Train the model
# -----------------------------------
epochs = 200

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_model_state = None

patience = 20
counter = 0

print("\nStarting model training...")

for epoch in range(epochs):

    # Training mode
    model.train()

    # Forward pass
    train_predictions = model(X_train)

    # Calculate training loss
    train_loss = criterion(train_predictions, y_train)

    # Backpropagation
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation mode
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val)

        val_loss = criterion(val_predictions, y_val)

    # Store losses
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    # Save best model
    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        best_model_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

        counter = 0

    else:
        counter += 1

    # Display progress
    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss.item():.6f} "
            f"Val Loss: {val_loss.item():.6f}"
        )

    # Early stopping
    if counter >= patience:

        print("\nEarly stopping activated.")
        break


# -----------------------------------
# Step 6: Load best model
# -----------------------------------
model.load_state_dict(best_model_state)

# Save trained model
torch.save(
    model.state_dict(),
    "windowed_mlp_model.pth"
)

print("\nTraining completed!")
print("Best Validation Loss:", best_val_loss)
print("Model saved as windowed_mlp_model.pth")


# -----------------------------------
# Step 7: Plot training and validation loss
# -----------------------------------
plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig("training_loss.png")
plt.show()

print("Training loss graph saved!")

