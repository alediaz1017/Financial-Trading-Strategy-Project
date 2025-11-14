import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# 1. Data Loading and Merging

df_hai = pd.read_csv("hai_calculator_results.csv")
df_prices = pd.read_csv("past_median_household_prices.csv")

# Merge on "County" (inner join ensures counties appear in both files)
df_combined = pd.merge(df_hai[['County', 'HAI (Low-Income $118,650)']], 
                       df_prices, on="County", how="inner")
print("Merged dataset shape:", df_combined.shape)
print("Columns:", df_combined.columns.tolist())


# 2. Prepare Historical Price Data

# For multi-step forecasting, we now use six years of data.
# Example: historical price columns from 2019 to 2024.
price_columns = ['2019', '2020', '2021', '2022', '2023', '2024']
df_prices_only = df_combined.set_index("County")[price_columns]
print("Number of counties:", df_prices_only.shape[0])
print("Historical price columns:", df_prices_only.columns.tolist())


# 3. Create Rolling Window Samples for Multi-step Forecasting

# We define:
#   input_length = 3 (years) and output_length = 3 (years)
input_length = 3
output_length = 3

X_data, y_data = [], []
for county in df_prices_only.index:
    series = df_prices_only.loc[county].values.astype(float)
    # We can only form a sample if we have at least (input_length + output_length) years
    if len(series) < (input_length + output_length):
        continue
    # Create samples: sliding window that takes input_length years as input 
    # and the following output_length years as the target.
    for i in range(len(series) - input_length - output_length + 1):
        X_data.append(series[i : i + input_length])
        y_data.append(series[i + input_length : i + input_length + output_length])
X_data = np.array(X_data)  # shape: (samples, input_length)
y_data = np.array(y_data)  # shape: (samples, output_length)
print("Full dataset X shape:", X_data.shape)
print("Full dataset y shape:", y_data.shape)


# 4. Split the Data into Train, Validation, and Test Sets

n_samples = X_data.shape[0]
train_end = int(0.7 * n_samples)
val_end = int(0.85 * n_samples)
X_train = X_data[:train_end]
y_train = y_data[:train_end]
X_val = X_data[train_end:val_end]
y_val = y_data[train_end:val_end]
X_test = X_data[val_end:]
y_test = y_data[val_end:]
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")


# 5. Normalize the Data

# Flatten all values so that the scaler learns a global min and max.
all_values = np.concatenate([X_data.flatten(), y_data.flatten()])
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_values.reshape(-1, 1))
print("Scaler fitted on all values.")

# Transform X and y (flatten, scale, then reshape back).
X_train_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
X_val_scaled   = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
y_val_scaled   = scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
X_test_scaled  = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
y_test_scaled  = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# For RNN, reshape X to (samples, input_length, features); here features = 1.
X_train_scaled = X_train_scaled.reshape(-1, input_length, 1)
X_val_scaled   = X_val_scaled.reshape(-1, input_length, 1)
X_test_scaled  = X_test_scaled.reshape(-1, input_length, 1)
# y remains as (samples, output_length)
print("Shapes after scaling: X_train:", X_train_scaled.shape, "y_train:", y_train_scaled.shape)

# Convert arrays to PyTorch tensors.
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_scaled, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test_scaled, dtype=torch.float32)


# 6. Define the RNN Model for Multi-step Forecasting

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        # Update the final fully-connected layer to output output_size values
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

input_size = 1       # one feature per time step
hidden_size = 50
output_size = output_length  # now output is 3 values (for 3 future years)
model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 7. Train the RNN Model and Compute MAPE

num_epochs = 100
train_losses, val_losses = [], []
train_mapes, val_mapes = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # Compute training MAPE (inverse-transform predictions and true values)
    with torch.no_grad():
        outputs_np = outputs.detach().numpy()
    y_train_pred_orig = scaler.inverse_transform(outputs_np.reshape(-1,1)).reshape(outputs_np.shape)
    y_train_true_orig = scaler.inverse_transform(y_train_tensor.numpy().reshape(-1,1)).reshape(y_train_tensor.shape)
    mape_train = np.mean(np.abs((y_train_true_orig - y_train_pred_orig) / y_train_true_orig)) * 100
    train_mapes.append(mape_train)
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    val_losses.append(val_loss.item())
    with torch.no_grad():
        val_outputs_np = val_outputs.detach().numpy()
    y_val_pred_orig = scaler.inverse_transform(val_outputs_np.reshape(-1,1)).reshape(val_outputs_np.shape)
    y_val_true_orig = scaler.inverse_transform(y_val_tensor.numpy().reshape(-1,1)).reshape(y_val_tensor.shape)
    mape_val = np.mean(np.abs((y_val_true_orig - y_val_pred_orig) / y_val_true_orig)) * 100
    val_mapes.append(mape_val)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, " +
              f"Train MAPE: {mape_train:.2f}%, Val MAPE: {mape_val:.2f}%")


# 8. Evaluate on Test Set

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
print(f"Test Performance - MSE: {test_loss.item():.4f}")
with torch.no_grad():
    test_outputs_np = test_outputs.detach().numpy()
y_test_pred_orig = scaler.inverse_transform(test_outputs_np.reshape(-1,1)).reshape(test_outputs_np.shape)
y_test_true_orig = scaler.inverse_transform(y_test_tensor.numpy().reshape(-1,1)).reshape(y_test_tensor.shape)
test_mape = np.mean(np.abs((y_test_true_orig - y_test_pred_orig) / y_test_true_orig)) * 100
print(f"Test MAPE: {test_mape:.2f}%")


# 9. Plot Training and Validation Metrics

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(train_mapes, label='Train MAPE')
plt.plot(val_mapes, label='Validation MAPE')
plt.xlabel('Epoch')
plt.ylabel('Mean Abolsute Percentage Error Loss')
plt.legend()
plt.show()


# 10. Predict Future Housing Prices and Save to CSV

# For future prediction, we use the last 3 years of historical prices (2022-2024) as input.
# (Assume that the most recent columns in your CSV are 2022, 2023, 2024.)
future_input_years = ['2022', '2023', '2024']
predictions = {}
for county in df_prices_only.index:
    # We assume the county has data for these years.
    # Extract the values and ensure they are float.
    series = df_prices_only.loc[county][future_input_years].values.astype(float)
    if len(series) < input_length:
        continue
    # Reshape and scale the input.
    input_seq = series.reshape(-1, 1)  # shape (3, 1)
    input_seq_scaled = scaler.transform(input_seq).reshape(1, input_length, 1)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(input_seq_scaled, dtype=torch.float32)).numpy()
    # Inverse transform to obtain the predicted prices in original scale.
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).reshape(pred_scaled.shape)
    predictions[county] = pred[0].tolist()  # list of 3 predicted values

df_future_prices_rnn = pd.DataFrame({
    "County": list(predictions.keys()),
    "Predicted Housing Prices (RNN Future)": list(predictions.values())
})

df_future_prices_rnn.to_csv("future_housing_prices_rnn.csv", index=False)
print("Future RNN housing prices saved to future_housing_prices_rnn.csv")
print(df_future_prices_rnn)
