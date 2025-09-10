import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Load the data
data_path = 'Google_train_data.csv'  # Make sure this file is in the repo
data = pd.read_csv(data_path)

# Convert 'Close' and 'Volume' columns to numeric (cleaning commas if present)
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

# Drop rows with missing values
data.dropna(inplace=True)

# Add moving averages (5-day and 20-day)
data['5_day_MA'] = data['Close'].rolling(window=5).mean()
data['20_day_MA'] = data['Close'].rolling(window=20).mean()

# Add Exponential Moving Average (EMA)
data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()

# Drop rows with NaN after moving averages
data.dropna(inplace=True)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Define features and target
X = data[['Open', 'High', 'Low', 'Volume', '5_day_MA', '20_day_MA', 'EMA']]
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Predict on the full dataset for visualization
data['Predicted'] = model.predict(X)

# Save results to JSON (for frontend or visualization)
predictions = []
for i in range(len(data)):
    predictions.append({
        "date": data['Date'].iloc[i].strftime('%Y-%m-%d'),
        "actual": float(data['Close'].iloc[i]),
        "predicted": float(data['Predicted'].iloc[i])
    })

with open('predictions.json', 'w') as f:
    json.dump(predictions, f, indent=4)

print("Predictions saved to 'predictions.json'")
