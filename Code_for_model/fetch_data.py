import yfinance as yf

# Download stock data for Google
data = yf.download("GOOG", start="2015-01-01", end="2025-01-01")
data.reset_index(inplace=True)
data.to_csv("Google_train_data.csv", index=False)
print("Data downloaded and saved as Google_train_data.csv")