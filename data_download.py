import yfinance as yf
import pandas as pd

ticker = "AAPL"
data = yf.download(ticker, start="2023-01-01", end="2025-01-01")

print("First 5 rows of Apple stock data:")
print(data.head())

print("\nShape of data (rows, columns):")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())