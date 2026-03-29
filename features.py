import yfinance as yf
import pandas as pd
import pandas_ta as ta

ticker = "AAPL"
data = yf.download(ticker, start="2023-01-01", end="2025-01-01", auto_adjust=True)

data.columns = [col[0] for col in data.columns]

data.columns = ["close", "high", "low", "open", "volume"]

print("Raw data shape:", data.shape)
print(data.head(3))
## Add technical indicators

# RSI (Relative Strength Index) - 14 day
# Measures if a stock is overbought (>70) or oversold (<30)
# Values always between 0 and 100
data["rsi"] = ta.rsi(data["close"], length=14)

# MACD (Moving Average Convergence Divergence)
# Measures momentum - when the MACD line crosses the signal line
# it suggests a trend change is happening
macd = ta.macd(data["close"], fast=12, slow=26, signal=9)
data["macd"] = macd["MACD_12_26_9"]
data["macd_signal"] = macd["MACDs_12_26_9"]

# EMA ratio (Exponential Moving Average)
# EMA 9 = average of last 9 days (short term trend)
# EMA 21 = average of last 21 days (longer term trend)
# When EMA9 > EMA21 the stock is in a short term uptrend
data["ema9"] = ta.ema(data["close"], length=9)
data["ema21"] = ta.ema(data["close"], length=21)
data["ema_ratio"] = data["ema9"] / data["ema21"]

# Bollinger Bands %B
# Tells us where the price sits relative to its volatility range
# Above 1.0 = price broke above upper band (overbought signal)
# Below 0.0 = price broke below lower band (oversold signal)
bbands = ta.bbands(data["close"], length=20, std=2)
data["bb_percent"] = bbands["BBP_20_2.0_2.0"]

# ATR (Average True Range)
# Measures how volatile the stock is
# High ATR = big price swings, Low ATR = calm market
data["atr"] = ta.atr(data["high"], data["low"], data["close"], length=14)

# Volume ratio
# Compares today's volume to the 10 day average
# High ratio means unusual activity - often precedes big moves
data["volume_ratio"] = data["volume"] / data["volume"].rolling(10).mean()


##Create the label (what we want to predict)


# 1 means the price went UP the next day
# 0 means the price went DOWN the next day
# shift(-1) moves tomorrow's close into today's row so we can compare
data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)

## Clean up

# The first ~26 rows will have NaN (empty) values because indicators
# need a warmup period to calculate. We drop those rows.
data = data.dropna()

print("\nData with indicators shape:", data.shape)
print("\nFirst 3 rows with all features:")
print(data.head(3))

print("\nColumn names:")
print(data.columns.tolist())

print("\nLabel distribution (how many UP vs DOWN days):")
print(data["target"].value_counts())