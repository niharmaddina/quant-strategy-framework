import yfinance as yf

df = yf.download("SPY", start="2024-11-20", end="2025-11-20", auto_adjust=True, progress=False)
print(df.head())
print(df.tail())
print(df.shape)

