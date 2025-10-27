import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. Load & clean data
# -----------------------------
df = pd.read_csv("wfp_food_prices_nga.csv", header=0, low_memory=False)
df = df[df['date'] != '#date']
df['market'] = df['market'].str.strip()
df['commodity'] = df['commodity'].str.strip()
df['usdprice'] = pd.to_numeric(df['usdprice'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['usdprice','date'])

# -----------------------------
# 2. Filter market & commodity
# -----------------------------
market = "Dawanau"
commodity = "Maize (white)"
data = df[(df['market'] == market) & (df['commodity'] == commodity)]

if data.empty:
    print(f"No data for {commodity} in {market}.")
    exit()

# -----------------------------
# 3. Handle duplicates and sort
# -----------------------------
data = data.groupby('date', as_index=False)['usdprice'].mean()
data = data.sort_values('date')
historical = data.set_index('date')['usdprice']

# -----------------------------
# 4. Clip outliers (1% extremes)
# -----------------------------
lower, upper = historical.quantile([0.01, 0.99])
historical_clipped = historical.clip(lower=lower, upper=upper)

# -----------------------------
# 5. Trim historical to end Dec 2022
# -----------------------------
historical_trimmed = historical_clipped[historical_clipped.index <= "2022-12-31"]

# -----------------------------
# 6. Resample monthly & interpolate missing months
# -----------------------------
series = historical_trimmed.resample('MS').mean().interpolate()

# -----------------------------
# 7. Smooth historical series
# -----------------------------
series_smooth = series.rolling(window=3, center=True, min_periods=1).mean()

# -----------------------------
# 8. Fit SARIMA on full data
# -----------------------------
model = SARIMAX(series_smooth, order=(1,0,1), seasonal_order=(0,1,1,12))
result = model.fit(disp=False)
print("SARIMA model fitted successfully.")

# -----------------------------
# 9. Compute in-sample MAE
# -----------------------------
in_sample_pred = result.fittedvalues
mae = mean_absolute_error(series_smooth.dropna(), in_sample_pred.dropna())

# -----------------------------
# 10. Forecast 12 months ahead (from Jan 2023)
# -----------------------------
forecast_steps = 12
forecast = result.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# -----------------------------
# 11. Combine historical + forecast
# -----------------------------
full_series = pd.concat([series_smooth, forecast_mean])

# Y-axis limits
all_values = full_series[np.isfinite(full_series)]
ymin, ymax = all_values.min(), all_values.max()

# -----------------------------
# 12. Plot
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(series_smooth.index, series_smooth.values, label='Observed (smoothed)', color='blue', marker='o')
plt.plot(forecast_mean.index, forecast_mean.values, label='Forecast', color='red', marker='o')
plt.title(f"{commodity} Price Forecast - {market}\nMAE: {mae:.2f}")
plt.xlabel("Date")
plt.ylabel("USD Price")
plt.ylim(ymin*0.95, ymax*1.05)
plt.legend()
plt.show()
