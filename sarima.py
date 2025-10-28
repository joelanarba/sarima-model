# -----------------------------
# AGRICULTURAL PRICE FORECAST (SARIMA)
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. LOAD & CLEAN DATA
# -----------------------------
df = pd.read_csv("wfp_food_prices_gha.csv", header=0, low_memory=False)
df = df[df['date'] != '#date']
df['market'] = df['market'].str.strip()
df['commodity'] = df['commodity'].str.strip()

if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
else:
    df['price'] = pd.to_numeric(df['usdprice'], errors='coerce')

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['price', 'date'])

# -----------------------------
# 2. FILTER MARKET & COMMODITY
# -----------------------------
market = "Kumasi"
commodity = "Maize"
data = df[(df['market'] == market) & (df['commodity'] == commodity)]
if data.empty:
    raise ValueError(f"No data found for {commodity} in {market}.")

data = data.groupby('date', as_index=False)['price'].mean().sort_values('date')
series = data.set_index('date')['price']

# -----------------------------
# 3. OUTLIER REMOVAL & SMOOTHING
# -----------------------------
q1, q3 = series.quantile([0.25, 0.75])
iqr = q3 - q1
series = series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
series = series.resample('M').mean().interpolate()
series_smooth = series.rolling(window=3, center=True, min_periods=1).mean()

# -----------------------------
# 4. LOG TRANSFORMATION
# -----------------------------
series_log = np.log1p(series_smooth)

# -----------------------------
# 5. TRAIN/TEST SPLIT
# -----------------------------
train = series_log[:-12]
test = series_log[-12:]

# -----------------------------
# 6. AUTO ARIMA MODEL
# -----------------------------
print("Finding best SARIMA parameters — please wait...\n")
auto_model = auto_arima(
    train,
    seasonal=True, m=12,
    stepwise=True,
    trace=True,
    max_p=5, max_q=5, max_P=3, max_Q=3,
    d=None, D=None,
    information_criterion='aicc',
    error_action='ignore',
    suppress_warnings=True
)
print(f"\nOptimal parameters: {auto_model.order} x {auto_model.seasonal_order}")

# -----------------------------
# 7. FIT MODEL
# -----------------------------
model = SARIMAX(train,
                order=auto_model.order,
                seasonal_order=auto_model.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)
result = model.fit(disp=False)
print("Model fitted successfully.\n")

# -----------------------------
# 8. FORECAST
# -----------------------------
forecast = result.get_forecast(steps=12)
forecast_mean = np.expm1(forecast.predicted_mean)
test_real = np.expm1(test)
mae = mean_absolute_error(test_real, forecast_mean)
print(f"Mean Absolute Error (MAE): {mae:.2f} GHS\n")

# ==================================================
#  EXTRA VISUALIZATIONS FOR REPORT
# ==================================================

# Figure 1 — Raw Historical Prices
plt.figure(figsize=(14,5))
plt.plot(series.index, series.values, color='steelblue')
plt.title(f"{commodity} Prices in {market} (Historical Monthly Series)")
plt.xlabel("Date")
plt.ylabel("Price (GHS)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 2 — Smoothed vs Raw Prices
plt.figure(figsize=(14,5))
plt.plot(series.index, series.values, label='Raw', color='lightgray')
plt.plot(series_smooth.index, series_smooth.values, label='Smoothed (3-mo avg)', color='green')
plt.title("Raw vs Smoothed Monthly Price Series")
plt.xlabel("Date")
plt.ylabel("Price (GHS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 3 — Seasonal Boxplot by Month
df_box = pd.DataFrame({'price': series.values})
df_box['month'] = series.index.month
df_box['month_name'] = series.index.month_name()
plt.figure(figsize=(12,5))
sns.boxplot(data=df_box, x='month_name', y='price', palette='viridis')
plt.title(f"Seasonal Price Variation by Month — {commodity} ({market})")
plt.xticks(rotation=45)
plt.ylabel("Price (GHS)")
plt.tight_layout()
plt.show()

# Figure 4 — Observed vs Forecast (Main)
plt.figure(figsize=(14,6))
plt.plot(series.index, series.values, label='Observed', color='blue')
plt.plot(forecast_mean.index, forecast_mean.values, label='Forecast (Next 12 months)', color='red', linestyle='--', marker='o')
plt.title(f"{commodity} Price Forecast — {market}\nSARIMA {auto_model.order}x{auto_model.seasonal_order}, MAE={mae:.2f} GHS")
plt.xlabel("Date")
plt.ylabel("Price (GHS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 5 — Residual Diagnostics
result.plot_diagnostics(figsize=(12,8))
plt.tight_layout()
plt.show()

# Figure 6 — Forecast vs Actual (Zoomed Test)
plt.figure(figsize=(10,5))
plt.plot(test_real.index, test_real.values, label='Actual', color='blue', marker='o')
plt.plot(forecast_mean.index, forecast_mean.values, label='Predicted', color='red', marker='x')
plt.title(f"Test Period Forecast Comparison ({market} - {commodity})")
plt.xlabel("Date")
plt.ylabel("Price (GHS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 7 — Model Residuals Over Time
residuals = result.resid
plt.figure(figsize=(12,5))
plt.plot(residuals.index, residuals.values, color='purple')
plt.title("Model Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 8 — Residual Distribution
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=20, kde=True, color='teal')
plt.title("Residual Distribution (Normality Check)")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
