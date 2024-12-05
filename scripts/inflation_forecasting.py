# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:24:02 2024

@author: cbrut
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from xgboost import XGBRegressor

# Set current working directory
dir_path = 'C:/Users/cbrut/OneDrive/Documentos/Inflation Forecasting'
os.chdir(dir_path)

# Load UK macroeconomic data (GDP Growth, inflation, FX Change, Unemployment, Oil Price)
data = pd.read_excel(
    'data/UK DATA.xlsx',
    sheet_name='UK DATA',
    header=None,
    skiprows=5,
    usecols='B:F'
)

# Assigning the columns to variables for easier reference
gdpg = data.iloc[:, 0]  # GDP Growth (YoY)
infl = data.iloc[:, 1]  # (CPI) Inflation (YoY)
fx = data.iloc[:, 2]    # Changes in the sterling/dollar foreign exchange (YoY)
urate = data.iloc[:, 3] # Unemployment Rate (in levels) (YoY)
poil = data.iloc[:, 4]  # Sterling Price of Oil (YoY)

# Set up the dates (quarterly from 1981 to 2017)
dates = pd.date_range(start='1981-01-01', periods=len(infl), freq='Q')

# Parameters
t = len(infl)
r = 4 * 15  # In-sample period (start forecasting from 1981Q1+R quarters)
horizon = 2  # Produce forecasts for 2 quarters ahead inflation
p = t - r - horizon  # Out-of-sample period
l = r  # Rolling window length
rolling = True  # Set rolling window flag

# Define target variable (YoY inflation)
y = infl
target = y[-p:]

"""
Compute the h-step ahead forecast for inflation using the following models:
    
a) The RW model
b) A naive flat forecast at 2%
c) An ARIMA(0,1,1) model
d) A bivariate VAR(2) model with unemployment
e) A trivariate VAR(2) model with the exchange rate and the price of oil
f) XGBoost
    
"""
# Initialize forecast arrays for different models
rw_forecast = np.full(p, np.nan)
naive_forecast = np.full(p, np.nan)
arima_forecast = np.full(p, np.nan)
var_bi_forecast = np.full(p, np.nan)
var_tri_forecast = np.full(p, np.nan)
xgb_forecast = np.full(p, np.nan)

# Rolling window forecast loop
for t_end in range(l, t - horizon):
    t_init = t_end - l + 1 if rolling else 0
    y_used = y[t_init:t_end]

    # Random Walk Forecast (RW)
    rw_forecast[t_end - l] = y_used.iloc[-1]

    # Naive Forecast (constant 2% inflation)
    naive_forecast[t_end - l] = 2

    # ARIMA Forecast
    model_arima = ARIMA(y_used, order=(0, 1, 1))
    arima_fitted = model_arima.fit()
    arima_forecast[t_end - l] = arima_fitted.forecast(
        steps=horizon
    ).iloc[-1]

    # Bivariate VAR Forecast (with Unemployment)
    predictors_used_bi = urate[t_init:t_end]
    model_var_bi = VAR(np.column_stack([y_used, predictors_used_bi]))
    var_bi_fitted = model_var_bi.fit(2)  # VAR(2)
    var_bi_forecast[t_end - l] = var_bi_fitted.forecast(
        np.column_stack([y_used, predictors_used_bi])[-2:],
        steps=horizon
    )[-1, 0]

    # Trivariate VAR Forecast (with FX and Price of Oil)
    predictors_used_tri = np.column_stack([fx[t_init:t_end], poil[t_init:t_end]])
    model_var_tri = VAR(np.column_stack([y_used, predictors_used_tri]))
    var_tri_fitted = model_var_tri.fit(2)  # VAR(2)
    var_tri_forecast[t_end - l] = var_tri_fitted.forecast(
        np.column_stack([y_used, predictors_used_tri])[-2:],
        steps=horizon
    )[-1, 0]
    
    # XGBoost
    # Prepare data: features X and target y
    X_train = np.column_stack([urate[t_init:t_end], fx[t_init:t_end], poil[t_init:t_end]])
    y_train = y_used.to_numpy()

    # Features from most recent period used to predict
    X_test = np.array([urate[t_end - 1], fx[t_end - 1], poil[t_end - 1]]).reshape(1, -1)

    # Define and traing XGBoost model
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100,
                             max_depth=3, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)

    # Generate forecast
    xgb_forecast[t_end - l] = xgb_model.predict(X_test)[0]
    
    
"""
Plot real inflation and model predictions
"""
# Combine the real series (in-sample) with the forecasts (out-of-sample)
forecast_df = pd.DataFrame({
    'Real': y.to_numpy(),
    'RW': np.concatenate([y[:r+horizon], rw_forecast]),
    'Naive': np.concatenate([y[:r+horizon], naive_forecast]),
    'ARIMA': np.concatenate([y[:r+horizon], arima_forecast]),
    'VAR_Bi': np.concatenate([y[:r+horizon], var_bi_forecast]),
    'VAR_Tri': np.concatenate([y[:r+horizon], var_tri_forecast]),
    'XGBoost': np.concatenate([y[:r+horizon], xgb_forecast])
}, index=dates)

# Plot real inflation vs. inflation forecasts
plt.figure(figsize=(15, 8))
plt.plot(forecast_df['Real'], label='Real Inflation', color='black', linewidth=2)
plt.plot(forecast_df['RW'], label='Random Walk', linestyle='--', alpha=0.7)
plt.plot(forecast_df['Naive'], label='Naive', linestyle=':', alpha=0.7)
plt.plot(forecast_df['ARIMA'], label='ARIMA', linestyle='-.', alpha=0.7)
plt.plot(forecast_df['VAR_Bi'], label='VAR (Bivariate)', linestyle='--', alpha=0.7)
plt.plot(forecast_df['VAR_Tri'], label='VAR (Trivariate)', linestyle=':', alpha=0.7)
plt.plot(forecast_df['XGBoost'], label='XGBoost', linestyle='-.', alpha=0.7)

# Customize the plot
plt.axvline(dates[r], color='red', linestyle='--', label='Forecast Start')  # Mark start of forecast
plt.title('Real Inflation vs. Inflation Forecasts', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Inflation (YoY)', fontsize=16)
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)
plt.legend(loc='best', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot
plt.savefig('figures/inflation_forecasts_plot.png')

# Show the plot
plt.show()

"""
Compute forecast errors and calculate RMSE and MAE for each model
"""
# Compute forecast errors
forecast_errors = np.column_stack([
    rw_forecast,
    naive_forecast,
    arima_forecast,
    var_bi_forecast,
    var_tri_forecast,
    xgb_forecast
]) - np.array(target)[:, None]

# Calculate RMSE and MAE for each model
rmse = np.sqrt(np.mean(forecast_errors ** 2, axis=0))
mae = np.mean(np.abs(forecast_errors), axis=0)

# Compare models
model_names = ['Random Walk', 'Naive Forecast', 'ARIMA', 'Bivariate VAR',
               'Trivariate VAR', 'XGBoost']

# Display RMSE and MAE results for each model
for i, model in enumerate(model_names):
    print(f"{model}: RMSE = {rmse[i]:.4f}, MAE = {mae[i]:.4f}")

# Identify the model with the lowest RMSE and MAE
best_rmse_model = model_names[np.argmin(rmse)]
best_mae_model = model_names[np.argmin(mae)]

print(f"\nBest model based on RMSE: {best_rmse_model}")
print(f"Best model based on MAE: {best_mae_model}")

# Plot RMSE and MAE comparison for each model
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# RMSE plot
ax[0].bar(model_names, rmse, color='skyblue')
ax[0].set_title('RMSE Comparison')
ax[0].set_ylabel('RMSE')
ax[0].tick_params(axis='x', rotation=45)

# MAE plot
ax[1].bar(model_names, mae, color='salmon')
ax[1].set_title('MAE Comparison')
ax[1].set_ylabel('MAE')
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/rmse_mae_models_comparison_plot.png')
plt.show()

"""
Plot forecast errors
"""
plt.figure(figsize=(12, 6))
plt.plot(forecast_errors[:, 0], label='RW Errors', linestyle='--', alpha=0.7)
plt.plot(forecast_errors[:, 1], label='Naive Errors', linestyle=':', alpha=0.7)
plt.plot(forecast_errors[:, 2], label='ARIMA Errors', linestyle='-.', alpha=0.7)
plt.plot(forecast_errors[:, 3], label='Bivariate VAR Errors', linestyle='--', alpha=0.7)
plt.plot(forecast_errors[:, 4], label='Trivariate VAR Errors', linestyle=':', alpha=0.7)
plt.plot(forecast_errors[:, 5], label='XGboost Errors', linestyle='-.', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title('Forecast Errors Comparison', fontsize=16)
plt.xlabel('Period', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/inflation_forecasts_errors_plot.png')
plt.show()

"""
Combine all the models considering using weights proportional to RMSE 
"""

# Calculate weights as the inverse of RMSE (normalized)
weights = 1 / rmse
weights /= weights.sum()

# Combine the models using the weights
weighted_forecast = (
    weights[0] * rw_forecast +
    weights[1] * naive_forecast +
    weights[2] * arima_forecast +
    weights[3] * var_bi_forecast +
    weights[4] * var_tri_forecast +
    weights[5] * xgb_forecast
)

# Compute forecast errors for the combined forecast
weighted_forecast_errors = weighted_forecast - np.array(target)

# Calculate RMSE and MAE for the combined forecast
weighted_forecast_rmse = np.sqrt(np.mean(weighted_forecast_errors ** 2))
weighted_forecast_mae = np.mean(np.abs(weighted_forecast_errors))

# Display results
print(f"Combined Forecast (Including Random Walk): "
      f"RMSE = {weighted_forecast_rmse:.4f}, MAE = {weighted_forecast_mae:.4f}")
print(f"Benchmark (Random Walk): RMSE = {rmse[0]:.4f}, MAE = {mae[0]:.4f}")

# Compare to the benchmark model
if weighted_forecast_rmse < rmse[0]:
    print("The combined forecast (including Random Walk) has a lower RMSE "
          "compared to the benchmark.")
else:
    print("The benchmark model has a lower RMSE than the combined forecast "
          "(including Random Walk).")

if weighted_forecast_mae < mae[0]:
    print("The combined forecast (including Random Walk) has a lower MAE " 
          "compared to the benchmark.")
else:
    print("The benchmark model has a lower MAE than the combined forecast "
          "(including Random Walk).")

# Final forecast plot
# Combine the real series (in-sample) with the forecasts (out-of-sample)
weighted_forecast_df = pd.DataFrame({
    'Real': y.to_numpy(),
    'Weighted Forecast': np.concatenate([y[:r+horizon], weighted_forecast])
}, index=dates)

# Plot real inflation vs. inflation forecast
plt.figure(figsize=(15, 8))
plt.plot(weighted_forecast_df['Real'], label='Real Inflation', color='black', linewidth=2)
plt.plot(weighted_forecast_df['Weighted Forecast'], label='Weighted Model Inflation Forecast', linestyle='--', alpha=0.7)

# Customize the plot
plt.axvline(dates[r], color='red', linestyle='--', label='Forecast Start')  # Mark start of forecast
plt.title('Real Inflation vs Weighted Model Inflation Forecast', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Inflation (YoY)', fontsize=16)
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)
plt.legend(loc='best', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot
plt.savefig('figures/inflation_weighted_forecast_plot.png')

# Show the plot
plt.show()

