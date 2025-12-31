import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Load and preprocess the data
file_path = r"C:\Users\dhadwama\Downloads\Data\weather\Espoo Tapiola_ 30.12.2014 - 30.12.2024.csv"

# Try different encodings and separators to read the CSV properly
try:
    df = pd.read_csv(file_path, encoding='utf-8', sep=',', on_bad_lines='skip')
except:
    try:
        df = pd.read_csv(file_path, encoding='latin1', sep=',', on_bad_lines='skip')
    except:
        df = pd.read_csv(file_path, encoding='iso-8859-1', sep=';', on_bad_lines='skip')

print(f"Dataset shape: {df.shape}")
print("Columns:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Drop rows with missing values that might affect date parsing
df.dropna(subset=['Year', 'Month', 'Day'], inplace=True)

# Convert date components to integers
df[['Year', 'Month', 'Day']] = df[['Year', 'Month', 'Day']].astype(int)

# Create a date column
df['ds'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# Rename columns for Prophet
df_max = df[['ds', 'Maximum temperature [°C]']].rename(columns={'Maximum temperature [°C]': 'y'})
df_min = df[['ds', 'Minimum temperature [°C]']].rename(columns={'Minimum temperature [°C]': 'y'})

# Calculate average temperature
df['avg_temp'] = (df['Maximum temperature [°C]'] + df['Minimum temperature [°C]']) / 2
df_avg = df[['ds', 'avg_temp']].rename(columns={'avg_temp': 'y'})

# 2. Train Prophet models
# Max temperature model
model_max = Prophet()
model_max.fit(df_max)

# Min temperature model
model_min = Prophet()
model_min.fit(df_min)

# Avg temperature model
model_avg = Prophet()
model_avg.fit(df_avg)

# 3. Generate forecasts for 2025
# Calculate days needed to reach end of 2025
last_date = df['ds'].max()
end_2025 = pd.Timestamp('2025-12-31')
days_to_forecast = (end_2025 - last_date).days + 1

print(f"Last date in dataset: {last_date}")
print(f"Days to forecast: {days_to_forecast}")

future_dates = model_max.make_future_dataframe(periods=days_to_forecast)
print(f"Future dates range: {future_dates['ds'].min()} to {future_dates['ds'].max()}")

forecast_max = model_max.predict(future_dates)
forecast_min = model_min.predict(future_dates)
forecast_avg = model_avg.predict(future_dates)

# Filter for December 2025
forecast_dec_2025_max = forecast_max[forecast_max['ds'].dt.to_period('M') == '2025-12']
forecast_dec_2025_min = forecast_min[forecast_min['ds'].dt.to_period('M') == '2025-12']
forecast_dec_2025_avg = forecast_avg[forecast_avg['ds'].dt.to_period('M') == '2025-12']

print(f"December 2025 forecast rows: {len(forecast_dec_2025_max)}")

# 4. Visualize the results
# Plot max temperature forecast
fig_max = model_max.plot(forecast_max)
plt.title('Maximum Temperature Forecast for 2025')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.show()

# Plot min temperature forecast
fig_min = model_min.plot(forecast_min)
plt.title('Minimum Temperature Forecast for 2025')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.show()

# Plot avg temperature forecast
fig_avg = model_avg.plot(forecast_avg)
plt.title('Average Temperature Forecast for 2025')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.show()

# Plot components
fig_max_comp = model_max.plot_components(forecast_max)
plt.show()

fig_min_comp = model_min.plot_components(forecast_min)
plt.show()

fig_avg_comp = model_avg.plot_components(forecast_avg)
plt.show()


# Combine forecasts into a single DataFrame
if len(forecast_dec_2025_max) > 0:
    forecast_dec_2025 = pd.DataFrame({
        'Date': forecast_dec_2025_max['ds'].reset_index(drop=True),
        'Max_Temperature_Forecast': forecast_dec_2025_max['yhat'].reset_index(drop=True),
        'Min_Temperature_Forecast': forecast_dec_2025_min['yhat'].reset_index(drop=True),
        'Avg_Temperature_Forecast': forecast_dec_2025_avg['yhat'].reset_index(drop=True)
    })
    
    # Save the combined forecast to a CSV file
    output_path = 'forecast_december_2025.csv'
    forecast_dec_2025.to_csv(output_path, index=False)
    
    print(f"Forecast for December 2025 saved to {output_path}")
    print(f"Number of days forecasted: {len(forecast_dec_2025)}")
    print("First few rows of forecast:")
    print(forecast_dec_2025.head())
else:
    print("No December 2025 data found in forecast. Check date range and filtering.")
    print("Available forecast date range:")
    print(f"From: {forecast_max['ds'].min()} To: {forecast_max['ds'].max()}")


