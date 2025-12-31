import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

# 1. Load the actual 2025 data
actual_2025_file = r"C:\Users\dhadwama\Downloads\Data\weather\Espoo Tapiola_ 1.1.2025 - 30.12.2025.csv"

# Try different encodings and separators to read the CSV properly
try:
    actual_2025 = pd.read_csv(actual_2025_file, encoding='utf-8', sep=',', on_bad_lines='skip')
except:
    try:
        actual_2025 = pd.read_csv(actual_2025_file, encoding='latin1', sep=',', on_bad_lines='skip')
    except:
        actual_2025 = pd.read_csv(actual_2025_file, encoding='iso-8859-1', sep=';', on_bad_lines='skip')

print("Actual 2025 data shape:", actual_2025.shape)
print("Columns:", actual_2025.columns.tolist())
print("First few rows:")
print(actual_2025.head())

# Clean and prepare actual 2025 data
actual_2025.dropna(subset=['Year', 'Month', 'Day'], inplace=True)
actual_2025[['Year', 'Month', 'Day']] = actual_2025[['Year', 'Month', 'Day']].astype(int)
actual_2025['Date'] = pd.to_datetime(actual_2025[['Year', 'Month', 'Day']])
actual_2025['Actual_Avg_Temp'] = (actual_2025['Maximum temperature [°C]'] + actual_2025['Minimum temperature [°C]']) / 2

# 2. Load the forecast data
forecast_2025 = pd.read_csv('forecast_december_2025.csv')
forecast_2025['Date'] = pd.to_datetime(forecast_2025['Date'])

print("\nForecast data shape:", forecast_2025.shape)
print("Forecast date range:", forecast_2025['Date'].min(), "to", forecast_2025['Date'].max())

# 3. Filter actual data for December 2025 to match forecast
actual_dec_2025 = actual_2025[actual_2025['Date'].dt.month == 12]

print("\nActual December 2025 data shape:", actual_dec_2025.shape)
print("Actual December date range:", actual_dec_2025['Date'].min(), "to", actual_dec_2025['Date'].max())

# 4. Merge forecast and actual data
comparison_data = pd.merge(
    forecast_2025, 
    actual_dec_2025[['Date', 'Maximum temperature [°C]', 'Minimum temperature [°C]', 'Actual_Avg_Temp']], 
    on='Date', 
    how='inner'
)

print(f"\nComparison data shape: {comparison_data.shape}")

if len(comparison_data) > 0:
    # Rename actual columns for clarity
    comparison_data = comparison_data.rename(columns={
        'Maximum temperature [°C]': 'Actual_Max_Temp',
        'Minimum temperature [°C]': 'Actual_Min_Temp'
    })
    
    # 5. Calculate metrics
    mae_max = mean_absolute_error(comparison_data['Actual_Max_Temp'], comparison_data['Max_Temperature_Forecast'])
    mae_min = mean_absolute_error(comparison_data['Actual_Min_Temp'], comparison_data['Min_Temperature_Forecast'])
    mae_avg = mean_absolute_error(comparison_data['Actual_Avg_Temp'], comparison_data['Avg_Temperature_Forecast'])
    
    rmse_max = np.sqrt(mean_squared_error(comparison_data['Actual_Max_Temp'], comparison_data['Max_Temperature_Forecast']))
    rmse_min = np.sqrt(mean_squared_error(comparison_data['Actual_Min_Temp'], comparison_data['Min_Temperature_Forecast']))
    rmse_avg = np.sqrt(mean_squared_error(comparison_data['Actual_Avg_Temp'], comparison_data['Avg_Temperature_Forecast']))
    
    print("\n=== MODEL PERFORMANCE METRICS ===")
    print(f"Maximum Temperature - MAE: {mae_max:.2f}°C, RMSE: {rmse_max:.2f}°C")
    print(f"Minimum Temperature - MAE: {mae_min:.2f}°C, RMSE: {rmse_min:.2f}°C")
    print(f"Average Temperature - MAE: {mae_avg:.2f}°C, RMSE: {rmse_avg:.2f}°C")
    
    # 6. Create visualizations
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Maximum temperature comparison
    axes[0].plot(comparison_data['Date'], comparison_data['Actual_Max_Temp'], 'b-', label='Actual Max Temp', linewidth=2)
    axes[0].plot(comparison_data['Date'], comparison_data['Max_Temperature_Forecast'], 'r--', label='Predicted Max Temp', linewidth=2)
    axes[0].set_title('Maximum Temperature: Actual vs Predicted (December 2025)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Minimum temperature comparison
    axes[1].plot(comparison_data['Date'], comparison_data['Actual_Min_Temp'], 'b-', label='Actual Min Temp', linewidth=2)
    axes[1].plot(comparison_data['Date'], comparison_data['Min_Temperature_Forecast'], 'r--', label='Predicted Min Temp', linewidth=2)
    axes[1].set_title('Minimum Temperature: Actual vs Predicted (December 2025)')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Average temperature comparison
    axes[2].plot(comparison_data['Date'], comparison_data['Actual_Avg_Temp'], 'b-', label='Actual Avg Temp', linewidth=2)
    axes[2].plot(comparison_data['Date'], comparison_data['Avg_Temperature_Forecast'], 'r--', label='Predicted Avg Temp', linewidth=2)
    axes[2].set_title('Average Temperature: Actual vs Predicted (December 2025)')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('temperature_comparison_december_2025.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Create scatter plots for correlation analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Max temperature scatter
    axes[0].scatter(comparison_data['Actual_Max_Temp'], comparison_data['Max_Temperature_Forecast'], alpha=0.7, color='red')
    axes[0].plot([comparison_data['Actual_Max_Temp'].min(), comparison_data['Actual_Max_Temp'].max()], 
                 [comparison_data['Actual_Max_Temp'].min(), comparison_data['Actual_Max_Temp'].max()], 
                 'k--', alpha=0.5, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Max Temperature (°C)')
    axes[0].set_ylabel('Predicted Max Temperature (°C)')
    axes[0].set_title('Max Temperature: Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Min temperature scatter
    axes[1].scatter(comparison_data['Actual_Min_Temp'], comparison_data['Min_Temperature_Forecast'], alpha=0.7, color='blue')
    axes[1].plot([comparison_data['Actual_Min_Temp'].min(), comparison_data['Actual_Min_Temp'].max()], 
                 [comparison_data['Actual_Min_Temp'].min(), comparison_data['Actual_Min_Temp'].max()], 
                 'k--', alpha=0.5, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Min Temperature (°C)')
    axes[1].set_ylabel('Predicted Min Temperature (°C)')
    axes[1].set_title('Min Temperature: Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Average temperature scatter
    axes[2].scatter(comparison_data['Actual_Avg_Temp'], comparison_data['Avg_Temperature_Forecast'], alpha=0.7, color='green')
    axes[2].plot([comparison_data['Actual_Avg_Temp'].min(), comparison_data['Actual_Avg_Temp'].max()], 
                 [comparison_data['Actual_Avg_Temp'].min(), comparison_data['Actual_Avg_Temp'].max()], 
                 'k--', alpha=0.5, label='Perfect Prediction')
    axes[2].set_xlabel('Actual Avg Temperature (°C)')
    axes[2].set_ylabel('Predicted Avg Temperature (°C)')
    axes[2].set_title('Avg Temperature: Actual vs Predicted')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_correlation_december_2025.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Save detailed comparison to CSV
    comparison_data.to_csv('detailed_forecast_vs_actual_december_2025.csv', index=False)
    
    print(f"\nDetailed comparison saved to 'detailed_forecast_vs_actual_december_2025.csv'")
    print(f"Visualization plots saved as 'temperature_comparison_december_2025.png' and 'temperature_correlation_december_2025.png'")
    
    # 9. Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("Actual December 2025 temperatures:")
    print(f"  Max temp: {comparison_data['Actual_Max_Temp'].mean():.1f}°C (avg), {comparison_data['Actual_Max_Temp'].std():.1f}°C (std)")
    print(f"  Min temp: {comparison_data['Actual_Min_Temp'].mean():.1f}°C (avg), {comparison_data['Actual_Min_Temp'].std():.1f}°C (std)")
    print(f"  Avg temp: {comparison_data['Actual_Avg_Temp'].mean():.1f}°C (avg), {comparison_data['Actual_Avg_Temp'].std():.1f}°C (std)")
    
    print("\nPredicted December 2025 temperatures:")
    print(f"  Max temp: {comparison_data['Max_Temperature_Forecast'].mean():.1f}°C (avg), {comparison_data['Max_Temperature_Forecast'].std():.1f}°C (std)")
    print(f"  Min temp: {comparison_data['Min_Temperature_Forecast'].mean():.1f}°C (avg), {comparison_data['Min_Temperature_Forecast'].std():.1f}°C (std)")
    print(f"  Avg temp: {comparison_data['Avg_Temperature_Forecast'].mean():.1f}°C (avg), {comparison_data['Avg_Temperature_Forecast'].std():.1f}°C (std)")

else:
    print("No matching data found between forecast and actual data!")
    print("Forecast dates:", forecast_2025['Date'].tolist()[:5], "...")
    if len(actual_dec_2025) > 0:
        print("Actual dates:", actual_dec_2025['Date'].tolist()[:5], "...")