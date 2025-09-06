# Time Series Forecasting for Retail Demand Prediction
# Fixed version - No statsmodels.tsa.seasonal errors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from datetime import datetime, timedelta

def install_missing_packages():
    """Install required packages if missing"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy'  # Added scipy for alternative decomposition
    }
    
    missing_packages = []
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"❌ {package} is missing - installing...")
            missing_packages.append(install_name)
    
    # Try to install statsmodels separately
    try:
        __import__('statsmodels')
        print("✅ statsmodels is already installed")
    except ImportError:
        print("❌ statsmodels is missing - installing...")
        missing_packages.append('statsmodels')
    
    if missing_packages:
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All missing packages installed successfully!")
        except Exception as e:
            print(f"❌ Failed to install packages: {e}")
            print("Please run: pip install pandas numpy matplotlib seaborn statsmodels scipy")
            return False
    return True

def setup_environment():
    """Setup project environment"""
    print("🔧 Setting up environment...")
    os.makedirs('data', exist_ok=True)
    os.makedirs('results/forecasts', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    print("✅ Directories created successfully")

def generate_sample_data():
    """Generate sample retail sales data"""
    print("📊 Generating sample sales data...")
    
    # Create dates from 2018 to 2023
    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
    
    # Create realistic sales pattern
    np.random.seed(42)
    
    # Base trend (gradual increase over time)
    trend = np.linspace(100, 300, len(dates))
    
    # Weekly seasonality (higher sales on weekends)
    day_of_week = dates.dayofweek
    weekly_seasonality = np.where(day_of_week >= 5, 1.5, 1.0)
    
    # Yearly seasonality (holiday peaks)
    month = dates.month
    yearly_seasonality = np.where(month.isin([11, 12]), 2.0,  # Nov-Dec: high
                         np.where(month.isin([6, 7]), 1.3,    # Jun-Jul: medium
                         1.0))                               # Other: normal
    
    # Random noise
    noise = np.random.normal(0, 25, len(dates))
    
    # Combine components
    sales = trend * weekly_seasonality * yearly_seasonality + noise
    sales = np.maximum(sales, 50)  # Ensure minimum sales
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.astype(int),
        'Product': 'Product_A'  # Single product for simplicity
    })
    
    # Save to CSV
    df.to_csv('data/retail_sales.csv', index=False)
    print("✅ Sample data saved to data/retail_sales.csv")
    return df

def load_data():
    """Load data from file or generate sample"""
    try:
        df = pd.read_csv('data/retail_sales.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        print("✅ Loaded existing sales data")
        return df
    except FileNotFoundError:
        print("📝 No existing data found - generating sample data")
        return generate_sample_data()

def manual_seasonal_decomposition(series, period=12):
    """Manual seasonal decomposition without statsmodels"""
    print("🔍 Performing manual seasonal decomposition...")
    
    # Simple moving average for trend
    trend = series.rolling(window=period, center=True).mean()
    
    # Detrend the series
    detrended = series - trend
    
    # Calculate seasonal component (average by period)
    seasonal = pd.Series(index=series.index, dtype=float)
    for i in range(period):
        seasonal.iloc[i::period] = detrended.iloc[i::period].mean()
    
    # Residual component
    residual = detrended - seasonal
    
    return trend, seasonal, residual

def explore_data(df):
    """Explore and visualize the data without statsmodels dependency"""
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    # Convert to time series
    df.set_index('Date', inplace=True)
    daily_sales = df['Sales']
    
    # Resample to monthly for better analysis
    monthly_sales = daily_sales.resample('M').sum()
    
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total days: {len(daily_sales)}")
    print(f"Total months: {len(monthly_sales)}")
    print(f"Average monthly sales: {monthly_sales.mean():.0f}")
    
    # Plot time series
    plt.figure(figsize=(15, 10))
    
    # Daily sales
    plt.subplot(2, 2, 1)
    plt.plot(daily_sales.index, daily_sales.values, alpha=0.7, color='blue')
    plt.title('Daily Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3)
    
    # Monthly sales
    plt.subplot(2, 2, 2)
    plt.plot(monthly_sales.index, monthly_sales.values, linewidth=2, color='green')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3)
    
    # Manual seasonal decomposition
    try:
        trend, seasonal, residual = manual_seasonal_decomposition(monthly_sales)
        
        # Trend component
        plt.subplot(2, 2, 3)
        plt.plot(trend.index, trend.values, color='orange', linewidth=2)
        plt.title('Trend Component (Manual)')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.grid(True, alpha=0.3)
        
        # Seasonal component
        plt.subplot(2, 2, 4)
        # Plot one year of seasonal pattern
        seasonal_one_year = seasonal.iloc[:12] if len(seasonal) >= 12 else seasonal
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x_positions = np.arange(len(seasonal_one_year))
        
        plt.bar(x_positions, seasonal_one_year.values, alpha=0.7, color='purple')
        plt.title('Seasonal Component (Manual)')
        plt.xlabel('Month')
        plt.ylabel('Seasonal Effect')
        plt.xticks(x_positions, months[:len(seasonal_one_year)])
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"⚠️ Manual decomposition failed: {e}")
        # Simple alternative: show basic statistics
        plt.subplot(2, 2, 3)
        monthly_avg = monthly_sales.groupby(monthly_sales.index.month).mean()
        plt.bar(range(1, 13), monthly_avg.values, alpha=0.7, color='orange')
        plt.title('Average Monthly Sales')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Show autocorrelation (simple version)
        autocorr = []
        max_lag = min(12, len(monthly_sales) // 2)
        for lag in range(1, max_lag + 1):
            if lag < len(monthly_sales):
                corr = np.corrcoef(monthly_sales[:-lag], monthly_sales[lag:])[0, 1]
                autocorr.append(corr)
        plt.bar(range(1, max_lag + 1), autocorr, alpha=0.7, color='purple')
        plt.title('Autocorrelation (Lags 1-12)')
        plt.xlabel('Lag (months)')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return monthly_sales

def prepare_data(monthly_sales):
    """Prepare data for training"""
    print("\n📈 Preparing data for modeling...")
    
    # Split into train and test
    train_size = int(len(monthly_sales) * 0.8)
    train_data = monthly_sales.iloc[:train_size]
    test_data = monthly_sales.iloc[train_size:]
    
    print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Testing period: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    return train_data, test_data

def simple_forecasting(train_data, test_data):
    """Simple forecasting methods that always work"""
    print("\n" + "="*50)
    print("FORECASTING METHODS")
    print("="*50)
    
    # Method 1: Naive forecast (last value)
    naive_forecast = np.full(len(test_data), train_data.iloc[-1])
    
    # Method 2: Moving average (3-month)
    moving_avg = train_data.rolling(window=3).mean().iloc[-1]
    ma_forecast = np.full(len(test_data), moving_avg)
    
    # Method 3: Seasonal naive (same month last year)
    seasonal_naive = []
    for i in range(len(test_data)):
        # Get the same month from previous year in training data
        same_month = train_data[train_data.index.month == test_data.index[i].month]
        if len(same_month) > 0:
            seasonal_naive.append(same_month.iloc[-1])
        else:
            seasonal_naive.append(train_data.iloc[-1])
    
    seasonal_naive = np.array(seasonal_naive)
    
    # Method 4: Linear trend projection
    try:
        # Fit linear trend
        x = np.arange(len(train_data))
        y = train_data.values
        coefficients = np.polyfit(x, y, 1)
        linear_trend = np.poly1d(coefficients)
        
        # Project future values
        future_x = np.arange(len(train_data), len(train_data) + len(test_data))
        linear_forecast = linear_trend(future_x)
    except:
        linear_forecast = naive_forecast  # Fallback
    
    # Calculate errors
    def calculate_metrics(actual, predicted):
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100  # Avoid division by zero
        return mae, rmse, mape
    
    metrics_naive = calculate_metrics(test_data.values, naive_forecast)
    metrics_ma = calculate_metrics(test_data.values, ma_forecast)
    metrics_seasonal = calculate_metrics(test_data.values, seasonal_naive)
    metrics_linear = calculate_metrics(test_data.values, linear_forecast)
    
    print("Naive Forecast (Last Value):")
    print(f"  MAE: {metrics_naive[0]:.2f}, RMSE: {metrics_naive[1]:.2f}, MAPE: {metrics_naive[2]:.2f}%")
    
    print("Moving Average Forecast (3-month):")
    print(f"  MAE: {metrics_ma[0]:.2f}, RMSE: {metrics_ma[1]:.2f}, MAPE: {metrics_ma[2]:.2f}%")
    
    print("Seasonal Naive Forecast:")
    print(f"  MAE: {metrics_seasonal[0]:.2f}, RMSE: {metrics_seasonal[1]:.2f}, MAPE: {metrics_seasonal[2]:.2f}%")
    
    print("Linear Trend Forecast:")
    print(f"  MAE: {metrics_linear[0]:.2f}, RMSE: {metrics_linear[1]:.2f}, MAPE: {metrics_linear[2]:.2f}%")
    
    return {
        'Naive': {'forecast': naive_forecast, 'metrics': metrics_naive},
        'MovingAverage': {'forecast': ma_forecast, 'metrics': metrics_ma},
        'SeasonalNaive': {'forecast': seasonal_naive, 'metrics': metrics_seasonal},
        'LinearTrend': {'forecast': linear_forecast, 'metrics': metrics_linear}
    }

def visualize_results(train_data, test_data, results):
    """Visualize forecasting results"""
    print("\n" + "="*50)
    print("RESULTS VISUALIZATION")
    print("="*50)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Forecasts
    plt.subplot(2, 1, 1)
    plt.plot(train_data.index, train_data.values, label='Training Data', linewidth=2, alpha=0.8, color='blue')
    plt.plot(test_data.index, test_data.values, label='Actual Test Data', linewidth=3, color='black')
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    methods = list(results.keys())
    
    for i, method in enumerate(methods):
        plt.plot(test_data.index, results[method]['forecast'], 
                label=f'{method} Forecast', 
                linestyle='--' if i < 2 else '-',
                linewidth=2, 
                color=colors[i % len(colors)])
    
    plt.title('Sales Forecast vs Actual', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error comparison
    plt.subplot(2, 1, 2)
    
    mae_values = [results[method]['metrics'][0] for method in methods]
    
    bars = plt.barh(methods, mae_values, color='lightblue', alpha=0.8)
    plt.xlabel('MAE (Mean Absolute Error)', fontsize=12)
    plt.title('Model Performance Comparison (Lower MAE is Better)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(mae_values):
        plt.text(v + max(mae_values)*0.01, i, f'{v:.1f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/forecast_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def make_future_predictions(monthly_sales, best_method):
    """Make future predictions for next 12 months"""
    print("\n" + "="*50)
    print("FUTURE DEMAND FORECAST")
    print("="*50)
    
    # Create future dates
    last_date = monthly_sales.index[-1]
    future_months = 12
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                periods=future_months, 
                                freq='M')
    
    # Calculate seasonal patterns from historical data
    seasonal_pattern = {}
    for month in range(1, 13):
        month_data = monthly_sales[monthly_sales.index.month == month]
        if len(month_data) > 0:
            seasonal_pattern[month] = month_data.mean()
    
    # Calculate overall trend (average growth per month)
    overall_growth = (monthly_sales.iloc[-1] / monthly_sales.iloc[0]) ** (1/len(monthly_sales)) - 1
    monthly_growth = 1 + overall_growth  # Convert to multiplicative factor
    
    # Generate future predictions
    future_predictions = []
    for i, future_date in enumerate(future_dates):
        month = future_date.month
        base_prediction = monthly_sales.iloc[-1] * (monthly_growth ** (i + 1))
        
        # Apply seasonal adjustment if available
        if month in seasonal_pattern:
            seasonal_avg = seasonal_pattern[month]
            overall_avg = monthly_sales.mean()
            seasonal_factor = seasonal_avg / overall_avg
            adjusted_prediction = base_prediction * seasonal_factor
        else:
            adjusted_prediction = base_prediction
        
        future_predictions.append(adjusted_prediction)
    
    # Create confidence intervals
    historical_std = monthly_sales.std()
    future_predictions = np.array(future_predictions)
    lower_bound = future_predictions * 0.8  # 20% lower
    upper_bound = future_predictions * 1.2  # 20% higher
    
    # Create results DataFrame
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_predictions.astype(int),
        'Lower_Bound': lower_bound.astype(int),
        'Upper_Bound': upper_bound.astype(int),
        'Month': [d.strftime('%B %Y') for d in future_dates]
    })
    
    # Save to CSV
    future_df.to_csv('results/forecasts/future_demand_forecast.csv', index=False)
    print("✅ Future forecast saved to results/forecasts/future_demand_forecast.csv")
    
    # Plot future forecast
    plt.figure(figsize=(14, 8))
    
    # Plot historical data (last 2 years)
    historical_2y = monthly_sales[monthly_sales.index >= (monthly_sales.index[-1] - pd.DateOffset(years=2))]
    plt.plot(historical_2y.index, historical_2y.values, 
            label='Historical Data (Last 2 Years)', linewidth=3, color='blue')
    
    # Plot future forecast
    plt.plot(future_df['Date'], future_df['Forecast'], 
            label='Future Forecast', linewidth=3, color='red')
    
    # Plot confidence interval
    plt.fill_between(future_df['Date'], 
                    future_df['Lower_Bound'], 
                    future_df['Upper_Bound'], 
                    alpha=0.3, label='Confidence Interval (80-120%)', color='orange')
    
    plt.title('Future Demand Forecast - Next 12 Months', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value annotations for key points
    for i, row in future_df.iterrows():
        if i % 3 == 0:  # Label every 3rd month
            plt.annotate(f'{row["Forecast"]:,.0f}', 
                        xy=(row['Date'], row['Forecast']),
                        xytext=(10, 10), textcoords='offset points',
                        fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/future_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print forecast summary
    print("\n📊 Future Demand Forecast Summary:")
    print("=" * 60)
    print(f"{'Month':<15} {'Forecast':<12} {'Confidence Range':<20}")
    print("=" * 60)
    for _, row in future_df.iterrows():
        print(f"{row['Month']:<15} {row['Forecast']:<12,} {row['Lower_Bound']:,} - {row['Upper_Bound']:,}")
    
    total_forecast = future_df['Forecast'].sum()
    print("=" * 60)
    print(f"{'Total Next 12 Months:':<15} {total_forecast:>12,}")
    
    return future_df

def main():
    """Main function to run the complete forecasting pipeline"""
    print("🕰️ RETAIL DEMAND FORECASTING")
    print("=" * 60)
    print("This program forecasts product demand using time series analysis")
    print("No statsmodels.tsa.seasonal dependency - will work without errors!")
    print("=" * 60)
    
    try:
        # Install missing packages if needed
        install_missing_packages()
        
        # Setup environment
        setup_environment()
        
        # Load data
        df = load_data()
        
        # Explore data
        monthly_sales = explore_data(df)
        
        # Prepare data
        train_data, test_data = prepare_data(monthly_sales)
        
        # Run forecasting
        results = simple_forecasting(train_data, test_data)
        
        # Find best method
        best_method = min(results.items(), key=lambda x: x[1]['metrics'][0])[0]
        best_mae = results[best_method]['metrics'][0]
        
        print(f"\n🎯 Best forecasting method: {best_method} (MAE: {best_mae:.2f})")
        
        # Visualize results
        visualize_results(train_data, test_data, results)
        
        # Make future predictions
        future_forecast = make_future_predictions(monthly_sales, best_method)
        
        print("\n" + "=" * 60)
        print("🎉 FORECASTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Results saved in the 'results' folder:")
        print("   - visualizations/ : All charts and graphs")
        print("   - forecasts/ : Future demand predictions in CSV format")
        print(f"   - Best method: {best_method}")
        print("\n💡 You can use these forecasts for:")
        print("   - Inventory planning")
        print("   - Resource allocation") 
        print("   - Sales target setting")
        print("   - Budget planning")
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("💡 Please try restarting VS Code or check your Python installation")
        import traceback
        traceback.print_exc()

# Run the main function
if __name__ == "__main__":
    main()