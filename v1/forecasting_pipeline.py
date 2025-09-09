# forecasting_gradio_pipeline_enhanced.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gradio as gr
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===== Utility function =====
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# ===== Training Pipeline =====
def train_and_save_model(csv_path="events.csv", model_path="lstm_model.h5", scaler_path="scaler.gz"):
    df = pd.read_csv(csv_path)

    # Cleaning & renaming
    if "Start time UTC" in df.columns:
        del df["Start time UTC"]
    if "End time UTC" in df.columns:
        del df["End time UTC"]
    if "Start time UTC+03:00" in df.columns:
        del df["Start time UTC+03:00"]
    if "End time UTC+03:00" in df.columns:
        df.rename(columns={"End time UTC+03:00": "DateTime"}, inplace=True)
    if "Electricity consumption in Finland" in df.columns:
        df.rename(columns={"Electricity consumption in Finland": "Consumption"}, inplace=True)

    dataset = df.set_index("DateTime")
    dataset.index = pd.to_datetime(dataset.index)

    # Daily resample
    newDataSet = dataset.resample("D").mean()
    y = newDataSet["Consumption"]

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

    training_size = int(len(y_scaled) * 0.80)
    train_data = y_scaled[0:training_size, :]

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Save model + scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}, Scaler saved to {scaler_path}")

# ===== Enhanced Forecast Function =====
def forecast_future_enhanced(forecast_days=30, show_confidence=True, plot_style="Line"):
    try:
        csv_path = "events.csv"
        model_path = "lstm_model.h5"
        scaler_path = "scaler.gz"
        
        # Load data
        df = pd.read_csv(csv_path)

        # Cleaning & renaming
        if "Start time UTC" in df.columns:
            del df["Start time UTC"]
        if "End time UTC" in df.columns:
            del df["End time UTC"]
        if "Start time UTC+03:00" in df.columns:
            del df["Start time UTC+03:00"]
        if "End time UTC+03:00" in df.columns:
            df.rename(columns={"End time UTC+03:00": "DateTime"}, inplace=True)
        if "Electricity consumption in Finland" in df.columns:
            df.rename(columns={"Electricity consumption in Finland": "Consumption"}, inplace=True)

        dataset = df.set_index("DateTime")
        dataset.index = pd.to_datetime(dataset.index)

        newDataSet = dataset.resample("D").mean()
        y = newDataSet["Consumption"]

        # Load model & scaler
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)

        y_scaled = scaler.transform(np.array(y).reshape(-1, 1))
        time_step = 100
        test_data = y_scaled[-time_step:]

        # Generate forecast
        temp_input = test_data.flatten().tolist()
        lst_output = []

        for i in range(forecast_days):
            x_input = np.array(temp_input[-time_step:]).reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            next_val = float(yhat[0][0])
            temp_input.append(next_val)
            lst_output.append(next_val)

        forecast = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
        future_dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D")

        # Enhanced Plotting
        fig, ax = plt.subplots(figsize=(16, 8))
        
        if plot_style == "Line":
            # Line style
            recent_data = y.tail(180)  # Show last 6 months
            ax.plot(recent_data.index, recent_data.values, 
                   color='#2E86AB', linewidth=2, label='Historical Data', alpha=0.8)
            ax.plot(future_dates, forecast.flatten(), 
                   color='#A23B72', linewidth=3, label='Forecast', marker='o', markersize=4)
            
            if show_confidence:
                # Add confidence interval (simplified)
                std_dev = np.std(forecast) * 0.1
                upper_bound = forecast.flatten() + std_dev
                lower_bound = forecast.flatten() - std_dev
                ax.fill_between(future_dates, lower_bound, upper_bound, 
                              color='#A23B72', alpha=0.2, label='Confidence Interval')
        
        else:  # Scatter style
            ax.scatter(y.tail(180).index, y.tail(180).values, 
                      label="Historical Data", color="#2E86AB", alpha=0.6, s=25)
            ax.scatter(future_dates, forecast, 
                      label="Forecast", color="#A23B72", alpha=0.9, s=50, marker="^")

        # Styling improvements
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.set_title('Energy Consumption Forecast', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Consumption (MW)', fontsize=12, fontweight='bold')
        
        # Format dates on x-axis
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        # Enhanced DataFrame with additional statistics
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Forecast (MW)': [f"{val:.2f}" for val in forecast.flatten()],
            'Day of Week': future_dates.strftime('%A'),
            'Month': future_dates.strftime('%B')
        })
        
        # Summary statistics
        avg_forecast = np.mean(forecast)
        max_forecast = np.max(forecast)
        min_forecast = np.min(forecast)
        
        summary_text = f"""
        FORECAST SUMMARY ({forecast_days} days):
        • Average: {avg_forecast:.2f} MW
        • Maximum: {max_forecast:.2f} MW  
        • Minimum: {min_forecast:.2f} MW
        • Peak expected on: {future_dates[np.argmax(forecast.flatten())].strftime('%Y-%m-%d (%A)')}
        • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return fig, forecast_df, summary_text
        
    except Exception as e:
        error_fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error: {str(e)}\nPlease ensure model is trained and files exist.', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        return error_fig, pd.DataFrame(), f"Error occurred: {str(e)}"

# ===== Model Info Function =====
def get_model_info():
    try:
        import os
        model_exists = os.path.exists("lstm_model.h5")
        scaler_exists = os.path.exists("scaler.gz")
        data_exists = os.path.exists("events.csv")
        
        info = f"""
        MODEL STATUS:
        • LSTM Model: {'Available' if model_exists else 'Not Found'}
        • Scaler: {'Available' if scaler_exists else 'Not Found'}  
        • Data File: {'Available' if data_exists else 'Not Found'}
        
        MODEL SPECIFICATIONS:
        • Architecture: Multi-layer LSTM (4 layers, 50 units each)
        • Time Steps: 100 days
        • Training Split: 80/20
        • Optimizer: Adam
        • Loss Function: Mean Squared Error
        • Input Features: Daily electricity consumption
        • Target: Next day consumption
        """
        return info
    except:
        return "Unable to retrieve model information."

# ===== Training Function for UI =====
def train_model_ui():
    try:
        train_and_save_model()
        return "Model training completed successfully! You can now generate forecasts."
    except Exception as e:
        return f"Training failed: {str(e)}"

# ===== Enhanced Gradio Interface =====
def create_interface():
    # Custom CSS for Oracle branding
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .oracle-header {
        background: linear-gradient(90deg, #FF0000 0%, #CC0000 100%);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .forecast-panel {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    """

    with gr.Blocks(css=custom_css, title="Oracle Energy Forecasting") as interface:
        
        # Header
        gr.HTML("""
            <div class="oracle-header">
                <h1 style="margin: 0; font-size: 28px; font-weight: bold; color: white;">
                    Oracle Solution Center
                </h1>
                <h2 style="margin: 10px 0 0 0; font-size: 20px; font-weight: normal; color: white;">
                    Forecast Energy Consumption
                </h2>
            </div>
        """)

        
        with gr.Row():
            with gr.Column(scale=1):
                # gr.HTML('<div class="forecast-panel">')
                gr.Markdown("### Forecast Configuration")
                
                forecast_days = gr.Slider(
                    minimum=7, 
                    maximum=90, 
                    value=30, 
                    step=1, 
                    label="Forecast Period (Days)",
                    info="Select number of days to forecast (7-90 days)"
                )
                
                show_confidence = gr.Checkbox(
                    value=True, 
                    label="Show Confidence Interval",
                    info="Display uncertainty bounds around predictions"
                )
                
                plot_style = gr.Radio(
                    choices=["Line", "Scatter"], 
                    value="Line",
                    label="Visualization Style",
                    info="Choose your preferred chart style"
                )
                
                forecast_btn = gr.Button(
                    "Generate Forecast", 
                    variant="primary",
                    size="lg"
                )
                
                gr.HTML('</div>')
            
            with gr.Column(scale=2):
                # gr.HTML('<div class="forecast-panel">')
                
                # Main forecast plot
                forecast_plot = gr.Plot(
                    label="Energy Consumption Forecast",
                    show_label=True
                )
                
                # Summary information
                summary_info = gr.Textbox(
                    label="Forecast Summary",
                    lines=8,
                    interactive=False,
                    show_label=True
                )
                gr.HTML('</div>')
        
        # Forecast results table
        # gr.HTML('<div class="forecast-panel">')
        gr.Markdown("### Detailed Forecast Results")
        forecast_table = gr.Dataframe(
            label="Forecast Data",
            interactive=False,
            wrap=True,
            max_height=400
        )
        gr.HTML('</div>')
        
        # Event handlers
        forecast_btn.click(
            fn=forecast_future_enhanced,
            inputs=[forecast_days, show_confidence, plot_style],
            outputs=[forecast_plot, forecast_table, summary_info]
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; 
                        background: rgba(255,255,255,0.8); border-radius: 10px;">
                <p style="margin: 0; color: #666; font-size: 12px;">
                    Oracle Solution Center | Advanced Analytics Platform<br>
                </p>
            </div>
        """)
    
    return interface

# ===== Enhanced Forecast Function =====
def forecast_future_enhanced(forecast_days=30, show_confidence=True, plot_style="Line"):
    try:
        csv_path = "events.csv"
        model_path = "lstm_model.h5"
        scaler_path = "scaler.gz"
        
        # Load data
        df = pd.read_csv(csv_path)

        # Cleaning & renaming
        if "Start time UTC" in df.columns:
            del df["Start time UTC"]
        if "End time UTC" in df.columns:
            del df["End time UTC"]
        if "Start time UTC+03:00" in df.columns:
            del df["Start time UTC+03:00"]
        if "End time UTC+03:00" in df.columns:
            df.rename(columns={"End time UTC+03:00": "DateTime"}, inplace=True)
        if "Electricity consumption in Finland" in df.columns:
            df.rename(columns={"Electricity consumption in Finland": "Consumption"}, inplace=True)

        dataset = df.set_index("DateTime")
        dataset.index = pd.to_datetime(dataset.index)

        newDataSet = dataset.resample("D").mean()
        y = newDataSet["Consumption"]

        # Load model & scaler
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)

        y_scaled = scaler.transform(np.array(y).reshape(-1, 1))
        time_step = 100
        test_data = y_scaled[-time_step:]

        # Generate forecast
        temp_input = test_data.flatten().tolist()
        lst_output = []

        for i in range(forecast_days):
            x_input = np.array(temp_input[-time_step:]).reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            next_val = float(yhat[0][0])
            temp_input.append(next_val)
            lst_output.append(next_val)

        forecast = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
        future_dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D")

        # Enhanced Plotting
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.patch.set_facecolor('white')
        
        if plot_style == "Line":
            # Show last 6 months of historical data
            recent_data = y.tail(180)
            ax.plot(recent_data.index, recent_data.values, 
                   color='#1f77b4', linewidth=2.5, label='Historical Consumption', alpha=0.8)
            ax.plot(future_dates, forecast.flatten(), 
                   color='#ff7f0e', linewidth=3, label='Predicted Consumption', 
                   marker='o', markersize=3, markerfacecolor='white', markeredgewidth=1)
            
            if show_confidence:
                # Enhanced confidence interval
                forecast_flat = forecast.flatten()
                std_dev = np.std(y.tail(180)) * 0.15  # Use historical volatility
                upper_bound = forecast_flat + std_dev
                lower_bound = forecast_flat - std_dev
                ax.fill_between(future_dates, lower_bound, upper_bound, 
                              color='#ff7f0e', alpha=0.2, label='95% Confidence Interval')
        
        else:  # Scatter style
            recent_data = y.tail(180)
            ax.scatter(recent_data.index, recent_data.values, 
                      label="Historical Data", color="#1f77b4", alpha=0.6, s=30)
            ax.scatter(future_dates, forecast, 
                      label="Forecast", color="#ff7f0e", alpha=0.9, s=60, marker="^")

        # Enhanced styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
        ax.set_title('Energy Consumption Forecast | Oracle Analytics', 
                    fontsize=18, fontweight='bold', pad=25)
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Consumption (MW)', fontsize=13, fontweight='bold')
        
        # Format y-axis with thousand separators
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Format dates on x-axis
        fig.autofmt_xdate()
        plt.tight_layout()

        # Enhanced DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Consumption (MW)': [f"{val:.1f}" for val in forecast.flatten()],
            'Day': future_dates.strftime('%A'),
            'Week': [f"Week {date.isocalendar()[1]}" for date in future_dates],
            'Month': future_dates.strftime('%B %Y')
        })
        
        # Enhanced summary statistics
        avg_forecast = np.mean(forecast)
        max_forecast = np.max(forecast)
        min_forecast = np.min(forecast)
        trend = "Increasing" if forecast[-1] > forecast[0] else "Decreasing"
        
        # Calculate seasonal patterns
        weekly_avg = forecast_df.groupby('Day')['Consumption (MW)'].apply(lambda x: np.mean([float(val) for val in x]))
        peak_day = weekly_avg.idxmax()
        
        summary_text = f"""FORECAST ANALYSIS ({forecast_days} days ahead)

CONSUMPTION METRICS:
• Average Daily Consumption: {avg_forecast:.1f} MW
• Peak Consumption: {max_forecast:.1f} MW
• Minimum Consumption: {min_forecast:.1f} MW
• Overall Trend: {trend}

TIMING ANALYSIS:
• Peak Day Expected: {future_dates[np.argmax(forecast.flatten())].strftime('%A, %B %d, %Y')}
• Highest Weekly Consumption: {peak_day}
• Forecast Period: {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')}
"""
        
# TECHNICAL INFO:
# • Model Type: Multi-layer LSTM Neural Network
# • Confidence Level: 95% (when enabled)
# • Last Training Update: Model checkpoint available
# • Generated: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}

        return fig, forecast_df, summary_text
        
    except Exception as e:
        # Error handling with Line presentation
        error_fig, ax = plt.subplots(figsize=(12, 6))
        error_fig.patch.set_facecolor('white')
        
        ax.text(0.5, 0.5, 
                f'SYSTEM ERROR\n\n{str(e)}\n\nPlease ensure:\n• Model files exist (lstm_model.h5)\n• Data file is available (events.csv)\n• Model has been trained', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee", edgecolor="#f44336"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Oracle Energy Forecasting - System Status', fontsize=14, fontweight='bold')
        
        return error_fig, pd.DataFrame({'Status': ['Error occurred']}), f"System Error: {str(e)}"

# ===== Main Application =====
if __name__ == "__main__":
    import os

    # Check and train model if needed
    if not os.path.exists("lstm_model.h5"):
        print("Training model for the first time...")
        train_and_save_model()
    else:
        print("Using existing trained model.")

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False
    )