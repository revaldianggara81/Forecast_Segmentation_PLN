# forecast.py
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ========== Load Dataset ==========
df_raw = pd.read_csv("data/global_energy_consumption.csv")

# Normalisasi nama kolom
if "Electricity_Consumption" in df_raw.columns:
    value_col = "Electricity_Consumption"
elif "Energy_Consumption" in df_raw.columns:
    value_col = "Energy_Consumption"
else:
    value_col = df_raw.columns[-1]

country_col = "Country" if "Country" in df_raw.columns else "Entity"
year_col = "Year"

# Buat kolom Date dari Year (pakai Januari tiap tahun)
df_raw["Date"] = pd.to_datetime(df_raw[year_col].astype(str) + "-01-01")
df_raw = df_raw[[country_col, "Date", value_col]].dropna()

available_countries = sorted(df_raw[country_col].unique().tolist())


# ========== AI Forecasting Algorithms ==========
def arima_forecast(train_data, steps=12):
    try:
        model = ARIMA(train_data, order=(2, 1, 2))
        model_fit = model.fit()
        return np.array(model_fit.forecast(steps=steps)).ravel()
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None

def sarima_forecast(train_data, steps=12):
    try:
        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()
        return np.array(model_fit.forecast(steps=steps)).ravel()
    except Exception as e:
        print(f"SARIMA Error: {e}")
        return None

def lstm_forecast(train_data, steps=12):
    """Pseudo LSTM using trend + seasonal pattern"""
    try:
        data = np.array(train_data).ravel()
        x = np.arange(len(data))
        trend_coef = np.polyfit(x, data, 1)[0]
        last_12 = data[-12:]
        seasonal_pattern = last_12 - np.mean(last_12)

        forecast, last_value = [], data[-1]
        for i in range(steps):
            trend_component = trend_coef * (i + 1)
            seasonal_component = seasonal_pattern[i % 12]
            next_value = last_value + trend_component + seasonal_component * 0.5
            forecast.append(next_value)
        return np.array(forecast)
    except Exception as e:
        print(f"LSTM Error: {e}")
        return None

def nbeats_forecast(train_data, steps=12):
    """Pseudo N-BEATS using trend + seasonality blocks"""
    try:
        data = np.array(train_data).ravel()
        x = np.arange(len(data))
        trend_coef = np.polyfit(x, data, 2)
        trend_forecast = np.polyval(trend_coef, np.arange(len(data), len(data) + steps))

        seasonal_period = 12
        if len(data) >= seasonal_period:
            seasonal_data = data[-seasonal_period:]
            seasonal_pattern = seasonal_data - np.mean(seasonal_data)
            seasonal_forecast = np.tile(seasonal_pattern, (steps // seasonal_period) + 1)[:steps]
        else:
            seasonal_forecast = np.zeros(steps)

        return trend_forecast + seasonal_forecast
    except Exception as e:
        print(f"N-BEATS Error: {e}")
        return None


# ========== Assumption Algorithms ==========
def monthly_assumption_forecast(train_data, steps=12):
    data = np.array(train_data).ravel()
    last_12 = data[-12:]
    monthly_avg_growth = (last_12[-1] - last_12[0]) / 12
    predictions, last_val = [], last_12[-1]
    for i in range(steps):
        new_val = last_val + monthly_avg_growth
        predictions.append(new_val)
        last_val = new_val
    return np.array(predictions)

def yearly_assumption_forecast(train_data, steps=12):
    data = np.array(train_data).ravel()
    last_5y = data[-60:] if len(data) >= 60 else data
    yearly_growth = (last_5y[-1] - last_5y[0]) / 5 if len(last_5y) >= 24 else 0
    base = np.mean(data[-12:])
    return np.linspace(base, base + yearly_growth/12, steps)


# ========== Core Forecast Function ==========
def forecast_last_year(country, ai_algorithm="ARIMA"):
    data = df_raw[df_raw[country_col] == country].copy()
    data = data.groupby("Date", as_index=False)[value_col].sum()
    data = data.rename(columns={value_col: "consumption"})

    # Interpolasi tahunan → bulanan
    data = data.set_index("Date").sort_index().resample("MS").interpolate(method="linear")

    if len(data) < 72:
        return go.Figure(), pd.DataFrame(), pd.DataFrame()

    train, test = data[:-12], data[-12:]

    # AI forecast
    ai_algorithms = {
        "ARIMA": arima_forecast,
        "SARIMA": sarima_forecast,
        "LSTM": lstm_forecast,
        "N-BEATS": nbeats_forecast
    }
    ai_func = ai_algorithms.get(ai_algorithm, arima_forecast)
    ai_forecast = ai_func(train.values.ravel(), steps=12)
    if ai_forecast is None:
        ai_forecast = np.repeat(train.iloc[-1], 12)
    ai_forecast = pd.Series(ai_forecast, index=test.index)

    # Assumptions
    monthly_pred = pd.Series(monthly_assumption_forecast(train.values.ravel(), 12), index=test.index)
    yearly_pred = pd.Series(yearly_assumption_forecast(train.values.ravel(), 12), index=test.index)

    # Metrics
    mse_ai = mean_squared_error(test.values.ravel(), ai_forecast.values.ravel())
    mae_ai = mean_absolute_error(test.values.ravel(), ai_forecast.values.ravel())
    mse_monthly = mean_squared_error(test.values.ravel(), monthly_pred.values.ravel())
    mae_monthly = mean_absolute_error(test.values.ravel(), monthly_pred.values.ravel())
    mse_yearly = mean_squared_error(test.values.ravel(), yearly_pred.values.ravel())
    mae_yearly = mean_absolute_error(test.values.ravel(), yearly_pred.values.ravel())

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train[-24:].index, y=train[-24:].values.ravel(),
                             mode="lines", name="Historical Data", line=dict(color="lightgray")))
    fig.add_trace(go.Scatter(x=test.index, y=test.values.ravel(),
                             mode="lines+markers", name="Real Data", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=test.index, y=ai_forecast,
                             mode="lines+markers", name=f"{ai_algorithm} Forecast", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=test.index, y=monthly_pred,
                             mode="lines+markers", name="Monthly Assumption", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=test.index, y=yearly_pred,
                             mode="lines+markers", name="Yearly Assumption", line=dict(color="orange")))

    fig.update_layout(
        title=f"Electricity Consumption Forecast - {country} ({ai_algorithm})<br>"
              f"<sub>MSE: AI={mse_ai:.2e}, Monthly={mse_monthly:.2e}, Yearly={mse_yearly:.2e}</sub>",
        xaxis_title="Date", yaxis_title="Consumption",
        template="plotly_white", height=600
    )

    # Result tables
    result_table = pd.DataFrame({
        "Month": test.index.strftime("%Y-%m"),
        "Real": test.values.ravel(),
        "AI_Predicted": ai_forecast.values.ravel(),
        "Monthly_Assumption": monthly_pred.values.ravel(),
        "Yearly_Assumption": yearly_pred.values.ravel()
    })

    summary_stats = pd.DataFrame({
        "Metric": ["MSE", "MAE"],
        f"{ai_algorithm}": [f"{mse_ai:.2e}", f"{mae_ai:.2f}"],
        "Monthly Assumption": [f"{mse_monthly:.2e}", f"{mae_monthly:.2f}"],
        "Yearly Assumption": [f"{mse_yearly:.2e}", f"{mae_yearly:.2f}"]
    })

    return fig, result_table, summary_stats


# ========== Gradio UI ==========
def create_forecasting_ui():
    gr.Markdown("## Electricity Consumption Forecasting (AI vs Assumptions)")

    default_country = "Indonesia" if "Indonesia" in available_countries else available_countries[0]

    with gr.Row():
        country_dropdown = gr.Dropdown(choices=available_countries, label="Select Country", value=default_country)
        algorithm_dropdown = gr.Dropdown(choices=["ARIMA", "SARIMA", "LSTM", "N-BEATS"],
                                         label="AI Algorithm", value="ARIMA")

    with gr.Row():
        submit_btn = gr.Button("Run Forecast", variant="primary")
        clear_btn = gr.Button("Reset", variant="secondary")

    with gr.Row():
        forecast_chart = gr.Plot(label="Forecast Comparison Chart")

    # with gr.Row():
    #     result_table = gr.Dataframe(label="Detailed Results (Last 12 months)", interactive=False)
    #     summary_table = gr.Dataframe(label="Summary Statistics", interactive=False)
    with gr.Row():
        gr.Markdown("### Detailed Results (Last 12 Months)")
    result_table = gr.Dataframe(label="Monthly Data & Cost Analysis", interactive=False)

    with gr.Row():
        gr.Markdown("### Summary Statistics")
    summary_table = gr.Dataframe(label="Performance Comparison", interactive=False)

    submit_btn.click(
        forecast_last_year,
        inputs=[country_dropdown, algorithm_dropdown],
        outputs=[forecast_chart, result_table, summary_table]
    )

    clear_btn.click(lambda: (go.Figure(), pd.DataFrame(), pd.DataFrame()),
                    outputs=[forecast_chart, result_table, summary_table])

    return forecast_chart, result_table, summary_table
