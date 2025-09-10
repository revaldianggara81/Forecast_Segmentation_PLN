# forecast.py
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ===================== LOAD DATA =====================
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

# Buat kolom Date dari Year → Januari tiap tahun
df_raw["Date"] = pd.to_datetime(df_raw[year_col].astype(str) + "-01-01")
df_raw = df_raw[[country_col, "Date", value_col]].dropna()

available_countries = sorted(df_raw[country_col].unique().tolist())


# ===================== AI MODELS =====================
def arima_forecast(train_data, steps=12):
    try:
        model = ARIMA(train_data, order=(2, 1, 2))
        model_fit = model.fit()
        return np.array(model_fit.forecast(steps=steps)).ravel()
    except:
        return None

def sarima_forecast(train_data, steps=12):
    try:
        model = SARIMAX(train_data, order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        return np.array(model_fit.forecast(steps=steps)).ravel()
    except:
        return None

def lstm_like_forecast(train_data, steps=12):
    """Pseudo-LSTM: tren linear + pola musiman"""
    data = np.array(train_data).ravel()
    x = np.arange(len(data))
    trend_coef = np.polyfit(x, data, 1)[0]
    last_12 = data[-12:]
    seasonal_pattern = last_12 - np.mean(last_12)

    forecast, last_val = [], data[-1]
    for i in range(steps):
        trend_component = trend_coef * (i + 1)
        seasonal_component = seasonal_pattern[i % 12]
        next_val = last_val + trend_component + 0.5 * seasonal_component
        forecast.append(next_val)
    return np.array(forecast)

def nbeats_like_forecast(train_data, steps=12):
    """Pseudo N-BEATS: quadratic trend + pola musiman"""
    data = np.array(train_data).ravel()
    x = np.arange(len(data))
    trend_coef = np.polyfit(x, data, 2)
    trend_forecast = np.polyval(trend_coef,
                                np.arange(len(data), len(data) + steps))

    if len(data) >= 12:
        seasonal_data = data[-12:]
        seasonal_pattern = seasonal_data - np.mean(seasonal_data)
        seasonal_forecast = np.tile(seasonal_pattern,
                                    (steps // 12) + 1)[:steps]
    else:
        seasonal_forecast = np.zeros(steps)

    return trend_forecast + seasonal_forecast


# ===================== ASSUMPTIONS =====================
def monthly_assumption_forecast(train_data, steps=12):
    """
    Monthly assumption:
    - Ambil 12 bulan terakhir
    - Hitung rata-rata konsumsi bulanan
    - Tambah ke nilai terakhir, ulangi sampai 12 bulan
    """
    data = np.array(train_data).ravel()
    last_12 = data[-12:] if len(data) >= 12 else data
    monthly_avg = np.sum(last_12) / 12

    preds, last_val = [], data[-1]
    for i in range(steps):
        next_val = last_val + (monthly_avg - last_val) * 0.1
        preds.append(next_val)
        last_val = next_val
    return np.array(preds)


def yearly_assumption_forecast(train_data, steps=12):
    """
    Yearly assumption:
    - Ambil 5 tahun terakhir (60 bulan)
    - Hitung rata-rata growth tahunan
    - Tambahkan ke total tahun terakhir
    - Distribusi ke 12 bulan berikutnya pakai proporsi tahun terakhir
    """
    data = np.array(train_data).ravel()
    last_5y = data[-60:] if len(data) >= 60 else data
    years = len(last_5y) // 12

    if years >= 2:
        yearly_totals = [np.sum(last_5y[i * 12:(i + 1) * 12])
                         for i in range(years)]
        yearly_growth = (yearly_totals[-1] - yearly_totals[0]) / (years - 1)
    else:
        yearly_growth = 0

    last_year = data[-12:] if len(data) >= 12 else data
    last_year_total = np.sum(last_year)
    next_year_total = last_year_total + yearly_growth

    proportions = (last_year / last_year_total
                   if len(last_year) == 12 else np.ones(12) / 12)
    return next_year_total * proportions[:steps]


# ===================== FORECAST FUNCTION =====================
def forecast_last_year(country, ai_algorithm="ARIMA"):
    df = df_raw[df_raw[country_col] == country].copy()
    df = df.groupby("Date", as_index=False)[value_col].sum()
    df = df.rename(columns={value_col: "consumption"})

    # Resample ke bulanan
    df = df.set_index("Date").resample("MS").interpolate(method="linear")

    if len(df) < 72:  # minimal 6 tahun data
        return go.Figure(), pd.DataFrame(), pd.DataFrame()

    train, test = df[:-12], df[-12:]

    # AI model
    models = {
        "ARIMA": arima_forecast,
        "SARIMA": sarima_forecast,
        "LSTM": lstm_like_forecast,
        "N-BEATS": nbeats_like_forecast
    }
    ai_func = models.get(ai_algorithm, arima_forecast)
    ai_pred = ai_func(train.values.ravel(), 12)
    ai_pred = (np.repeat(train.iloc[-1], 12)
               if ai_pred is None else ai_pred)
    ai_pred = pd.Series(ai_pred, index=test.index)

    # Assumptions
    monthly_pred = pd.Series(monthly_assumption_forecast(train.values.ravel(), 12),
                             index=test.index)
    yearly_pred = pd.Series(yearly_assumption_forecast(train.values.ravel(), 12),
                            index=test.index)

    # Metrics
    mse_ai = mean_squared_error(test.values.ravel(), ai_pred.values)
    mae_ai = mean_absolute_error(test.values.ravel(), ai_pred.values)
    mse_monthly = mean_squared_error(test.values.ravel(), monthly_pred.values)
    mae_monthly = mean_absolute_error(test.values.ravel(), monthly_pred.values)
    mse_yearly = mean_squared_error(test.values.ravel(), yearly_pred.values)
    mae_yearly = mean_absolute_error(test.values.ravel(), yearly_pred.values)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train[-24:].index, y=train[-24:].values.ravel(),
                             mode="lines", name="Historical Data",
                             line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=test.index, y=test.values.ravel(),
                             mode="lines", name="Actual Data",
                             line=dict(color="blue", width=3)))
    fig.add_trace(go.Scatter(x=test.index, y=ai_pred,
                             mode="lines", name=f"{ai_algorithm} Forecast",
                             line=dict(color="orange", dash="dash")))
    fig.add_trace(go.Scatter(x=test.index, y=monthly_pred,
                             mode="lines", name="Monthly Assumption",
                             line=dict(color="green", dash="dot")))
    fig.add_trace(go.Scatter(x=test.index, y=yearly_pred,
                             mode="lines", name="Yearly Assumption",
                             line=dict(color="red", dash="dashdot")))

    fig.update_layout(
        title=f"Energy Consumption Forecast - {country} ({ai_algorithm})",
        xaxis_title="Date", yaxis_title="Energy Consumption",
        template="plotly_white", height=600
    )

    # Tables
    detail = pd.DataFrame({
        "Month": test.index.strftime("%Y-%m"),
        "Actual": test.values.ravel(),
        "AI_Forecast": ai_pred.values,
        "Monthly_Assumption": monthly_pred.values,
        "Yearly_Assumption": yearly_pred.values
    }).round(2)

    summary = pd.DataFrame({
        "Metric": ["MSE", "MAE"],
        ai_algorithm: [f"{mse_ai:.2e}", f"{mae_ai:.2f}"],
        "Monthly Assumption": [f"{mse_monthly:.2e}", f"{mae_monthly:.2f}"],
        "Yearly Assumption": [f"{mse_yearly:.2e}", f"{mae_yearly:.2f}"]
    })

    return fig, detail, summary


# ===================== GRADIO UI =====================
def create_forecasting_ui():
    gr.Markdown("## Energy Consumption Forecasting (AI vs Assumptions)")
    default_country = "Indonesia" if "Indonesia" in available_countries else available_countries[0]

    with gr.Row():
        country = gr.Dropdown(choices=available_countries, value=default_country,
                              label="Select Country")
        algo = gr.Dropdown(choices=["ARIMA", "SARIMA", "LSTM", "N-BEATS"],
                           value="ARIMA", label="AI Algorithm")

    with gr.Row():
        run_btn = gr.Button("Run Forecast", variant="primary")
        reset_btn = gr.Button("Reset", variant="secondary")

    chart = gr.Plot(label="Forecast Comparison Chart")
    detail = gr.Dataframe(label="Detailed Results (12 months)", interactive=False)
    summary = gr.Dataframe(label="Summary Statistics", interactive=False)

    run_btn.click(forecast_last_year,
                  inputs=[country, algo],
                  outputs=[chart, detail, summary])
    reset_btn.click(lambda: (go.Figure(), pd.DataFrame(), pd.DataFrame()),
                    outputs=[chart, detail, summary])

    return chart, detail, summary
