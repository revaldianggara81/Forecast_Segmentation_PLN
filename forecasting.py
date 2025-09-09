# predictive_hajj_planning.py
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# ===== Data & model =====
# Expected columns: Year, Pilgrim_Count, Accommodation_Units, Buses_Needed
df = pd.read_csv("data/hajj_annual_forecast.csv")
X = df["Year"].values.reshape(-1, 1)

model_pilgrims = LinearRegression().fit(X, df["Pilgrim_Count"])
model_accommodation = LinearRegression().fit(X, df["Accommodation_Units"])
model_buses = LinearRegression().fit(X, df["Buses_Needed"])


def create_initial_charts():
    # Chart Pilgrims
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["Year"], y=df["Pilgrim_Count"],
        mode='lines+markers', name='Historical Data'
    ))
    fig1.update_layout(
        title="Pilgrim Volume Trend Analysis",
        xaxis_title="Year", yaxis_title="Pilgrim Count",
        template="plotly_white", height=400
    )

    # Chart Accommodation
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["Year"], y=df["Accommodation_Units"],
        mode='lines+markers', name='Historical Data'
    ))
    fig2.update_layout(
        title="Accommodation Units Trend Analysis",
        xaxis_title="Year", yaxis_title="Units Needed",
        template="plotly_white", height=400
    )

    # Chart Buses
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["Year"], y=df["Buses_Needed"],
        mode='lines+markers', name='Historical Data'
    ))
    fig3.update_layout(
        title="Transportation (Buses) Trend Analysis",
        xaxis_title="Year", yaxis_title="Buses Needed",
        template="plotly_white", height=400
    )

    # Teks kosong saat awal
    return fig1, "", fig2, "", fig3, ""


def forecast_and_update_charts(years_ahead: int):
    future_years = [int(df["Year"].max()) + i for i in range(1, years_ahead + 1)]
    X_future = np.array(future_years).reshape(-1, 1)

    y_pilgrims = model_pilgrims.predict(X_future).astype(int)
    y_accommodation = model_accommodation.predict(X_future).astype(int)
    y_buses = model_buses.predict(X_future).astype(int)

    # Pilgrims
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["Year"], y=df["Pilgrim_Count"],
        mode='lines+markers', name='Historical Data'
    ))
    fig1.add_trace(go.Scatter(
        x=future_years, y=y_pilgrims,
        mode='lines+markers', name='Forecast'
    ))
    fig1.update_layout(
        title="Pilgrim Volume Forecast",
        xaxis_title="Year", yaxis_title="Pilgrim Count",
        template="plotly_white", height=400
    )

    # Accommodation
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["Year"], y=df["Accommodation_Units"],
        mode='lines+markers', name='Historical Data'
    ))
    fig2.add_trace(go.Scatter(
        x=future_years, y=y_accommodation,
        mode='lines+markers', name='Forecast'
    ))
    fig2.update_layout(
        title="Accommodation Units Forecast",
        xaxis_title="Year", yaxis_title="Units Needed",
        template="plotly_white", height=400
    )

    # Buses
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["Year"], y=df["Buses_Needed"],
        mode='lines+markers', name='Historical Data'
    ))
    fig3.add_trace(go.Scatter(
        x=future_years, y=y_buses,
        mode='lines+markers', name='Forecast'
    ))
    fig3.update_layout(
        title="Transportation (Buses) Forecast",
        xaxis_title="Year", yaxis_title="Buses Needed",
        template="plotly_white", height=400
    )

    fy = future_years[-1]
    pilgrims_text = f"**Forecast for {fy}**: Estimated Pilgrims = **{y_pilgrims[-1]:,}**"
    accommodation_text = f"**Forecast for {fy}**: Accommodation Units Needed = **{y_accommodation[-1]:,}**"
    buses_text = f"**Forecast for {fy}**: Buses Needed = **{y_buses[-1]:,}**"

    return fig1, pilgrims_text, fig2, accommodation_text, fig3, buses_text


def create_predictive_hajj_planning_ui():
    """
    Bangun UI di konteks Blocks/Group pemanggil (tanpa membuat Blocks baru).
    Layout: setiap chart di kiri, teks ringkasan di kanan (3 baris).
    """
    gr.Markdown("## Predictive Hajj Planning")
    with gr.Row():
        with gr.Column(scale=2):
            forecast_years = gr.Radio(
                choices=[1, 2],
                label="📅 Forecast Years Ahead",
                value=1,
                info="Select how many years ahead to forecast",
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Generate Forecast", variant="primary")
            clear_btn = gr.Button("🔄 Reset Charts", variant="secondary")

    # Row untuk Pilgrims
    with gr.Row():
        with gr.Column(scale=3):
            pilgrims_chart = gr.Plot(label="Pilgrim Forecast")
        with gr.Column(scale=2):
            pilgrims_text = gr.Markdown(label="Pilgrim Summary")

    # Row untuk Accommodation
    with gr.Row():
        with gr.Column(scale=3):
            accommodation_chart = gr.Plot(label="Accommodation Forecast")
        with gr.Column(scale=2):
            accommodation_text = gr.Markdown(label="Accommodation Summary")

    # Row untuk Buses
    with gr.Row():
        with gr.Column(scale=3):
            buses_chart = gr.Plot(label="Transportation Forecast")
        with gr.Column(scale=2):
            buses_text = gr.Markdown(label="Transportation Summary")

    # Wiring events
    submit_btn.click(
        forecast_and_update_charts,
        inputs=[forecast_years],
        outputs=[
            pilgrims_chart, pilgrims_text,
            accommodation_chart, accommodation_text,
            buses_chart, buses_text
        ],
    )
    clear_btn.click(
        create_initial_charts,
        outputs=[
            pilgrims_chart, pilgrims_text,
            accommodation_chart, accommodation_text,
            buses_chart, buses_text
        ],
    )

    return (
        pilgrims_chart, pilgrims_text,
        accommodation_chart, accommodation_text,
        buses_chart, buses_text
    )
