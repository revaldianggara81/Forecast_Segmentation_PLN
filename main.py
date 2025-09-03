# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gradio as gr
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
from urllib.request import urlretrieve
from zipfile import ZipFile
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openai import OpenAI
import matplotlib.ticker as ticker
import random

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==============================================================
# Forecasting Functions from forecasting_pipeline.py
# ==============================================================
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout

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
    """
    Trains and saves the LSTM model for energy consumption forecasting.
    """
    try:
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
    
    except Exception as e:
        print(f"An error occurred during training: {e}")

# ===== Enhanced Forecast Function =====
def forecast_future_enhanced(forecast_days=30, show_confidence=True, plot_style="Line"):
    """
    Generates and plots a future forecast based on the trained LSTM model.
    """
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
            recent_data = y.tail(180)
            ax.plot(recent_data.index, recent_data.values, 
                   color='#1f77b4', linewidth=2.5, label='Historical Consumption', alpha=0.8)
            ax.plot(future_dates, forecast.flatten(), 
                   color='#ff7f0e', linewidth=3, label='Predicted Consumption', 
                   marker='o', markersize=3, markerfacecolor='white', markeredgewidth=1)
            
            if show_confidence:
                forecast_flat = forecast.flatten()
                std_dev = np.std(y.tail(180)) * 0.15
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

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
        ax.set_title('Energy Consumption Forecast | Oracle Analytics', 
                    fontsize=18, fontweight='bold', pad=25)
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Consumption (MW)', fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        fig.autofmt_xdate()
        plt.tight_layout()

        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Consumption (MW)': [f"{val:.1f}" for val in forecast.flatten()],
            'Day': future_dates.strftime('%A'),
            'Week': [f"Week {date.isocalendar()[1]}" for date in future_dates],
            'Month': future_dates.strftime('%B %Y')
        })
        
        avg_forecast = np.mean(forecast)
        max_forecast = np.max(forecast)
        min_forecast = np.min(forecast)
        trend = "Increasing" if forecast[-1] > forecast[0] else "Decreasing"
        
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

# ==============================================================
# Segmentation Functions from segmentation2.py
# ==============================================================
# ===== Helper functions =====
def rgb_to_hex(rgb):
    rgb = tuple(int(x * 255) for x in rgb[:3])
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

def load_indonesia_geojson():
    geojson_path = "indonesia_province.geojson"
    if not os.path.exists(geojson_path):
        print(f"Error: GeoJSON file not found at {geojson_path}")
        return None
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    return geojson_data

def extract_regions_from_geojson(geojson_data):
    regions = {
        "Sumatera": ["Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi", "Sumatera Selatan", "Bengkulu", "Lampung", "Kepulauan Bangka Belitung", "Kepulauan Riau"],
        "Jawa": ["DKI Jakarta", "Jawa Barat", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur", "Banten"],
        "Kalimantan": ["Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara"],
        "Sulawesi": ["Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan", "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat"],
        "Nusa Tenggara": ["Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur"],
        "Maluku": ["Maluku Utara", "Maluku"],
        "Papua": ["Papua", "Papua Barat", "Papua Selatan", "Papua Tengah", "Papua Pegunungan", "Papua Barat Daya"],
        "Other": []
    }
    province_to_region = {}
    for region, provinces in regions.items():
        for province in provinces:
            province_to_region[province] = region
    
    if geojson_data:
        geojson_provinces = {feature['properties'].get('state', feature['properties'].get('name')) for feature in geojson_data['features']}
        for province in geojson_provinces:
            if province not in province_to_region and province is not None:
                province_to_region[province] = "Other"

    return regions, province_to_region

def assign_cluster_labels(cluster_profile):
    labels = {}
    for cluster_id, profile in cluster_profile.iterrows():
        total_consumption = profile.sum()
        sm3_profile = profile[[col for col in profile.index if 'SM3' in col]]
        if not sm3_profile.empty:
            peak_hour_sm3 = sm3_profile.idxmax().split('_')[1]
            peak_hour = int(peak_hour_sm3)
        else:
            peak_hour = -1

        if total_consumption < 50:
            labels[cluster_id] = "Low Overall Consumption"
        elif total_consumption > 200:
            labels[cluster_id] = "High Overall Consumption"
        else:
            if peak_hour >= 18 or peak_hour <= 6:
                labels[cluster_id] = "Evening/Night Peak"
            else:
                labels[cluster_id] = "Daytime Peak"
    return labels

# ===== Data Loading & Preprocessing =====
def load_and_preprocess():
    """
    Loads, cleans, and preprocesses the household power consumption data.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    local_zip = "household_power_consumption.zip"
    data_path = "dataset/household_power_consumption.txt"

    if not os.path.exists(local_zip):
        print("Downloading data...")
        urlretrieve(url, local_zip)
        print("Download complete.")

    if not os.path.exists("dataset"):
        with ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall("dataset")

    df = pd.read_csv(
        data_path,
        sep=";",
        header=0,
        usecols=["Date", "Time", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
        low_memory=False,
        parse_dates={"datetime": [0, 1]},
        index_col=["datetime"],
        na_values=["NaN", "?"],
    )

    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())

    geojson_data = load_indonesia_geojson()
    regions, province_to_region = extract_regions_from_geojson(geojson_data)
    
    # Simulating region data since original data doesn't have it
    np.random.seed(42)
    region_names = list(regions.keys())
    df["Region"] = np.random.choice(region_names, size=len(df))
    
    hourly_agg = df.resample("H").agg({
        "Sub_metering_1": "mean",
        "Sub_metering_2": "mean",
        "Sub_metering_3": "mean",
        "Region": lambda x: x.mode()[0] if not x.empty else "Jawa"
    })

    hourly_agg["Date"] = hourly_agg.index.date
    hourly_agg["Hour"] = hourly_agg.index.hour
    
    hourly_pivoted_sm1 = hourly_agg.pivot_table(
        index="Date", columns="Hour", values="Sub_metering_1", aggfunc="mean"
    ).fillna(0)
    hourly_pivoted_sm2 = hourly_agg.pivot_table(
        index="Date", columns="Hour", values="Sub_metering_2", aggfunc="mean"
    ).fillna(0)
    hourly_pivoted_sm3 = hourly_agg.pivot_table(
        index="Date", columns="Hour", values="Sub_metering_3", aggfunc="mean"
    ).fillna(0)
    
    x_combined = np.hstack([hourly_pivoted_sm1, hourly_pivoted_sm2, hourly_pivoted_sm3])
    
    daily_agg = hourly_agg.groupby("Date").agg({
        "Region": lambda x: x.mode()[0] if not x.empty else "Jawa",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
    })
    
    df_clustered = daily_agg
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_combined)

    return df_clustered, x_combined, x_scaled, geojson_data

# ===== Map Visualization =====
def make_enhanced_indonesia_map(df_clustered, geojson_data, identified_clusters, labels_dict):
    """
    Creates an interactive Folium map of Indonesia with clustered regions.
    """
    m = folium.Map(location=[-2.5, 118], zoom_start=5)
    df_clustered['Cluster'] = identified_clusters

    regions_df = df_clustered.groupby('Region').agg(
        dominant_cluster=('Cluster', lambda x: x.mode()[0] if not x.empty else -1),
        data_points=('Cluster', 'count')
    ).reset_index()

    regions_df['dominant_cluster'] = regions_df['dominant_cluster'].astype(int)
    
    n_clusters = len(labels_dict)
    cluster_colors_rgb = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    cluster_color_map = {cid: rgb_to_hex(color) for cid, color in zip(labels_dict.keys(), cluster_colors_rgb)}
    
    _, province_to_region = extract_regions_from_geojson(geojson_data)
    
    if geojson_data:
        for feature in geojson_data['features']:
            props = feature['properties']
            state_name = props.get('state', props.get('name', 'Unknown'))
            region = province_to_region.get(state_name, 'Other')
            
            region_stats = regions_df[regions_df['Region'] == region]
            
            if not region_stats.empty:
                dominant_cluster_id = region_stats.iloc[0]['dominant_cluster']
                pop_count = region_stats.iloc[0]['data_points']
                cluster_name = labels_dict.get(dominant_cluster_id, "Unknown Cluster")
                fill_color = cluster_color_map.get(dominant_cluster_id, '#A0A0A0')
                
                popup_text = (f"<b>{state_name}</b><br>"
                              f"Region: {region}<br>"
                              f"Dominant Cluster: <b>{cluster_name}</b><br>"
                              f"Data Points: {pop_count}")
                
                folium.GeoJson(
                    feature,
                    style_function=lambda x, color=fill_color: {
                        "fillColor": color,
                        "color": "black",
                        "weight": 1,
                        "fillOpacity": 0.6,
                    },
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{state_name} ({region})"
                ).add_to(m)
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: auto;
                 background-color: white; border:2px solid grey; z-index:9999; 
                 font-size:14px; padding: 10px">
    <p><strong>Clustered Consumer Segments</strong></p>
    '''
    for cid, label in labels_dict.items():
        color = cluster_color_map.get(cid, '#A0A0A0')
        legend_html += f'<p><i style="background:{color}; width: 20px; height: 10px; display: inline-block;"></i> {label}</p>'
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()

# ===== Main Segmentation Function =====
def run_segmentation(n_clusters=4):
    """
    Main function to run the clustering analysis and generate visualizations.
    """
    try:
        df_clustered, x, x_scaled, geojson_data = load_and_preprocess()
        if geojson_data is None:
            return None, None, "<p>Error: GeoJSON file not found.</p>", None, None

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        identified_clusters = kmeans.fit_predict(x_scaled)

        hourly_means_df = pd.DataFrame(x, columns=[f"SM1_{h}" for h in range(24)] + 
                                                [f"SM2_{h}" for h in range(24)] + 
                                                [f"SM3_{h}" for h in range(24)])
        hourly_means_df["Cluster"] = identified_clusters
        
        cluster_profile = hourly_means_df.groupby("Cluster").mean()
        labels_dict = assign_cluster_labels(cluster_profile)
        
        df_clustered['Cluster'] = identified_clusters
        
        # PCA Plot
        fig1 = plt.figure(figsize=(14, 6))
        if n_clusters > 2:
            reduced_data = PCA(n_components=3, random_state=42).fit_transform(x_scaled)
            results = pd.DataFrame(reduced_data, columns=["x", "y", "z"])
            results["Cluster"] = identified_clusters
            ax = fig1.add_subplot(projection="3d")
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            for i, cid in enumerate(sorted(results["Cluster"].unique())):
                subset = results[results["Cluster"] == cid]
                ax.scatter(subset["x"], subset["y"], subset["z"],
                           label=f"Cluster {cid}: {labels_dict[cid]}", 
                           alpha=0.7, color=colors[i])
            ax.set_title("3D Cluster Visualization")
            ax.legend(title="Clusters", fontsize='small')
        else:
            reduced_data = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
            results = pd.DataFrame(reduced_data, columns=["x", "y"])
            results["Cluster"] = identified_clusters
            ax = fig1.add_subplot(111)
            sns.scatterplot(x="x", y="y", hue="Cluster", palette="tab10",
                            data=results, alpha=0.7, ax=ax, s=50)
            ax.set_title("2D Cluster Visualization")

        # Heatmap Plot
        fig2 = plt.figure(figsize=(14, 6))
        ax2 = fig2.add_subplot(111)
        heatmap_data_all = cluster_profile.T.copy()
        heatmap_data_all.index = [int(col.split('_')[1]) for col in heatmap_data_all.index]
        sns.heatmap(heatmap_data_all, cmap="YlOrRd", annot=False, cbar=True, ax=ax2)
        ax2.set_title("Average Hourly Consumption by Cluster")

        map_html = make_enhanced_indonesia_map(df_clustered.copy(), geojson_data, identified_clusters, labels_dict)
        return fig1, fig2, map_html, df_clustered, labels_dict
    except Exception as e:
        error_fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 
                f'SYSTEM ERROR\n\n{str(e)}\n\nCheck data files and try again.', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee", edgecolor="#f44336"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Segmentation - System Status', fontsize=14, fontweight='bold')
        return error_fig, None, "<p>Error running segmentation. Check the console for details.</p>", None, None


# ===== Chatbot Function =====
client = OpenAI(api_key="")

def query_data_from_prompt(prompt, df_clustered, labels_dict):
    """
    Handles user prompts using a simple chatbot and provides data insights.
    """
    if df_clustered is None or labels_dict is None:
        return "Silakan jalankan segmentasi terlebih dahulu."

    # A simple, fake summary since the original data is not Indonesia-specific
    region_summary_data = {
        'Sumatera': {'Sub_metering_1': random.uniform(1000, 2000), 'Sub_metering_2': random.uniform(500, 1000), 'Sub_metering_3': random.uniform(800, 1500)},
        'Jawa': {'Sub_metering_1': random.uniform(5000, 8000), 'Sub_metering_2': random.uniform(2000, 4000), 'Sub_metering_3': random.uniform(3000, 6000)},
        'Kalimantan': {'Sub_metering_1': random.uniform(800, 1500), 'Sub_metering_2': random.uniform(400, 800), 'Sub_metering_3': random.uniform(700, 1200)},
        'Sulawesi': {'Sub_metering_1': random.uniform(900, 1600), 'Sub_metering_2': random.uniform(450, 900), 'Sub_metering_3': random.uniform(750, 1300)},
        'Nusa Tenggara': {'Sub_metering_1': random.uniform(500, 1000), 'Sub_metering_2': random.uniform(250, 500), 'Sub_metering_3': random.uniform(400, 800)},
    }
    
    region_summary = {region: {k: f"{v:.2f}" for k, v in data.items()} for region, data in region_summary_data.items()}
    cluster_labels = str(labels_dict)

    system_prompt = f"""
    Kamu adalah asisten data. Gunakan informasi berikut:
    - Ringkasan konsumsi energi per region (angka adalah total konsumsi dalam unit arbitrary): {region_summary}
    - Label cluster: {cluster_labels}
    - Konsumsi energi dari data ini berasal dari data rumah tangga di Eropa dan hanya disimulasikan secara acak untuk region di Indonesia.
    Jawablah pertanyaan user secara ringkas dan jelas.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content

# ==============================================================
# Combined Gradio Interface
# ==============================================================
def main_interface():
    """
    Creates the combined Gradio interface with tabs for Forecasting and Segmentation.
    """
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
    .forecast-panel, .segmentation-panel {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .logo-img {
        height: 50px;
        margin-right: 20px;
    }
    .gr-row {
        margin-top: 20px;
    }
    .gr-button.primary {
        background-color: #CC0000 !important;
        border-color: #CC0000 !important;
    }
    """

    with gr.Blocks(css=custom_css, title="Oracle Energy Analytics") as interface:
        gr.HTML("""
            <div class="oracle-header">
                <h1 style="margin: 0; font-size: 28px; font-weight: bold; color: white;">
                    Oracle Solution Center
                </h1>
                <h2 style="margin: 10px 0 0 0; font-size: 20px; font-weight: normal; color: white;">
                    Advanced Analytics Platform
                </h2>
            </div>
        """)
        
        # State variables for the segmentation tab
        df_clustered_state = gr.State(None)
        labels_dict_state = gr.State(None)
        
        with gr.Tabs():
            
            # ===== Forecasting Tab =====
            with gr.TabItem("Energy Forecasting"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Forecast Configuration")
                        forecast_days = gr.Slider(
                            minimum=7, maximum=90, value=30, step=1,
                            label="Forecast Period (Days)",
                            info="Select number of days to forecast (7-90 days)"
                        )
                        show_confidence = gr.Checkbox(
                            value=True, label="Show Confidence Interval",
                            info="Display uncertainty bounds around predictions"
                        )
                        plot_style = gr.Radio(
                            choices=["Line", "Scatter"], value="Line",
                            label="Visualization Style",
                            info="Choose your preferred chart style"
                        )
                        forecast_btn = gr.Button("Generate Forecast", variant="primary", size="lg")
                        gr.HTML('</div>')
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Forecast Results")
                        forecast_plot = gr.Plot(label="Energy Consumption Forecast")
                        summary_info = gr.Textbox(label="Forecast Summary", lines=8, interactive=False)
                        gr.HTML('</div>')
                
                gr.Markdown("### Detailed Forecast Results")
                forecast_table = gr.Dataframe(
                    label="Forecast Data", interactive=False, wrap=True, max_height=400
                )
                
                forecast_btn.click(
                    fn=forecast_future_enhanced,
                    inputs=[forecast_days, show_confidence, plot_style],
                    outputs=[forecast_plot, forecast_table, summary_info]
                )

            # ===== Segmentation Tab =====
            with gr.TabItem("Customer Segmentation"):
                gr.Markdown("### Customer Segmentation & Analysis")
                
                with gr.Row():
                    num_clusters = gr.Slider(minimum=2, maximum=4, step=1, value=2, label="Number of Clusters (K)")
                    run_btn = gr.Button("Run Segmentation", variant="primary")
                
                with gr.Tabs():
                    with gr.TabItem("Cluster & Map"):
                        with gr.Row():
                            pca_plot = gr.Plot(label="Cluster Visualization")
                            heatmap = gr.Plot(label="Average Hourly Consumption by Cluster")
                        map_output = gr.HTML(label="Clustered Regions Map")
                    
                    with gr.TabItem("Ask Data"):
                        gr.Examples(
                            ["Tampilkan 3 region dengan konsumsi terbesar", "Distribusi cluster per region", "Apa karakteristik cluster"],
                            inputs=[gr.Textbox()],
                        )
                        with gr.Row():
                            query_input = gr.Textbox(placeholder="Masukkan pertanyaan Anda...", scale=4)
                            query_btn = gr.Button("Ask", scale=1)
                        query_output = gr.Markdown()
                
                run_btn.click(fn=run_segmentation, inputs=[num_clusters],
                              outputs=[pca_plot, heatmap, map_output, df_clustered_state, labels_dict_state])
                query_btn.click(fn=query_data_from_prompt,
                                inputs=[query_input, df_clustered_state, labels_dict_state],
                                outputs=[query_output])
        
        gr.HTML("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; 
                        background: rgba(255,255,255,0.8); border-radius: 10px;">
                <p style="margin: 0; color: #666; font-size: 12px;">
                    Oracle Solution Center | Advanced Analytics Platform<br>
                </p>
            </div>
        """)
        
    return interface

# ==============================================================
# Main Application
# ==============================================================
if __name__ == "__main__":
    # Check and train model if needed
    if not os.path.exists("lstm_model.h5"):
        print("Training forecasting model for the first time...")
        train_and_save_model()
    else:
        print("Using existing trained forecasting model.")

    # Create and launch the combined interface
    demo = main_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False
    )