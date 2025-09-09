import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
from zipfile import ZipFile
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import gradio as gr
from openai import OpenAI

# ==============================================================
# Helper functions
# ==============================================================
def rgb_to_hex(rgb):
    rgb = tuple(int(x * 255) for x in rgb[:3])
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

def load_indonesia_geojson():
    geojson_path = os.path.join(os.path.dirname(__file__), "indonesia_province.geojson")
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

# ==============================================================
# Data Loading & Preprocessing
# ==============================================================
def load_and_preprocess():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    local_zip = "household_power_consumption.zip"
    data_path = "dataset/household_power_consumption.txt"

    if not os.path.exists(local_zip):
        urlretrieve(url, local_zip)

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

# ==============================================================
# Map Visualization
# ==============================================================
def make_enhanced_indonesia_map(df_clustered, geojson_data, identified_clusters, labels_dict):
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

# ==============================================================
# Main Gradio Interface Function
# ==============================================================
def run_segmentation(n_clusters=4):
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
    else:
        reduced_data = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
        results = pd.DataFrame(reduced_data, columns=["x", "y"])
        results["Cluster"] = identified_clusters
        ax = fig1.add_subplot(111)
        sns.scatterplot(x="x", y="y", hue="Cluster", palette="tab10",
                        data=results, alpha=0.7, ax=ax, s=50)
        ax.set_title("2D Cluster Visualization")

    fig2 = plt.figure(figsize=(14, 6))
    ax2 = fig2.add_subplot(111)
    heatmap_data_all = cluster_profile.T.copy()
    heatmap_data_all.index = [int(col.split('_')[1]) for col in heatmap_data_all.index]
    sns.heatmap(heatmap_data_all, cmap="YlOrRd", annot=False, cbar=True, ax=ax2)
    ax2.set_title("Average Hourly Consumption by Cluster")

    map_html = make_enhanced_indonesia_map(df_clustered.copy(), geojson_data, identified_clusters, labels_dict)
    return fig1, fig2, map_html, df_clustered, labels_dict

# ==============================================================
# Chatbot Function pakai ChatGPT API
# ==============================================================
client = OpenAI(api_key="")

def query_data_from_prompt(prompt, df_clustered, labels_dict):
    if df_clustered is None or labels_dict is None:
        return "Silakan jalankan segmentasi terlebih dahulu."

    region_summary = df_clustered.groupby("Region")[["Sub_metering_1","Sub_metering_2","Sub_metering_3"]].sum().head().to_dict()
    cluster_labels = str(labels_dict)

    system_prompt = f"""
    Kamu adalah asisten data. Gunakan informasi berikut:
    - Ringkasan konsumsi energi per region: {region_summary}
    - Label cluster: {cluster_labels}
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
# Gradio Interface
# ==============================================================
def create_gradio_interface():
    with gr.Blocks(title="Customer Segmentation & Forecast") as demo:
        gr.Markdown("# Customer Segmentation & Analysis")
        df_clustered_state = gr.State(None)
        labels_dict_state = gr.State(None)

        with gr.Row():
            num_clusters = gr.Slider(minimum=2, maximum=4, step=1, value=2, label="Number of Clusters (K)")
        with gr.Row():
            run_btn = gr.Button("Run Segmentation", variant="primary")

        with gr.Tabs():
            with gr.TabItem("Cluster & Map"):
                with gr.Row():
                    pca_plot = gr.Plot()
                    heatmap = gr.Plot()
                map_output = gr.HTML()
            with gr.TabItem("Ask Data"):
                gr.Examples(
                    ["Tampilkan 3 region dengan konsumsi terbesar", "Distribusi cluster per region", "Apa karakteristik cluster"],
                    inputs=[gr.Textbox()],
                )
                with gr.Row():
                    query_input = gr.Textbox(placeholder="Masukkan pertanyaan Anda...")
                    query_btn = gr.Button("Ask")
                query_output = gr.Markdown()

        run_btn.click(fn=run_segmentation, inputs=[num_clusters],
                      outputs=[pca_plot, heatmap, map_output, df_clustered_state, labels_dict_state])
        query_btn.click(fn=query_data_from_prompt,
                        inputs=[query_input, df_clustered_state, labels_dict_state],
                        outputs=[query_output])
    demo.launch()

if __name__ == "__main__":
    create_gradio_interface()
