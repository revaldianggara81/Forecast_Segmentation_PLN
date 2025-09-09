# segmentation_gradio.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from urllib.request import urlretrieve
from zipfile import ZipFile
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gradio as gr

# ============================================================== #
# Utility: Load & preprocess dataset
# ============================================================== #
def load_and_preprocess():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    local_zip = "household_power_consumption.zip"

    if not os.path.exists(local_zip):
        urlretrieve(url, local_zip)

    with ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall("./dataset/")

    data_path = os.path.join("./dataset/household_power_consumption.txt")
    df = pd.read_csv(
        data_path,
        sep=";",
        header=0,
        usecols=["Date", "Time", "Global_active_power"],
        low_memory=False,
        infer_datetime_format=True,
        parse_dates={"datetime": [0, 1]},
        index_col=["datetime"],
        na_values=["NaN", "?"],
    )

    # Fill missing values
    df.iloc[:, -1] = df.iloc[:, -1].fillna(df.iloc[:, -1].mean())

    # Resample to hourly
    hourly = df.resample("H").sum()
    hourly["Time"] = hourly.index.hour
    hourly.index = hourly.index.date
    hourly.index.name = "Date"

    hourly_pivoted = hourly.pivot(columns="Time").dropna()

    # Normalize with StandardScaler
    scaler = StandardScaler()
    x = hourly_pivoted.values
    x_scaled = scaler.fit_transform(x)

    return x, x_scaled


# ============================================================== #
# Cluster labeling function
# ============================================================== #
def assign_cluster_labels(cluster_profile):
    labels = {}
    for cid, row in cluster_profile.iterrows():
        peak_hour = row.values.argmax()
        if 0 <= peak_hour <= 5:
            labels[cid] = "peak midnight"
        elif 6 <= peak_hour <= 12:
            labels[cid] = "Morning Users"
        elif 13 <= peak_hour <= 17:
            labels[cid] = "Daytime Users"
        elif 18 <= peak_hour <= 23:
            labels[cid] = "Evening Peak Users"
        else:
            labels[cid] = "Balanced Users"
    return labels


# ============================================================== #
# Main clustering + visualization
# ============================================================== #
def run_segmentation(n_clusters=4):
    x, x_scaled = load_and_preprocess()

    # Clustering with fixed random_state
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    identified_clusters = kmeans.fit_predict(x_scaled)

    # Profiles per cluster
    hourly_means = pd.DataFrame(x, columns=[f"Hour_{i}" for i in range(24)])
    hourly_means["Cluster"] = identified_clusters
    cluster_profile = hourly_means.groupby("Cluster").mean()

    # Assign English labels
    labels_dict = assign_cluster_labels(cluster_profile)

    # PCA for visualization
    if n_clusters > 2:
        pca_model = PCA(n_components=3, random_state=42).fit(x_scaled)
        reduced_data = pca_model.transform(x_scaled)
        results = pd.DataFrame(reduced_data, columns=["pca1", "pca2", "pca3"])
        results["Cluster"] = identified_clusters

        fig1 = plt.figure(figsize=(9, 7))
        ax = fig1.add_subplot(projection="3d")

        for cid in sorted(results["Cluster"].unique()):
            subset = results[results["Cluster"] == cid]
            ax.scatter(
                subset["pca1"], subset["pca2"], subset["pca3"],
                label=f"Cluster {cid}: {labels_dict[cid]}", alpha=0.7
            )

        ax.set_title(
            "3D Cluster Visualization\n"
            "(X=Overall Intensity, Y=Day/Night, Z=Morning vs Afternoon Usage)"
        )
        ax.set_xlabel("X (Overall Consumption Intensity)")
        ax.set_ylabel("Y (Daytime vs Nighttime)")
        ax.set_zlabel("Z (Morning vs Afternoon Usage)")
        ax.legend(
            title="Cluster Label",
            bbox_to_anchor=(1.25, 1),
            loc="upper left",
            fontsize=10
        )
    else:
        reduced_data = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
        results = pd.DataFrame(reduced_data, columns=["x", "y"])  # ✅ rename langsung
        results["Cluster"] = identified_clusters

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x="x", y="y",   # ✅ sudah sesuai dengan kolom di DataFrame
            hue="Cluster", palette="tab10",
            data=results, alpha=0.7, ax=ax1
        )
        ax1.set_title(
            "2D PCA Cluster Visualization\n"
            "(x=Overall Intensity, y=Day/Night Pattern)"
        )
        ax1.set_xlabel("x (Overall Consumption Intensity)")
        ax1.set_ylabel("y (Daytime vs Nighttime)")

        # Custom legend mapping cluster ID ke English label
        handles, _ = ax1.get_legend_handles_labels()
        new_labels = [f"Cluster {cid}: {labels_dict[cid]}" for cid in sorted(labels_dict.keys())]
        ax1.legend(handles, new_labels, title="Cluster Label", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Heatmap profile
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_profile, cmap="YlOrRd", annot=False, cbar=True, ax=ax2)
    ax2.set_title("Average Hourly Consumption per Cluster")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Cluster")

    return fig1, fig2


# ============================================================== #
# Gradio UI
# ============================================================== #
with gr.Blocks(title="Electricity Segmentation") as demo:
    gr.Markdown("## Electricity Consumer Segmentation")

    with gr.Row():
        n_clusters = gr.Slider(2, 8, value=4, step=1, label="Number of Clusters")
        run_btn = gr.Button("Run Segmentation")

    with gr.Row():
        pca_plot = gr.Plot(label=" Cluster Visualization (2D/3D)")
        heatmap_plot = gr.Plot(label="Cluster Hourly Profile")

    run_btn.click(fn=run_segmentation,
                  inputs=[n_clusters],
                  outputs=[pca_plot, heatmap_plot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
