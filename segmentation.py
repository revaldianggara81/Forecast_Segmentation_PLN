# segmentation.py
import pandas as pd
import numpy as np
import plotly.express as px
import gradio as gr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# Base Class for Preprocessing
# ============================================================
class CustomerSegmentationBase:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.clusters = None

    def load_data(self, file):
        try:
            if file.name.endswith('.csv'):
                self.data = pd.read_csv(file.name)
            else:
                return "Error: Please upload a CSV file", None, None
            return "Dataset loaded successfully!", self.data.head(), self.data.describe()
        except Exception as e:
            return f"Error loading data: {str(e)}", None, None

    def preprocess_data(self):
        if self.data is None:
            return "No data loaded"

        df = self.data.copy()

        # handle customer_since
        if 'customer_since' in df.columns:
            df['customer_since'] = pd.to_datetime(df['customer_since'])
            df['customer_year'] = df['customer_since'].dt.year
            df['days_as_customer'] = (datetime.now() - df['customer_since']).dt.days
            df = df.drop('customer_since', axis=1)

        # payment_status logic: negative = Late, positive or zero = OnTime
        if 'average_days_to_bill_payment' in df.columns:
            df['payment_status'] = df['average_days_to_bill_payment'].apply(
                lambda x: "Late" if x < 0 else "OnTime"
            )

        # encode categorical
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'user_id' in numerical_cols:
            numerical_cols.remove('user_id')

        self.processed_data = df[numerical_cols].fillna(df[numerical_cols].median())
        self.data = df
        return f"Data preprocessed successfully! Features used: {numerical_cols}"


# ============================================================
# Segmentation (Custom)
# ============================================================
class CustomerSegmentationCustom(CustomerSegmentationBase):
    def perform_clustering(self):
        if self.processed_data is None:
            return None, None, None, None, None, None

        # gunakan hanya average_daily_consumption
        features = ['average_daily_consumption']
        scaled_data = self.scaler.fit_transform(self.processed_data[features])

        # fix cluster = 3
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.clusters = model.fit_predict(scaled_data)

        result_data = self.data.copy()
        result_data['Cluster'] = self.clusters

        # mapping cluster label
        cluster_map = {0: "Low Consumption", 1: "Middle Consumption", 2: "High Consumption"}
        result_data['Cluster_Label'] = result_data['Cluster'].map(cluster_map)

        # Mean Daily Consumption per Cluster (Horizontal bar)
        mean_vals = result_data.groupby("Cluster_Label")["average_daily_consumption"].mean().reset_index()
        mean_chart = px.bar(
            mean_vals,
            x="average_daily_consumption",
            y="Cluster_Label",
            orientation="h",
            color="Cluster_Label",
            category_orders={"Cluster_Label": ["Low Consumption", "Middle Consumption", "High Consumption"]},
            title="Mean Daily Consumption per Cluster"
        )
        mean_chart.update_traces(texttemplate='%{x:.2f}', textposition="outside")
        mean_chart.update_layout(xaxis_title="average_daily_consumption (KWh)", yaxis_title="Cluster_Label")

        # Total Customers per Cluster (Horizontal bar)
        total_vals = result_data.groupby("Cluster_Label")["user_id"].count().reset_index()
        total_chart = px.bar(
            total_vals,
            x="user_id",
            y="Cluster_Label",
            orientation="h",
            color="Cluster_Label",
            category_orders={"Cluster_Label": ["Low Consumption", "Middle Consumption", "High Consumption"]},
            title="Total Customers per Cluster"
        )
        total_chart.update_traces(texttemplate='%{x}', textposition="outside")
        total_chart.update_layout(xaxis_title="Total Customers", yaxis_title="Cluster_Label")

        # Payment Status Breakdown per Cluster (percentage, Horizontal bar)
        payment_counts = result_data.groupby(["Cluster_Label", "payment_status"])["user_id"].count().reset_index()
        total_per_cluster = result_data.groupby("Cluster_Label")["user_id"].count().reset_index()
        payment_counts = payment_counts.merge(total_per_cluster, on="Cluster_Label", suffixes=("", "_total"))
        payment_counts["percentage"] = (payment_counts["user_id"] / payment_counts["user_id_total"]) * 100

        payment_chart = px.bar(
            payment_counts,
            x="percentage",
            y="Cluster_Label",
            orientation="h",
            color="payment_status",
            barmode="group",
            category_orders={"Cluster_Label": ["Low Consumption", "Middle Consumption", "High Consumption"]},
            title="Payment Status Breakdown per Cluster"
        )
        payment_chart.update_traces(texttemplate='%{x:.1f}%', textposition="outside")
        payment_chart.update_layout(xaxis_title="Percentage of Customers", yaxis_title="Cluster_Label")

        # Heatmap Cluster vs Age
        heatmap_age = px.density_heatmap(
            result_data, x="Cluster_Label", y="age", z="user_id", histfunc="count",
            category_orders={"Cluster_Label": ["Low Consumption", "Middle Consumption", "High Consumption"]},
            color_continuous_scale="Reds",
            title="Heatmap: Cluster vs Age"
        )

        # Heatmap Cluster vs Marital Status
        heatmap_marital = px.density_heatmap(
            result_data, x="Cluster_Label", y="marital_status", z="user_id", histfunc="count",
            category_orders={"Cluster_Label": ["Low Consumption", "Middle Consumption", "High Consumption"]},
            color_continuous_scale="Reds",
            title="Heatmap: Cluster vs Marital Status"
        )

        return result_data, mean_chart, total_chart, payment_chart, heatmap_age, heatmap_marital


def create_segmentation_ui_custom():
    segmentation = CustomerSegmentationCustom()

    def load_and_preview(file):
        _, head, desc = segmentation.load_data(file)
        return head, desc

    def preprocess():
        return segmentation.preprocess_data()

    def perform_clustering():
        result_data, mean_chart, total_chart, payment_chart, heatmap_age, heatmap_marital = segmentation.perform_clustering()
        return result_data, mean_chart, total_chart, payment_chart, heatmap_age, heatmap_marital

    with gr.Blocks(title="Customer Segmentation Dashboard (Custom)") as interface:
        gr.Markdown("## Customer Segmentation & Analysis")

        with gr.Tab("Data Loading & Preprocessing"):
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            load_btn = gr.Button("Load Data", variant="primary")
            preprocess_btn = gr.Button("Preprocess Data", variant="secondary")
            data_head = gr.Dataframe(label="Data Preview")
            data_desc = gr.Dataframe(label="Statistics")
            preprocess_info = gr.Textbox(label="Preprocessing Information", lines=3)

        with gr.Tab("Clustering & Analysis"):
            cluster_btn = gr.Button("Perform Clustering", variant="primary")

            with gr.Row():
                heatmap_age = gr.Plot(label="Heatmap Cluster vs Age")
                heatmap_marital = gr.Plot(label="Heatmap Cluster vs Marital Status")

            clustered_data = gr.Dataframe(label="Clustered Data", visible=False)

            mean_chart = gr.Plot(label="Mean Daily Consumption per Cluster")
            total_chart = gr.Plot(label="Total Customers per Cluster")
            payment_chart = gr.Plot(label="Payment Status Breakdown per Cluster")

        load_btn.click(fn=load_and_preview, inputs=[file_input], outputs=[data_head, data_desc])
        preprocess_btn.click(fn=preprocess, outputs=[preprocess_info])
        cluster_btn.click(fn=perform_clustering,
                          outputs=[clustered_data, mean_chart, total_chart, payment_chart, heatmap_age, heatmap_marital])

    return (mean_chart, total_chart, payment_chart, heatmap_age, heatmap_marital)
