# segmentation_2d.py
import pandas as pd
import plotly.express as px
import gradio as gr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class CustomerSegmentation2D:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()

    def load_data(self, file):
        try:
            if file.name.endswith(".csv"):
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

        # Handle customer_since
        if "customer_since" in df.columns:
            df["customer_since"] = pd.to_datetime(df["customer_since"])
            df["customer_year"] = df["customer_since"].dt.year
            df["days_as_customer"] = (datetime.now() - df["customer_since"]).dt.days
            df = df.drop("customer_since", axis=1)

        # Encode categorical
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col])

        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if "user_id" in numerical_cols:
            numerical_cols.remove("user_id")

        self.processed_data = df[numerical_cols].fillna(df[numerical_cols].median())
        self.data = df
        return f"Data preprocessed successfully! Features used: {numerical_cols}"

    def perform_segmentation(self):
        if self.processed_data is None:
            return None, None

        df = self.data.copy()

        # thresholds for 2D segmentation
        median_consumption = df["average_daily_consumption"].median()
        median_area = df["area"].median()

        def assign_cluster(row):
            if row["average_daily_consumption"] <= median_consumption and row["area"] <= median_area:
                return "Cluster 1: Small house, low consumption"
            elif row["average_daily_consumption"] > median_consumption and row["area"] <= median_area:
                return "Cluster 2: Small house, high consumption"
            elif row["average_daily_consumption"] <= median_consumption and row["area"] > median_area:
                return "Cluster 3: Large house, low consumption"
            else:
                return "Cluster 4: Large house, high consumption"

        df["Cluster_Label"] = df.apply(assign_cluster, axis=1)

        # Scatter plot 2D
        scatter_plot = px.scatter(
            df,
            x="average_daily_consumption",
            y="area",
            color="Cluster_Label",
            title="2D Segmentation: Consumption vs Area",
            labels={
                "average_daily_consumption": "Average Daily Consumption (KWh)",
                "area": "Area (m²)",
                "Cluster_Label": "Cluster",
            },
        )

        return df, scatter_plot


def create_segmentation_ui_2d():
    segmentation = CustomerSegmentation2D()

    def load_and_preview(file):
        _, head, desc = segmentation.load_data(file)
        return head, desc

    def preprocess():
        return segmentation.preprocess_data()

    def perform_segmentation():
        _, scatter_plot = segmentation.perform_segmentation()
        return scatter_plot

    with gr.Blocks(title="Customer Segmentation 2D") as interface:
        gr.Markdown("## Customer Segmentation 2D (Consumption vs Area)")

        with gr.Tab("Data Loading & Preprocessing"):
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            load_btn = gr.Button("Load Data", variant="primary")
            preprocess_btn = gr.Button("Preprocess Data", variant="secondary")
            data_head = gr.Dataframe(label="Data Preview")
            data_desc = gr.Dataframe(label="Statistics")
            preprocess_info = gr.Textbox(label="Preprocessing Information", lines=3)

        with gr.Tab("2D Segmentation"):
            segment_btn = gr.Button("Perform 2D Segmentation", variant="primary")
            scatter_plot = gr.Plot(label="Segmentation 2D")

        load_btn.click(fn=load_and_preview, inputs=[file_input], outputs=[data_head, data_desc])
        preprocess_btn.click(fn=preprocess, outputs=[preprocess_info])
        segment_btn.click(fn=perform_segmentation, outputs=[scatter_plot])

    return (scatter_plot,)
