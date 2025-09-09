# segmentation.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.clusters = None
        self.cluster_centers = None
        self.pca = PCA(n_components=2)
        
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
        if 'customer_since' in df.columns:
            df['customer_since'] = pd.to_datetime(df['customer_since'])
            df['days_as_customer'] = (datetime.now() - df['customer_since']).dt.days
            df = df.drop('customer_since', axis=1)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'user_id' in numerical_cols:
            numerical_cols.remove('user_id')
            
        self.processed_data = df[numerical_cols].fillna(df[numerical_cols].median())
        return f"Data preprocessed successfully! Features used: {numerical_cols}"
    
    def find_optimal_clusters(self, method='kmeans', max_clusters=10):
        if self.processed_data is None:
            return "Please preprocess data first", None
        
        scaled_data = self.scaler.fit_transform(self.processed_data)
        k_range = range(2, min(max_clusters + 1, len(self.processed_data)))
        
        inertias, silhouette_scores, calinski_scores, davies_bouldin_scores = [], [], [], []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_data, labels))
            calinski_scores.append(calinski_harabasz_score(scaled_data, labels))
            davies_bouldin_scores.append(davies_bouldin_score(scaled_data, labels))
        
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=('Elbow Method', 'Silhouette Score',
                            'Calinski-Harabasz Score', 'Davies-Bouldin Score'))
        fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers'), row=1, col=2)
        fig.add_trace(go.Scatter(x=list(k_range), y=calinski_scores, mode='lines+markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(k_range), y=davies_bouldin_scores, mode='lines+markers'), row=2, col=2)
        fig.update_layout(height=600, showlegend=False, title_text="Clustering Evaluation Metrics")
        
        optimal_k_sil = k_range[np.argmax(silhouette_scores)]
        optimal_k_calinski = k_range[np.argmax(calinski_scores)]
        optimal_k_db = k_range[np.argmin(davies_bouldin_scores)]
        candidates = [optimal_k_sil, optimal_k_calinski, optimal_k_db]
        final_k = max(set(candidates), key=candidates.count)
        
        info = f"""
Recommended clusters:
- Silhouette → {optimal_k_sil}
- Calinski-Harabasz → {optimal_k_calinski}
- Davies-Bouldin → {optimal_k_db}

Final recommendation: {final_k}
"""
        return info, fig
    
    def perform_clustering(self, method='kmeans', n_clusters=3, eps=0.5, min_samples=5):
        if self.processed_data is None:
            return "Please preprocess data first", None, None
        
        scaled_data = self.scaler.fit_transform(self.processed_data)
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = model.fit_predict(scaled_data)
            self.cluster_centers = model.cluster_centers_
        elif method == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples)
            self.clusters = model.fit_predict(scaled_data)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            self.clusters = model.fit_predict(scaled_data)
        
        result_data = self.data.copy()
        result_data['Cluster'] = self.clusters
        info = "Clustering completed!"
        if len(set(self.clusters)) > 1:
            sil_score = silhouette_score(scaled_data, self.clusters)
            info = f"Clustering completed! Silhouette Score: {sil_score:.3f}"
        return info, result_data, self.create_cluster_summary()
    
    def create_cluster_summary(self):
        if self.clusters is None or self.processed_data is None:
            return None
        summary_data = self.processed_data.copy()
        summary_data['Cluster'] = self.clusters
        return summary_data.groupby('Cluster').agg(['mean', 'std', 'count']).round(3)
    
    def visualize_clusters_2d(self):
        if self.clusters is None or self.processed_data is None:
            return None
        scaled_data = self.scaler.fit_transform(self.processed_data)
        pca_data = self.pca.fit_transform(scaled_data)
        plot_data = pd.DataFrame({
            'PC1': pca_data[:, 0],
            'PC2': pca_data[:, 1],
            'Cluster': self.clusters.astype(str)
        })
        return px.scatter(plot_data, x='PC1', y='PC2', color='Cluster',
                          title='Customer Clusters (PCA Visualization)')
    
    def visualize_cluster_features(self, selected_features=None):
        if self.clusters is None or self.processed_data is None:
            return None
        if selected_features is None:
            selected_features = self.processed_data.columns[:4]
        
        plot_data = self.processed_data.copy()
        plot_data['Cluster'] = self.clusters.astype(str)
        
        n_features = len(selected_features)
        fig = make_subplots(rows=(n_features + 1) // 2, cols=2, subplot_titles=selected_features)
        colors = px.colors.qualitative.Set1[:len(set(self.clusters))]
        
        for i, feature in enumerate(selected_features):
            row, col = (i // 2) + 1, (i % 2) + 1
            for j, cluster in enumerate(sorted(set(self.clusters))):
                cluster_data = plot_data[plot_data['Cluster'] == str(cluster)][feature]
                fig.add_trace(go.Box(y=cluster_data, name=f'Cluster {cluster}',
                                     boxpoints='outliers',
                                     line_color=colors[j % len(colors)],
                                     showlegend=(i == 0)),
                              row=row, col=col)
        return fig
    
    def create_cluster_profiles(self):
        if self.clusters is None or self.data is None:
            return "No clustering results available"
        profiles = []
        result_data = self.data.copy()
        result_data['Cluster'] = self.clusters
        for cluster in sorted(set(self.clusters)):
            cluster_data = result_data[result_data['Cluster'] == cluster]
            size = len(cluster_data)
            profile = f"\n{'='*40}\nCLUSTER {cluster} (n={size})\n{'='*40}\n"
            for col in self.processed_data.columns:
                mean_val = self.processed_data.loc[cluster_data.index, col].mean()
                profile += f"{col}: {mean_val:.2f}\n"
            for col in self.data.select_dtypes(include=['object']).columns:
                if col in cluster_data.columns:
                    mode_val = cluster_data[col].mode().iloc[0] if not cluster_data[col].mode().empty else "N/A"
                    profile += f"{col}: {mode_val}\n"
            profiles.append(profile)
        return '\n'.join(profiles)


# ========== Gradio UI Wrapper ==========
def create_segmentation_ui():
    segmentation = CustomerSegmentation()

    def load_and_preview_data(file):
        _, head, desc = segmentation.load_data(file)
        return head, desc

    def preprocess_data():
        return segmentation.preprocess_data()

    def find_optimal_k(method, max_clusters):
        return segmentation.find_optimal_clusters(method, max_clusters)

    def perform_clustering_analysis(method, n_clusters):
        if method == "dbscan":
            info, data, summary = segmentation.perform_clustering("dbscan", eps=0.5, min_samples=5)
        else:
            info, data, summary = segmentation.perform_clustering(method, n_clusters=n_clusters)
        pca_plot = segmentation.visualize_clusters_2d()
        feature_plot = segmentation.visualize_cluster_features()
        profiles = segmentation.create_cluster_profiles()
        return info, data, summary, pca_plot, feature_plot, profiles

    with gr.Blocks(title="Customer Segmentation Dashboard") as interface:
        gr.Markdown("## Customer Segmentation & Analysis")

        with gr.Tab("Data Loading & Preprocessing"):
            with gr.Row():
                file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                load_btn = gr.Button("Load Data", variant="primary")
                preprocess_btn = gr.Button("Preprocess Data", variant="secondary")
            with gr.Row():
                data_head = gr.Dataframe(label="Data Preview")
                data_desc = gr.Dataframe(label="Statistics")
            preprocess_info = gr.Textbox(label="Preprocessing Information", lines=3)

        with gr.Tab("Optimal Clusters Analysis"):
            with gr.Row():
                method_select = gr.Dropdown(choices=["kmeans"], value="kmeans", label="Clustering Method")
                max_k = gr.Slider(2, 15, value=10, step=1, label="Maximum Clusters")
                optimize_btn = gr.Button("Find Optimal Clusters", variant="primary")
            optimal_info = gr.Textbox(label="Optimization Results", lines=8)
            optimization_plot = gr.Plot(label="Clustering Metrics")

        with gr.Tab("Clustering & Analysis"):
            cluster_method = gr.Dropdown(choices=["kmeans", "dbscan", "hierarchical"], value="kmeans")
            n_clusters_input = gr.Slider(2, 10, value=3, step=1, label="Number of Clusters")
            cluster_btn = gr.Button("Perform Clustering", variant="primary")

            cluster_info = gr.Textbox(label="Clustering Results", lines=3)
            clustered_data = gr.Dataframe(label="Clustered Data")
            cluster_summary = gr.Dataframe(label="Cluster Summary Statistics")
            pca_visualization = gr.Plot(label="PCA Cluster Visualization")
            feature_visualization = gr.Plot(label="Feature Distribution by Cluster")
            cluster_profiles_text = gr.Textbox(label="Cluster Profiles", lines=15)

        # Wiring events
        load_btn.click(fn=load_and_preview_data, inputs=[file_input], outputs=[data_head, data_desc])
        preprocess_btn.click(fn=preprocess_data, outputs=[preprocess_info])
        optimize_btn.click(fn=find_optimal_k, inputs=[method_select, max_k],
                           outputs=[optimal_info, optimization_plot])
        cluster_btn.click(
            fn=perform_clustering_analysis,
            inputs=[cluster_method, n_clusters_input],
            outputs=[cluster_info, clustered_data, cluster_summary,
                     pca_visualization, feature_visualization, cluster_profiles_text]
        )

    return (clustered_data, cluster_summary, pca_visualization, feature_visualization, cluster_profiles_text)
