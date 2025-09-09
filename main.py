# main.py
import gradio as gr
from forecasting import create_forecasting_ui
from segmentation import create_segmentation_ui

MENU_ITEMS = ["Forecasting", "Segmentation"]

def switch_view(selected):
    return tuple(gr.update(visible=(item == selected)) for item in MENU_ITEMS)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.Markdown("### PLN Demo")
        menu = gr.Radio(MENU_ITEMS, value=MENU_ITEMS[0], label=None)

    with gr.Group(visible=True) as p1:
        forecast_chart, result_table, summary_table = create_forecasting_ui()

    with gr.Group(visible=False) as p2:
        (clustered_data, cluster_summary,
         pca_visualization, feature_visualization,
         cluster_profiles_text) = create_segmentation_ui()

    menu.change(switch_view, inputs=menu, outputs=[p1, p2])

if __name__ == "__main__":
    demo.launch(share=True)
