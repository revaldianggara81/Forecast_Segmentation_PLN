# main.py
import gradio as gr
from forecasting import create_forecasting_ui
from segmentation import create_segmentation_ui_custom
from segmentation_2d import create_segmentation_ui_2d

MENU_ITEMS = ["Forecasting", "Segmentation 1D", "Segmentation 2D"]

def switch_view(selected):
    return tuple(gr.update(visible=(item == selected)) for item in MENU_ITEMS)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.Markdown(
            """
            <div style="display:flex;align-items:center;gap:8px">
                <img src="http://web.pln.co.id/statics/img/logo-header-20170501a.jpg" 
                     alt="Logo" style="height:32px">
            </div>
            """,
            elem_id="sidebar-title"
        )
        menu = gr.Radio(MENU_ITEMS, value=MENU_ITEMS[0], label=None)

    # Forecasting tab
    with gr.Group(visible=True) as p1:
        forecast_chart, result_table, summary_table = create_forecasting_ui()

    # Segmentation (Custom)
    with gr.Group(visible=False) as p2:
        (mean_chart, total_chart, payment_chart, heatmap_age, heatmap_marital) = create_segmentation_ui_custom()

    # Segmentation 2D
    with gr.Group(visible=False) as p3:
        (scatter_plot,) = create_segmentation_ui_2d()

    menu.change(switch_view, inputs=menu, outputs=[p1, p2, p3])

if __name__ == "__main__":
    demo.launch(share=True)
