# pip install gradio>=4.29.0
import gradio as gr
from automation_doc_processing import create_automation_docs_processing_ui
from predictive_hajj_planning import (
    create_predictive_hajj_planning_ui,
    create_initial_charts,
)
from health_monitoring_pilgrims import create_health_monitoring_ui
from fraud_detection_risk import create_fraud_detection_ui
from sentiment_analysis import create_sentiment_analysis_ui
from qa_chatbot import create_qa_chatbot_ui

MENU_ITEMS = [
    "Predictive Hajj Planning",
    "Fraud Detection & Risk",
    "Automatic document Processing",
    "Health Monitoring for Pilgrims",
    "Sentiment Analysis",
    "Smart Pilgrim QA Chatbot for Hajj"
]

def switch_view(selected):
    return tuple(gr.update(visible=(item == selected)) for item in MENU_ITEMS)

def predict_hajj_plan(inputs):
    return f"👳‍♂️ Rekomendasi rencana haji (dummy) untuk: {inputs}"

def fraud_check(txn_text):
    score = 0.23
    note = "Risiko rendah (contoh)."
    return f"Skor risiko: {score:.2f}\nCatatan: {note}"

def health_monitor(name, heart_rate, temp):
    flags = []
    if heart_rate and (heart_rate < 50 or heart_rate > 110):
        flags.append("Detak jantung di luar rentang normal")
    if temp and temp >= 37.8:
        flags.append("Demam terdeteksi")
    status = "⚠️ " + "; ".join(flags) if flags else "✅ Stabil"
    return f"👤 {name or 'Peziarah'} — {status}"

CSS = """
html, body, #root, .gradio-container { height: 100%; }
.gradio-container { max-width: none; }
.app .main { display:flex; min-height: 100dvh; }
.app .main > .block { flex:1 1 auto; width:100%; padding:clamp(8px,2vw,24px); }

.panel {
  border: 1px solid var(--block-border-color);
  border-radius: 12px;
  padding: 16px;
  min-height: calc(100dvh - 140px);
  box-sizing: border-box;
}

/* Sidebar radio styling */
.gradio-container .sidebar [role="radiogroup"] [role="radio"]{
  background: transparent !important;
  border: 1px solid transparent !important;
  box-shadow: none !important;
}
.gradio-container .sidebar [role="radiogroup"] [role="radio"][aria-checked="true"]{
  background: var(--color-accent) !important;
  color: #fff !important;
  border-color: var(--color-accent) !important;
}
.gradio-container .sidebar [role="radiogroup"] [role="radio"]:hover{
  background: color-mix(in srgb, var(--color-accent) 12%, transparent) !important;
}
"""

with gr.Blocks(theme=gr.themes.Default(), css=CSS) as demo:
    with gr.Sidebar():
        gr.Markdown("### ☰ Menu")
        menu = gr.Radio(MENU_ITEMS, value=MENU_ITEMS[0], label=None)
        gr.Markdown("---")
        # gr.Markdown("**Context**")
        # gr.Markdown('<span class="badge">Demo UI</span> <span class="badge">Gradio</span>')

    with gr.Group(visible=True, elem_classes="panel") as p1:
        (pilgrims_chart, pilgrims_text,
         accommodation_chart, accommodation_text,
         buses_chart, buses_text) = create_predictive_hajj_planning_ui()

    # with gr.Group(visible=False, elem_classes="panel") as p2:
    #     fraud_ui = create_segmentation_ui()

    menu.change(switch_view, inputs=menu, outputs=[p1])

    demo.load(
        fn=create_initial_charts,
        outputs=[pilgrims_chart, pilgrims_text,
                 accommodation_chart, accommodation_text,
                 buses_chart, buses_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)
