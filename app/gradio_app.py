import joblib
from pathlib import Path
import pandas as pd
import gradio as gr
from matplotlib import pyplot as plt
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


def predict_structured(area, bedrooms, bathrooms, stories,
                       mainroad, guestroom, basement, hotwaterheating,
                       airconditioning, parking, prefarea, furnishingstatus):
    data = {
        "area": float(area),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "stories": int(stories),
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": int(parking),
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return f"Estimated price: {pred:,.0f}"


def plot_prediction_scatter(y_true, y_pred, out_path):
    if len(y_true) == 0 or len(y_pred) == 0 :
        return
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.xlabel("Ture Prices")
    plt.ylabel("Predicted Prices")
    plt.title("True vs Predicted House Prices")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.savefig(out_path)

def run_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# House Price Chat-like Predictor\nFill inputs then click Predict. Chat shows input and model answer.")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column():
                area = gr.Number(value=6000, label="area")
                bedrooms = gr.Number(value=3, label="bedrooms")
                bathrooms = gr.Number(value=2, label="bathrooms")
                stories = gr.Number(value=2, label="stories")
                parking = gr.Number(value=1, label="parking")
            with gr.Column():
                mainroad = gr.Dropdown(choices=["yes", "no"], value="yes", label="mainroad")
                guestroom = gr.Dropdown(choices=["yes", "no"], value="no", label="guestroom")
                basement = gr.Dropdown(choices=["yes", "no"], value="no", label="basement")
                hotwaterheating = gr.Dropdown(choices=["yes", "no"], value="no", label="hotwaterheating")
                airconditioning = gr.Dropdown(choices=["yes", "no"], value="no", label="airconditioning")
                prefarea = gr.Dropdown(choices=["yes", "no"], value="no", label="prefarea")
                furnishingstatus = gr.Dropdown(choices=["furnished", "semi-furnished", "unfurnished"], value="semi-furnished", label="furnishingstatus")

        def on_predict(a, b, c, d, m, g, bs, hw, ac, p, pf, fs):
            human = f"Inputs -> area:{a}, bedrooms:{b}, bathrooms:{c}, stories:{d}, mainroad:{m}, guestroom:{g}, basement:{bs}, hotwaterheating:{hw}, airconditioning:{ac}, parking:{p}, prefarea:{pf}, furnishingstatus:{fs}"
            bot = predict_structured(a, b, c, d, m, g, bs, hw, ac, p, pf, fs)

            with gr.Blocks():
                gr.Markdown("### Prediction Scatter Plot")
                plot_prediction_scatter([], [], MODEL_PATH.parent / "prediction_scatter.png")
                gr.Image(MODEL_PATH.parent / "prediction_scatter.png")
            # Newer Gradio versions expect messages as dicts with 'role' and 'content'
            return [{"role": "user", "content": human}, {"role": "assistant", "content": bot}]

        predict_btn = gr.Button("Predict")
        predict_btn.click(on_predict, inputs=[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus], outputs=chatbot)


    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    run_gradio()
