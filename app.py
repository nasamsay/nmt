import os
import logging
from flask import Flask, request, render_template
from model.model import NMT
import gradio as gr

app = Flask(__name__)
model = NMT()
logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/v1/predict", methods=["POST"])
def predict():
    """Provide main prediction API route. Responds to POST requests."""
    logging.info("Predict request received!")
    input_text = request.form.get("input_text")
    prediction = model.predict(input_text)
    
    logging.info("prediction from model= {}".format(prediction))
    return render_template("index.html", output_text=prediction)

@app.route("/gradio")
def gradio_interface():
    iface = gr.Interface(fn=predict_gr, inputs="text", outputs="text", title="Machine Translation")
    return iface.launch(share=True)


def predict_gr(input_text):
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")
    prediction = model.predict(input_text)

    logging.info("prediction from model= {}".format(prediction))
    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
