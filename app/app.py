from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model("app/model/fake_news_lstm_model.h5")
with open("app/model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 300

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["news"]
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_len)
        result = model.predict(padded)[0][0]
        prediction = "Fake" if result >= 0.5 else "Real"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
