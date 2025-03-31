import numpy as np
from flask import Flask, request, render_template, render_template_string
import pickle

# Create flask app
flask_app = Flask(__name__, template_folder="templates")
model = pickle.load(open("model.pkl", "rb"))
import os
print("Current working directory:", os.getcwd())
print("Templates folder exists:", os.path.exists("templates"))
print("index.html exists:", os.path.exists("templates/index.html"))


@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The Predicted Crop is {}".format(prediction[0]))

if __name__ == "__main__":
    flask_app.run(debug=True)