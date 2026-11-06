from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load imputer
with open("imputer.pkl", "rb") as f:
    fill_zeros = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction_text=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        field_names = [
            "num_preg",
            "glucose_conc",
            "diastolic_bp",
            "insulin",
            "bmi",
            "diab_pred",
            "age",
            "skin"
        ]

        # Read form values
        form_data = {
            field: request.form[field]
            for field in field_names
        }

        # Convert values into a NumPy array
        features = [
            float(form_data[field])
            for field in field_names
        ]

        input_array = np.asarray(features).reshape(1, -1)

        # Apply the same preprocessing used during training
        input_array = fill_zeros.transform(input_array)
        input_array = scaler.transform(input_array)

        # Predict
        prediction = rf_model.predict(input_array)

        result = (
            "Diabetic"
            if prediction[0] == 1
            else "Not Diabetic"
        )

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result}",
            form_data=form_data
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            form_data=request.form
        )


if __name__ == "__main__":
    app.run(debug=True)
