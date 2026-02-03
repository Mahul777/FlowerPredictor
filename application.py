from flask import Flask, render_template, request
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the trained model
model = joblib.load("artifacts/models/model.pkl")

# Define the mapping from model output to flower species
class_names = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = {}

    if request.method == 'POST':
        try:
            # Get the input values from the form
            sepal_length_cm = request.form.get('sepal_length_cm')
            sepal_width_cm = request.form.get('sepal_width_cm')
            petal_length_cm = request.form.get('petal_length_cm')
            petal_width_cm = request.form.get('petal_width_cm')

            # Debug: Print form data to check if values are received
            app.logger.debug(f"Form data received: {request.form}")

            # Check if any input is missing or empty
            if not sepal_length_cm or not sepal_width_cm or not petal_length_cm or not petal_width_cm:
                prediction['prediction'] = "Error: Please enter values for all fields."
            else:
                # Convert inputs to float
                sepal_length_cm = float(sepal_length_cm)
                sepal_width_cm = float(sepal_width_cm)
                petal_length_cm = float(petal_length_cm)
                petal_width_cm = float(petal_width_cm)

                # Prepare input data for prediction (ensure it's a 2D array)
                input_data = [[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]]  # 2D list

                # Debug: Print input data to check the format
                app.logger.debug(f"Input data for prediction: {input_data}")

                # Make the prediction
                model_prediction = model.predict(input_data)

                # Debug: Print the model's raw prediction output
                app.logger.debug(f"Model Prediction: {model_prediction}")

                # Map the class index to the flower species name
                prediction['prediction'] = class_names[model_prediction[0]]  # Map index to species name

                # Debug: Print the final prediction
                app.logger.debug(f"Final Prediction: {prediction['prediction']}")

        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            prediction['prediction'] = f"Error: {str(e)}. Please enter valid numeric values for all inputs."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


