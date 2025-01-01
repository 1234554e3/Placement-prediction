from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    """Handle predictions."""
    try:
        # Get form data
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))

        # Make a prediction
        prediction = model.predict(np.array([cgpa, iq]).reshape(1, -1))[0]

        # Map result to a readable output
        if prediction == 1:
            return jsonify({'placement': 'Placed'})
        else:
            return jsonify({'placement': 'Not Placed'})
    except Exception as e:
        return jsonify({'placement': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
