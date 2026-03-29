from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load the trained model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model_pipeline.pkl')
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
else:
    pipeline = None

# Labels Mapping
LABEL_MAP = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from form
    text = request.form.get('text', '')
    
    if not pipeline:
        return render_template('index.html', error="Model not trained yet. Please run src/model.py first.", text=text)
        
    if not text.strip():
        return render_template('index.html', error="Please enter some news text.", text=text)
    
    # Predict probabilities
    try:
        probabilities = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        
        results = []
        for cls, prob in zip(classes, probabilities):
            results.append({
                'category': LABEL_MAP.get(cls, f'Category {cls}'),
                'probability': round(prob * 100, 2)
            })
            
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        return render_template('index.html', results=results, text=text)
    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}", text=text)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
