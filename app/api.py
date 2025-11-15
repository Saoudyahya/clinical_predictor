from flask import Flask, request, jsonify
from core.model import BaseModel
import numpy as np
import pickle

app = Flask(__name__)

# Variable globale pour le modèle
model = None
preprocessor = None


def load_trained_model(filepath: str):
    """Charge le modèle entraîné"""
    global model, preprocessor
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        preprocessor = data['preprocessor']


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        # Récupérer les données
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Prétraitement
        features = preprocessor.normalize(features, fit=False)

        # Prédiction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        diagnosis = "Infecté" if prediction == 1 else "Sain"

        return jsonify({
            'diagnosis': diagnosis,
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': float(abs(probability - 0.5) * 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Prédiction en batch"""
    try:
        data = request.get_json()
        features = np.array(data['features'])

        features = preprocessor.normalize(features, fit=False)
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                'diagnosis': "Infecté" if pred == 1 else "Sain",
                'prediction': int(pred),
                'probability': float(prob)
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Charger le modèle au démarrage
    load_trained_model('models/clinical_model.pkl')
    app.run(debug=True, host='0.0.0.0', port=5000)
