from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS
import subprocess
import json

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solarflare')
def solarflare_home():
    return render_template('solarflare.html')


@app.route('/weather')
def weather_home():
    return render_template('weather.html')


@app.route('/energy')
def energy_home():
    return render_template('energy.html')


@app.route('/load_energy_models')
def load_energy_models():
    try:
        subprocess.run(['python3', 'scripts/energy/models.py'])
        return "Script exécuté avec succès"
    except Exception as e:
        print(f"Erreur lors de l'exécution du script : {e}")
        return "Erreur lors de l'exécution du script"


@app.route('/energy/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()

        # Extract features from the data
        features = [
            data['dayOfYear'],
            data['Year'], data['Month'], data['Day'],
            data['StartHour'], data['EndHour'],
            data['dayName'], data['monthName']
        ]

        # Load the model
        model = joblib.load(f'joblib/joblib_energy/modele_voting_solar.joblib')

        # Make predictions with the loaded model
        prediction = model.predict([features])[0]

        # Return the prediction in JSON format
        return jsonify({'prediction': prediction})

    except Exception as e:
        # Handle errors here
        print('Error:', e)
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/energy/predict-future', methods=['POST'])
def predict_future():
    try:
        data = request.json

        source = data.get('source')
        model = data.get('model')
        num_periods = int(data.get('numPeriods'))

        model_name = f'{model}_{source}'
        loaded_model = joblib.load(f'joblib/joblib_energy/{model_name}.joblib')

        script_arguments = {
            'source': source,
            'model': model,
            'numPeriods': num_periods
        }
        # Convertir le dictionnaire en une chaîne JSON pour le passer en tant qu'argument
        script_arguments_json = json.dumps(script_arguments)

        # Exécute le script permettant de créer le graphique
        subprocess.run(['python3', 'scripts/energy/periods.py', script_arguments_json])

        return jsonify({'graph_path': 'static/images/energy/'+source+'_'+model+'.png'})

    except Exception as e:
        print('Error:', e)
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(port=3000, debug=True)