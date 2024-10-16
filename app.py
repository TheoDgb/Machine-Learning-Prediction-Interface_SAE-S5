from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import subprocess
import json
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')



# ==================== SOLARFLARE ====================

@app.route('/solarflare')
def solarflare_home():
    return render_template('solarflare.html')

@app.route('/solarflare/predict', methods=['POST'])
def predictsolar():
    try:
        model_name = request.json.get('model')

        # Load the model based on the selected model name
        if model_name == 'gradientboost':
            model = joblib.load('joblib/joblib_solarflare/gb_model.joblib')
        elif model_name == 'randomforest':
            model = joblib.load('joblib/joblib_solarflare/rf_reg_model.joblib')
        else:
            return jsonify({'error': 'Invalid model selection'}), 400

        # Extract features from the data
        data = request.json
        features = [
            data['Year'], data['Month'], data['Day'],
            data['Hour'], data['Minute'], data['duration_s'],
            data['total_counts'], data['x_pos_asec'], data['y_pos_asec'],
            data['radial'], data['active_region_ar']
        ]

        # Convert features to numpy array (assuming your model expects a numpy array)
        features = np.array(features).reshape(1, -1)

        # Make predictions with the loaded model
        prediction = model.predict(features)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f'Error in predictsolar route: {str(e)}')
        print(f'Type of exception: {type(e)}')
        print(f'Exception args: {e.args}')
        return jsonify({'error': 'Internal Server Error'}), 500

# ==================== HEART ======================= 

@app.route('/heart')
def heart_home():
    return render_template('heart.html')

@app.route('/heart/predict', methods=['POST'])
def predictheart():
    try:
        model_name = request.json.get('model')
        if model_name == 'knn':
            model = joblib.load('joblib/joblib_heart/knn_classifier.joblib')
        elif model_name == 'dt':
            model = joblib.load('joblib/joblib_heart/best_dt_classifier.joblib')
        elif model_name == 'stacking':
            model = joblib.load('joblib/joblib_heart/stacking_classifier.joblib')
        elif model_name == 'voting':
            model = joblib.load('joblib/joblib_heart/voting_classifier.joblib')
        else:
            return jsonify({'error': 'Invalid model selection'}), 400
        
        data = request.json
        features = [
            data['age'],
            data['sex'],
            data['chest_pain_type'],
            data['resting_bp'],
            data['cholesterol'],
            data['fasting_bs'],
            data['resting_ecg'],
            data['max_hr'],
            data['exercise_angina'],
            data['oldpeak'],
            data['st_slope'],
        ]
        features = [float(i) for i in features]
        features = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print(f'Error in predictheart route: {str(e)}')
        print(f'Type of exception: {type(e)}')
        print(f'Exception args: {e.args}')
        return jsonify({'error': 'Internal Server Error'}), 500

# ==================== WEATHER ====================
class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

@app.route('/weather')
def weather_home():
    return render_template('weather.html')

@app.route('/weather/predict', methods=['POST'])
def predictweather():
    try:
        model_name = request.json.get('model')

        print(f"Model name received: {model_name}")

        if model_name == 'modele_classification_random_forest':
            model_path = 'joblib/joblib_weather/classification_random_forest.joblib'
        elif model_name == 'modele_random_forest':
            model_path = 'joblib/joblib_weather/Random_Forest.joblib'
        elif model_name == 'modele_regression_lineaire':
            model_path = 'joblib/joblib_weather/regression_lineaire.joblib'
        elif model_name == 'modele_stacked_regressor':
            model_path = 'joblib/joblib_weather/stacked_regressor.joblib'
        elif model_name == 'modele_ridge':
            model_path = 'joblib/joblib_weather/ridge_model.joblib'
        else:
            return jsonify({'error': 'Invalid model selection'}), 400

        print(f"Loading model from path: {model_path}")

        if not os.path.isfile(model_path):
            print(f"Model file not found: {model_path}")
            return jsonify({'error': 'Model file not found'}), 404

        model = joblib.load(model_path)
        print("Model loaded successfully")

        data = request.json
        features = [
            data['Year'], data['Month'], data['Day'],
            data['Humidity'], data['Pression']
        ]

        print(f"Features received: {features}")

        features = np.array(features).reshape(1, -1)
        print(f"Features reshaped: {features}")

        prediction = model.predict(features)
        print(f"Prediction made: {prediction}")

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f'Error in predictweather route: {str(e)}')
        print(f'Type of exception: {type(e)}')
        print(f'Exception args: {e.args}')
        return jsonify({'error': 'Internal Server Error'}), 500



# ==================== ENERGY ====================
# -------------------- ENERGY PRODUCTION --------------------
@app.route('/energy')
def energy_home():
    if not os.path.exists('static/images/energy/solar_wind_presentation_graph.png'):
        subprocess.run(['python3', 'scripts/energy/solarwindpresentation.py'])

    # Charger les fichiers CSV
    solar_data = pd.read_csv('scripts/energy/res_models_solar.csv')
    wind_data = pd.read_csv('scripts/energy/res_models_wind.csv')

    return render_template('energy.html', solar_data=solar_data, wind_data=wind_data)

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

        source = data.get('source')
        model = data.get('model')
        model_name = f'{model}_{source}'

        # Load the model
        loaded_model = joblib.load(f'joblib/joblib_energy/{model_name}.joblib')

        # Make predictions with the loaded model
        prediction = loaded_model.predict([features])[0]

        # Return the prediction in JSON format
        return jsonify({'prediction': prediction})

    except Exception as e:
        # Handle errors here
        print('Error:', e)
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/energy/predict-future-by-periods', methods=['POST'])
def predict_future():
    try:
        data = request.json

        source = data.get('source')
        model = data.get('model')
        num_periods = int(data.get('numPeriods'))
        print(f'num_periods: {num_periods}')
        print(f'source: {source}')
        print(f'model: {model}')
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

# -------------------- WIND TURBINE FAULTS --------------------
@app.route('/wind-turbine')
def wind_turbine_home():
    if not os.path.exists('static/images/energy/windTurbineFaults/wind_turbine_plot_time_span.png') or not os.path.exists('static/images/energy/windTurbineFaults/wind_turbine_plot_nb_fault_per_month.png'):
        subprocess.run(['python3', 'scripts/energy/windTurbineFaults/wind_turbine_analysis.py'])

    subprocess.run(['python3', 'scripts/energy/windTurbineFaults/wind_turbine_prepare_data.py'])

    # Charger les fichiers CSV
    solar_data = pd.read_csv('scripts/energy/res_models_solar.csv')
    wind_data = pd.read_csv('scripts/energy/res_models_wind.csv')

    html_table = open('static/images/energy/windTurbineFaults/df_combine_subset_table.html', 'r').read()

    return render_template('windturbine.html', solar_data=solar_data, wind_data=wind_data, html_table=html_table)

@app.route('/load_wind_turbine_fault_models')
def load_wind_turbine_faults_models():
    try:
        subprocess.run(['python3', 'scripts/energy/windTurbineFaults/wind_turbine_fault_models.py'])
        return "Script exécuté avec succès"
    except Exception as e:
        print(f"Erreur lors de l'exécution du script : {e}")
        return "Erreur lors de l'exécution du script"

@app.route('/predict_wind_turbine_fault')
def predict_wind_turbine_fault():
    try:
        result = subprocess.run(['python3', 'scripts/energy/windTurbineFaults/wind_turbine_fault_prediction.py'], capture_output=True, text=True)
        output = result.stdout
        return jsonify(json.loads(output))
    except Exception as e:
        print(f"Erreur lors de l'exécution du script : {e}")
        return jsonify({"error": "Erreur lors de l'exécution du script"})



if __name__ == '__main__':
    # Run the Flask app
    app.run(port=3000, debug=True)
