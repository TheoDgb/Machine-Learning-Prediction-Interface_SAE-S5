import pandas as pd
import joblib
import json

# Charger les modèles sauvegardés
fault_detection_model = joblib.load('joblib/joblib_energy/joblib_wind_turbine_fault/joblib_wind_turbine_fault_detection.joblib')
fault_diagnostic_model = joblib.load('joblib/joblib_energy/joblib_wind_turbine_fault/joblib_wind_turbine_fault_diagnostic.joblib')

# Charger les nouvelles données à prédire
data_to_predict_fault = pd.read_csv('scripts/energy/windTurbineFaults/predict_df.csv', sep=',')


# Utiliser le modèle de détection des anomalies pour prédire les anomalies
anomaly_predictions = fault_detection_model.predict(data_to_predict_fault)

# Ajouter les prédictions d'anomalie au DataFrame
data_to_predict_fault['Anomaly'] = anomaly_predictions

# Filtrer les données pour ne garder que les prédictions d'anomalies
anomalous_data_df = data_to_predict_fault[data_to_predict_fault['Anomaly'] == 1]


# Reprendre les mêmes données que celles utilisées pour l'entraînement du modèle de diagnostic
X_predict = anomalous_data_df.drop(columns=['Anomaly'])

# Prédictions du type d'anomalie
anomaly_type_predictions = fault_diagnostic_model.predict(X_predict)

# Ajouter les prédictions du type d'anomalie au DataFrame original
anomalous_data_df['Anomaly_Type'] = anomaly_type_predictions

# Afficher les données anormales avec leur type prédit
# print(anomalous_data_df[['Anomaly', 'Anomaly_Type']])

# Convertir les résultats en JSON
# result_json = anomalous_data_df[['Anomaly', 'Anomaly_Type']].to_json(orient='records')


# Réinitialiser l'index pour inclure le numéro de ligne
anomalous_data_df = anomalous_data_df.reset_index()

# Convertir les résultats en JSON
result_json = anomalous_data_df[['index', 'Anomaly', 'Anomaly_Type']].to_json(orient='records')


# Retourner les résultats sous forme de JSON
print(result_json)