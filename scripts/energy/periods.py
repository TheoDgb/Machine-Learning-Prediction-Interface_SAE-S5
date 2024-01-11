import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class CONFIG:

    NAMES_DTYPES = {
        "Source" : str,
        "Production" : np.float32
    }

production_fr_df = pd.read_csv('scripts/energy/intermittent-renewables-production-france.csv', sep=',',
    index_col="Date and Hour",
    parse_dates=["Date and Hour", "Date"],
    infer_datetime_format=True,
    dtype=CONFIG.NAMES_DTYPES
    )



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_energy_data(production_df, source_name):
    print('Pour la source : '+source_name)
    
    # Créer un dataframe de la source en paramètre
    production_filtered_energy = production_df[production_df['Source'] == source_name].copy()

    nan_count = production_filtered_energy.isna().sum().sum()
    print(f'Nombre de lignes contenant des NaN : {nan_count}')
    zero_production_count = production_filtered_energy[production_filtered_energy['Production'] == 0].shape[0]
    print(f'Nombre de lignes avec production égale à 0 : {zero_production_count}')

    # Supprimer les lignes avec des NaN
    production_filtered_energy = production_filtered_energy.dropna(subset=['Production'])
    # Supprimer les lignes avec 0 de Production
    production_filtered_energy = production_filtered_energy[production_filtered_energy['Production'] != 0]

    nan_count = production_filtered_energy.isna().sum().sum()
    print(f'Nombre de lignes contenant des NaN après suppression : {nan_count}')
    zero_production_count = production_filtered_energy[production_filtered_energy['Production'] == 0].shape[0]
    print(f'Nombre de lignes avec production égale à 0 après suppression : {zero_production_count}')

    print(f'Nouvelle taille du DataFrame : {production_filtered_energy.shape}')

    # Réaliser un label encoding des jours et mois car ce sont des données catégoriques (object)
    # et il faut qu'elles soient numériques pour les algorithmes d'apprentissage
    label_encoder = LabelEncoder()
    production_filtered_energy['dayName'] = label_encoder.fit_transform(production_filtered_energy['dayName'])
    production_filtered_energy['monthName'] = label_encoder.fit_transform(production_filtered_energy['monthName'])
    
    # # Réaliser un one-hot des jours et mois car ce sont des données catégoriques (object)
    # production_filtered_energy = pd.get_dummies(production_filtered_energy, columns=['dayName', 'monthName'], drop_first=False)
    
    # Extraire l'année, le mois et le jour de la colonne Date
    production_filtered_energy['Year'] = production_filtered_energy['Date'].dt.year
    production_filtered_energy['Month'] = production_filtered_energy['Date'].dt.month
    production_filtered_energy['Day'] = production_filtered_energy['Date'].dt.day

    # Il y a des EndHour à 24:00:00 qui représente en réalité 00:00:00 du jour suivant
    # Remplacer l'horaire 24:00:00 par 00:00:00
    # REMARQUE : je me suis rendu compte que ca devait devenir 00:00:00 MAIS DU JOUR SUIVANT (pas corrigé)
    production_filtered_energy['EndHour'] = production_filtered_energy['EndHour'].replace('24:00:00', '00:00:00')

    # Convertir les colonnes StartHour et EndHour (objects) en datetime
    production_filtered_energy['StartHour'] = pd.to_datetime(production_filtered_energy['StartHour'], format='%H:%M:%S')
    production_filtered_energy['EndHour'] = pd.to_datetime(production_filtered_energy['EndHour'], format='%H:%M:%S')

    # Extraire l'heure à partir de datetime en nombre entier de 0 à 23
    production_filtered_energy['StartHour'] = production_filtered_energy['StartHour'].dt.hour
    production_filtered_energy['EndHour'] = production_filtered_energy['EndHour'].dt.hour
    
    # Commencer à partir du 2020-04-01 afin d'éviter le trou de données (peut être dû au COVID)
    production_filtered_energy = production_filtered_energy[production_filtered_energy['Date'] >= '2020-04-01']
    
    return production_filtered_energy

# Prétraitement des données pour l'énergie solaire
production_filtered_energy_solar = preprocess_energy_data(production_fr_df, 'Solar')

# Prétraitement des données pour l'énergie éolienne
production_filtered_energy_wind = preprocess_energy_data(production_fr_df, 'Wind')



import sys
import json
import joblib
# Récupérer les arguments passés au script
script_arguments_json = sys.argv[1]
script_arguments = json.loads(script_arguments_json)
# Extraire les informations nécessaires
source = script_arguments.get('source')
model = script_arguments.get('model')
num_periods = int(script_arguments.get('numPeriods'))
model_name = f'{model}_{source}'
loaded_model = joblib.load(f'joblib/joblib_energy/{model_name}.joblib')

def prep_prediction_future(production_filtered_energy, num_periods):
    last_date = production_filtered_energy.index[-1]

    # Générer les dates futures
    future_dates = pd.date_range(start=last_date, periods=num_periods+1, freq='H')[1:]

    # Créer un DataFrame avec les caractéristiques pour les prédictions futures
    X_future = pd.DataFrame({
        'dayOfYear': future_dates.dayofyear,
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day,
        'StartHour': future_dates.hour,
        'EndHour': future_dates.hour + 1
    }, index=future_dates)

    # Ajouter les colonnes 'dayName' et 'monthName' avec les mêmes valeurs que l'encodage de label
    label_encoder = LabelEncoder()
    X_future['dayName'] = label_encoder.fit_transform(future_dates.day_name())
    X_future['monthName'] = label_encoder.fit_transform(future_dates.month_name())

    return X_future

if source == 'solar':
    X_future_solar = prep_prediction_future(production_filtered_energy_solar, num_periods)
    # Faire des prédictions pour le nombre de périodes donné en paramètre
    y_future_pred_solar = loaded_model.predict(X_future_solar)
    print('Predictions for Solar Source:', y_future_pred_solar)

    # Afficher les prédictions futures sous forme de graphique
    plt.figure(figsize=(10, 6))
    production_filtered_energy_solar['Production'].plot(label='Production réelle')
#     plt.plot(X_test_solar.index, y_pred_solar_stacking, label='Prédictions de test', color='lightgreen')
    plt.plot(X_future_solar.index, y_future_pred_solar, label='Prédictions futures', color='blue')
    plt.xlabel('Date and Hour')
    plt.ylabel('Production (MWh)')
    plt.title('Production Solar en fonction de la Date et de l\'Heure ('+model+')')
    plt.legend()
#     plt.show()
    plt.savefig('static/images/energy/'+source+'_'+model+'.png')

elif source == 'wind':
    X_future_wind = prep_prediction_future(production_filtered_energy_wind, num_periods)
    # Faire des prédictions pour le nombre de périodes donné en paramètre
    y_future_pred_wind = loaded_model.predict(X_future_wind)
    print('Predictions for Wind Source:', y_future_pred_wind)

    # Afficher les prédictions futures sous forme de graphique
    plt.figure(figsize=(10, 6))
    production_filtered_energy_wind['Production'].plot(label='Production réelle')
    plt.plot(X_future_wind.index, y_future_pred_wind, label='Prédictions futures', color='blue')
    plt.xlabel('Date and Hour')
    plt.ylabel('Production (MWh)')
    plt.title('Production Wind en fonction de la Date et de l\'Heure ('+model+')')
    plt.legend()
#     plt.show()
    plt.savefig('static/images/energy/'+source+'_'+model+'.png')



# def display_pred_future(production_filtered_energy, X_test, y_pred, X_future, y_future_pred, source_name, model):
#
#     # Ajouter les prédictions futures au graphique
#     plt.figure(figsize=(10, 6))
#     production_filtered_energy['Production'].plot(label='Production réelle')
#     plt.plot(X_test.index, y_pred, label='Prédictions de test', color='lightgreen')
#     plt.plot(X_future.index, y_future_pred, label='Prédictions futures', color='blue')  # Ajout des prédictions futures
#     plt.xlabel('Date and Hour')
#     plt.ylabel('Production (MWh)')
#     plt.title('Production '+source_name+' en fonction de la Date et de l\'Heure (modèle '+model+')')
#     plt.legend()
#     plt.show()
#
# display_pred_future(production_filtered_energy_solar, X_test_solar, y_pred_solar_stacking, X_future_solar, y_future_pred_solar, 'Solar', 'Stacking ensemble')
# display_pred_future(production_filtered_energy_wind, X_test_wind, y_pred_wind_stacking, X_future_wind, y_future_pred_wind, 'Wind', 'Stacking ensemble')