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

# plt.figure(figsize=(10, 6))
#
# # Tracer la production éolienne en bleu et solaire en rouge
# production_fr_df[production_fr_df['Source'] == 'Wind']['Production'].plot(label='Wind Production', color='blue')
# production_fr_df[production_fr_df['Source'] == 'Solar']['Production'].plot(label='Solar Production', color='red')
#
# plt.xlabel('Date and Hour')
# plt.ylabel('Production (MWh)')
# plt.title('Production en fonction de la Date et de l\'Heure')
# plt.legend()
# plt.show()
# A ENREGISTRER ET AFFICHER SUR L INTERFACE



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
    
    production_filtered_energy.info()
    # dayName et monthName sont désormais des int
    
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
    print('\n\n\n')
    
    return production_filtered_energy

# Prétraitement des données pour l'énergie solaire
production_filtered_energy_solar = preprocess_energy_data(production_fr_df, 'Solar')

# Prétraitement des données pour l'énergie éolienne
production_filtered_energy_wind = preprocess_energy_data(production_fr_df, 'Wind')



def param_models(production_filtered_energy):
    # Caractéristiques utiles pour les modèles de prédictions
    features = ['dayOfYear',
                'Year', 'Month', 'Day',
                'StartHour', 'EndHour',
                'dayName', 'monthName']
    # Ce que je veux prédire : production d'énergie
    target = 'Production'

    X = production_filtered_energy[features]
    Y = production_filtered_energy[target]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Tri par l'index (remettre les dates en ordre pour les graphiques)
    X_test = X_test.sort_index()
    y_test = y_test.sort_index()
    
    return X_train, X_test, y_train, y_test

# Paramètres pour les modèles
X_train_solar, X_test_solar, y_train_solar, y_test_solar = param_models(production_filtered_energy_solar)
X_train_wind, X_test_wind, y_train_wind, y_test_wind = param_models(production_filtered_energy_wind)



import joblib
from sklearn.metrics import mean_squared_error, r2_score
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingRegressor, StackingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

def train_and_evaluate_models(X_train, X_test, y_train, y_test, source):

    # Entraînement du modèle LightGBM avec recherche bayesian (tuning)
    lgbm_model = LGBMRegressor(force_col_wise=True, verbose=0, random_state=42)
    param_space_lgbm = {
        'num_leaves': (100, 300),
        'n_estimators': (100, 400),
        'learning_rate': (0.001, 0.3),
        'subsample': (0.8, 1.0),
        'colsample_bytree': (0.8, 1.0),
        'reg_alpha': (0, 1.0),
        'reg_lambda': (0, 1.0)
    }
    bayes_lgbm = BayesSearchCV(estimator=lgbm_model, search_spaces=param_space_lgbm,
                               scoring='neg_mean_squared_error', cv=5, n_iter=5,
                               return_train_score=False, refit=True, random_state=42)
    bayes_lgbm.fit(X_train, y_train)

    # Entraînement du modèle CatBoost avec recherche bayesian (tuning)
    catboost_model = CatBoostRegressor(verbose=0, random_state=42)
    param_space_catboost = {
        'max_depth': (6, 12),
        'n_estimators': (100, 400),
        'learning_rate': (0.001, 0.3)
    }
    bayes_catboost = BayesSearchCV(estimator=catboost_model, search_spaces=param_space_catboost,
                                   scoring='neg_mean_squared_error', cv=5, n_iter=10,
                                   return_train_score=False, refit=True, random_state=42)
    bayes_catboost.fit(X_train, y_train)
    # Évaluer les prédictions avec RMSE

    # Entraînement du modèle XGBoost avec recherche bayesian (tuning)
    xgboost_model = XGBRegressor(random_state=42)
    param_space_xgboost = {
        'max_depth': (3, 12),
        'n_estimators': (100, 500),
        'learning_rate': (0.001, 0.3),
        'subsample': (0.8, 1.0),
        'colsample_bytree': (0.8, 1.0),
        'gamma': (0, 5)
    }
    bayes_xgboost = BayesSearchCV(estimator=xgboost_model, search_spaces=param_space_xgboost,
                                  scoring='neg_mean_squared_error', cv=5, n_iter=10,
                                  return_train_score=False, refit=True, random_state=42)
    bayes_xgboost.fit(X_train, y_train)

    # Entraînement du modèle RandomForest avec recherche bayesian (tuning)
    randomforest_model = RandomForestRegressor(random_state=42)
    param_space_randomforest = {
        'n_estimators': (50, 250),
        'max_depth': (5, 40),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
        'max_features': ('sqrt', 'log2')
    }
    bayes_randomforest = BayesSearchCV(estimator=randomforest_model, search_spaces=param_space_randomforest,
                                       scoring='neg_mean_squared_error', cv=5, n_iter=10,
                                       return_train_score=False, refit=True, random_state=42)
    bayes_randomforest.fit(X_train, y_train)

    # Entraînement du modèle ExtraTrees avec recherche bayesian (tuning)
    extratrees_model = ExtraTreesRegressor(random_state=42)
    param_space_extratrees = {
        'n_estimators': (50, 250),
        'max_depth': (5, 40),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
        'max_features': ('sqrt', 'log2')
    }
    bayes_extratrees = BayesSearchCV(estimator=extratrees_model, search_spaces=param_space_extratrees,
                                     scoring='neg_mean_squared_error', cv=5, n_iter=10,
                                     return_train_score=False, refit=True, random_state=42)
    bayes_extratrees.fit(X_train, y_train)

    # Entraînement du modèle DecisionTree avec recherche bayesian (tuning)
    decisiontree_model = DecisionTreeRegressor(random_state=42)
    param_space_decisiontree = {
        'max_depth': (5, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4)
    }
    bayes_decisiontree = BayesSearchCV(estimator=decisiontree_model, search_spaces=param_space_decisiontree,
                                       scoring='neg_mean_squared_error', cv=5, n_iter=10,
                                       return_train_score=False, refit=True, random_state=42)
    bayes_decisiontree.fit(X_train, y_train)



    # Faire des prédictions pour chaque modèle (sans voting, stacking, bagging)
    y_pred_lgbm = bayes_lgbm.best_estimator_.predict(X_test)
    y_pred_catboost = bayes_catboost.best_estimator_.predict(X_test)
    y_pred_xgboost = bayes_xgboost.best_estimator_.predict(X_test)
    y_pred_randomforest = bayes_randomforest.best_estimator_.predict(X_test)
    y_pred_extratrees = bayes_extratrees.best_estimator_.predict(X_test)
    y_pred_decisiontree = bayes_decisiontree.best_estimator_.predict(X_test)



    # Création des modèles de base pour le stacking
    base_models = [
        ('lgbm', bayes_lgbm.best_estimator_),
        ('catboost', bayes_catboost.best_estimator_),
        ('xgboost', bayes_xgboost.best_estimator_),
        ('randomforest', bayes_randomforest.best_estimator_),
        ('extratrees', bayes_extratrees.best_estimator_),
        ('decisiontree', bayes_decisiontree.best_estimator_)
    ]

    # Création du modèle de stacking
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

    # Entraînement du modèle de stacking
    stacking_model.fit(X_train, y_train)

    # Faire des prédictions avec le modèle de stacking
    y_pred_stacking = stacking_model.predict(X_test)



    # Création du modèle de vote
    voting_model = VotingRegressor(estimators=[
        ('lgbm', bayes_lgbm.best_estimator_),
        ('catboost', bayes_catboost.best_estimator_),
        ('xgboost', bayes_xgboost.best_estimator_),
        ('randomforest', bayes_randomforest.best_estimator_),
        ('extratrees', bayes_extratrees.best_estimator_),
        ('decisiontree', bayes_decisiontree.best_estimator_)
    ])

    # Entraînement du modèle de vote
    voting_model.fit(X_train, y_train)

    # Faire des prédictions avec le modèle de vote
    y_pred_voting = voting_model.predict(X_test)



    # Entraînement des modèles de bagging
    bagging_lgbm = BaggingRegressor(estimator=bayes_lgbm.best_estimator_)
    bagging_lgbm.fit(X_train, y_train)

    bagging_catboost = BaggingRegressor(estimator=bayes_catboost.best_estimator_)
    bagging_catboost.fit(X_train, y_train)

    bagging_xgboost = BaggingRegressor(estimator=bayes_xgboost.best_estimator_)
    bagging_xgboost.fit(X_train, y_train)

    bagging_randomforest = BaggingRegressor(estimator=bayes_randomforest.best_estimator_)
    bagging_randomforest.fit(X_train, y_train)

    bagging_extratrees = BaggingRegressor(estimator=bayes_extratrees.best_estimator_)
    bagging_extratrees.fit(X_train, y_train)

    bagging_decisiontree = BaggingRegressor(estimator=bayes_decisiontree.best_estimator_)
    bagging_decisiontree.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test avec chaque modèle Bagging
    y_pred_bagging_lgbm = bagging_lgbm.predict(X_test)
    y_pred_bagging_catboost = bagging_catboost.predict(X_test)
    y_pred_bagging_xgboost = bagging_xgboost.predict(X_test)
    y_pred_bagging_randomforest = bagging_randomforest.predict(X_test)
    y_pred_bagging_extratrees = bagging_extratrees.predict(X_test)
    y_pred_bagging_decisiontree = bagging_decisiontree.predict(X_test)



    # Création du modèle de vote avec les modèles stacking et bagging
    voting_stacking_bagging_model = VotingRegressor(estimators=[
        ('bagging_stacking', BaggingRegressor(estimator=stacking_model)),
        ('bagging_lgbm', BaggingRegressor(estimator=bayes_lgbm.best_estimator_)),
        ('bagging_catboost', BaggingRegressor(estimator=bayes_catboost.best_estimator_)),
        ('bagging_xgboost', BaggingRegressor(estimator=bayes_xgboost.best_estimator_)),
        ('bagging_randomforest', BaggingRegressor(estimator=bayes_randomforest.best_estimator_)),
        ('bagging_extratrees', BaggingRegressor(estimator=bayes_extratrees.best_estimator_)),
        ('bagging_decisiontree', BaggingRegressor(estimator=bayes_decisiontree.best_estimator_))
    ])

    # Entraînement du modèle de vote avec les modèles stacking et bagging
    voting_stacking_bagging_model.fit(X_train, y_train)

    # Faire des prédictions avec le modèle de vote avec les modèles stacking et bagging
    y_pred_voting_stacking_bagging_model = voting_stacking_bagging_model.predict(X_test)



    # Évaluer les prédictions avec RMSE
    rmse = mean_squared_error(y_test, y_pred_lgbm, squared=False)
    print(f'RMSE avec LightGBM : {rmse}')
    rmse = mean_squared_error(y_test, y_pred_catboost, squared=False)
    print(f'RMSE avec CatBoost : {rmse}')
    rmse = mean_squared_error(y_test, y_pred_xgboost, squared=False)
    print(f'RMSE avec XGBoost : {rmse}')
    rmse = mean_squared_error(y_test, y_pred_randomforest, squared=False)
    print(f'RMSE avec RandomForest : {rmse}')
    rmse = mean_squared_error(y_test, y_pred_extratrees, squared=False)
    print(f'RMSE avec ExtraTrees : {rmse}')
    rmse = mean_squared_error(y_test, y_pred_decisiontree, squared=False)
    print(f'RMSE avec DecisionTree : {rmse}')

    rmse = mean_squared_error(y_test, y_pred_voting, squared=False)
    print(f'RMSE avec le modèle de vote : {rmse}')

    rmse = mean_squared_error(y_test, y_pred_stacking, squared=False)
    print(f'RMSE avec le modèle de stacking : {rmse}')

    rmse = mean_squared_error(y_test, y_pred_bagging_lgbm, squared=False)
    print(f'RMSE avec le modèle de bagging LightGBM : {rmse}')
    rmse = mean_squared_error(y_test, y_pred_bagging_catboost, squared=False)
    print(f'RMSE avec le modèle de bagging CatBoost : {rmse}')
    mse = mean_squared_error(y_test, y_pred_bagging_xgboost, squared=False)
    print(f'RMSE avec le modèle de bagging XGBoost : {rmse}')
    mse = mean_squared_error(y_test, y_pred_bagging_randomforest, squared=False)
    print(f'RMSE avec le modèle de bagging RandomForest : {rmse}')
    mse = mean_squared_error(y_test, y_pred_bagging_decisiontree, squared=False)
    print(f'RMSE avec le modèle de bagging DecisionTree : {rmse}')

    rmse = mean_squared_error(y_test, y_pred_voting_stacking_bagging_model, squared=False)
    print(f'RMSE avec le modèle de vote / stacking / bagging : {rmse}')

    print('\n')

    # Calculer le R-squared
    r2 = r2_score(y_test, y_pred_lgbm)
    print(f'R-squared avec LightGBM: {r2}')
    r2 = r2_score(y_test, y_pred_catboost)
    print(f'R-squared avec CatBoost: {r2}')
    r2 = r2_score(y_test, y_pred_xgboost)
    print(f'R-squared avec XGBoost: {r2}')
    r2 = r2_score(y_test, y_pred_randomforest)
    print(f'R-squared avec RandomForest: {r2}')
    r2 = r2_score(y_test, y_pred_extratrees)
    print(f'R-squared avec ExtraTrees: {r2}')
    r2 = r2_score(y_test, y_pred_decisiontree)
    print(f'R-squared avec DecisionTree: {r2}')

    r2 = r2_score(y_test, y_pred_voting)
    print(f'R-squared avec le modèle de vote : {r2}')

    r2 = r2_score(y_test, y_pred_stacking)
    print(f'R-squared avec le modèle de stacking : {r2}')

    r2 = r2_score(y_test, y_pred_bagging_lgbm)
    print(f'R-squared avec le modèle de bagging LightGBM : {r2}')
    r2 = r2_score(y_test, y_pred_bagging_catboost)
    print(f'R-squared avec le modèle de bagging CatBoost : {r2}')
    r2 = r2_score(y_test, y_pred_bagging_xgboost)
    print(f'R-squared avec le modèle de bagging XGBoost : {r2}')
    r2 = r2_score(y_test, y_pred_bagging_randomforest)
    print(f'R-squared avec le modèle de bagging RandomForest : {r2}')
    r2 = r2_score(y_test, y_pred_bagging_extratrees)
    print(f'R-squared avec le modèle de bagging ExtraTrees : {r2}')
    r2 = r2_score(y_test, y_pred_bagging_decisiontree)
    print(f'R-squared avec le modèle de bagging DecisionTree : {r2}')

    r2 = r2_score(y_test, y_pred_voting_stacking_bagging_model)
    print(f'R-squared avec le modèle de vote / stacking / bagging : {r2}')

    print('\n')

    # Afficher les résultats sous forme de dataframe
    results = {
        'Model': ['LightGBM', 'CatBoost', 'XGBoost', 'RandomForest', 'ExtraTrees', 'DecisionTree',
                  'Voting', 'Stacking', 'Bagging_LightGBM', 'Bagging_CatBoost', 'Bagging_XGBoost',
                  'Bagging_RandomForest', 'Bagging_ExtraTrees', 'Bagging_DecisionTree', 'Voting_Stacking_Bagging'],
        'RMSE': [mean_squared_error(y_test, y_pred_lgbm, squared=False),
                 mean_squared_error(y_test, y_pred_catboost, squared=False),
                 mean_squared_error(y_test, y_pred_xgboost, squared=False),
                 mean_squared_error(y_test, y_pred_randomforest, squared=False),
                 mean_squared_error(y_test, y_pred_extratrees, squared=False),
                 mean_squared_error(y_test, y_pred_decisiontree, squared=False),
                 mean_squared_error(y_test, y_pred_voting, squared=False),
                 mean_squared_error(y_test, y_pred_stacking, squared=False),
                 mean_squared_error(y_test, y_pred_bagging_lgbm, squared=False),
                 mean_squared_error(y_test, y_pred_bagging_catboost, squared=False),
                 mean_squared_error(y_test, y_pred_bagging_xgboost, squared=False),
                 mean_squared_error(y_test, y_pred_bagging_randomforest, squared=False),
                 mean_squared_error(y_test, y_pred_bagging_extratrees, squared=False),
                 mean_squared_error(y_test, y_pred_bagging_decisiontree, squared=False),
                 mean_squared_error(y_test, y_pred_voting_stacking_bagging_model, squared=False)],
        'R-squared': [r2_score(y_test, y_pred_lgbm),
                      r2_score(y_test, y_pred_catboost),
                      r2_score(y_test, y_pred_xgboost),
                      r2_score(y_test, y_pred_randomforest),
                      r2_score(y_test, y_pred_extratrees),
                      r2_score(y_test, y_pred_decisiontree),
                      r2_score(y_test, y_pred_voting),
                      r2_score(y_test, y_pred_stacking),
                      r2_score(y_test, y_pred_bagging_lgbm),
                      r2_score(y_test, y_pred_bagging_catboost),
                      r2_score(y_test, y_pred_bagging_xgboost),
                      r2_score(y_test, y_pred_bagging_randomforest),
                      r2_score(y_test, y_pred_bagging_extratrees),
                      r2_score(y_test, y_pred_bagging_decisiontree),
                      r2_score(y_test, y_pred_voting_stacking_bagging_model)]
    }

    results_df = pd.DataFrame(results)
    print("Dataframe des résultats : ")
    print(results_df)

    # Sauvegarder chaque modèle individuellement
    base_directory = "joblib/joblib_energy"
    joblib.dump(bayes_lgbm.best_estimator_, f'{base_directory}/modele_lgbm_{source}.joblib')
    joblib.dump(bayes_catboost.best_estimator_, f'{base_directory}/modele_catboost_{source}.joblib')
    joblib.dump(bayes_xgboost.best_estimator_, f'{base_directory}/modele_xgboost_{source}.joblib')
    joblib.dump(bayes_randomforest.best_estimator_, f'{base_directory}/modele_randomforest_{source}.joblib')
    joblib.dump(bayes_extratrees.best_estimator_, f'{base_directory}/modele_extratrees_{source}.joblib')
    joblib.dump(bayes_decisiontree.best_estimator_, f'{base_directory}/modele_decisiontree_{source}.joblib')
    joblib.dump(voting_model, f'{base_directory}/modele_voting_{source}.joblib')
    joblib.dump(stacking_model, f'{base_directory}/modele_stacking_{source}.joblib')
    joblib.dump(bagging_lgbm, f'{base_directory}/modele_bagging_lgbm_{source}.joblib')
    joblib.dump(bagging_catboost, f'{base_directory}/modele_bagging_catboost_{source}.joblib')
    joblib.dump(bagging_xgboost, f'{base_directory}/modele_bagging_xgboost_{source}.joblib')
    joblib.dump(bagging_randomforest, f'{base_directory}/modele_bagging_randomforest_{source}.joblib')
    joblib.dump(bagging_extratrees, f'{base_directory}/modele_bagging_extratrees_{source}.joblib')
    joblib.dump(bagging_decisiontree, f'{base_directory}/modele_bagging_decisiontree_{source}.joblib')
    joblib.dump(voting_stacking_bagging_model, f'{base_directory}/modele_voting_stacking_bagging_{source}.joblib')

    return y_pred_stacking, stacking_model

y_pred_solar_stacking, stacking_model_solar = train_and_evaluate_models(X_train_solar, X_test_solar, y_train_solar, y_test_solar, "solar")

# y_pred_wind_stacking, stacking_model_wind = train_and_evaluate_models(X_train_wind, X_test_wind, y_train_wind, y_test_wind, "wind")


def prep_prediction_future(production_filtered_energy):
    last_date = production_filtered_energy.index[-1]

    # Générer les dates futures
    future_dates = pd.date_range(start=last_date, periods=20001, freq='H')[1:]

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

    # Afficher les premières lignes du DataFrame résultant
    print(X_future.head())
    print('\n')

    return X_future

X_future_solar = prep_prediction_future(production_filtered_energy_solar)
# X_future_wind = prep_prediction_future(production_filtered_energy_wind)

# Faire des prédictions pour les 2000 périodes suivantes
y_future_pred_solar = stacking_model_solar.predict(X_future_solar)
# y_future_pred_wind = stacking_model_wind.predict(X_future_wind)

def display_pred_future(production_filtered_energy, X_test, y_pred, X_future, y_future_pred, source_name, model):

    # Ajouter les prédictions futures au graphique
    plt.figure(figsize=(10, 6))
    production_filtered_energy['Production'].plot(label='Production réelle')
    plt.plot(X_test.index, y_pred, label='Prédictions de test', color='lightgreen')
    plt.plot(X_future.index, y_future_pred, label='Prédictions futures', color='blue')  # Ajout des prédictions futures
    plt.xlabel('Date and Hour')
    plt.ylabel('Production (MWh)')
    plt.title('Production '+source_name+' en fonction de la Date et de l\'Heure (modèle '+model+')')
    plt.legend()
    plt.show()

display_pred_future(production_filtered_energy_solar, X_test_solar, y_pred_solar_stacking, X_future_solar, y_future_pred_solar, 'Solar', 'Stacking ensemble')
# display_pred_future(production_filtered_energy_wind, X_test_wind, y_pred_wind_stacking, X_future_wind, y_future_pred_wind, 'Wind', 'Stacking ensemble')
