import numpy as np
import pandas as pd
import joblib
# pip install imbalanced-learn
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Load scada_data.csv, status_data.csv, and fault_data.csv

# scada_df = pd.read_csv('kaggle/input/iiot-data-of-wind-turbine/scada_data.csv')
# status_df = pd.read_csv('kaggle/input/iiot-data-of-wind-turbine/status_data.csv')
# fault_df = pd.read_csv('kaggle/input/iiot-data-of-wind-turbine/fault_data.csv')

scada_df = pd.read_csv('scripts/energy/windTurbineFaults/scada_data.csv', sep=',')
scada_df['DateTime'] = pd.to_datetime(scada_df['DateTime'])

status_df = pd.read_csv('scripts/energy/windTurbineFaults/status_data.csv', sep=',')
status_df['Time'] = pd.to_datetime(status_df['Time'])
status_df.rename(columns={'Time': 'DateTime'}, inplace=True)

fault_df = pd.read_csv('scripts/energy/windTurbineFaults/fault_data.csv', sep=',')
fault_df['DateTime'] = pd.to_datetime(fault_df['DateTime'])


# Combine scada and fault data and keep all rows
df_combine = scada_df.merge(fault_df, on='Time', how='outer')


# There are lots of NaNs, or unmatched SCADA timestamps with fault timestamps, simply because there are no faults happen at certain time. For these NaNs, we will replace with "NF".
# NF is No Fault (normal condition)
# Replace records that has no fault label (NaN) as 'NF' (no fault)
df_combine['Fault'] = df_combine['Fault'].replace(np.nan, 'NF')


# To have a balanced dataset, we will pick 300 samples of each fault mode data
# Pick 300 samples of NF (No Fault) mode data
df_nf = df_combine[df_combine.Fault=='NF'].sample(300, random_state=42)

# With fault mode data
df_f = df_combine[df_combine.Fault!='NF']

# Combine no fault and faulty dataframes
df_combine = pd.concat((df_nf, df_f), axis=0).reset_index(drop=True)


# Drop irrelevant features
train_df = df_combine.drop(columns=['DateTime_x', 'Time', 'Error', 'WEC: ava. windspeed',
                                    'WEC: ava. available P from wind',
                                    'WEC: ava. available P technical reasons',
                                    'WEC: ava. Available P force majeure reasons',
                                    'WEC: ava. Available P force external reasons',
                                    'WEC: max. windspeed', 'WEC: min. windspeed',
                                    'WEC: Operating Hours', 'WEC: Production kWh',
                                    'WEC: Production minutes', 'DateTime_y'])


# Convert 'Fault' column to binary for anomaly detection. NF = 0 Else 1
train_df['Anomaly'] = train_df['Fault'].apply(lambda x: 0 if x == 'NF' else 1)


def param_model_fault_detection(train_df):
    # Feature and target
    X = train_df.drop(columns=['Fault', 'Anomaly'])
    y = train_df['Anomaly']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# Paramètres pour le modèle de détection d'anomalies
X_train, X_test, y_train, y_test = param_model_fault_detection(train_df)


def train_and_evaluate_model_fault_detection(X_train, X_test, y_train, y_test):
    base_directory = "joblib/joblib_energy/joblib_wind_turbine_fault"

    # Make pipeline of SMOTE, scaling, and classifier
    # SMOTE: Suréchantillonne (oversampling) les classes minoritaires dans l'ensemble d'entraînement.
    pipe = make_pipeline(SMOTE(), StandardScaler(), LGBMClassifier(random_state=42))

    # Define multiple scoring metrics
    scoring = {
        'acc': 'accuracy', # précision globale
        'prec_macro': 'precision_macro', # moyenne de la précision pour chaque classe
        'rec_macro': 'recall_macro', # moyenne du recall pour chaque classe (recall : rapport entre le nombre d'exemples correctement prédits comme appartenant à une class donnée)
        'f1_macro': 'f1_macro' # moyenne du score F1 pour chaque classe
    }

    # Stratified K-Fold (Divise les données en 5 folds avec une proportion égale de chaque classe dans chaque fold)
    stratkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #Cross-Validation: évaluer la performance d'un modèle de manière plus robuste en divisant le jeu de données en plusieurs sous-ensembles (folds)
    cv_scores = cross_validate(pipe, X_train, y_train, cv=stratkfold, scoring=scoring)

    # Fit the pipeline on training data
    pipe.fit(X_train, y_train)

    # Predict anomalies on test set
    y_pred = pipe.predict(X_test)

    # Save the trained model to a file
    model_filename = f"{base_directory}/joblib_wind_turbine_fault_detection.joblib"
    joblib.dump(pipe, model_filename)
    print(f"Model saved to {model_filename}")

    return cv_scores

cv_scores = train_and_evaluate_model_fault_detection(X_train, X_test, y_train, y_test)

# loaded_model = joblib.load(model_filename)
# # Now you can use loaded_model to make predictions
# predictions = loaded_model.predict(X_test)


# Préparation des données pour le diagnostic des types d'anomalies
# Ne garder que les observations qui sont marquées comme anomalies (1)
anomalies_df = train_df[train_df['Anomaly'] == 1]


def param_model_fault_diagnostic(anomalies_df):
    # Features and target
    X = anomalies_df.drop(columns=['Fault', 'Anomaly'])
    y = anomalies_df['Fault']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# Paramètres pour le modèle de diagnostic des types d'anomalies
X_train, X_test, y_train, y_test = param_model_fault_diagnostic(anomalies_df)


def train_and_evaluate_model_fault_diagnostic(X_train, X_test, y_train, y_test):
    base_directory = "joblib/joblib_energy/joblib_wind_turbine_fault"

    # Model training
    fault_diag_model = RandomForestClassifier(random_state=42)
    fault_diag_model.fit(X_train, y_train)

    # Predict fault types on test set
    y_pred = fault_diag_model.predict(X_test)

    # Save the trained model to a file
    model_filename = f"{base_directory}/joblib_wind_turbine_fault_diagnostic.joblib"
    joblib.dump(fault_diag_model, model_filename)
    print(f"Model saved to {model_filename}")

train_and_evaluate_model_fault_diagnostic(X_train, X_test, y_train, y_test)

# loaded_model = joblib.load(model_filename)
# # Now you can use loaded_model to make predictions
# predictions = loaded_model.predict(X_test)