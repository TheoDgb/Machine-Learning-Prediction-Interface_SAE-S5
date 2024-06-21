import numpy as np
import pandas as pd
# pip install imbalanced-learn
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from lightgbm import LGBMClassifier

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

# Save the table as HTML
df_combine_subset = df_combine.head(5)
html_table = df_combine_subset.to_html(classes='table table-striped', index=False)
html_file_path = 'static/images/energy/windTurbineFaults/df_combine_subset_table.html'
# Write the HTML table to a file
with open(html_file_path, 'w') as file:
    file.write(html_table)


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

# Imbalanced fault modes
plt.figure(figsize=(10, 5))
train_df.Fault.value_counts().plot.pie(title='Imbalanced Fault Modes')
plt.savefig('static/images/energy/windTurbineFaults/imbalanced_fault_modes_pie.png')