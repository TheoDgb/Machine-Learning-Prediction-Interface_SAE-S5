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

# Plot time span of all data
time_scada = scada_df.DateTime
time_fault = fault_df.DateTime
time_status = status_df.DateTime

plt.figure(figsize=(10,4))
plt.plot(time_scada, np.full(len(scada_df), 1), label='scada data')
plt.plot(time_fault, np.full(len(fault_df), 2), label='fault data')
plt.plot(time_status, np.full(len(status_df), 3), label='status data')

plt.legend(loc='lower right')
plt.title('Time Span of SCADA, Fault, and Status Data')

plt.savefig('static/images/energy/windTurbineFaults/wind_turbine_plot_time_span.png')



# Plot the number of faults per month
fig, ax = plt.subplots(figsize=(10, 5))
c = ['red', 'orange', 'green', 'blue', 'violet']
fault_df.resample('M', on='DateTime').Fault.value_counts().unstack().plot.bar(
    stacked=True,
    width=0.8,
    ax=ax,
    color=c,
    rot=45,
    title='Wind Turbine Faults per Month',
    ylabel='Fault Counts')

plt.subplots_adjust(bottom=0.3)

fig.savefig('static/images/energy/windTurbineFaults/wind_turbine_plot_nb_fault_per_month.png')