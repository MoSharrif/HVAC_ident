import numpy as np
import pandas as pd
import pickle
from pathlib import Path


# Functions
def load_synth_time_series_from_file(synthProfilesPath):
    arr = np.load(synthProfilesPath)
    df = pd.DataFrame(arr, index = dateRange)
    return df

def J_to_kWh(df):
    df = df/3.6e+6
    return df

def load_real_time_series_from_file(realProfilesPath):
    df = pd.read_csv(realProfilesPath, parse_dates = ['Date/Time'])
    df = df.set_index('Date/Time')
    return df


# Load the model from the file
with open('HVAC_ident_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file) 


# Select a run
### Synthetic data
runName = '2024_05_21_232700562'
synthProfilesPath = Path().absolute().parent / 'GAN' / 'runs' / runName / 'synth_profiles.npy'

### Real data
folder = '100 Renovated Building with PV and HP'
filename = 'electricityConsumptionrPVHPRen.csv'
realProfilesPath = Path().absolute().parent / 'GAN_data' / 'Time-series end use consumer profiles' / folder / filename


# Specify the date range of the imported data
drStart = '2021-01-01 00:15:00' #ts... date range
drEnd = '2022-01-01 00:00:00'
drFreq = '15T'  #15 minutes
dateRange = pd.date_range(start = drStart, end = drEnd, freq = drFreq)


# Load the time series from the file
### Synthetic data
#test_data = load_synth_time_series_from_file(synthProfilesPath) #synthetic data
#test_data = J_to_kWh(test_data)

### Real data
test_data = load_real_time_series_from_file(realProfilesPath)
test_data = J_to_kWh(test_data)


# Predict
Month_start = 4
Month_end = 12
predictions = []
for col in test_data:
    x = test_data[col].to_frame().resample('M').mean()/test_data[col].to_frame().resample('M').max()
    predictions.append(loaded_model.predict(x[Month_start:Month_end].transpose())[0])
for item in predictions:
    print(item)