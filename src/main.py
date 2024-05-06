"""
    Main source code

    Returns:
        
 """

#%%
import numpy as np
import pandas as pd
from Reader import DataReader
from Segmenter import MarkerSegmenter
from Normalizer import Normalizer

def trim_trials(segmented_trials):
    negative_trials = [trial for trial in segmented_trials.values() if isinstance(trial, dict) and trial.get('Label') == 0]
    
    if negative_trials:
        min_num_steps = min(len(trial) for trial in negative_trials)
    else:
        min_num_steps = 0

    for key, value in segmented_trials.items():
        if isinstance(value, dict) and value.get('Label') == 0:
            segmented_trials[key] = {k: v for i, (k, v) in enumerate(value.items()) if i >= len(value) - min_num_steps}
            
    return segmented_trials

def package_subjectwise(segmented_trials):
    X = {}
    y = {}
    for trial, step_dict in segmented_trials.items():
        instances = []
        y_subject = []
        for step_number, data in step_dict.items():
            if step_number != 'Label':
                data = data.drop(columns = ['Time'])
                instances.append(data)
                y_subject.append(step_dict['Label'])
        
        n_instances = len(instances)
        n_dims = len(instances[0].columns)
        n_points = len(instances[0].index)

        X_subject = np.empty([n_instances, n_dims, n_points])
        
        for i, data in enumerate(instances):
            for j, dim in enumerate(data.columns):
                X_subject[i,j,:] = data[dim].values
        
        X[trial] = X_subject
        y[trial] = y_subject
        
    return X, y 
    
def check_nans(X):            
    nans_per_trial = {}
    for trial, data in X.items(): 
        for i in range(data.shape[0]):
            instance_data = data[i, :, :]
            num_nan = np.sum(np.isnan(instance_data))
            print(f"Instance {i+1}: Number of NaN values = {num_nan}")
        nans_per_trial[trial] = num_nan
        
def package_series(marker_data):
    marker_data = normalize_amplitude(marker_data)
    
    X = marker_data
    y = []
    
    for trial, data in marker_data.items():
        data = data.drop(columns = ['Time'])
        y.append(meta.loc[meta['ID'] == trial, 'Label'].values[0])
        print(y)
    return X, y

def normalize_amplitude(marker_data):
    normalized_marker = {}
    for trial, data in marker_data.items():
        normalized_data = []
        for col in data.columns:
            if col != 'Time':
                max_val = data[col].max()
                min_val = data[col].min()
                    
                normalized_dim = data[col] - min_val
                normalized_dim /= (max_val - min_val)
                    
                normalized_data.append(normalized_dim)
            else:
                normalized_data.append(data[col]) 
            normalized_df = pd.concat(normalized_data, axis=1)       
        normalized_df.columns = data.columns 
        normalized_marker[trial] = normalized_df
    
    return normalized_marker
        

path_to_meta = 'C:/Users/hartmann/Desktop/AIchilles/GaIF/conf/meta.xlsx'
path_to_whitelist = 'C:/Users/hartmann/Desktop/AIchilles/GaIF/conf/columns_to_keep.xlsx'
path_to_data = 'C:/Users/hartmann/Desktop/AIchilles/GaIF/data/'

whitelist = pd.read_excel(path_to_whitelist)
allowed_dims = list(whitelist.loc[0,:])
allowed_dims = [dim for dim in allowed_dims if 'Angle' not in dim or 'Time' in dim]

meta = pd.read_excel(path_to_meta)

reader = DataReader(path_to_data, allowed_dims=allowed_dims)
blocked_ids = []
       
marker_data = {}
marker_data_raw = {}
for index, trial in meta.iterrows():
    print(f"Currently processing: trial_id = {trial['ID']} \n")
    if trial['ID'] not in blocked_ids:
        marker = reader.read(trial_id=trial['ID'], origin=trial['Origin'], num_frames=37200, exclude_n_frames=0)
        marker_data[trial['ID']] = Normalizer.center_body(marker)
        marker_data_raw[trial['ID']] = marker
    else:
        print(f"trial_id = {trial['ID']} is blocked\n")
        

touchdowns_complete = {}  
segmented_trials = {}
for trial, data in marker_data.items():
    print(f"Processing trial {trial}:\n")

    segmenter = MarkerSegmenter(data)
    touchdowns = segmenter.fca()
    touchdowns_complete[trial] = touchdowns
    
    steps = segmenter.seperate_steps(touchdowns, cleanout=True, normalize_amplitude=True, normalize_time=False)
    
    steps['Label'] = meta.loc[meta['ID'] == trial, 'Label'].values[0]
    segmented_trials[trial] = steps
        

import matplotlib.pyplot as plt
trial = '5BqY7'
window = [0, 1000]
tds = [td for td in touchdowns_complete[trial] if td < window[-1]]

x = np.arange(window[-1]+1)/200
y1 = marker_data[trial].loc[window[0]:window[-1], 'LHEE Z']
y2 = marker_data[trial].loc[window[0]:window[-1], 'LTOE Z']

plt.figure(1)
plt.plot(x, y1, color='g', label = 'LHEE Z')
plt.plot(x, y2, color='b', label = 'LTOE Z')
for td in tds:
    plt.axvline(td/200 ,color='r', label='_nolegend')
plt.axvline(tds[-1]/200 ,color='r', label='Erstkontakt')
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude [mm]')
plt.xlim(min(x), max(x))
plt.title('Detektion der Erstkontakte', fontsize=12, fontweight='bold')
plt.legend(loc='upper right')
plt.show()



