"""
    Reader class for reading maker data from csv file considering the given data structure.
    
    Example usage:
        reader = DataReader(path_to_data, allowed_dims=allowed_dims)
        marker = reader.read(trial_id=trial['ID'], origin=trial['Origin'], num_frames=None, exclude_n_frames=None)
"""

import numpy as np
import pandas as pd

class DataReader:
    def __init__(self, path_to_data, allowed_dims):
        self.path_to_data = path_to_data
        self.allowed_dims = allowed_dims
            
    def __get_marker(self, trial_id, origin, num_frames, exclude_n_frames):
        if exclude_n_frames is None:
            exclude_n_frames = 0
            
        path_to_trial = self.path_to_data + trial_id + '_run.csv'
        
        header = self.__create_header(path_to_trial, origin)
        
        total_frames = sum(1 for _ in open(path_to_trial))
        if num_frames is None:
            num_frames = total_frames  
        skip_frames = max(20, total_frames - (num_frames + exclude_n_frames))
        nrows = max(0, total_frames - skip_frames - exclude_n_frames)
        if skip_frames == 0:
            raise Warning(f"num_frames ({num_frames}) is greater than the number of frames in this trial ({total_frames})\n trial_id: {trial_id}\n)")
        print(f"skip_frames = {skip_frames}\n")
        
        marker = pd.read_csv(path_to_trial, delimiter=',', skiprows=lambda x: x < skip_frames, nrows=nrows, header=None)
        
        if marker.shape[1] == len(header):
            marker.columns = header
        else:
            raise ValueError(f"Column count in CSV (count={marker.shape[1]}) does not meet the requirements (header={len(header)})")
        
        trimmed_marker = marker[[col for col in self.allowed_dims if col in marker.columns]]
        
        return trimmed_marker 

    def __create_header(self, path_to_trial, origin):
        header_row = 5 if origin=='BFH' else 7
        df = pd.read_csv(path_to_trial, delimiter=',', skiprows=header_row, nrows=1, header=None)
        columns = df.loc[0,:].dropna().tolist()
        header = [f"{col} {component}" for col in columns for component in ["X", "Y", "Z"] if col != 'Time']
        header.insert(0, 'Time')
        return header

    def read(self, trial_id, origin, num_frames, exclude_n_frames):
        marker = self.__get_marker(trial_id, origin, num_frames, exclude_n_frames)
        return marker