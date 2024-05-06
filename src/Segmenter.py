"""
    Segmenter class to seperate steps from marker data using the fca (Maiwald et al. 2009)
    Also normalizes time and amplitude
    
    Example usage:
        segmenter = MarkerSegmenter(data)
        steps = segmenter.seperate_steps(touchdowns, cleanout=True, normalize_amplitude=True, normalize_time=False)
             
    """


import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

class MarkerSegmenter:
    def __init__(self, marker):
        self.marker = marker
        if not isinstance(marker, dict):
            self.heel_z = marker['LHEE Z']
            self.toe_z = marker['LTOE Z']
        
    def fca(self):
        TDapprox_windows = self._get_TDapprox_windows()
        heel_acceleration = self.heel_z.diff().diff()
        toe_acceleration = self.toe_z.diff().diff()
        touchdowns = []
        for key, window in TDapprox_windows.items():
            if 'heel' in key:
                touchdown = self._determine_td(heel_acceleration, window)
            elif 'toe' in key: 
                touchdown = self._determine_td(toe_acceleration, window)
            else:
                raise Exception('Something went wrong - might wanna check Segmenter class')
            touchdowns.append(touchdown)
        return touchdowns
                               
    def _determine_td(self, acceleration, window):         
        acc_window = acceleration[window[0]:window[-1]]
        touchdown = np.argmax(acc_window) + window[0]
        return touchdown
                           
    def _get_TDapprox_windows(self):
        if np.mean(self.heel_z) < 0:
            heel_peaks, _ = find_peaks(-self.heel_z, prominence = 50, distance = 40)
            toe_peaks, _ = find_peaks(-self.toe_z, prominence = 30, distance = 80)  
        else:
            heel_peaks, _ = find_peaks(-self.heel_z, prominence = 50, distance = 40)
            toe_peaks, _ = find_peaks(-self.toe_z, prominence = 30, distance = 80) 
        heel_peaks, toe_peaks = self._synchronise(heel_peaks, toe_peaks)
        if len(heel_peaks) != len(toe_peaks):
            print(f"Found different number of peaks in the signal (heel_peaks = {len(heel_peaks)} and toe_peaks = {len(toe_peaks)}). Trimming the longer array.")
            min_length = min(len(heel_peaks), len(toe_peaks))
            heel_peaks = heel_peaks[:min_length]
            toe_peaks = toe_peaks[:min_length]
            
        min_values = np.minimum(heel_peaks, toe_peaks)
        window_dict = {}
        for i in range(len(heel_peaks)):
            if heel_peaks[i] < toe_peaks[i]:
                key = 'heel_' + str(i+1)
            else:
                key = 'toe_' + str(i+1)
            window_dict[key] = None
        
        index = 0       
        for key, _ in window_dict.items():     
            window = np.asarray(np.arange(min_values[index]-5, min_values[index]+10))
            window_dict[key] = window
            index += 1
        return window_dict
                
    def _synchronise(self, heel_peaks, toe_peaks):
        if heel_peaks[0] > toe_peaks[0]: 
            if abs((heel_peaks[0] - toe_peaks[0])) > abs((heel_peaks[0] - toe_peaks[1])):
                toe_peaks = toe_peaks[1:]         
        else:
            if abs((toe_peaks[0] - heel_peaks[0])) > abs((toe_peaks[0] - heel_peaks[1])):
                heel_peaks = heel_peaks[1:]
        return heel_peaks, toe_peaks 
       
    def seperate_steps(self, touchdowns, cleanout=True, normalize_amplitude=True, normalize_time=True):
        step_dict = {}
        index = 0
        for i, j in zip([0] + touchdowns, touchdowns + [None]):
            step = {}
            for col in self.marker.columns:
                segment = self.marker[col].iloc[i:j]
                step[col] = segment
            index += 1
            key = 'Step_' + str(index)             
            step_dict[key] = pd.DataFrame(step)
        if cleanout:
            del step_dict[next(iter(step_dict))]
            del step_dict[next(reversed(step_dict))]
        if normalize_amplitude:
            step_dict = self._normalize_amplitude(step_dict)
        if normalize_time:
            step_dict = self._normalize_time(step_dict)
        return step_dict
    
    def _normalize_time(self, step_dict):
        normalized_steps = {}
        for key, step in step_dict.items():
            interp_functions = [interp1d(np.linspace(0, 1, len(step[dim])), step[dim], kind = 'cubic', fill_value = 'extrapolate') for dim in step.columns]
            new_time = np.linspace(0, 1, 100)
            normalized_steps[key] = pd.DataFrame({dim: f(new_time) for dim, f in zip(step.columns, interp_functions)})
        return normalized_steps
           
    def _normalize_amplitude(self, step_dict):
        normalized_steps = {}
        for step, data in step_dict.items():
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
            normalized_steps[step] = normalized_df
    
        return normalized_steps

