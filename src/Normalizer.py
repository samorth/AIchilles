"""
    Normalizer class to transform coordinates into the hip center.
        
    Returns:
        centered marker data
"""


import pandas as pd

class Normalizer:
    def center_body(marker):      
        hip_marker_columns = ['LASI X', 'LASI Y', 'LASI Z',
                              'RASI X', 'RASI Y', 'RASI Z', 
                              'LPSI X', 'LPSI Y', 'LPSI Z',
                              'RPSI X', 'RPSI Y', 'RPSI Z'] 
               
        hip_marker = marker[hip_marker_columns]

        hip_center_x = hip_marker[['LASI X', 'RASI X', 'LPSI X', 'RPSI X']].mean(axis=1)
        hip_center_y = hip_marker[['LASI Y', 'RASI Y', 'LPSI Y', 'RPSI Y']].mean(axis=1)
        hip_center_z = hip_marker[['LASI Z', 'RASI Z', 'LPSI Z', 'RPSI Z']].mean(axis=1)


        offset = pd.DataFrame({
            'x': hip_center_x,
            'y': hip_center_y,
            'z': hip_center_z
            })
                
        centered_columns = {}
        for col in marker.columns:
            if 'X' in col:
                centered_columns[col] = marker[col] - offset['x']
            elif 'Y' in col:
                centered_columns[col] = marker[col] - offset['y']
            elif 'Z' in col:
                centered_columns[col] = marker[col] - offset['z']
            else:
                centered_columns[col] = marker[col]        
        centered_marker = pd.DataFrame(centered_columns)
        
        return centered_marker
            
                      
