"""
Marker EDA Class 
Class used for perfoming simple Data Exporation steps on marker data
Data input format: Pandas Dataframe (column names are equivalent to marker names)

Example Usage: 
  >>> eda = MarkerEDA(marker)
  >>> analysis_objects = ["RANK", "LASI"]
  >>> stats = eda.summary_statistics(analysis_objects)
  >>> eda.visualize_statistics(stats)
  >>> eda.plot_time_series(analysis_objects, start_time=0, end_time=1)
  >>> eda.plot_marker_3d_trajectory(analysis_objects)
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MarkerEDA:
    def __init__(self, marker):
        self.marker = marker

    def show_head(self, n=5):
        print(self.marker.head(n))

    def summary_statistics(self, marker_list=None):
        if marker_list is None:
            marker_list = self.marker
        elif type(marker_list) is not list:
            marker_list = [marker_list]
    
        stats_data = []

        for marker in marker_list:
            cols = [col for col in self.marker.columns if col.startswith(marker)]
            for col in cols:
                stats_series = pd.Series([
                    self.marker[col].mean(),
                    self.marker[col].std(),
                    self.marker[col].median(),
                    self.marker[col].max(),
                    self.marker[col].min()
                ], index=['mean', 'std', 'median', 'max', 'min'], name=col)
                stats_data.append(stats_series)

        stats = pd.concat(stats_data, axis=1)
        return stats

    def visualize_statistics(self, stats):
        stats.T.plot(kind='bar', subplots=True, layout=(5,1), figsize=(10, 15), legend=False)
        plt.tight_layout()
        plt.show()
            
    def plot_time_series(self, marker_list, start_time=None, end_time=None):
        if type(marker_list) is not list:
            marker_list = [marker_list]
        if start_time is not None and end_time is not None:
            data_to_plot = self.marker[(self.marker['Time'] >= start_time) & (self.marker['Time'] <= end_time)]
        else:
            data_to_plot = self.marker

        for marker in marker_list:
            cols = [col for col in self.marker.columns if col.startswith(marker)]
            if not cols:
                print(f"Marker {marker} not found.")
                continue

            fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
            titles = ['X Component', 'Y Component', 'Z Component', 'XY Plane']
            for i in range(3):
                axs[i].plot(data_to_plot['Time'], data_to_plot[cols[i]])
                axs[i].set_title(f"{marker} {titles[i]}")
                axs[i].grid(True)

            axs[3].plot(data_to_plot[cols[0]], data_to_plot[cols[1]])
            axs[3].set_title(f"{marker} {titles[3]}")
            axs[3].set_xlabel('X Position')
            axs[3].set_ylabel('Y Position')
            axs[3].grid(True)

            plt.tight_layout()
            plt.show()

    def plot_marker_3d_trajectory(self, marker_list):
        if type(marker_list) is not list:
            marker_list = [marker_list]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for marker_name in marker_list:
            cols = [col for col in self.marker.columns if col.startswith(marker_name)]
            if not cols:
                print(f"Marker {marker_name} not found.")
                continue

            ax.plot(self.marker[cols[0]], self.marker[cols[1]], self.marker[cols[2]], label=marker_name)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f"3D Trajectories of Markers: {', '.join(marker_list)}")
        ax.legend()
        plt.show()
        