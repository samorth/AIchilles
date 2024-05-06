"""
    Source code for plotting temporal importance curves

        
"""
#%%
import numpy as np
import pickle
from sktime.classification.interval_based import CanonicalIntervalForest
import matplotlib.pyplot as plt
from sktime.classification.plotting.temporal_importance_diagram import plot_cif
from sklearn.metrics import ConfusionMatrixDisplay
from sktime.transformations.panel import catch22
from sktime.utils.validation._dependencies import _check_soft_dependencies

path = 'C:/Users/hartmann/Desktop/AIchilles/GaIF/results/models/Model April05/'

with open(path + 'accs.pkl', "rb") as file:
    accs = pickle.load(file)

with open(path + 'reports.pkl', "rb") as file:
    reports = pickle.load(file)
    
with open(path + 'confusion.pkl', "rb") as file:
    confusion_matrices = pickle.load(file)
     
with open(path + 'clf.pkl', "rb") as file:
    	clf = pickle.load(file)
     
with open(path + 'folds.pkl', "rb") as file:
    	folds = pickle.load(file)

path = 'C:/Users/hartmann/Desktop/AIchilles/GaIF/'

with open(path + 'X_subjectwise_balanced.pkl', "rb") as file:
    X = pickle.load(file)


def plot_curves(curves, curve_names, top_curves_shown=None, plot_mean=True):
    
    import matplotlib.pyplot as plt

    top_curves_shown = len(curves) if top_curves_shown is None else top_curves_shown
    max_ig = [max(i) for i in curves]
    top = sorted(range(len(max_ig)), key=lambda i: max_ig[i], reverse=True)[
        :top_curves_shown
    ]

    top_curves = [curves[i] for i in top]
    top_names = [curve_names[i] for i in top]
    
    if plot_mean:
        mean_curve = list(np.mean(curves, axis=0))
        
    return mean_curve, top_curves, top_names


def plot_cif(cif, normalise_time_points=False, top_curves_shown=None, plot_mean=True):

    curves = cif._temporal_importance_curves(
        normalise_time_points=normalise_time_points
    )
    curves = curves.reshape((25 * cif.n_dims_, cif.series_length_))
    features = catch22.feature_names + ["Mean", "Standard Deviation", "Slope"]
    curve_names = []
    for feature in features:
        for i in range(cif.n_dims_):
            name = feature if cif.n_dims_ == 1 else feature + " Dim " + str(i)
            curve_names.append(name)
            
    mean_curve, top_curves, top_names = plot_curves(
                                curves,
                                curve_names,
                                top_curves_shown=top_curves_shown,
                                plot_mean=plot_mean,
                                )
    
    
    return mean_curve, top_curves, top_names



curves_complete = []
names_complete = []
for classifier in clf.values():
    mean_curve, top_curves, top_names = plot_cif(classifier, top_curves_shown=1, plot_mean=True)
    curves_complete.append(top_curves)
    names_complete.append(top_names)

curves = [curves[0] for curves in curves_complete]
names = [names[0] for names in names_complete]

name_dim = [str(name[-6:]) for name in names]
name_attribute = ['Attribut 6, ', 'Attribut 6, ', 'Attribut 6, ', 'Attribut 18, ', 'Attribut 6, ']
for i in range(0, len(curves)):
    plt.plot(
        curves[i],
        label=str(name_attribute[i]) + str(name_dim[i])
    )

a= 'a'
title = f'Wichtigkeit W{a}(t)'
plt.title(title, fontweight='bold', loc='center', pad=20)
         
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),  # Adjusted position to be under the plot
    ncol=2,
    borderaxespad=0.0,
)

plt.xlabel("Zeitreihenpunkte")
plt.ylabel("IG")

plt.xlim(0, 100)


mean_curves = []
for classifier in clf.values():
    mean_curve, _ , _ = plot_cif(classifier, top_curves_shown=1, plot_mean=True)
    mean_curves.append(mean_curve)

for i in range(0, len(mean_curves)):
    plt.plot(
        mean_curves[i],
        label='Falte' + str(i+1)
    )

a= 'a'
title = f'Mittlere Wichtigkeiten W(t)'
plt.title(title, fontweight='bold', loc='center', pad=20)
         
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),  # Adjusted position to be under the plot
    ncol=2,
    borderaxespad=0.0,
)

plt.xlabel("Zeitreihenpunkte")
plt.ylabel("IG")

plt.xlim(0, 99)
plt.xticks(np.arange(0,99,10))
plt.grid(True)
