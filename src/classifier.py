#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sktime.classification.interval_based import CanonicalIntervalForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, train_test_split

path = 'C:/Users/hartmann/Desktop/AIchilles/GaIF/'

with open(path + 'X_subjectwise_balanced.pkl', "rb") as file:
    X = pickle.load(file)
    
with open(path + 'y_subjectwise_balanced.pkl', "rb") as file:
    y = pickle.load(file)

reports_cif = {}
clf_cif = {}
accs_cif = {}
#loo = LeaveOneOut()
ids = list(X.keys())
for train_index, test_index in loo.split(ids):
    X_train = {ids[i]: list(X.values())[i] for i in train_index}
    X_train = np.vstack([X_train[key] for key in X_train])
    X_test = X[ids[test_index[0]]]
    
    y_train = {ids[i]: list(y.values())[i] for i in train_index}
    y_train = np.concatenate([np.array(y_subject) for y_subject in y_train.values()], axis=0).ravel()
    y_test = np.array(y[ids[test_index[0]]])

    classifier = CanonicalIntervalForest(
        n_estimators=200, 
        att_subsample_size=8, 
        n_jobs=-1,
        base_estimator='CIT'
    )
    
    classifier.fit(X_train, y_train)
    
    y_pred= classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accs_cif[ids[test_index[0]]] = accuracy
    print(f"Accuracy for leaving out trial {ids[test_index[0]]} is {accuracy}")
    
    reports_cif[ids[test_index[0]]] = classification_report(y_test, y_pred)
    clf_cif[ids[test_index[0]]] = classifier




#%%
import joblib

#joblib.dump(reports_cif, 'reports_March21')
#joblib.dump(accs_cif, 'accs_March21')
joblib.dump(clf_cif, 'models_March21.pkl')
# %%
