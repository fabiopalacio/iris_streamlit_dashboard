"""
    Importations:
    pandas -> read_csv() and get_dummies()
    sklearn -> RandomForestClassifier() => machine learning algorithm chosen
    pickle -> dump() -> to save the model in a file
"""

import json

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

# Reading the data
iris_data = datasets.load_iris()

# Making a dataframe of the data
iris = datasets.load_iris()
X = iris.data
Y = iris.target

with open("./model/model_params.json") as f:
    params = json.load(f)

# Create the Random Forest Classifier
clf = RandomForestClassifier(
    **params,
    random_state=42
)

# Fit the model
clf.fit(X, Y)

# Save the model
pickle.dump(clf, open("./model/iris_rfc.pkl", "wb"))
