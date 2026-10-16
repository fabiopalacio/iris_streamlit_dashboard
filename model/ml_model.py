"""
    Importations:
    pandas -> read_csv() and get_dummies()
    sklearn -> RandomForestClassifier() => machine learning algorithm chosen
    pickle -> dump() -> to save the model in a file
"""

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

# Reading the data
iris_data = datasets.load_iris()

# Making a dataframe of the data
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Create the Random Forest Classifier
clf = RandomForestClassifier()

# Fit the model
clf.fit(X, Y)

# Save the model
pickle.dump(clf, open("iris_rfc.pkl", "wb"))
