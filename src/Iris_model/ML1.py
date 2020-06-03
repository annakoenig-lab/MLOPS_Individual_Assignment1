
#imports
import numpy as np
import pandas as pd
import sklearn

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
##from sklearn.metrics import accuracy_score



def train_model():
	# Load Iris dataset from the internet
	url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pd.read_csv(url, names=names)

	# Split-out validation dataset
	array = dataset.values
	X = array[:,0:4]
    	y = array[:,4]
    	X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


    	model = SVC(gamma='auto')

    	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    	cv_result = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    	return model, cv_result
