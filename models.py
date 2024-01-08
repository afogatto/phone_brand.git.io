import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib

data = pd.read_csv('dataset/phone_brand.csv')


X = data.drop(['id','brand', 'model'], axis = 1)
y = data['brand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X, y)
joblib.dump(rfc, "rfc.pkl")


tree = DecisionTreeClassifier() 
tree.fit(X, y)
joblib.dump(tree, "tree.pkl")