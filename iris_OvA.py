# One Vs All classifier on Iris Dataset

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix 


# Read the data
df = pd.read_csv("iris_data.txt", header = None)


# Shuffle data to split into train and test
np.random.seed(1)
new_df = df.iloc[np.random.permutation(len(df))]

x_train = new_df.iloc[:105,[0,1,2,3]]
y_train = new_df.iloc[:105,4]
x_test = new_df.iloc[105:,[0,1,2,3]]
y_test = new_df.iloc[105:,4]


# One Vs All Classifier
OvA_clf = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(x_train, y_train)
OvA_predictions = OvA_clf.predict(x_test)


# Confusion Matrix
cm_OvA = confusion_matrix(y_test, OvA_predictions)
print cm_OvA