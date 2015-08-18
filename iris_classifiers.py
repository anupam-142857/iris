import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv("iris_data.txt", header = None)


# Size of the dataset
print df.shape


# Print the histogram of all the columns
df.plot(kind='hist')
plt.show()


# Shuffle data to split into train and test
np.random.seed(1)
new_df = df.iloc[np.random.permutation(len(df))]

x_train = new_df.iloc[:105,[0,1,2,3]]
y_train = new_df.iloc[:105,4]
x_test = new_df.iloc[105:,[0,1,2,3]]
y_test = new_df.iloc[105:,4]


# Build SVM classifier
svm_clf = svm.SVC(kernel = 'linear')
svm_clf.fit(x_train, y_train)
svm_predictions = svm_clf.predict(x_test)


# Build Logistic Regression classifier
logit_clf = LogisticRegression()
logit_clf.fit(x_train, y_train)
logit_predictions = logit_clf.predict(x_test)


# Plot confusion matrix
cm_svm = confusion_matrix(y_test, svm_predictions)
cm_logit = confusion_matrix(y_test, logit_predictions)
print cm_svm
print cm_logit
