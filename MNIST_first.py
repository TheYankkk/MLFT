from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import urllib
train_image="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_label="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
test_image="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_label="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"



x_train, x_test, y_train, y_test = train_test_split(mnist['data'], mnist['target'], random_state=0)
clf=SVC(kernel ='rbf')
clf.fit(x_train,y_train)
pred=cross_val_predict(clf, x_train,y_train, cv=3)

precision = precision_score(y_test,pred)
recall = recall_score(y_test, pred)
print('Random Forest Accuracy Score: %f' %accuracy_score(y_test,pred))
print('Random Forest Precision: %f' %precision)
print('Random Forest Recall: %f' %recall)



