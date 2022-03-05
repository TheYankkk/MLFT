from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#print(test_images.shape)

x_train=train_images.reshape((60000,784))
y_train=train_labels
x_test=test_images.reshape((10000,784))
y_test=test_labels


#x_train, x_test, y_train, y_test = train_test_split(mnist['data'], mnist['target'], random_state=0)
clf=SVC(kernel ='rbf')#using SVM model
clf.fit(x_train,y_train)
#pred=cross_val_predict(clf, x_train,y_train, cv=3)
pred=clf.predict(x_test)
print("predict size:",pred.shape)
print("test size:",y_test.shape)
#precision = precision_score(y_test,pred)
#recall = recall_score(y_test, pred)
print('SVM Accuracy Score: %f' %accuracy_score(y_test,pred))
#print('Random Forest Precision: %f' %precision)
#print('Random Forest Recall: %f' %recall)




