import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import cv2
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#build the network
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),#CNN
                    activation="relu",
                    input_shape=(28,28,1)))
                    # kernalsize = 3*3 并没有改变数据维度
model.add(Conv2D(16,kernel_size=(3,3),
                    activation="relu"
                    ))
model.add(MaxPooling2D(pool_size=(2,2)))
                    # 进行数据降维操作
model.add(Flatten())#Flatten层用来将输入“压平”，即把多维的输入一维化，
                    #常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Dense(32,activation="relu"))
                    #全连接层
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',
                optimizer='Adadelta',
                metrics=['accuracy'])
train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")
train_images  /= 255
test_images /= 255
from keras.utils import to_categorical

train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels,10)
model.fit(train_images,
            train_labels,
            batch_size=32,
            epochs=5,
            verbose=1,
            validation_data=(test_images,test_labels),
            shuffle=True
            )

score = model.evaluate(test_images,test_labels,verbose=1)

print('test loss:',score[0])
print('test accuracy:',score[1])