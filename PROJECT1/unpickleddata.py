#!/usr/bin/python


import pickle
import tensorflow
import numpy as np
from keras import Sequential
from keras.layers import Dropout, Flatten, MaxPooling2D, Dense, Conv2D
import tensorflow.keras.models
import tensorflow.keras.layers
#from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
with open('mazes1.pkl', 'rb') as f:
    my_new_list1 = pickle.load(f)

with open('paths1.pkl', 'rb') as f:
    my_new_list2 = pickle.load(f)
with open('pathshortest.pkl', 'rb') as f:
    my_new_list3 = pickle.load(f)

#print(len(my_new_list2))
#print(*my_new_list2, sep="\n")
#print(len(my_new_list1))
#print(*my_new_list1, sep="\n")
#print(len(my_new_list3))
#print(*my_new_list3, sep="\n")

x =np.asarray(my_new_list1)
x1 =x.reshape(200,121)
x = x.reshape(200,11,11,1)

y = np.asarray(my_new_list3)
y = y.reshape(200,121)
y1=y
#print(y)
#y = np.unit8(my_new_list2)

'''def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

x_val, x_t = np.split(trim_dataset(x,10),2)
y_val, y_t = np.split(trim_dataset(y,10),2)'''

length_train=round(len(x)*0.75)
length_t=len(x)-length_train
length_test=round(length_t/2)
length_validate=length_t-length_test

X_train_nn =x[0:length_train, :]
X_train_nn1 =x1[0:length_train, :]
Y_train_nn =y[0:length_train]
Y_train_nn1 =y1[0:length_train]
X_test_nn =x[length_train:length_train+length_test, :]
X_test_nn1 =x1[length_train:length_train+length_test, :]
Y_test_nn =y[length_train:length_train+length_test]
Y_test_nn1 =y1[length_train:length_train+length_test]
X_val_nn =x[length_test+length_train:length_validate+length_test+length_train, :]
X_val_nn1 =x1[length_test+length_train:length_validate+length_test+length_train, :]
Y_val_nn =y[length_test+length_train:length_validate+length_test+length_train]
Y_val_nn1 =y1[length_test+length_train:length_validate+length_test+length_train]

#When drop out is added after the 1st convolution and the dropout =0.5 :Accuracy 71.83
#drop out of 0.25 : Accuracy 71.83
#drop out of 0.75 :Accuracy 72.40
#drop out of 0.65 :Accuracy 72.40
#drop out of 0.95 :Accuracy 71.83
#Adding extra convolution layer doesnt improve accuracy
#The accuracy depends on the loss function
#Above mentioned are for binary cross entropy
#using mean_squared error - accuracy = 100%
#using mean_absolute_error - accuracy =100%
#mean_absolute_percentage_error -accuracy =100%
#mean_squared_logarithmic_error -accuracy = 100%
#squared_hinge =0
#hinge_loss =0
#categorical_hinge =0
#huber_loss=100
#categorical_cross_entropy =100
#sparse_categorical_crossentropy (throws an error)
#kullback_leibler_divergence =0
#poisson =1

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(11,11,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.65))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
'''
#Adding extra convolution layers
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
#tensorflow.keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
'''
model.add(Flatten())
model.add(Dense(242, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(121, input_shape=(11, 11), activation='relu'))
#model.add(Dense(121, activation='relu'))
#model.add(Dense(121, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='poisson', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_nn, Y_train_nn, epochs=150, batch_size=2, verbose=1, validation_data=(X_val_nn, Y_val_nn))
#print(X_val_nn)
#print(len(X_val_nn))
#print(len(Y_val_nn))
#score = model.evaluate(X_test_nn, Y_test_nn, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
_, accuracy = model.evaluate(X_test_nn, Y_test_nn)
print('Accuracy: %.2f' % (accuracy*100))
'''
#print('Test accuracy:', score)
model_json = model.to_json()
with open("cnn_model.json", "w") as json_file:
	json_file.write(model_json)
# saving weights to HDF5
model.save_weights("cnn_model.h5")
print("Saved cnn_model to disk")

model1 = Sequential()
model1.add(Dense(121, input_dim=121*1, activation='relu'))
model1.add(Dense(242, activation='relu'))
model1.add(Dense(121, activation='relu'))
# compile the keras model
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.fit(x1, y1, epochs=150)
_, accuracy1 = model1.evaluate(X_test_nn1, Y_test_nn1)
print('Accuracy: %.2f' % (accuracy1*100))

model_json1 = model1.to_json()
with open("nn_model.json", "w") as json_file:
	json_file.write(model_json)
# saving weights to HDF5
model.save_weights("nn_model.h5")
print("Saved nn_model to disk")

'''
print('Accuracy of cnn: %.2f' % (accuracy*100))
print('Accuracy of ann: %.2f' % (accuracy1*100))