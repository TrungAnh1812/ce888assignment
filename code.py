## READING THE MNIST DATA FROM FILES##

import pickle
# Reading X
x_train = pickle.load(open ('x_train', 'rb'))
x1_train = pickle.load(open ('x1_train', 'rb'))
x2_train = pickle.load(open ('x2_train', 'rb'))
x3_train = pickle.load(open ('x3_train', 'rb'))
x_test = pickle.load(open ('x_test', 'rb'))
x1_test = pickle.load(open ('x1_test', 'rb'))
x2_test = pickle.load(open ('x2_test', 'rb'))
x3_test = pickle.load(open ('x3_test', 'rb'))

# Reading y
y_train = pickle.load(open ('y_train', 'rb'))
y_test = pickle.load(open ('y_test', 'rb'))

##CREATING THE FIRST INSTANCE FOR MNIST DATA##

from keras.datasets import mnist 
from keras.models import Model 
from keras.layers import Input, Dense 
from keras.utils import np_utils 
import numpy as np

num_train = 60000 #The number of observations in the training data
num_test = 10000 #The number of observations in the testing data

height, width = 28, 28 # MNIST images are 28x28
num_classes = 10 # there are 10 classes

#Train the first instance with the original data set
(X_train, y_train), (X_test, y_test) = (x_train, y_train), (x_test, y_test)

X_train = X_train.reshape(num_train, height * width)
X_test = X_test.reshape(num_test, height * width)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1]
X_test /= 255 # Normalise data to [0, 1]

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

input_img = Input(shape=(height * width,))

x = Dense(height * width, activation='relu')(input_img)

encoded1 = Dense(height * width//2, activation='relu')(x)
encoded2 = Dense(height * width//8, activation='relu')(encoded1)

y = Dense(height * width//256, activation='relu')(encoded2)

decoded2 = Dense(height * width//8, activation='relu')(y)
decoded1 = Dense(height * width//2, activation='relu')(decoded2)

z = Dense(height * width, activation='sigmoid')(decoded1)
autoencoder = Model(input_img, z)

encoder = Model(input_img, y)

autoencoder1.compile(optimizer='adadelta', loss='mse')

autoencoder1.fit(X_train, X_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

out2 = Dense(num_classes, activation='softmax')(encoder.output)
classifier1 = Model(encoder.input,out2)

classifier1.compile(loss='categorical_crossentropy',
          optimizer='adam', 
          metrics=['accuracy']) 
classifier1.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

scores = classifier1.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1]) #The accuracy: 0.9692


#Train the first instance with the second data set while validating with the first data set

(X_train, y_train), (X_test, y_test) = (x1_train, y_train), (x_test, y_test)

X_train = X_train.reshape(num_train, height * width)
X_test = X_test.reshape(num_test, height * width)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

autoencoder1.fit(X_train, X_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

classifier1.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

scores = classifier1.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1]) #The accuracy: 0.9802

#Train the first instance with the third data set while validating with the first data set

(X_train, y_train), (X_test, y_test) = (x2_train, y_train), (x_test, y_test)

X_train = X_train.reshape(num_train, height * width)
X_test = X_test.reshape(num_test, height * width)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

autoencoder1.fit(X_train, X_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

classifier1.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

scores = classifier1.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1]) #The accuracy: 0.9776

#Train the first instance with the forth data set while validating with the first data set

(X_train, y_train), (X_test, y_test) = (x3_train, y_train), (x_test, y_test)

X_train = X_train.reshape(num_train, height * width)
X_test = X_test.reshape(num_test, height * width)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

autoencoder1.fit(X_train, X_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

classifier1.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

scores = classifier1.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1]) #The accuracy: 0.9783


## READING THE CIFAR-10 DATA FROM FILES##

# Reading X
cifar_x_train = pickle.load(open ('cifar_x_train', 'rb'))
cifar_x1_train = pickle.load(open ('cifar_x1_train', 'rb'))
cifar_x2_train = pickle.load(open ('cifar_x2_train', 'rb'))
cifar_x3_train = pickle.load(open ('cifar_x3_train', 'rb'))
cifar_x_test = pickle.load(open ('cifar_x_test', 'rb'))
cifar_x1_test = pickle.load(open ('cifar_x1_test', 'rb'))
cifar_x2_test = pickle.load(open ('cifar_x2_test', 'rb'))
cifar_x3_test = pickle.load(open ('cifar_x3_test', 'rb'))

# Reading y
cifar_y_train = pickle.load(open ('cifar_y_train', 'rb'))
cifar_y_test = pickle.load(open ('cifar_y_test', 'rb'))

#Reduce the size of the training and testing sets
cifar_x_train = cifar_x_train[:500,]
cifar_x1_train = cifar_x1_train[:500,]
cifar_x2_train = cifar_x2_train[:500,]
cifar_x3_train = cifar_x3_train[:500,]
cifar_x_test = cifar_x_test[:100,]
cifar_x1_test = cifar_x1_test[:100,]
cifar_x2_test = cifar_x2_test[:100,]
cifar_x3_test = cifar_x3_test[:100,]
cifar_y_train = cifar_y_train[:500,]
cifar_y_test = cifar_y_test[:100,]


##CREATING THE FIRST INSTANCE FOR CIFAR-10 DATA##

cifar_num_train = 500
cifar_num_test = 100

height, width, depth = 32, 32, 3 # CIFAR10 images are 32x32x3
num_classes = 10 # there are 10 classes (1 per image)

#Train the first instance with the original data set
(X_train, y_train), (X_test, y_test) = (cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test)

X_train = X_train.reshape(cifar_num_train, height * width * depth)
X_test = X_test.reshape(cifar_num_test, height * width * depth)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(cifar_y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(cifar_y_test, num_classes) # One-hot encode the labels

input_img = Input(shape=(height * width * depth,))

x = Dense(height * width * depth, activation='relu')(input_img)

encoded1 = Dense(height * width * depth//2, activation='relu')(x)
encoded2 = Dense(height * width * depth//8, activation='relu')(encoded1)

y = Dense(height * width * depth//256, activation='relu')(encoded2)

decoded2 = Dense(height * width * depth//8, activation='relu')(y)
decoded1 = Dense(height * width * depth//2, activation='relu')(decoded2)

z = Dense(height * width * depth, activation='sigmoid')(decoded1)
autoencoder = Model(input_img, z)

encoder = Model(input_img, y)

autoencoder2.compile(optimizer='adadelta', loss='mse')

autoencoder2.fit(X_train, X_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

out2 = Dense(num_classes, activation='softmax')(encoder.output)
classifier2 = Model(encoder.input,out2)


classifier2.compile(loss='categorical_crossentropy',
          optimizer='adam', 
          metrics=['accuracy']) 

classifier2.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

scores = classifier2.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1]) #The accuracy: 0.89


#Train the first instance with the second data set while testing on the first data set
(X_train, y_train), (X_test, y_test) = (cifar_x1_train, cifar_y_train), (cifar_x_test, cifar_y_test)

X_train = X_train.reshape(cifar_num_train, height * width * depth)
X_test = X_test.reshape(cifar_num_test, height * width * depth)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

autoencoder2.fit(X_train, X_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

classifier2.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

scores = classifier2.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1]) #The accuracy: 0.915

#Train the first instance with the third data set while testing on the first data set
(X_train, y_train), (X_test, y_test) = (cifar_x2_train, cifar_y_train), (cifar_x1_test, cifar_y_test)

X_train = X_train.reshape(cifar_num_train, height * width * depth)
X_test = X_test.reshape(cifar_num_test, height * width * depth)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

autoencoder2.fit(X_train, X_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

classifier2.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

scores = classifier2.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1]) #The accuracy: 0.9407

#Train the first instance with the forth data set while testing on the first data set
(X_train, y_train), (X_test, y_test) = (cifar_x3_train, cifar_y_train), (cifar_x1_test, cifar_y_test)

X_train = X_train.reshape(cifar_num_train, height * width * depth)
X_test = X_test.reshape(cifar_num_test, height * width * depth)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

autoencoder2.fit(X_train, X_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, X_test))

classifier2.fit(X_train, Y_train,
      epochs=10,
      batch_size=128,
      shuffle=True,
      validation_data=(X_test, Y_test))

scores = classifier2.evaluate(X_test, Y_test, verbose=1) 
print("Accuracy: ", scores[1]) #The accuracy: 0.9218


