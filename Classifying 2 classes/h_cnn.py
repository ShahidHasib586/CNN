
#CNN in tumor detection
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initilizzing the CNN
classifier = Sequential()

#convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 1), activation= 'relu'))

#Pooling step (reducing the size of future map)-max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#flattening(huge single vector that will be a input of ANN)
classifier.add(Flatten())

#Full connection
classifier.add(Dense(units = 128, activation= 'relu'))
classifier.add(Dense(units = 1, activation= 'sigmoid'))

# compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('training_set',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


classifier.fit_generator(train_set,
                    steps_per_epoch=200,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=200)