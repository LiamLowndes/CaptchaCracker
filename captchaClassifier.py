#Author: Liam Lowndes
#Student Number: 101041818
#Date: April 15, 2023

#Base code taken from Assignment #4

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
import keras

pathToCsv = "./csv/captcha.csv"
pathToAns = "./answers/captcha-answers.txt"

#Make the captcha readable by the model by one hot encoding it
def oneHotEncode(captcha):
  chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

  capOneHot = []

  for char in captcha:
    charEncode = [0] * len(chars)
    charEncode[chars.index(char)] = 1
    capOneHot.append(charEncode)

  return(capOneHot)

#Get all the names of the files within the given folder path
def getNames(filePath):
  fileNames = []
  with open(filePath, 'r') as file:
    for line in file:
        fileNames.append(line.strip())

  return(fileNames)

print("Data Loading...")
data = pd.read_csv(pathToCsv, header=None)
print("Load Complete!")

#Shape and save data values
X = data.values
number_of_samples = X.shape[0]
height, width = 50, 200
X = X.reshape(number_of_samples, height, width, 1)

#Retrieve the answers to the captchas
captchaAnswers = getNames(pathToAns)

#Create a one hot encoded label out of each captcha answer
oneHotLabels = []
for captcha in captchaAnswers:
  oneHotLabels.append(oneHotEncode(captcha))
oneHotLabels = np.array(oneHotLabels)

#Split the dataset into trianing and validation
x_train, x_test, y_train, y_test = sk.train_test_split(X, oneHotLabels, test_size=0.2)

#Sequential neural netowrk model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(50, 200, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(5 * 62, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Reshape((5, 62))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.fit(x_train, np.array(y_train), epochs=100, validation_data=(x_test, np.array(y_test)), verbose=1, callbacks=[early_stopping])

training_performance = model.evaluate(x_test, y_test)
print(f"Training loss: {training_performance}")

model.save("captcha-cracker.h5")
