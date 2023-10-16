#Author: Liam Lowndes
#Student Number: 101041818
#Date: April 15, 2023

import numpy as np
import pandas as pd
import keras

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
pathToCsv = "./csv/captcha-test-outside.csv"
pathToAns = "./answers/captcha-test-outside-answers.txt"

#Make the captcha readable by the model by one hot encoding it
def oneHotEncode(captcha):
  capOneHot = []

  for char in captcha:
    charEncode = [0] * len(CHARS)
    if str.isalpha(char):
       char = str.upper(char)
    charEncode[CHARS.index(char)] = 1
    capOneHot.append(charEncode)
  
  return(capOneHot)

#Decode predictions that are one hot encoded
def predDecode(pred):
    decode = []
    for p in pred:
        decode.append(CHARS[np.argmax(p)])

    return(decode)

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

#Load the model and make the predictions
loaded_model = keras.models.load_model("captcha-cracker.h5")
predictions = loaded_model.predict(X)

#Decode the predictions that are one hot encoded
pDecode = []
for pred in predictions:
   pDecode.append(''.join(predDecode(pred)))

#Output results
wrong = []
correct = 0
for i in range(len(pDecode)):
    if (pDecode[i] == captchaAnswers[i]):
        correct += 1
    else:
       wrong.append([pDecode[i], captchaAnswers[i]])


right = 0
for w in wrong:
   subw = list(w[0])
   subr = list(w[1])
   for i in range(len(subw)):
      if subw[i] == subr[i]:
         right += 1
right += ((len(captchaAnswers) - len(wrong)) * 5)

print(pDecode[:10])
print(captchaAnswers[:10])

print("-" * 100)

print(wrong[:10])

print("-" * 100)

print(str(correct) + " / " + str(len(captchaAnswers)))
print("Percent Correct (Complete): " + str((correct/len(captchaAnswers))*100) + "%")
print("Percent Correct (Characters): " + str((right/((len(captchaAnswers)*5)))*100) + "%")