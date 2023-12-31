Author: Liam Lowndes
Date: April 15, 2023

Video Demo: https://youtu.be/esf6RL5KTf0
Google Drive link to download the model yourself: https://drive.google.com/drive/folders/1lMM7q5PDZZpZmLHnfnhAtq0EQ1-Duqwg?usp=sharing

NOTE:   There are two test sets you can test the trained model on:
        1) captcha-test.csv and captcha-test-answers.txt  
         - this will test my model on captchas that it was trained on.
         - The expected success rate should be ~70% correct on complete guesses and ~91% on character guesses as explained in my report
         - to achieve this leave pathToCsv and pathToAns as it is in the file
        2) captcha-test-outside.csv and captcha-test-outside-answers.txt
        - This will return a result of 0% for complete guesses and ~13% for character guesses as explained in my report
        - To achieve this, change the pathToCsv and pathToAns in captchaTest to

        pathToCsv = "./csv/captcha-test-outside.csv"
        pathToAns = "./answers/captcha-test-outside-answers.txt"

In this folder you will find 2 folders and 4 additional files, there contents are:

answers: A folder that contains three text files which are the answer keys to their respective csv files
captcha-samples: A folder of some images to show an example of what my model was trained on

captchaClassifier.py: The file used to train the neural network model
captchaTest.py The file used to test the neural network model
Lowndes_Liam_Final.pdf: My report
README.txt: This file
