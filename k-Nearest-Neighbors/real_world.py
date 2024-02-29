import pandas as pd
import numpy as np
from main import *

print("\n--------------------------------------------\n")

pregnancies = input("Pregnancies: ")
glucose = input("Glucose: ")
bloodPressure = input("Blood Pressure: ")
skinThickness = input("Skin Thickness: ")
insulin = input("Insulin: ")
bmi = input("BMI: ")
diabetesPedigreeFunction = input("Diabetes PedigreeFunction: ")
age = input("Age: ")

# Assuming inputData is initially defined as a list of lists
inputData = [[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]]

inputData = np.array(inputData).astype(np.float64)

X_train, X_test = X, inputData
y_train = y

# Fit the classifier to the data
knn.fit(X_train.values, y_train)


y_pred = knn.predict(X_test)
print("\n\nPredict: ", y_pred)

if(y_pred == 0):
    print("The patient is not Diabetic")
else:
    print("The patient is Diabetic")