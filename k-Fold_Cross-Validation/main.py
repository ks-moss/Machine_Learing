import pandas as pd
import numpy as np

#read in the data using pandas
df = pd.read_csv('k-Fold_Cross-Validation/data/diabetes.csv')

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

#check data has been read in properly
print(df.head())

#check number of rows and columns in dataset
print(df.shape)

#create a dataframe with all training data except the target column
X = df.drop(columns=['Outcome'])
#check that the target variable has been removed
print(X.head())

#separate target values
y = df['Outcome'].values
#view target values
print(y[0:5])


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print("cv_scores mean: ", format(np.mean(cv_scores)))