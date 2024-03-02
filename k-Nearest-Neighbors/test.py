import pandas as pd
import numpy as np

df = pd.read_csv('k-Nearest-Neighbors/data/survey_lung_cancer.csv')

print(df.iloc[:, 0])
df['GENDER'] = df['GENDER'].replace('M', 0)
df['GENDER'] = df['GENDER'].replace('F', 1)
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace('NO', 0)
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace('YES', 1)


# Save the modified DataFrame to a new CSV file
df.to_csv('k-Nearest-Neighbors/data/survey_lung_cancer_modified.csv', index=False)

print("Modified DataFrame:\n", df.head())
print("\nCSV file saved successfully.")


print(df.head())