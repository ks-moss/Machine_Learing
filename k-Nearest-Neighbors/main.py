import pandas as pd
import numpy as np

# read in the data using pandas
df = pd.read_csv('k-Nearest-Neighbors/data/diabetes.csv')

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

# create a dataframe with all training data except the target column
X = df.drop(columns=['Outcome'])

# separate target values
y = df['Outcome'].values

# set a random seed for reproducibility
np.random.seed(1)

# get unique classes and their counts
unique_classes, class_counts = np.unique(y, return_counts=True)

# set the split ratio (e.g., 80% train, 20% test)
split_ratio = 0.8

# initialize variables to store train and test indices
train_indices = []
test_indices = []

# iterate over unique classes
for class_label in unique_classes:
    # get indices for each class
    class_indices = np.where(y == class_label)[0]

    # calculate the number of samples to include in the training set
    num_train_samples = int(len(class_indices) * split_ratio)

    # randomly shuffle the indices
    np.random.shuffle(class_indices)

# split the indices into train and test

# [:num_train_samples]: This slice includes elements from the beginning of the array up to, 
    # but not including, the element at index num_train_samples. It extracts the first 
    # num_train_samples elements of the array.
    train_indices.extend(class_indices[:num_train_samples])

# [num_train_samples:]: This slice includes elements starting from the element at index 
    # num_train_samples and goes until the end of the array. It extracts elements from index 
    # num_train_samples onward.

    test_indices.extend(class_indices[num_train_samples:])

# split the data
X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

print("\n\nSplit DATA:\nX.iloc[train_indices] ---> ", len(X_train), "   X.iloc[test_indices] ---> ", len(X_test))
print("\nSplit LABEL:\ny[train_indices] ---> ", len(y_train), "    y[test_indices]] ---> ", len(y_test))
print("\n\nTotal X (DATA): ", len(X_train) + len(X_test))
print("Total y (LABEL): ", len(y_train) + len(y_test))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNeighborsClassifierCustom:
    def __init__(self, n_neighbors=23):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):


        # Calculate distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get index of k-nearest training data points
        k_nearest_indices = np.argsort(distances)[:self.n_neighbors]

        # Get the labels of the k-nearest training data points
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # Return the most common class label among the k-nearest neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
    
    # Function to calculate confusion matrix
    def calculate_confusion_matrix(self, y_true, y_pred):
        classes = np.unique(np.concatenate((y_true, y_pred)))
        confusion_matrix = np.zeros((len(classes), len(classes)))

        for i in range(len(classes)):
            for j in range(len(classes)):
                confusion_matrix[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))

        return confusion_matrix
    
    # Function to calculate precision, recall, and F1 score
    def calculate_f1_score(self, conf_matrix):
        
        true_positives = conf_matrix[0][0] # np.diag(conf_matrix)
        false_positives = conf_matrix[0][1] #. np.sum(conf_matrix, axis=0) - true_positives
        false_negatives = conf_matrix[1][0] # np.sum(conf_matrix, axis=1) - true_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        # Avoid division by zero for precision and recall
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)

        f1_score = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1_score


# Usage:
# Create the custom KNN classifier
knn = KNeighborsClassifierCustom(n_neighbors=3)

# Fit the classifier to the data
knn.fit(X_train.values, y_train)

# Show first 5 model predictions on the test data
y_pred = knn.predict(X_test.values)
print("\n\nPredict of X_test:\n", y_pred)

# Check accuracy of the model on the test data
print("\nScore: ", knn.score(X_test.values, y_test))

# Calculate confusion matrix
conf_matrix = knn.calculate_confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Calculate precision, recall, and F1 score
precision, recall, f1_score = knn.calculate_f1_score(conf_matrix)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score, "\n\n")