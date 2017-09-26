# import random

from scipy.spatial import distance
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            # label = random.choice(self.y_train)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])  # initial best value
        best_index = 0
        for i in range(1, len(self.X_train)):  # iterate over each point to find the closest
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i  # sets the index of the shortest distance so we can obtain the label for it
        return self.y_train[best_index]


# Step 1: Collect Training Data

iris = datasets.load_iris()

X = iris.data  # features
y = iris.target  # labels

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.5)  # splits the data by 50% to be used for testing

# Step 2: Train Classifier

my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

# Step 3: Make Predictions

predictions = my_classifier.predict(X_test)

# Step 4: Check Accuracy

print(accuracy_score(y_test, predictions))
