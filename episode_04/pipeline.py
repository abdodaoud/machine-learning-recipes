from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from sklearn import tree # replaced with KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Collect Training Data

iris = datasets.load_iris()

X = iris.data  # features
y = iris.target  # labels

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.5)  # splits the data by 50% to be used for testing

# Step 2: Train Classifier

# my_classifier = tree.DecisionTreeClassifier() #replaced with KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

# Step 3: Make Predictions

predictions = my_classifier.predict(X_test)

# Step 4: Check Accuracy

print(accuracy_score(y_test, predictions))
