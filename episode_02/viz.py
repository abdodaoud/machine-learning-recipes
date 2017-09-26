import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import pydotplus

# Step 1: Import dataset

iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Step 2: Train a classifier

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Step 3: Predict label for new flower.

print(test_target)  # expected outcome
print(clf.predict(test_data))  # model predicted outcome

# Step 4: Visualize the tree

# viz code
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
