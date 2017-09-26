from sklearn import tree

# Step 1: Collect Training Data

features = [[140, 1], [130, 1], [150, 0], [170, 0]]  # 0 = bumpy, 1 = smooth
labels = [0, 0, 1, 1]  # 0 = apple, 1 = orange

# Step 2: Train Classifier

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Step 3: Make Predictions

print(clf.predict([[150, 0]]))
