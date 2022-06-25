import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree, datasets


iris = datasets.load_iris()
X, y = datasets.load_iris(return_X_y=True)
X_train = X[(range(0,150,2)),:]
y_train = y[(range(0,150,2))]
X_test = X[(range(1,150,2)),:]
y_test = y[(range(1,150,2))]

clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)

print("\nPredictions:\n")
print(clf_pred)
print("\nAccuracy:", metrics.accuracy_score(y_test, clf_pred, normalize=True))
print("\nClassification report :\n", metrics.classification_report(y_test, clf_pred))
print("\n Confusion matrix :\n", metrics.confusion_matrix(y_test, clf_pred))


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)

tree.plot_tree(clf, feature_names=iris.feature_names,
               class_names=iris.target_names, filled=True)
fig.savefig('Iris.png')

"""

Predictions:

[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 2 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 1 2 2 2 2 2 2 2
 1]

Accuracy: 0.9466666666666667

Classification report :
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        25
           1       0.89      0.96      0.92        25
           2       0.96      0.88      0.92        25

    accuracy                           0.95        75
   macro avg       0.95      0.95      0.95        75
weighted avg       0.95      0.95      0.95        75


 Confusion matrix :
 [[25  0  0]
 [ 0 24  1]
 [ 0  3 22]]

"""
