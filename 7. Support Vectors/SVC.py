from sklearn import svm, datasets, metrics
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    X, y = datasets.load_iris(return_X_y=True)

    X_train = X[range(0, 150, 2), :]
    y_train = y[range(0, 150, 2)]

    X_test = X[range(1, 150, 2), :]
    y_test = y[range(1, 150, 2)]

    clf = svm.LinearSVC(max_iter=5000)  # default = 1000
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)

    print("Predicted Output : \n", prediction)
    print("\nAccuracy: ", (metrics.accuracy_score(prediction, y_test, normalize=True)) * 100)
    print("\nClassification report :\n", classification_report(y_test, prediction))
    print("\nConfusion Matrix :\n", confusion_matrix(y_test, prediction))

"""
For Multiple CLasses :

Linear SVC means one vs one
SVC means one vs all
"""

"""

Predicted Output : 
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 2 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 1 2 2 2 2 2 2 2
 2]

Accuracy:  96.0

Classification report :
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        25
           1       0.92      0.96      0.94        25
           2       0.96      0.92      0.94        25

    accuracy                           0.96        75
   macro avg       0.96      0.96      0.96        75
weighted avg       0.96      0.96      0.96        75


Confusion Matrix :
 [[25  0  0]
 [ 0 24  1]
 [ 0  2 23]]

"""