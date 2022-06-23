from sklearn import datasets, metrics
from sklearn.naive_bayes import GaussianNB

X, y = datasets.load_iris(return_X_y=True)
X_train = X[(range(0,150,2)),:]
y_train = y[(range(0,150,2))]
X_test = X[(range(1,150,2)),:]
y_test = y[(range(1,150,2))]

print("-------------------------- Gaussian --------------------------")
clf = GaussianNB()
clf.fit(X_train,y_train)
params = clf.get_params(deep = True)
print(params)
predictions = clf.predict(X_test)
print(predictions)

print("Accuracy : ", metrics.accuracy_score(y_test, predictions, normalize = True))

print("Report : \n", metrics.classification_report(y_test, predictions))
print("Confusion Matrix : \n", metrics.confusion_matrix(y_test, predictions))

"""

-------------------------- Gaussian --------------------------
{'priors': None, 'var_smoothing': 1e-09}
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 2 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 1 2 2 2 2 2 2 2
 2]
Accuracy :  0.96
Report : 
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


