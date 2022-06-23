from sklearn import datasets, metrics,neighbors
from sklearn.model_selection import GridSearchCV


X, y = datasets.load_iris(return_X_y=True)
X_train = X[(range(0,150,2)),:]
y_train = y[(range(0,150,2))]
X_test = X[(range(1,150,2)),:]
y_test = y[(range(1,150,2))]

kc=neighbors.KNeighborsClassifier()

gs=GridSearchCV(kc,{'n_neighbors':[1,4,7,10,13],'weights':['uniform','distance']},cv=3)

gs.fit(X_train,y_train)

gs.best_params_

predictions = gs.predict(X_test)


print("-------------------------- KNN --------------------------")
print("\nPredictions:\n",predictions)
print("\nAccuracy:",metrics.accuracy_score(y_test, predictions, normalize=True))
print("\nClassification report :\n",metrics.classification_report(y_test, predictions))
print("\n Confusion matrix :\n",metrics.confusion_matrix(y_test, predictions))

"""

-------------------------- KNN --------------------------

Predictions:
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 2 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 1 2 2 2 1 2 2 1 2 2 2 2 2 2 2
 2]

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