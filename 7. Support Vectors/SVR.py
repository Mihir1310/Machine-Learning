import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


X, y = datasets.load_boston(return_X_y=True)

X_train_temp1 = X[0:400, :]
X_train = np.ones((X_train_temp1.shape[0], X_train_temp1.shape[1] + 1))
X_train[:, 1:] = X_train_temp1
y_train = y[:400]
X_test_temp1 = X[400:506, :]
X_test = np.ones((X_test_temp1.shape[0], X_test_temp1.shape[1] + 1))
X_test[:, 1:] = X_test_temp1
y_test = y[400:506]

scaler = StandardScaler()
scaler.fit(X_train[:, 1:])
X_train[:, 1:] = scaler.transform(X_train[:, 1:])
X_test[:, 1:] = scaler.transform(X_test[:, 1:])

clf1 = svm.LinearSVR(max_iter=1000, C=0.1)  # default = 1000
clf1.fit(X_train, y_train)

prediction = clf1.predict(X_test)

print("MAE: ", metrics.mean_absolute_error(y_true=y_test, y_pred=prediction))
print("MSE: ", metrics.mean_squared_error(y_true=y_test, y_pred=prediction))

"""
Grid Search For SVR:

param_grid = {"C": [25, 50, 100, 1000],
              "gamma": [0.001, 0.0001, 0.00001], 
              "kernel": ["rbf"]}

grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)

grid.fit(X_train, y_train)

print(grid.best_params_)
//gamma = 0.00001 and C = 50
print(grid.best_estimator_)

prediction = grid.predict(X_test)

print("MAE: ", metrics.mean_absolute_error(y_true=y_test, y_pred=prediction))
print("MSE: ", metrics.mean_squared_error(y_true=y_test, y_pred=prediction))
"""


"""

MAE:  3.197004730519652
MSE:  20.10695477659058

"""