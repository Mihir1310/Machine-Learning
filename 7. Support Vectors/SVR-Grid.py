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

param_grid = {"C": [0.1,0.01,0.001,1]}

grid = GridSearchCV(svm.LinearSVR(max_iter=1000), param_grid, refit=True)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

prediction = grid.predict(X_test)

print("MAE: ", metrics.mean_absolute_error(y_true=y_test, y_pred=prediction))
print("MSE: ", metrics.mean_squared_error(y_true=y_test, y_pred=prediction))