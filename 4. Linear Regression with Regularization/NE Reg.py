#Normal Equation with Regularization

import numpy as np
from sklearn import metrics, datasets
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    X, y = datasets.load_boston(return_X_y=True)

    X_train_temp1 = X[0:400, :]
    X_train = np.ones((X_train_temp1.shape[0], X_train_temp1.shape[1] + 1))
    X_train[:, 1:] = X_train_temp1

    y_train = y[:400]
    X_test_temp1 = X[400:506, :]
    X_test = np.ones((X_test_temp1.shape[0], X_test_temp1.shape[1] + 1))
    X_test[:, 1:] = X_test_temp1

    y_test = y[400:506]


    m = X_train.shape[0]
    n = X_train.shape[1]

    scaler = StandardScaler()
    scaler.fit(X_train[:, 1:])
    X_train[:, 1:] = scaler.transform(X_train[:, 1:])
    X_test[:, 1:] = scaler.transform(X_test[:, 1:])

    theta = np.zeros(X_train.shape[1])

    temp_matrix = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(1, X_train.shape[1]):
        temp_matrix[i, i] = 300  # Here as we increase lambda errors get decreasing upto some extent

    XTXi = np.linalg.inv(np.dot(X_train.T, X_train) + temp_matrix)

    XTy = np.dot(X_train.T, y_train)
    theta = np.dot(XTXi, XTy)
    prediction = np.dot(theta, X_test.T)
    print("MAE: ", metrics.mean_absolute_error(y_true=y_test, y_pred=prediction))
    print("MSE: ", metrics.mean_squared_error(y_true=y_test, y_pred=prediction))

"""
Output:

MAE:  3.8935297339273682
MSE:  22.4346470265702

"""