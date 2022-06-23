import numpy as np
from sklearn import datasets,metrics
from numpy.linalg import inv, pinv, LinAlgError
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    X, y = datasets.load_boston(return_X_y=True)

    x_train_temp1 = X[0:400, :]
    x_train = np.ones((x_train_temp1.shape[0], x_train_temp1.shape[1] + 1))
    x_train[:, 1:] = x_train_temp1

    y_train = y[:400]
    x_test_temp1 = X[400:506, :]
    x_test = np.ones((x_test_temp1.shape[0], x_test_temp1.shape[1] + 1))
    x_test[:, 1:] = x_test_temp1

    y_test = y[400:506]

    scaler = StandardScaler()
    scaler.fit(x_train[:, 1:])
    x_train[:, 1:] = scaler.transform(x_train[:, 1:])
    x_test[:, 1:] = scaler.transform(x_test[:, 1:])

    theta = np.zeros(x_train.shape[1])
    try:
        XTXi = np.linalg.inv(np.dot(x_train.T, x_train))
    except np.linalg.LinAlgError:
        XTXi = np.linalg.pinv(np.dot(x_train.T, x_train))

    XTy = np.dot(x_train.T, y_train)
    theta = np.dot(XTXi, XTy)

    print("Theta : ",theta)

    prediction = np.dot(theta, x_test.T)

    print("MAE: ", metrics.mean_absolute_error(y_true=y_test, y_pred=prediction))
    print("MSE: ", metrics.mean_squared_error(y_true=y_test, y_pred=prediction))

"""
OUTPUT:

Theta :  [24.3345     -1.14370921  1.12191092  0.35913222  0.48497247 -1.7061696
  3.58169796  0.07554815 -2.8156326   3.05189603 -1.97502535 -1.7937352
 -0.05252128 -3.50239563]
MAE:  5.142232214465314
MSE:  37.893778599602236

"""