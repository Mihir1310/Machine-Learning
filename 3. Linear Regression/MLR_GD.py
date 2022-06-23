# Multiple Linear Regression Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, datasets
from sklearn.preprocessing import StandardScaler


def Plot_CF(theta1_Val, Cf):
    plt.plot(theta1_Val, Cf)
    plt.show()


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
    scaler.fit(X_train[:,1:])
    X_train[:,1:] = scaler.transform(X_train[:,1:])
    X_test[:,1:] = scaler.transform(X_test[:, 1:])

    theta = np.random.uniform(0,1,n)
    alpha = 0.01
    nitterations = 1000

    theta1_Val = []
    Cf = list()

    for i in range(nitterations):
        update = np.zeros(X_train.shape[1])
        ypred = np.dot(X_train, theta)
        error = ypred - y_train
        for j in range(n):
            update[j] = np.sum(np.dot(error, X_train[:, j]))

        theta = theta - (1 / m) * alpha * update
        theta1_Val.append(theta[0])

        # Cost-Function
        sum1 = 0
        for k in range(m):
            sum2 = 0
            for i in range(n):
                sum2 += theta[i] * X_train[k, i]
            sum1 += (sum2 - y_train[k]) ** 2

        Cf.append(sum1 * (1 / (2 * m)))

    print("Theta : ", theta)
    print(("Theta[1] : ", len(theta1_Val)))

    print("CF : ", len(Cf))
    prediction = np.dot(theta, X_test.T)
    print("MAE: ", metrics.mean_absolute_error(y_true=y_test, y_pred=prediction))
    print("MSE: ", metrics.mean_squared_error(y_true=y_test, y_pred=prediction))


    print("\nPlotting of Graphs : ")
    print("""    1. Cost Function
    2. EXIT""")
    while True:
        n = int(input("Enter What to Plot : "))
        if n == 1:
            Plot_CF(theta1_Val, Cf)
        else:
            exit("Thanks for Using")

"""
OUTPUT:

Theta :  [ 2.43334807e+01 -1.00515791e+00  9.15022551e-01  1.13357521e-02
  5.61123609e-01 -1.20836710e+00  3.73442039e+00 -2.73681723e-02
 -2.54183088e+00  2.06095317e+00 -1.08449809e+00 -1.65437094e+00
  5.40425479e-02 -3.47470775e+00]
('Theta[1] : ', 1000)
CF :  1000
MAE:  4.771415037813911
MSE:  32.86957950538916

Plotting of Graphs : 
    1. Cost Function
    2. EXIT
Enter What to Plot : 1
Enter What to Plot : 2
Thanks for Using

"""