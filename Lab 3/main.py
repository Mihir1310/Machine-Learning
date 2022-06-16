#Simple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def Plot_CF(theta1_Val, Cf):
    plt.plot(theta1_Val, Cf, color='red')
    plt.show()


def Plot_TrainSet(x_train, y_train, predytrain):
    plt.scatter(x_train[:, 1], y_train)
    plt.plot(x_train[:, 1], predytrain, color='red')
    plt.show()


def Plot_TestSet(x_test, y_test, predytest):
    plt.scatter(x_test[:, 1], y_test)
    plt.plot(x_test[:, 1], predytest, color='red')
    plt.show()


if __name__ == '__main__':
    df1 = pd.read_csv("Train.csv")
    df2 = pd.read_csv("Test.csv")
    x_train = np.ones((699, 2))
    x_train[:, 1] = df1['x']
    y_train = np.array(df1['y'])

    x_test = np.ones((300, 2))
    x_test[:, 1] = df2['x']
    y_test = np.array(df2['y'])

    m = x_train.shape[0]
    n = x_train.shape[1]

    theta = (-3, 2)
    alpha = 0.0001
    nitterations = 100

    theta1_Val = []
    Cf = list()

    for i in range(nitterations):
        update = np.zeros(x_train.shape[1])
        ypred = np.dot(x_train, theta)
        error = ypred - y_train
        for j in range(n):
            update[j] = np.sum(np.dot(error, x_train[:, j]))

        theta = theta - (1 / m) * alpha * update
        theta1_Val.append(theta[1])

        # Cost-Function
        sum1 = 0
        for k in range(m):
            sum1 += (theta[0] + theta[1] * x_train[k, 1] - y_train[k]) ** 2

        Cf.append(sum1 * (1 / (2 * m)))

    prediction = np.dot(theta, x_test.T)
    print("MAE: ", metrics.mean_absolute_error(y_true=y_test, y_pred=prediction))
    print("MSE: ", metrics.mean_squared_error(y_true=y_test, y_pred=prediction))

    predytrain = theta[0] + theta[1] * x_train[:, 1]
    predytest = theta[0] + theta[1] * x_test[:, 1]

    print("\nPlotting of Graphs : ")
    print("""        1. Cost Function
        2. Training Set Model
        3. Test Set Model
        4. EXIT""")
    while True:
        n = int(input("Enter What to Plot : "))
        if n == 1:
            Plot_CF(theta1_Val, Cf)
        if n == 2:
            Plot_TrainSet(x_train, y_train, predytrain)
        elif n == 3:
            Plot_TestSet(x_test, y_test, predytest)