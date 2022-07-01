# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('AND-3.csv')
print(df)

data = df.drop('AND', axis=1)
target = df['AND']

data = data.values
target = target.values

n_datapoints = data.shape[0]
n_dimension = data.shape[1]
print(n_datapoints, n_dimension)

# initialize weight W randomly value from -1 to 1
W = 2 * np.random.random_sample(n_dimension) - 1
# define bias term
b = np.random.random()

print('Weight W : ', W)
print('Bias b : ', b)

# set learning rate and epochs
lr = 0.1
n_epoch = 50

# train model
for ep in range(n_epoch):
    for i in range(n_datapoints):
        # net_input = XW + b
        net_input = np.dot(data[i], W) + b
        # a=1 if net_input>=0 else a=0
        a = net_input >= 0
        # error = target - actual
        e = target[i] - a
        # update weight and Bias using perceptron learning rule
        W = W + lr * e * (data[i].T)
        b = b + lr * e
    print("Epoch : ", ep, "Weight : ", W, "Bias : ", b)

# print Weight and Bias
print("Final weight : ", W)
print("Final Bias : ", b)

# make predictuion
predictions = (np.dot(data, W) + b) >= 0

# prediction in numeric scale
finalPrediction = []
for predict in predictions:
    if predict == True:
        finalPrediction.append(1)
    else:
        finalPrediction.append(0);
print("Final prediction : ", finalPrediction)

"""

Output:

   a  b  c  AND
0  0  0  0    0
1  0  0  1    0
2  0  1  0    0
3  0  1  1    0
4  1  0  0    0
5  1  0  1    0
6  1  1  0    0
7  1  1  1    1
8 3
Weight W :  [-0.62228227 -0.90514447 -0.93955054]
Bias b :  0.24632407101692666
Epoch :  0 Weight :  [-0.52228227 -0.80514447 -0.83955054] Bias :  0.24632407101692666
Epoch :  1 Weight :  [-0.42228227 -0.70514447 -0.73955054] Bias :  0.24632407101692666
Epoch :  2 Weight :  [-0.32228227 -0.60514447 -0.63955054] Bias :  0.24632407101692666
Epoch :  3 Weight :  [-0.22228227 -0.50514447 -0.53955054] Bias :  0.24632407101692666
Epoch :  4 Weight :  [-0.12228227 -0.40514447 -0.43955054] Bias :  0.24632407101692666
Epoch :  5 Weight :  [-0.12228227 -0.30514447 -0.33955054] Bias :  0.14632407101692665
Epoch :  6 Weight :  [-0.02228227 -0.20514447 -0.23955054] Bias :  0.14632407101692665
Epoch :  7 Weight :  [-0.02228227 -0.10514447 -0.13955054] Bias :  0.04632407101692665
Epoch :  8 Weight :  [ 0.07771773 -0.00514447 -0.03955054] Bias :  0.04632407101692665
Epoch :  9 Weight :  [0.07771773 0.09485553 0.06044946] Bias :  -0.05367592898307336
Epoch :  10 Weight :  [0.07771773 0.09485553 0.06044946] Bias :  -0.15367592898307333
Epoch :  11 Weight :  [0.17771773 0.09485553 0.06044946] Bias :  -0.15367592898307333
Epoch :  12 Weight :  [0.27771773 0.09485553 0.06044946] Bias :  -0.15367592898307333
Epoch :  13 Weight :  [0.27771773 0.09485553 0.06044946] Bias :  -0.25367592898307334
Epoch :  14 Weight :  [0.27771773 0.19485553 0.16044946] Bias :  -0.25367592898307334
Epoch :  15 Weight :  [0.27771773 0.09485553 0.16044946] Bias :  -0.3536759289830733
Epoch :  16 Weight :  [0.27771773 0.19485553 0.16044946] Bias :  -0.3536759289830733
Epoch :  17 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.3536759289830733
Epoch :  18 Weight :  [0.37771773 0.09485553 0.16044946] Bias :  -0.4536759289830733
Epoch :  19 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.4536759289830733
Epoch :  20 Weight :  [0.37771773 0.29485553 0.16044946] Bias :  -0.4536759289830733
Epoch :  21 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.5536759289830733
Epoch :  22 Weight :  [0.37771773 0.19485553 0.26044946] Bias :  -0.5536759289830733
Epoch :  23 Weight :  [0.37771773 0.29485553 0.26044946] Bias :  -0.5536759289830733
Epoch :  24 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  25 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  26 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  27 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  28 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  29 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  30 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  31 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  32 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  33 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  34 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  35 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  36 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  37 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  38 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  39 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  40 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  41 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  42 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  43 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  44 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  45 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  46 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  47 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  48 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Epoch :  49 Weight :  [0.37771773 0.19485553 0.16044946] Bias :  -0.6536759289830733
Final weight :  [0.37771773 0.19485553 0.16044946]
Final Bias :  -0.6536759289830733
Final prediction :  [0, 0, 0, 0, 0, 0, 0, 1]

"""
