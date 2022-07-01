# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Or-3.csv')
print(df)

data = df.drop('OR', axis=1)
target = df['OR']

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

# set learning rate and epoches
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

   a  b  c  OR
0  0  0  0   0
1  0  0  1   1
2  0  1  0   1
3  0  1  1   1
4  1  0  0   1
5  1  0  1   1
6  1  1  0   1
7  1  1  1   1
8 3
Weight W :  [ 0.45243687  0.45595124 -0.38874355]
Bias b :  0.07903244376082019
Epoch :  0 Weight :  [ 0.45243687  0.45595124 -0.28874355] Bias :  0.07903244376082019
Epoch :  1 Weight :  [ 0.45243687  0.45595124 -0.18874355] Bias :  0.07903244376082019
Epoch :  2 Weight :  [ 0.45243687  0.45595124 -0.08874355] Bias :  0.07903244376082019
Epoch :  3 Weight :  [0.45243687 0.45595124 0.01125645] Bias :  0.07903244376082019
Epoch :  4 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  0.07903244376082019
Epoch :  5 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  6 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  7 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  8 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  9 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  10 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  11 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  12 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  13 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  14 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  15 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  16 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  17 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  18 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  19 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  20 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  21 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  22 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  23 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  24 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  25 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  26 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  27 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  28 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  29 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  30 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  31 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  32 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  33 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  34 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  35 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  36 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  37 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  38 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  39 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  40 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  41 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  42 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  43 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  44 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  45 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  46 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  47 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  48 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Epoch :  49 Weight :  [0.45243687 0.45595124 0.11125645] Bias :  -0.020967556239179813
Final weight :  [0.45243687 0.45595124 0.11125645]
Final Bias :  -0.020967556239179813
Final prediction :  [0, 1, 1, 1, 1, 1, 1, 1]


"""
