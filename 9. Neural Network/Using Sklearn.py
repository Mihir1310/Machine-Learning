from sklearn.neural_network import MLPClassifier
import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0]).reshape(4,)


clf = MLPClassifier(hidden_layer_sizes=(4,2), max_iter=30000)
clf.fit(X, y)
y_predict = clf.predict(X)
print("Predicted Output : ",y_predict)

"""

Output:
Predicted Output :  [0 1 1 0]

"""