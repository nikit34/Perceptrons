import numpy as np
from v3 import Perceptron


X = [
    [1,1,0,0,1,1,1,1],
    [1,1,0,0,0,1,1,1],
    [1,0,0,0,1,1,1,1],
    [0,1,1,1,0,0,0,1],
    [0,0,0,1,0,0,0,0],
    [0,0,0,1,1,0,0,0]
]

labels = [0, 1, 0, 0, 0, 1]

Y = [1, 1, 1, 0, 0, 0]
test = [1, 1, 0, 1, 0, 1]

perceptron = Perceptron()
wt_matrix = perceptron.fit(X, labels, 1000, 0.01)
print(perceptron.predict(Y))
print(perceptron.predict(test))
