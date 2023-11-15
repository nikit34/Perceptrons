import numpy as np
from v2 import Perceptron


X = [
    [1,1,0,0,1,1,1,1],
    [1,1,0,0,0,1,1,1],
    [1,0,0,0,1,1,1,1],
    [0,1,1,1,0,0,0,1],
    [0,0,0,1,0,0,0,0],
    [0,0,0,1,1,0,0,0]
]

labels = np.array([0, 1, 0, 0])

Y = [1, 1, 1, 0, 0, 0]
test = np.array([1, 1, 0, 1, 0, 1, 0, 1])

perceptron = Perceptron(len(labels))
perceptron.train(X, labels)
perceptron.predict(Y)
perceptron.predict(test)
