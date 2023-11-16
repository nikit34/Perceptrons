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

labels = [0, 1, 0, 0, 0, 1, 0, 0]

Y = [1, 1, 1, 0, 0, 0, 1, 1]
test = [1, 1, 0, 1, 0, 1, 0, 1]

perceptron = Perceptron(len(labels))
perceptron.fit(X, labels)
print(perceptron.predict(Y))
print(perceptron.predict(test))
