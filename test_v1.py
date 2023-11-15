from v1 import create_neural_net, predict, train_network
import numpy as np


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
test = np.array([1,1,0,1,0,1,0,1])

layer_array = [[len(labels), 'sigmoid']]
input_dims = 8
neural_net = create_neural_net(layer_array, input_dims)

print(' weights:', neural_net[0],'\n\n biases:',neural_net[1],'\n\n activations:', neural_net[2])

print(predict(X[1], neural_net))

neural_net = train_network(X, Y, labels, neural_net, epochs=1000)

for i in range(len(X)):
    print(predict(X[i], neural_net))

print(predict(test, neural_net))
