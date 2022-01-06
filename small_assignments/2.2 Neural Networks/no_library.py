import re
from random import Random
import math

# use seeded random to reproduce results
rng = Random(1)

data = []

with open('ecoli.data') as f:
    whitespace_re = re.compile(r' +')
    cp_or_im_re = re.compile(r'^(cp|im)$', re.IGNORECASE)
    for line in f:
        splitline = re.split(whitespace_re, line.strip())
        if re.match(cp_or_im_re, splitline[-1]) is not None:
            data.append(splitline)

X = [list(map(float, i[1:8])) for i in data]
Y = [1.0 if i[8] == 'im' else 0.0 for i in data]

indexes = list(range(0, 220))
rng.shuffle(indexes)
X = [X[i] for i in indexes]
Y = [Y[i] for i in indexes]

X_train = X[:210]
X_test = X[210:]
Y_train = Y[:210]
Y_test = Y[210:]


weights1 = [[rng.random()/7 for _ in range(7)] for _ in range(10)]
weights2 = [[rng.random()/10 for _ in range(10)] for _ in range(5)]
weights3 = [[rng.random()/5 for _ in range(5)] for _ in range(1)]


def layer1(X: list[float]) -> list[float]:
    output = [0.0 for _ in range(10)]
    for output_i in range(10):
        for input_i in range(7):
            output[output_i] += X[input_i] * weights1[output_i][input_i]
    return output


def layer2(X: list[float]) -> list[float]:
    output = [0.0 for _ in range(5)]
    for output_i in range(5):
        for input_i in range(10):
            output[output_i] += X[input_i] * weights2[output_i][input_i]
    return output


def layer3(X: list[float]) -> list[float]:
    output = [0.0 for _ in range(1)]
    for output_i in range(1):
        for input_i in range(5):
            output[output_i] += X[input_i] * weights3[output_i][input_i]
    return output


def sigmoid(z):
    if(z < -100):
        return 0
    if(z > 100):
        return 1
    return 1.0/(1.0+math.exp(-z))


def network(X: list[float]) -> float:
    X = layer1(X)
    X = list(map(sigmoid, X))
    X = layer2(X)
    X = list(map(sigmoid, X))
    X = layer3(X)
    X = sigmoid(X[0])
    return X


print(list(map(network, X_test)))
