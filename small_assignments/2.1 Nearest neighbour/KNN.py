from heapq import heappush, heappushpop, heappop
from functools import reduce
import numpy
import matplotlib.pyplot as plt
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8
K_MAX = 100

# split into input (X) and output (Y) variables
X_train = dataset[:int(len(dataset)*splitratio), 0:8]
X_val = dataset[int(len(dataset)*splitratio):, 0:8]
Y_train = dataset[:int(len(dataset)*splitratio), 8]
Y_val = dataset[int(len(dataset)*splitratio):, 8]
# print(X_train)
# print(Y_train)


def distance(one, two):
    return numpy.linalg.norm(one-two)


def shortestDistance(x, x_rest, y_rest):
    shortest = distance(x, x_rest[0])
    predicted = y_rest[0]
    for i in range(len(x_rest)):
        if distance(x, x_rest[i]) <= shortest:
            shortest = distance(x, x_rest[i])
            predicted = y_rest[i]
    return predicted, shortest


# Returns an list of (predicted, distance) with length k
def shortestKDistances(x, x_rest, y_rest, k) -> list[tuple[int, int]]:
    class PredictedAndDistance:
        def __init__(self, predicted: int, distance: int):
            self.predicted = predicted
            self.distance = distance

        # lt returns True if self has bigger distance to keep elements with bigger distances on top of the heap
        def __lt__(self, other: 'PredictedAndDistance') -> bool:
            return self.distance > other.distance

        def __eq__(self, other: 'PredictedAndDistance') -> bool:
            return self.distance == other.dictance

        def to_tuple(self) -> tuple[int, int]:
            return (self.predicted, self.distance)

    heap: list[PredictedAndDistance] = []
    for i in range(len(x_rest)):
        new_elem = PredictedAndDistance(y_rest[i], distance(x, x_rest[i]))
        if len(heap) < k:
            # Add element because the heap isn't k elements long yet
            heappush(heap, new_elem)
        else:
            # Push the new_elem into heap, then remove the element with the biggest distance
            heappushpop(heap, new_elem)

    return [heappop(heap).to_tuple() for _ in range(len(heap))]

accuracies = []
recalls = []
precisions = []
f1s = []

for k in range(1, K_MAX+1):
    print("K:", k)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(X_val)):
        x = X_val[i]
        y = Y_val[i]

        l = shortestKDistances(x, X_train, Y_train, k)

        pred = 1 if \
            reduce(lambda acc, elem: acc + elem[0], l, 0) > k // 2 \
            else 0

        if(y == 1 and pred == 1):
            TP += 1

        if(y == 0 and pred == 0):
            TN += 1

        if(y == 1 and pred == 0):
            FN += 1

        if(y == 0 and pred == 1):
            FP += 1

    print("Accuracy:", (TP+TN)/(TP+TN+FP+FN))
    accuracies.append((TP+TN)/(TP+TN+FP+FN))
    print("Recall", TP/(TP+FN))
    recalls.append(TP/(TP+FN))
    print("Precision", TP/(TP+FP))
    precisions.append(TP/(TP+FP))
    print("F1", (2*TP)/(2*TP+FP+FN))
    f1s.append((2*TP)/(2*TP+FP+FN))
    print()

ks = list(range(1, K_MAX+1))


plt.plot(ks, accuracies, label="Accuracy")
plt.plot(ks, recalls, label="Recall")
plt.plot(ks, precisions, label="Precision")
plt.plot(ks, f1s, label="F1")
plt.xlabel("K")
# IDK what label would fit
#plt.ylabel("")
plt.title("Accuracy of the K-NN")
plt.legend()
plt.grid(linestyle='-', linewidth=1)
plt.xticks(range(0, 101, 5))
plt.show()