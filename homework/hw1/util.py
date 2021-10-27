import numpy as np

def readFile(path):
    X = []
    y = []
    for line in open(path).readlines():
        line = line.replace('\t', ' ')
        data = line.strip().split(' ')
        y.append(float(data[-1]))
        X.append(list(map(float, data[:-1])))

    return np.array(X), np.array(y)

def addW0(x):
    return np.insert(x, 0, np.ones(x.shape[0]), 1)

def unison_shuffled_copies(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]