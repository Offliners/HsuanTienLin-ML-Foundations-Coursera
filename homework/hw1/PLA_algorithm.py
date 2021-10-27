import numpy as np

class PLA_naive_cycle:
    """
        Initial weight w is 0
        Take sign(0) = -1
        Halt condition is that all datas are classified correctly.
        This program will return update times (T) of PLA in naive cycle order. 
    """
    def __init__(self):
        pass

    def run(self, X, y):
        w = np.zeros(X.shape[1])
        T = 0
        while True:
            correct = 0
            for i in range(X.shape[0]):
                if np.sign(w.T.dot(X[i])) == 0.0:
                    sign = -1
                else:
                    sign = np.sign(w.T.dot(X[i]))

                if sign != y[i]:
                    w += y[i] * X[i]
                    T += 1
                else:
                    correct += 1

            if correct == X.shape[0]:
                break
        
        return T

class PLA_naive_cycle_with_LR:
    """
        Initial weight w is 0
        Take sign(0) = -1
        You can set your own learning rate.
        Halt condition is that all datas are classified correctly.
        This program will return update times (T) of PLA in naive cycle order. 
    """
    def __init__(self, lr):
        self.lr = lr

    def run(self, X, y):
        w = np.zeros(X.shape[1])
        T = 0
        while True:
            correct = 0
            for i in range(X.shape[0]):
                if np.sign(w.T.dot(X[i])) == 0.0:
                    sign = -1
                else:
                    sign = np.sign(w.T.dot(X[i]))

                if sign != y[i]:
                    w += self.lr * y[i] * X[i]
                    T += 1
                else:
                    correct += 1

            if correct == X.shape[0]:
                break
        
        return T