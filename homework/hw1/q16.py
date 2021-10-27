import numpy as np
from util import readFile, addW0, unison_shuffled_copies
from PLA_algorithm import PLA

filePath = 'homework\hw1\data\hw1_15_train.dat'

X, y = readFile(filePath)
X = addW0(X)

print(X.shape)
print(y.shape)

pla = PLA()

T_sum = 0
expTimes = 2000
for i in range(expTimes):
    np.random.seed(i)
    X, y = unison_shuffled_copies(X, y)
    T_sum += pla.run(X, y)

print('Average update times : ' + str(T_sum / 2000))