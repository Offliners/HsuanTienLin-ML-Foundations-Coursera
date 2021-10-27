import numpy as np
from util import readFile, addW0
from PLA_algorithm import PLA_naive_cycle

filePath = 'hw1\data\hw1_15_train.dat'

X, y = readFile(filePath)
X = addW0(X)

print(X.shape)
print(y.shape)

pla = PLA_naive_cycle()
print('Update time : ' + str(pla.run(X, y)))