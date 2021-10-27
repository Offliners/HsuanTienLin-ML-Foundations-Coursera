import numpy as np
from util import readFile, addW0
from PLA_algorithm import PLA

filePath = 'homework\hw1\data\hw1_15_train.dat'

X, y = readFile(filePath)
X = addW0(X)

print(X.shape)
print(y.shape)

pla = PLA()
print('Update times : ' + str(pla.run(X, y)))