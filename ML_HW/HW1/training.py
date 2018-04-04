import numpy as np
import csv
from numpy.linalg import inv
import random
import math
import sys

data =[]
for i in range(18):
    data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5')
row = csv.reader(text, delimiter=",")

for r in row:
    if n_row != 0:
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1
text.close()


x = []
y = []
for i in range(12):
    for j in range(471):
        x.append([])
        for t in range(18):
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s])
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)


w = np.zeros(len(x[0]))
lr = 10
repeat = 1000

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x, w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a = math.sqrt(cost)
    gra = np.dot(x_t, loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - lr * gra / ada
    print('iteration: %d | cost: %f ' % (i, cost_a))


np.save('model.npy', w)
