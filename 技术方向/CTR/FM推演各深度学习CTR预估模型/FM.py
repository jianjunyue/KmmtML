from __future__ import division
from math import exp
import pandas as pd
from numpy import *
from random import normalvariate
from datetime import datetime
from sklearn import preprocessing

def load_train_data(data):
    global min_max_scaler
    data = pd.read_csv(data)
    labelMat = data.ix[:,-1]* 2 - 1
    X_train = np.array(data.ix[:, :-1])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    return X_train_minmax,labelMat

def laod_test_data(data):

    data = pd.read_csv(data)
    labelMat = data.ix[:, -1] * 2 - 1
    X_test = np.array(data.ix[:, :-1])
    X_tset_minmax = min_max_scaler.transform(X_test)
    return X_tset_minmax, labelMat

def sigmoid(inx):
    return 1. / (1. + exp(-max(min(inx, 10), -10)))

def FM_function(dataMatrix, classLabels, k, iter):
    m, n = shape(dataMatrix)
    alpha = 0.01
    w = zeros((n, 1))
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))
    for it in range(iter):
        print(it)
        for x in range(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            p = w_0 + dataMatrix[x] * w + interaction  #
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            w_0 = w_0 - alpha * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (
                        dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

    return w_0, w, v

def Assessment(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)
        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue
    print(result)
    return float(error) / allItem

if __name__ == '__main__':
    #-------读取数据----------
    trainData = 'train.txt'
    testData = 'test.txt'
    #------模型训练----
    dataTrain, labelTrain = load_train_data(trainData)
    dataTest, labelTest = laod_test_data(testData)
    w_0, w, v = FM_function(mat(dataTrain), labelTrain, 15, 100)