#!/usr/bin/python
# -*- encoding:utf-8 -*-

from numpy import *
import operator


def createDataSet():
    group = array([[1.0,1.0],[1.0,1.1],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat  = diffMat ** 2
    sqDiffDistance = sqDiffMat.sum(axis=1)
    distance = sqDiffDistance ** 0.5
    sortedDistIndice = distance.argsort()
    classCount = {}
    for x in range(k):
        voteLabel = labels[sortedDistIndice[x]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)



if __name__ == '__main__':
    # group,labels = createDataSet()
    # classify0([0,0],group,labels,3)
    with open("./dataset/datingTestSet2.txt") as fr:
        print(fr.readlines())