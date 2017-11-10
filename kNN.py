#!/usr/bin/python
# -*- encoding:utf-8 -*-

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


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
    with open("./dataset/datingTestSet2.txt") as fr:
        arrayOLines = fr.readlines()
        numberOLines = len(arrayOLines)
        returnMat = zeros((numberOLines,3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line =  line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat,classLabelVector


def showplot(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(array(datingLabels))
    ax.scatter(x,y, s=array(datingLabels) ** 3, c=10 * array(datingLabels))
    plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


def datingClassTest():
    datingDataMat, datingLabels = file2matrix("./dataset/datingTestSet2.txt")
    hoRatio = 0.10
    norMat, ranges , minVals = autoNorm(datingDataMat)
    m = norMat.shape[0]
    numOTest = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numOTest):
        classFierResult  =classify0(datingDataMat[i,:],datingDataMat[numOTest:,:],datingLabels[numOTest:],3)
        print ("the classfiner come back with %d,the real answer is : %d" % (classFierResult,datingLabels[i]))
        if classFierResult != datingLabels[i]:
            errorCount += 1.0
    print("errorCount: is %f" %(errorCount/float(numOTest)))


if __name__ == '__main__':
    '''
    group,labels = createDataSet()
    classify0([0,0],group,labels,3)
    datingDataMat,datingLabels = file2matrix("./dataset/datingTestSet2.txt")
    showplot(datingDataMat[:,0],datingDataMat[:,1])
     datingDataMat,datingLabels = file2matrix("./dataset/datingTestSet2.txt")
    print(autoNorm(datingDataMat))
    '''
    datingClassTest()

