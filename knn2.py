'''
# 计算欧式距离
import math
def ComputeEuclideanDistance(x, y):
    x1, y1 = x
    x2, y2 = y
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
dis = ComputeEuclideanDistance((3, 104), (18, 90))
print(dis)
'''

'''
花萼分类
from sklearn import neighbors
from sklearn import datasets

knn=neighbors.KNeighborsClassifier()
iris=datasets.load_iris()

# print(iris)

knn.fit(iris.data,iris.target)

predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])

print(predictedLabel)
'''
import numpy as np
import operator


def createDateSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, K):
    # 获取样本个数
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    sortedDisindicies = distance.argsort()
    print(sortedDisindicies)
    classCount = {}
    for i in range(K):
        voteLabel = labels[sortedDisindicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDateSet()
    test = classify0([0, 0], group, labels, 3)
    print(test)
