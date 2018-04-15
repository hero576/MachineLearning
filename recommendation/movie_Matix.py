from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating

sc = SparkContext('local')

# 引入数据
rawData = sc.textFile("../movie_test/ml-100k/u.data")

# 对数据进行分割，去除时间戳数据
rawRatings = rawData.map(lambda x: x.split("\t")[:3])

# 生成ratings类对象，后续输入给训练模型
# USER:ID,PRODUCT:ID,RATING:VALUE
ratings = rawRatings.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))

x = rawRatings.take(1).collect()
Rating(int(x[0]), int(x[1]), float(x[2]))

# 模型训练
# rank：50，表示ALS模型的因子个数，20~2000个就可以表现出很好的模型训练的效果
# iteration：10，迭代次数，10次足够了
# lambda：0.01，正则化系数，控制过拟合，值越大表示正则化越严厉
model = ALS.train(ratings, 50, 10, 0.01)

# 得到用户和物品的因子，通过count查看两因子数组
model.userFeatures().count()
model.productFeatures().count()

# 求出该模型预测用户789对电影123的评级
predictedRating = model.predict(789, 123)

# 用户789推荐的前10个物品：
topKRecs = model.recommendProducts(789, 10)

# 检验推荐内容
# 要直观地检验推荐的效果，可以简单比对下用户所评级过的电影的标题和被推荐的那些电影的电影名。
#
# 首先，我们需要读入电影数据（这是在上一章探索过的数据集）。这些数据会导入为Map[Int, String]类型，即从电影ID到标题的映射：
movies = sc.textFile("./movie_test/ml-100k/u.item")

# '1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0'
titles = movies.map(lambda x: x.split('|')).map(lambda x: (int(x[0]), x[1])).collectAsMap()

# 对用户789，我们可以找出他所接触过的电影、给出最高评级的前10部电影及名称。
# 利用lookup函数来只返回给定键值（即特定用户ID）对应的那些评级数据到驱动程序。
moviesForUser = ratings.keyBy(lambda x: x.user).lookup(789)
moviesForUser = sc.parallelize(moviesForUser)

# 接下来，我们要获取评级最高的前10部电影。具体做法是利用Rating对象的rating属性来对moviesForUser集合进行排序并选出排名前10的评级（含相应电影ID）。之后以其为输入，借助titles映射为“ (电影名称，具体评级)”形式。再将名称与具体评级打印出来
y = sc.parallelize(moviesForUser.sortBy(lambda x: x.rating, ascending=False, ).take(10)).map(
    lambda x: (titles[int(x.product)], x.rating)).collect()

# 现在看下对该用户的前10个推荐，并利用上述相同的方式来查看它们的电影名（注意这些推建基于 Spark 的推荐引擎荐已排序）：
z = sc.parallelize(topKRecs).map(lambda x: (titles[int(x.product)], x.rating)).collect()

import numpy as np
from numpy import linalg


# 定义余弦相似度函数
def cosineSimilarity(A, B):
    num = np.dot(A.T, B)
    denom = linalg.norm(A) * linalg.norm(B)
    return num / denom


# 获取商品567对应因子，取出一个数组，我们只需要一个值，就是物品因子的向量
itemFactor = model.productFeatures().lookup(567)[0]
itemVector = np.array(itemFactor)
s = cosineSimilarity(itemVector, itemVector)

# 相似度s=1.0
# 求其他物品余弦相似度
sims = model.productFeatures().map(lambda id, factor: (id, cosineSimilarity(np.array(factor), np.array(itemFactor))))

sims = model.productFeatures().map(lambda x: (x[0], cosineSimilarity(np.array(x[1]), np.array(itemFactor))))

# 对物品按照相似度排序，然后取出与物品567最相似的前10个物品：
sortedSims = sims.sortBy(lambda x: x[1], ascending=False).take(10)

# 检查推荐的相似物品
# 查看推荐的那些电影名称，取前11部最相似电影，以排除给定的那部。所以，可以选取列表中的第1到11项：
sortedSims2 = sims.sortBy(lambda x: x[1], ascending=False).take(11)
sortedSims2 = sc.parallelize(sortedSims2[1:]).map(lambda x: (titles[x[0]], x[1]))

# 推荐模型效果的评估
# 均方差

# 求789用户均方误差的一个示例
actual_Rating_Product = moviesForUser.take(1)[0].product
actual_Rating = moviesForUser.take(1)[0].rating
predict_Rating = model.predict(789, actual_Rating_Product)

import math

aquaredError = math.pow(actual_Rating - predict_Rating, 2)

# 首先从ratings RDD里提取用户和物品的ID，并使用model.predict来对各个“用户物品”对做预测。所得的RDD以“用户和物品ID”对作为主键，对应的预计评级作为值：

userProducts = ratings.map(lambda rating: (rating.user, rating.product))
predictions = model.predictAll(userProducts).map(lambda rating: ((rating.user, rating.product), rating.rating))

ratingsAndPredictions = ratings.map(lambda rating: ((rating.user, rating.product), rating.rating)).join(predictions)

from operator import add

# MSE均方误差Mean Squared Error
MSE = ratingsAndPredictions.map(lambda x: np.power(x[1][0] - x[1][1], 2)).reduce(add) / ratingsAndPredictions.count()

# RMSE均方根误差Root Mean Squared Error
RMSE = np.power(MSE, 0.5)


# APK:衡量的是模型对用户感兴趣和会去接触的物品的预测能力
def avgPrecisionK(A, B, k):
    #两个数组。一个以各个物品及其评级为内容，另一个以模型所预测的物品及其评级为内容。
    if len(B) > k:
        predk = sc.parallelize[:k]
    else:
        predk = B
    score = 0.0
    numHits = 0.0
    for i, p in enumerate(predk):
        if p in A and p not in predk[:i]:
            numHits += 1.0
            score += numHits / (i + 1.0)
    if not A:
        return 1.0
    else:
        return score / min(len(A), k)

#计算对用户789推荐的APK指标怎么样。首先提取出用户实际评级过的电影的ID：
actualMovies = moviesForUser.map(lambda x:x.product)

#提取出推荐的物品列表
predictedMovies = sc.parallelize(topKRecs).map(lambda x:x.product)

apk10 = avgPrecisionK(actualMovies.collect(), predictedMovies.collect(), 10)

#这里， APK的得分为0，这表明该模型在为该用户做相关电影预测上的表现并不理想

#全局MAPK的求解
#为每一个用户都生成相应的推荐列表，计算量较大,需要通过Spark将该计算分布式进行
#不过，这就会有一个限制，即每个工作节点都要有完整的物品因子矩阵。这样它们才能独立地计算某个物品向量与其他所有物品向量之间的相关性。
#然而当物品数量众多时，单个节点的内存可能保存不下这个矩阵。此时，这个限制也就成了问题。（没有其他简单的途径来应对这个问题。一种可能的方式是只计算与所有物品中的一部分物品的相关性。这可通过局部敏感哈希算法）


#取回物品因子向量并用它来构建一个DoubleMatrix对象
itemFactors = model.productFeatures().map(lambda x:x[1])
#{ case (id, factor)=> factor }.collect()
itemMatrix = np.array(itemFactors)
itemMatrix.shape
#说明itemMatrix的行列数分别为1682和50

#接下来，我们将该矩阵以一个广播变量的方式分发出去，以便每个工作节点都能访问到：
imBroadcast = sc.broadcast(itemMatrix)
#广播变量创建之后，应该在所有函数中替代v来使用，以免v多次被发送到集群节点。另外，对象v广播之后，不应该被修改，从而保证所有的节点看到的是相同的广播变量值。

#现在可以计算每一个用户的推荐。这会对每一个用户因子进行一次map操作。在这个操作里，会对用户因子矩阵和电影因子矩阵做乘积，其结果为一个表示各个电影预计评级的向量（长度为1682，即电影的总数目）。之后，用预计评级对它们排序：

userVector=model.userFeatures().map(lambda x:(x[0],np.array(x[1])))
userVector=userVector.map(lambda x:(
    x[0],np.dot(imBroadcast.value,np.array(x[1]).T)
))

userVectorId =userVector.map(lambda x:(
    x[0],[(xx,i) for i,xx in enumerate(x[1].tolist())]
))

sortUserVectorId = userVectorId.map(lambda x:(x[0],sorted(x[1],key=lambda x:x[0],reverse=True)))

sortUserVectorRecId = sortUserVectorId.map(lambda x: (x[0],[xx[1] for xx in x[1]]))

sortUserVectorRecId.count()

userMovies = ratings.map(lambda rating: (rating.user,rating.product)).groupBy(lambda x:x[0])
userMovies = userMovies.map(lambda x:(x[0], [xx[1] for xx in x[1]] ))
allAPK=sortUserVectorRecId.join(userMovies).map(lambda x:avgPrecisionK(x[1][1].collect(),x[1][0].collect(),2000))
# print allAPK.take(10)
allRecs=sc.parallelize([1,12,3,4])
MAPK = allRecs.join(userMovies).map()


# { case (userId, (predicted,
# actualWithIds)) =>
# val actual = actualWithIds.map(_._2).toSeq
# avgPrecisionK(actual, predicted, K)
# }.reduce(_ + _) / allRecs.count

def datasave(filename,data):
    if filename:
        path='d:/user/gm/desktop/'+filename+'.txt'
    else:
        path='d:/user/gm/desktop/123.txt'
    with open(path,'w') as f:
        f.write(data)


from pyspark.mllib.evaluation import RegressionMetrics,RankingMetrics

predictedAndTrue = ratingsAndPredictions.map(lambda x:(x[1][0],x[1][1]))

regressionMetrics = RegressionMetrics(predictedAndTrue)
predictedAndTrueForRanking = allRecs.join(userMovies).map()
# { case
# (userId, (predicted, actualWithIds)) =>
# val actual = actualWithIds.map(_._2)
# (predicted.toArray, actual.toArray)
# }
rankingMetrics = RankingMetrics(predictedAndTrueForRanking)