from pyspark import SparkContext

sc=SparkContext('local')
data = sc.textFile('./data/UserPurchaseHistory.csv').map(lambda line:line.split(",")).map(lambda record: (record[0],record[1],record[2]))
# 求总购买次数
numPurchases = data.count()
# 求有多少不同客户购买过商品
uniqueUsers = data.map(lambda record: record[0]).distinct().count()
# 求和得出总收入
totalRevenue = data.map(lambda record: float(record[2])).sum()
# 求最畅销的产品是什么
products = data.map(lambda record: (record[1], 1.0)).reduceByKey(lambda a, b: a + b).collect()
mostPopular = sorted(products, key=lambda x: x[1], reverse=True)[0]
print("Total purchases: %d" % numPurchases)
print("Unique users: %d" % uniqueUsers)
print("Total revenue: %2.2f" % totalRevenue)
print("Most popular product: %s with %d purchases" % (mostPopular[0], mostPopular[1]))