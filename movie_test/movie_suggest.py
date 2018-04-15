from pyspark import SparkContext
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
import numpy as np
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext('local')

user_data = sc.textFile('./movie_test/ml-100k/u.user')
user_fields = user_data.map(lambda line: line.split('|'))
user_data.first()

# '1|24|M|technician|85711'

user_fields = user_data.map(lambda line: line.split('|'))
num_users = user_fields.map(lambda fields: fields[0]).count()
num_genders = user_fields.map(lambda fields: fields[2]).distinct().count()
num_occupations = user_fields.map(lambda fields: fields[3]).distinct().count()
num_zipcodes = user_fields.map(lambda fields: fields[4]).distinct().count()
print(
    "Users: %d, genders: %d, occupations: %d, ZIP codes: %d" % (num_users, num_genders, num_occupations, num_zipcodes))

# 用户年龄段分布
ages = user_fields.map(lambda x: int(x[1])).collect()
hist(ages, bins=20, color='lightblue', normed=True)
fig = plt.pyplot.gcf()
fig.set_size_inches(16, 10)

# 用户职业分布
# 方法一
occupations_list = user_fields.map(lambda x: (x[3], 1)).reduceByKey(lambda x, y: x + y).collect()
x_axis1 = np.array([c[0] for c in occupations_list])
y_axis1 = np.array([c[1] for c in occupations_list])


x_axis = x_axis1[np.argsort(y_axis1)]
y_axis = y_axis1[np.argsort(y_axis1)]

# pos = np.array(len(x_axis))
# width = 1.0
# ax = plt.axes()
# ax.set_xticks()
# ax.set_xticklabels(x_axis)
# plt.bar(pos, y_axis, width
#         , color='lightblue')
# plt.xticks(rotation=30)
# fig2 = plt.gcf()
# fig2.set_size_inches(16, 10)
# fig2.show()

fig, ax = plt.subplots()
x = np.arange(len(x_axis))
plt.bar(x, y_axis)
plt.xticks(x, x_axis)
plt.xticks(rotation=30)
plt.show()

# 方法二
occupations_list = user_fields.map(lambda x: x[3]).countByValue().collect()
x_axis1 = np.array([c[0] for c in occupations_list])
y_axis1 = np.array([c[1] for c in occupations_list])
x_axis = x_axis1[np.argsort(y_axis1)]
y_axis = y_axis1[np.argsort(y_axis1)]

fig, ax = plt.subplots()
x = np.arange(len(x_axis))
plt.bar(x, y_axis)
plt.xticks(x, x_axis)
plt.xticks(rotation=30)
plt.show()

# 电影数据
# 1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
movie_data = sc.textFile('./movie_test/ml-100k/u.item')
movie_fields = movie_data.map(lambda x: x.split('|'))
movie_years = movie_fields.map(lambda x: x[2])
print(movie_years.collect())
num_movies = movie_data.count()


# 防止数据缺失
def convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900


years = movie_years.map(lambda x: convert_year(x))
year_filter = years.filter(lambda x: x != 1900)

# 计算电影的年龄，并统计数目
movie_ages = year_filter.map(lambda x: 2018 - x).countByValue()

# movie_ages=np.array(movie_ages)
# movie_ages1 = movie_ages[np.argsort(movie_ages.keys())]

# y_axis = y_axis1[np.argsort(y_axis1)]

# values = movie_ages.values()
# keys = movie_ages.keys()
#
# x=[]
#
# for i,v in enumerate(list(values)):
#     for j in range(v):
#         x.append(list(keys)[i])
#
#
# hist(x,bins=70,color='red',normed=True)
# fig=plt.gcf()
# plt.axis([0, 396, 0, 0.2])
# fig.show()

# n,bins,patches=plt.hist(movie_ages1,bins=key_len,color='red',normed=True)
# plt.grid(True)
# plt.show()
movie_ages = year_filter.map(lambda yr: 1998 - yr).countByValue()
values = movie_ages.values()
bins = movie_ages.keys()
hist(year_filter.map(lambda yr: 1998 - yr).collect(), bins=70, color='lightblue', normed=True)
fig = plt.gcf()
fig.set_size_inches(16, 10)

# 评级数据

rating_data = sc.textFile("./movie_test/ml-100k/u.data")
num_ratings = rating_data.count()
# '196\t242\t3\t881250949'

rating_data_mp = rating_data.map(lambda x: x.split('\t'))
ratings = rating_data_mp.map(lambda x: int(x[2]))
max_rating = ratings.reduce(lambda x, y: max(x, y))
min_rating = ratings.reduce(lambda x, y: min(x, y))
mean_rating = ratings.reduce(lambda x, y: x + y) / num_ratings
median_rating = np.median(ratings.collect())
ratings_per_user = num_ratings / num_users
ratings_per_movie = num_ratings / num_movies

ratings.stats()

count_by_rating = ratings.countByValue()
x_axis = np.array(count_by_rating.keys())
y_axis = np.array([float(c) for c in count_by_rating.values()])
# 这里对y轴正则化，使它表示百分比
y_axis_normed = y_axis / y_axis.sum()
pos = np.arange(len(x_axis))
width = 1.0
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(x_axis)
plt.bar(pos, y_axis_normed, width, color='lightblue')
plt.xticks(rotation=30)
fig = plt.gcf()
fig.set_size_inches(16, 10)

'''
kjlkj
'''

from pyspark import SparkContext
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
import numpy as np

sc = SparkContext('local')

movie_data = sc.textFile("./ml-100k/u.item")
num_movies = movie_data.count()


# 绘制电影年龄的分布图
def convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900  # 若数据缺失年份则将其年份设为1900。在后续处理中会过滤掉这类数据


movie_fields = movie_data.map(lambda x: x.split('|'))
years = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x))

years_filtered = years.filter(lambda x: x != 1900)

movie_ages = years_filtered.map(lambda yr: 1998 - yr).countByValue()

values = movie_ages.values()

plt.hist(values, 70, normed=True)
plt.show()

# 评级数据

rating_data = sc.textFile("./ml-100k/u.data")
num_ratings = rating_data.count()

rating_data = rating_data.map(lambda line: line.split("\t"))
ratings = rating_data.map(lambda fields: int(fields[2]))
max_rating = ratings.reduce(lambda x, y: max(x, y))
min_rating = ratings.reduce(lambda x, y: min(x, y))
mean_rating = ratings.reduce(lambda x, y: x + y) / num_ratings
median_rating = np.median(ratings.collect())

# RDD中，对数据的统计展示
ratings.stats()

count_by_rating = ratings.countByValue()
x_axis = np.array(count_by_rating.keys())
y_axis = np.array([float(c) for c in count_by_rating.values()])
# 这里对y轴正则化，使它表示百分比
y_axis_normed = y_axis / y_axis.sum()
pos = np.arange(len(x_axis.all()))
width = 1.0
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(x_axis.all())
plt.bar(pos, y_axis_normed, width, color='lightblue')
fig = plt.gcf()
fig.set_size_inches(16, 10)

plt.bar(pos, y_axis_normed)

# 用户评分统计
user_ratings_grouped = rating_data.map(lambda fields: (int(fields[0]), int(fields[2]))).groupByKey().mapValues(len)

user_ratings_grouped.take(5)

x = user_ratings_grouped.map(lambda x: x[1]).collect()

hist(x, 200, normed=True)
plt.show()

# 正则化
np.random.seed(42)
x = np.random.randn(10)
norm_x_2 = np.linalg.norm(x)
normalized_x = x / norm_x_2
np.linalg.norm(normalized_x)

rawData = sc.textFile("./ml-100k/u.data")
rawData.first()

data = [
    LabeledPoint(0.0, [0.0, 1.0]),
    LabeledPoint(1.0, [1.0, 0.0]),
]

lrm = LogisticRegressionWithSGD.train(sc.parallelize(data), iterations=10)
lrm.predict([1.0, 0.0])

from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating

'''
ALS.train()
r1 = (1, 1, 1.0)
r2 = (1, 2, 2.0)
r3 = (2, 1, 2.0)
ratings = sc.parallelize([r1, r2, r3])
model = ALS.trainImplicit(ratings, 1, seed=10)
model.predict(2, 2)

testset = sc.parallelize([(1, 2), (1, 1)])
model = ALS.train(ratings, 2, seed=0)
model.predictAll(testset).collect()
'''

rawRatings = rawData.map(lambda x: x.split("\t")[:3])

ratings = rawRatings.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))

x= rawRatings.take(1).collect()
Rating(int(x[0]), int(x[1]), float(x[2]))

model=ALS.train(ratings,50,10,0.01)

model.userFeatures()

predictedRating = model.predict(789, 123)

topKRecs = model.recommendProducts(789, 10)

movies = sc.textFile("./ml-100k/u.item")

#'1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0'
titles = movies.map(lambda x:x.split('|')).map(lambda x:(int(x[0]),x[1])).collectAsMap()

moviesForUser = ratings.keyBy(lambda x:x.user).lookup(789)


moviesForUser = sc.parallelize(moviesForUser)

y=sc.parallelize(moviesForUser.sortBy(lambda x:x.rating,ascending=False,).take(10)).map(lambda x:(titles[int(x.product)],x.rating)).collect()

y=sc.parallelize(moviesForUser.sortBy(lambda x:x.rating,ascending=False,).take(10)).map(lambda x:(titles[int(x.product)],x.rating)).collect()


z=sc.parallelize(topKRecs).map(lambda x:(titles[int(x.product)],x.rating)).collect()