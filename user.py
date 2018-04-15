from pyspark import SparkContext

sc = SparkContext('local')

# data = sc.parallelize([1,2,3,4,5,6,7,8,9],3)

# data2 = sc.textFile('E:/video/spark/01/用户安装列表数据.rar')
# data1 = sc.textFile('E:/video/spark/01/用户安装列表数据.rar',use_unicode=False)

# print(data2.count())

import os

file_list = "E:/video/spark/01/用户安装列表数据/".join(os.listdir("E:\\video\\spark\\01\\用户安装列表数据"))
print(file_list)

BASE_DIR = 'E:/video/spark/01/用户安装列表数据/'

file_list = []

for i in os.listdir(BASE_DIR):
    file_list.append(os.path.join(BASE_DIR, i))
print(file_list)

# data = sc.textFile(file_list)
# data = sc.wholeTextFiles(BASE_DIR)

# data.take(1)

data1 = sc.textFile('E:/video/spark/01/用户安装列表数据/000000_0.gz')
user_info = sc.parallelize(data1.collect(), data1.count())
user_info = sc.parallelize(data1.collect(), data1.count())


def split_x(x, n):
    return x.split('\t')[n]


data_user = data1.map(lambda x: x.split('\t')[1])

data_user_dic=data_user.distinct()
data_user.count()
# 45977
data_user_dic=data_user.distinct()
data_user_dic.count()
# 1114

data_date = data1.map(lambda x: x.split('\t')[0])
data_date.distinct().count()
# 1

data2 = sc.textFile('E:/video/spark/01/用户安装列表数据/000001_0.gz')
data=data1.union(data2)




user_data = sc.paralleclize(None)
user_data = sc.textFile('E:/video/spark/01/用户安装列表数据/000000_0.gz')
for i in os.listdir(BASE_DIR):
    if i == 'E:/video/spark/01/用户安装列表数据/000000_0.gz':
        continue
    user_data = user_data.union(sc.textFile(os.path.join(BASE_DIR, i)).collect())

user_data.union(sc.textFile('E:/video/spark/01/用户安装列表数据/000001_0.gz').collect())

user_data = sc.textFile('E:/video/spark/01/用户安装列表数据/*')
user_data_spilt = user_data.map(lambda: ())

user_info = sc.parallelize(user_data, user_data.count())
