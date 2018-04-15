# from sklearn.feature_extraction import DictVectorizer
# import csv
# from sklearn import preprocessing
# from sklearn import tree
# from sklearn.externals.six import StringIO
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
# allElectronicsData = open(r'./trainingDigits/AllElectronics.csv', 'rb')
# reader = csv.reader(allElectronicsData)
# # header = reader.next()
# # print(header)
'''
featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[header[i]] = row[i]
    featureList.append(rowDict)
print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print('dummyX' + str(dummyX))
print(vec.get_feature_names())

print('labelList' + str(labelList))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY" + str(dummyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print('clf', str(clf))

with open('allElectronicInformationGainori.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print('oneRowX', str(oneRowX))

newRowX = oneRowX

newRowX[0] = 1
newRowX[2] = 0

print('newRowX' + str(newRowX))
predictedY = clf.predict(newRowX)
print('predictedY' + str(predictedY))
'''
iris_data=pd.read_csv("./trainingDigits/iris.csv")
iris_data.columns=['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm','class']
print(iris_data.head())
print(iris_data.describe())

# 属性的关系
sb.pairplot(iris_data.dropna(),hue='class')

#分布的情况
plt.figure(figsize=(10,10))
for column_index,column in enumerate(iris_data.columns):
    if column=='class':
        continue
    plt.subplot(2,2,column_index+1)
    sb.violinplot(x='class',y=column,data=iris_data)

all_inputs=iris_data[['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm']].values
all_classes=iris_data['class'].values

(training_inputs,testing_inputs,training_classes,testing_classes)=train_test_split(all_inputs,all_classes,train_size=0.75,random_state=1)


# DecisionTreeClassifier
'''
1、指标：gini和熵
2、切分点：best和random，best使用所有特征中最好的切分点，random是部分特征中（数据量比较大的时候）
3、候选特征：
'''

decision_tree_classifier = DecisionTreeClassifier()

# Train the classifier on the training set
decision_tree_classifier.fit(training_inputs, training_classes)

# Validate the classifier on the testing set using classification accuracy
decision_tree_classifier.score(testing_inputs, testing_classes)



decision_tree_classifier = DecisionTreeClassifier()

# cross_val_score returns a list of the scores, which we can visualize
# to get a reasonable estimate of our classifier's performance
cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_classes, cv=10)
print(cv_scores)
# kde=False
sb.distplot(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))

decision_tree_classifier = DecisionTreeClassifier(max_depth=1)

cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_classes, cv=10)
print(cv_scores)
sb.distplot(cv_scores, kde=False)
plt.title('Average score: {}'.format(np.mean(cv_scores)))


decision_tree_classifier = DecisionTreeClassifier()

parameter_grid = {'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]}

cross_validation = StratifiedKFold(all_classes, n_folds=10)

grid_search = GridSearchCV(decision_tree_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(all_inputs, all_classes)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

grid_visualization = []

for grid_pair in grid_search.grid_scores_:
    grid_visualization.append(grid_pair.mean_validation_score)

grid_visualization = np.array(grid_visualization)
grid_visualization.shape = (5, 4)
sb.heatmap(grid_visualization, cmap='Blues')
plt.xticks(np.arange(4) + 0.5, grid_search.param_grid['max_features'])
plt.yticks(np.arange(5) + 0.5, grid_search.param_grid['max_depth'][::-1])
plt.xlabel('max_features')
plt.ylabel('max_depth')

decision_tree_classifier = grid_search.best_estimator_
decision_tree_classifier

import sklearn.tree as tree
from sklearn.externals.six import StringIO

with open('iris_dtc.dot', 'w') as out_file:
    out_file = tree.export_graphviz(decision_tree_classifier, out_file=out_file)
# http://www.graphviz.org/


from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier()

parameter_grid = {'n_estimators': [5, 10, 25, 50],
                  'criterion': ['gini', 'entropy'],
                  'max_features': [1, 2, 3, 4],
                  'warm_start': [True, False]}

cross_validation = StratifiedKFold(all_classes, n_folds=10)

grid_search = GridSearchCV(random_forest_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(all_inputs, all_classes)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

grid_search.best_estimator_


