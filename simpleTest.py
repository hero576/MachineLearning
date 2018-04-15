from NeuralNetwork import *
'''XOR预测'''
# nn=NeuralNetwork([2,2,1],'tanh')
# X=np.array([[0,0],[0,1],[1,0],[1,1]])
# y=np.array([0,1,1,0])
# nn.fit(X,y)
# for i in [[0,0],[0,1],[1,0],[1,1]]:
#     print(i,nn.predict(i))

'''数字展示'''
# from sklearn.datasets import load_digits
# digits=load_digits()
# print(digits.data.shape)
# import pylab as pl
# # pl.gray()
# pl.matshow(digits.images[0])
# pl.show()

import numpy as np
from sklearn.datasets import load_digits
# confusion:
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

digits=load_digits()
X=digits.data
y=digits.target
#数据预处理，将X变为0，1之间的值
X-=X.min()
X/=X.max()
nn=NeuralNetwork([64,100,10],"logistic")
X_train,X_test,y_train,y_test=train_test_split(X,y)
labels_train=LabelBinarizer().fit_transform(y_train)
labels_test=LabelBinarizer().fit_transform(y_test)
print('start fittig')
nn.fit(X_train,labels_train,epochs=3000)
predictions=[]
for i in range(X_test.shape[0]):
    o=nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))




