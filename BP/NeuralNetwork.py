# coding:utf-8
'''
Created on 2016/4/27

@author: wangdan
'''
import numpy as np
import pandas as pd

# 定义双曲函数和他们的导数
def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


# 定义NeuralNetwork 神经网络算法
class NeuralNetwork:
    # 初始化，layes表示的是一个list，eg[10,10,3]表示第一层10个神经元，第二层10个神经元，第三层3个神经元
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        # 循环从1开始，相当于以第二层为基准，进行权重的初始化
        for i in range(1, len(layers) - 1):
            # 对当前神经节点的前驱赋值
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            # 对当前神经节点的后继赋值
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    # 训练函数   ，X矩阵，每行是一个实例 ，y是每个实例对应的结果，learning_rate 学习率，
    # epochs，表示抽样的方法对神经网络进行更新的最大次数
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)  # 确定X至少是二维的数据
        temp = np.ones([X.shape[0], X.shape[1] + 1])  # 初始化矩阵
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)  # 把list转换成array的形式

        for k in range(epochs):
            # 随机选取一行，对神经网络进行更新
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            # 完成所有正向的更新
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            #
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # 开始反向计算误差，更新权重
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    # 预测函数
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
'''
[2,2,1]
第一个2:表示 数据的纬度，因为是二维的，表示两个神经元，所以是2
第二个2：隐藏层数据纬度也是2，表示两个神经元 
1：表示输入为一个神经元
tanh:表示用双曲函数里的tanh函数
'''
train = pd.read_csv('../data/trainset.csv',index_col='id')
test = pd.read_csv('../data/testset.csv',index_col='id')

xx = test.values.tolist()
yy = test['label'].tolist()

x = train.values.tolist()
y = train['label'].tolist()
for i in range(len(x)):
    del x[i][0]

for i in range(len(xx)):
    del xx[i][0]



nn = NeuralNetwork([9,2,1], 'tanh')
#X = np.array([[0, 0,0], [0, 1,0], [1, 0,0], [1, 1,0]])
#y = np.array([0, 1, 1, 0])

nn.fit(x, y)
result = []
for i in xx:
    if nn.predict(i)[0] < 0 :result.append(0)
    else:result.append(1)
    #print(i,nn.predict(i))


TP=0
FP=0
FN=0
TN=0
for i in range(len(result)):
    if yy[i]==1:
        if result[i]==1:
            TP = TP+1
        else:
            FN = FN+1
    if yy[i]==-1:
        if result[i]==-1:
            TN = TN+1
        else:
            FP = FP+1

print('TP:',TP,'FN:',FN,'TN:',TN,'FP:',FP)
accuracy = float(TP+TN)/float(TP+FN+FP+TN)
percision = float(TP)/float(TP+FP)
recall = float(TP)/float(TP+FN)
print(accuracy)
print(percision)
print(recall)
