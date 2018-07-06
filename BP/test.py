#coding:utf-8
'''
Created on 2016/4/27

@author: Gamer Think
'''

import numpy as np
from BP.NeuralNetwork import NeuralNetwork
'''
[2,2,1]
第一个2:表示 数据的纬度，因为是二维的，表示两个神经元，所以是2
第二个2：隐藏层数据纬度也是2，表示两个神经元 
1：表示输入为一个神经元
tanh:表示用双曲函数里的tanh函数
'''
nn = NeuralNetwork([2,2,1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i,nn.predict(i))