# coding=utf-8
import Employees
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def TocsvLeaveEmployees():
    employees = Employees.queryLeaveEmployees()
    LeaveEmployees=[]
    for item in employees:
        user = item[0]
        filename1 = '../data/BehaviorsFeaturesByUser/'+user+'.csv'
        lablefile = '../data/user_lable.csv'
        labledata = pd.read_csv(lablefile)

        for item2 in labledata.values:
            if user==item2[2]:
                lable = item2[3]
                break

        data = pd.read_csv(filename1)
        data = data.values.reshape(1,-1)
        data = np.insert(data, 0, lable, axis=1)
        print(data)
        LeaveEmployees.append(data[0])

    LeaveEmployees =pd.DataFrame(LeaveEmployees)
    print (LeaveEmployees.values)
    LeaveEmployees.to_csv('../data/LeaveUsersFeaturesAndLables/LeaveUsersFeaturesAndLbles.csv')

def Train_test_split():
    data = pd.read_csv('../data/LeaveUsersFeaturesAndLables/LeaveUsersFeaturesAndLbles.csv')
    data = data.fillna(0)
    data = np.matrix(data.values)
    data = np.transpose(data)
    #print(data)
    X = np.array(np.transpose(data[2:30]))
    y = np.array(np.transpose(data[1]))
    #pca
    pca = PCA(n_components=15)
    X = pca.fit(X).transform(X)

    #print (X)
    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.33, random_state = 42)
    return X_train, X_test, y_train, y_test

def Train_test_split_period():
    data = pd.read_csv('../data/FeaturesPeriod/Allfeatures.csv')
    data = data.fillna(0)
    data = np.matrix(data.values)
    data = np.transpose(data)
    #print(data)

    X = np.array(np.transpose(data[1:-1]))
    y = np.array(np.transpose(data[-1]))
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    print(X)
    print(y)
    #pca
    pca = PCA(n_components=2)
    X = pca.fit(X).transform(X)

    #print (X)
    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.33, random_state = 42)
    return X_train, X_test, y_train, y_test

if __name__ =='__main__':
    #TocsvLeaveEmployees()
    #Train_test_split()
    Train_test_split_period()