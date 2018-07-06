from sklearn import svm
import pandas as pd
import data
import Evaluation
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def MLP_model(X_train, X_test, y_train, y_test):
    print("--------MLP---------")
    #standar
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(X_train,y_train)
    predict=(clf.predict(X_test))
    return predict

def LeaveUserFeaturesHour():
    X_train, X_test, y_train, y_test = data.Train_test_split()
    predict = MLP_model(X_train, X_test, y_train, y_test)
    y_test = y_test.reshape(1,-1)[0]

    Evaluation.evalution(y_test,predict)
    Evaluation.AUC(y_test,predict)

def AllactionUserFeaturesPeriod():
    X_train, X_test, y_train, y_test = data.Train_test_split_period()
    predict = MLP_model(X_train, X_test, y_train, y_test)
    y_test = y_test.reshape(1,-1)[0]

    Evaluation.evalutionZero_One(y_test,predict)
    Evaluation.AUC(y_test,predict)

if __name__ =='__main__':
    LeaveUserFeaturesHour()
    #AllactionUserFeaturesPeriod()

