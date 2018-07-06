from sklearn import svm
import pandas as pd
import data
import Evaluation
from sklearn import svm

def SVM_model(X_train, X_test, y_train, y_test):
    print("--------SVM---------")
    #X_train, X_test, y_train, y_test = data.Train_test_split()
    clf = svm.SVC()
    #clf = SGDClassifier(n_iter=100, alpha=0.01)
    clf.fit(X_train,y_train)
    predict=(clf.predict(X_test))
    return predict



def LeaveUserFeaturesHour():
    X_train, X_test, y_train, y_test = data.Train_test_split()
    predict = SVM_model(X_train, X_test, y_train, y_test)
    y_test = y_test.reshape(1,-1)[0]

    Evaluation.evalution(y_test,predict)
    Evaluation.AUC(y_test,predict)

def AllactionUserFeaturesPeriod():
    X_train, X_test, y_train, y_test = data.Train_test_split_period()
    predict = SVM_model(X_train, X_test, y_train, y_test)
    y_test = y_test.reshape(1,-1)[0]

    Evaluation.evalutionZero_One(y_test,predict)
    Evaluation.AUC(y_test,predict)

if __name__ =='__main__':
    AllactionUserFeaturesPeriod()