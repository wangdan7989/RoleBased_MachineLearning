from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import data
import Evaluation

def RandomForest_model(X_train, X_test, y_train, y_test):
    print("----------Random Forest-------------")
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    return predict
def LeaveUserFeaturesHour():
    X_train, X_test, y_train, y_test = data.Train_test_split()
    predict = RandomForest_model(X_train, X_test, y_train, y_test)
    y_test = y_test.reshape(1,-1)[0]

    Evaluation.evalution(y_test,predict)
    Evaluation.AUC(y_test,predict)

def AllactionUserFeaturesPeriod():
    X_train, X_test, y_train, y_test = data.Train_test_split_period()
    predict = RandomForest_model(X_train, X_test, y_train, y_test)
    y_test = y_test.reshape(1,-1)[0]

    Evaluation.evalutionZero_One(y_test,predict)
    Evaluation.AUC(y_test,predict)

if __name__ =='__main__':
    AllactionUserFeaturesPeriod()