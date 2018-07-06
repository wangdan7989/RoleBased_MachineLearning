from sklearn import cross_validation,metrics

def evalution(y_test,predict):
    print("predict:",predict)
    print("y_test:",y_test)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(predict)):
        if y_test[i] == 1:
            if predict[i] == 1:
                TP = TP + 1
            else:
                FN = FN + 1
        if y_test[i] == -1:
            if predict[i] == -1:
                TN = TN + 1
            else:
                FP = FP + 1

    print('TP:', TP, 'FN:', FN, 'TN:', TN, 'FP:', FP)
    accuracy = float(TP + TN) / float(TP + FN + FP + TN)
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    F1 =(2*precision*recall)/(precision + recall)
    print("accuracy:",accuracy)
    print("precision:",precision)
    print("recall:",recall)
    print("F1 score:", F1)


def evalutionZero_One(y_test,predict):
    print("predict:",predict)
    print("y_test:",y_test)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(predict)):
        if y_test[i] == 1:
            if predict[i] == 1:
                TP = TP + 1
            else:
                FN = FN + 1
        if y_test[i] == 0:
            if predict[i] == 0:
                TN = TN + 1
            else:
                FP = FP + 1

    print('TP:', TP, 'FN:', FN, 'TN:', TN, 'FP:', FP)
    accuracy = float(TP + TN) / float(TP + FN + FP + TN)
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    F1 =(2*precision*recall)/(precision + recall)
    print("accuracy:",accuracy)
    print("precision:",precision)
    print("recall:",recall)
    print("F1 score:", F1)

def AUC(y_test,predict):
    test_auc = metrics.roc_auc_score(y_test, predict)
    print ("AUC:",test_auc)