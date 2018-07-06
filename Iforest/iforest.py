__author__ = 'wangdan'

import pandas as pd
from sklearn.ensemble import IsolationForest

data = pd.read_csv('data.csv', index_col="id")
#data = pd.read_csv('/Users/crystal/Documents/iForest/data.csv')
print data

ilf = IsolationForest(n_estimators=100,
                      n_jobs=-1,
                      verbose=2,
    )



data = data.fillna(0)
# select features
X_cols = ["age", "salary", "sex"]
print( data.shape)

# train
print("train begain")
ilf.fit(data[X_cols])
shape = data.shape[0]
batch = 10**6
print("train over")


all_pred = []
for i in range(shape/batch+1):
    start = i * batch
    end = (i+1) * batch
    test = data[X_cols][start:end]
    # predict
    pred = ilf.predict(test)
    all_pred.extend(pred)

data['pred'] = all_pred
print("outlier detect")
data.to_csv('/Users/crystal/Documents/iForest/outliers.csv', columns=["pred",], header=False)

