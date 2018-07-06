import data
from sklearn.decomposition import PCA


def pca():
    X_train, X_test, y_train, y_test = data.Train_test_split()

    pca = PCA(n_components=100)
    X_train = pca.fit(X_train).transform(X_train)
    pca1 = PCA(n_components=100)
    X_test = pca1.fit(X_test).transform(X_test)
    return X_train, X_test, y_train, y_test