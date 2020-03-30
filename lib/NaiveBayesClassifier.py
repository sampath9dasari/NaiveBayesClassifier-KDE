import numpy as np


class NaiveBayesClassifier():

    def __init__(self, bandwidth=1, kernel='radial', MultiBW=False):
        self.priors = dict()
        self.dim = 1
        self.MultiBW = MultiBW
        self.bandwidth = bandwidth
        if kernel == "radial":
            self.kernel = self.radial
        elif kernel == "hypercube":
            self.kernel = self.hypercube

    def hypercube(self, k):
        return np.all(k < 0.5, axis=1)

    def radial(self, k):
        const_part = (2 * np.pi) ** (-self.dim / 2)
        return const_part * np.exp(-0.5 * np.add.reduce(k ** 2, axis=1))

    def parzen_estimation(self, h, x, x_train):
        N = x_train.shape[0]
        dim = self.dim
        k = np.abs(x - x_train) * 1.0 / h
        summation = np.add.reduce(self.kernel(k))
        return summation / (N * (h ** dim))

    def KDE(self, h, x_test, x_train):
        P_x = np.zeros(len(x_test))
        N = x_train.shape[0]
        dim = self.dim
        for i in range(len(x_test)):
            P_x[i] = self.parzen_estimation(h, x_test[i], x_train)

        return P_x

    def fit(self, X, Y):
        self.x_train = X
        self.y_train = Y
        self.dim = X.shape[1]
        labels = set(Y)
        for c in labels:
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def predict(self, x_test):
        N, D = x_test.shape
        priors = self.priors
        K = len(priors)
        P = np.zeros((N, K))
        x_train = self.x_train
        y_train = self.y_train
        if self.MultiBW:
            bw = self.bandwidth
        else:
            bw = np.repeat(self.bandwidth, K)
        for c, p in priors.items():
            P[:, int(c)] = self.KDE(bw[int(c)], x_test, x_train[y_train == c]) * p

        pred_y = np.argmax(P, axis=1)
        self.pred_y = pred_y

        return pred_y

    def accuracy(self, y_test):
        pred_y = self.pred_y
        count = 0
        return np.array([pred_y == y_test]).mean()

    def score(self, x_test, y_test):
        self.pred_y = self.predict(x_test)
        accuracy_score = self.accuracy(y_test)
        return accuracy_score
