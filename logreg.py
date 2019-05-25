import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def batch_gd(X, y, w, fit_intercept, penalty, penalty_coef, gd_step, max_iters):
    epsilon = .000001
    if penalty == 'none':
        for _ in range(max_iters):
            shift = (gd_step * X.shape[1]) * np.dot(X.T, sigmoid(np.dot(X, w)) - y)
            w = w - shift

            if np.linalg.norm(shift) < epsilon:
                break

    if not fit_intercept and penalty == 'l2':
        for _ in range(max_iters):
            shift = (gd_step * X.shape[0]) * np.dot(X.T, sigmoid(np.dot(X, w)) - y)
            w - shift - (penalty_coef / X.shape[1]) * w

            if np.linalg.norm(shift) < epsilon:
                break

    if not fit_intercept and penalty == 'l1':
        for _ in range(max_iters):
            shift = (gd_step * X.shape[0]) * np.dot(X.T, sigmoid(np.dot(X, w)) - y)
            w - shift - (penalty_coef / X.shape[1]) * np.sign(w)

            if np.linalg.norm(shift) < epsilon:
                break

    if fit_intercept and penalty == 'l1':
        for _ in range(max_iters):
            shift = (gd_step / X.shape[0]) * np.dot(X.T, sigmoid(np.dot(X, w)) - y)
            w = w - shift - \
                np.vstack((np.array([0.0]), np.sign(w[1:, ...]) * gd_step * penalty_coef / X.shape[0]))

            if np.linalg.norm(shift) < epsilon:
                break

    if fit_intercept and penalty == 'l2':
        for _ in range(max_iters):
            shift = (gd_step / X.shape[0]) * np.dot(X.T, sigmoid(np.dot(X, w)) - y)
            w = w - shift - \
                np.vstack((np.array([0.0]), w[1:, ...] * gd_step * penalty_coef / X.shape[0]))

            if np.linalg.norm(shift) < epsilon:
                break

    return w


class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression model.
           :cost function - J = -y * log(h(x)) - (1 - y) * log(1 - h(x))
           :solving - 'Batch gradient descent' """
    def __init__(self, fit_inercept=True, penalty='l2', penalty_coef=1.0, gd_step=.000001, max_iters=1000):
        self.fit_intercept = fit_inercept
        self.penalty = penalty
        self.penalty_coef = penalty_coef
        self.gd_step = gd_step
        self.max_iters = max_iters

    def fit(self, X, y):
        assert (self.penalty in ['none', 'l1', 'l2']), "Unknown type of regularization"
        assert (type(self.penalty_coef) == float), "Type of penalty_coef should be 'float' "
        assert (type(self.gd_step) == float), "Type of gd_step should be 'float' "
        assert (type(self.max_iters) == int), "Type of max_iters should be 'int' "

        y = y.reshape((-1, 1))

        if not self.fit_intercept:
            self.X_ = X
            self.w_ = np.random.uniform(-1, 1, self.X_.shape[1]).reshape(-1, 1)

            self.w_ = batch_gd(self.X_, y, self.w_, self.fit_intercept, self.penalty, self.penalty_coef,
                               self.gd_step, self.max_iters)

        elif self.fit_intercept:

            self.X_ = np.hstack((np.ones((X.shape[0], 1)), X))
            self.w_ = np.random.uniform(-1, 1, self.X_.shape[1]).reshape(-1, 1)

            self.w_ = batch_gd(self.X_, y, self.w_, self.fit_intercept, self.penalty, self.penalty_coef,
                               self.gd_step, self.max_iters)

    def predict_proba(self, X):
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("Model should be learned first")

        if self.fit_intercept:
                return sigmoid(np.dot(np.vstack((np.ones((X.shape[0], 1)), X)), self.w_))

        if not self.fit_intercept:
                return sigmoid(np.dot(X, self.w_))

    def predict(self, X):
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("Model should be learned first")

        if self.fit_intercept:
                kek = np.hstack((np.ones((X.shape[0], 1)), X))
                return np.where(np.dot(kek, self.w_) < 0, 0, 1)

        if not self.fit_intercept:
                return np.where(np.dot(X, self.w_) < 0, 0, 1)





