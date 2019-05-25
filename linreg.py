import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import normalize


def batch_gd(X, y, w, fit_intercept, max_iters, gd_step, penalty, penalty_coef):
    epsilon = .000001
    if not penalty:
        for i in range(max_iters):

            w = w - (gd_step / (X.shape[0])) * np.dot(X.T, np.dot(X, w) - y)

    elif penalty and not fit_intercept:

        for _ in range(max_iters):
            shift = (gd_step / X.shape[0]) * np.dot(X.T, np.dot(X, w) - y)
            w = w - shift - w * gd_step * penalty_coef / X.shape[0]

            if np.linalg.norm(shift) < epsilon:
                print("converged\n")
                break

    elif penalty and fit_intercept:
        for _ in range(max_iters):
            shift = (gd_step / X.shape[0]) * np.dot(X.T, np.dot(X, w) - y)
            w = w - shift - \
                np.vstack((np.array([0.0]), w[1:, ...] * gd_step * penalty_coef / X.shape[0]))

            if np.linalg.norm(shift) < epsilon:
                print("converged\n")
                break
    return w


class MyLinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression model.
       :cost function - 'Mean squared error'
       :solving - 'Batch gradient descent' """

    def __init__(self, fit_intercept=True, max_iters=1000, gd_step=.0000001,
                 penalty=True, penalty_coef=1.0):

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iters = max_iters
        self.gd_step = gd_step
        self.penalty = penalty
        self.penalty_coef = penalty_coef


    # @staticmethod
    # def analytical_solution(X, y):
    #     return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    def fit(self, X, y):
        assert(type(self.max_iters) == int), "Type of number of iterations should be 'int' "
        assert(type(self.gd_step) == float), "Type of gradient descent step should be 'float' "
        assert(type(self.penalty_coef) == float), "Type of penalty coefficient  "

        if self.fit_intercept:
            self.X_ = np.hstack([np.ones((X.shape[0], 1)), X])
            self.w_ = np.random.uniform(-1, 1, self.X_.shape[1]).reshape(-1, 1)

            self.w_ = batch_gd(self.X_, y, self.w_, self.fit_intercept, self.max_iters, self.gd_step,
                               self.penalty, self.penalty_coef)
        else:
            self.X_ = X
            self.w_ = np.random.uniform(-1, 1, self.X_.shape[1]).reshape(-1, 1)

            self.w_ = batch_gd(self.X_, y, self.w_, self.fit_intercept, self.max_iters, self.gd_step,
                               self.penalty, self.penalty_coef)

        return self

    def predict(self, X):
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("The model should be learned first")

        if self.fit_intercept:
            return np.dot(np.hstack([np.ones((X.shape[0], 1)), X]), self.w_)
        else:
            return np.dot(X, self.w_)


