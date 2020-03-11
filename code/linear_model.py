import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature


            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class logLinearClassifier:
    def __init__(self, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i], f = findMin.findMin(self.logLoss, self.W[i], self.maxEvals, ytmp, X)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

    def logLoss(self, w, y, X):
        f = np.sum(np.log(1 + np.exp(-y * (X@w).T)))
        g = X.T@(-y/(1+np.exp(y * (X@w).T)))
        return f, g

class softmaxClassifier:
    def __init__(self, maxEvals=400):
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros(self.n_classes*d)

        self.W, f = findMin.findMin(self.softMax, self.W, self.maxEvals, y, X)
        self.W = np.reshape(self.W, (self.n_classes,d))

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

    def softMax(self, w, y, X):
        n, d = X.shape
        w = np.reshape(w, (self.n_classes, d))
        
        Xw_y = np.zeros(n)
        Xw_T = X@w.T

        i = np.arange(n)
        Xw_y[i] = Xw_T[i,y[i]]

        f = np.sum(-Xw_y + np.log(np.sum(np.exp(Xw_T), axis=1)))

        p = np.exp(Xw_T)/np.sum(np.exp(Xw_T), axis=1)[np.newaxis].T
        I = np.zeros((n,self.n_classes))
        I[i,y[i]] = 1

        g = (X.T@(p - I)).T
        
        return f, g.flatten()