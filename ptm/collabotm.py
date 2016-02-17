from __future__ import print_function
import time

import numpy as np
import numpy.linalg
import scipy.optimize
from six.moves import xrange

from .simplex_projection import euclidean_proj_simplex
from .formatted_logger import formatted_logger

logger = formatted_logger('CollaborativeTopicModel', 'info')

e = 1e-100
error_diff = 10


class CollaborativeTopicModel:
    """
    Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles."
    Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.

    Attributes
    ----------
    n_item: int
        number of items
    n_user: int
        number of users
    R: ndarray, shape (n_user, n_item)
        user x item rating matrix
    """

    def __init__(self, n_topic, n_voca, n_user, n_item, doc_ids, doc_cnt, ratings):
        self.lambda_u = 0.01
        self.lambda_v = 0.01
        self.alpha = 1
        self.eta = 0.01
        self.a = 1
        self.b = 0.01

        self.n_topic = n_topic
        self.n_voca = n_voca
        self.n_user = n_user
        self.n_item = n_item

        # U = user_topic matrix, U x K
        self.U = np.random.multivariate_normal(np.zeros(n_topic), np.identity(n_topic) * (1. / self.lambda_u),
                                               size=self.n_user)
        # V = item(doc)_topic matrix, V x K
        self.V = np.random.multivariate_normal(np.zeros(n_topic), np.identity(n_topic) * (1. / self.lambda_u),
                                               size=self.n_item)
        self.theta = np.random.random([n_item, n_topic])
        self.theta = self.theta / self.theta.sum(1)[:, np.newaxis]  # normalize
        self.beta = np.random.random([n_voca, n_topic])
        self.beta = self.beta / self.beta.sum(0)  # normalize

        self.doc_ids = doc_ids
        self.doc_cnt = doc_cnt

        self.C = np.zeros([n_user, n_item]) + self.b
        self.R = np.zeros([n_user, n_item])  # user_size x item_size

        for di in xrange(len(ratings)):
            rate = ratings[di]
            for user in rate:
                self.C[user, di] += self.a - self.b
                self.R[user, di] = 1

        self.phi_sum = np.zeros([n_voca, n_topic]) + self.eta

    def fit(self, doc_ids, doc_cnt, rating_matrix, max_iter=100):
        old_err = 0
        for iteration in xrange(max_iter):
            tic = time.clock()
            self.do_e_step()
            self.do_m_step()
            err = self.sqr_error()
            if self.verbose:
                logger.info('[ITER] %3d,\tElapsed time:%.2f,\tReconstruction error:%.3f', iteration,
                            time.clock() - tic, err)
            if abs(old_err - err) < error_diff:
                break

    # reconstructing matrix for prediction
    def predict_item(self):
        return np.dot(self.U, self.V.T)

    # reconstruction error
    def sqr_error(self):
        err = (self.R - self.predict_item()) ** 2
        err = err.sum()

        return err

    def do_e_step(self):
        self.update_u()
        self.update_v()
        self.update_theta()

    def update_theta(self):
        def func(x, v, phi, beta, lambda_v):
            return 0.5 * lambda_v * np.dot((v - x).T, v - x) - np.sum(np.sum(phi * (np.log(x * beta) - np.log(phi))))

        for vi in xrange(self.n_item):
            W = np.array(self.doc_ids[vi])
            word_beta = self.beta[W, :]
            phi = self.theta[vi, :] * word_beta + e  # W x K
            phi = phi / phi.sum(1)[:, np.newaxis]
            result = scipy.optimize.minimize(func, self.theta[vi, :], method='nelder-mead',
                                             args=(self.V[vi, :], phi, word_beta, self.lambda_v))
            self.theta[vi, :] = euclidean_proj_simplex(result.x)
            self.phi_sum[W, :] += np.array(self.doc_cnt[vi])[:, np.newaxis] * phi

    def update_u(self):
        for ui in xrange(self.n_user):
            left = np.dot(self.V.T * self.C[ui, :], self.V) + self.lambda_u * np.identity(self.n_topic)

            self.U[ui, :] = numpy.linalg.solve(left, np.dot(self.V.T * self.C[ui, :], self.R[ui, :]))

    def update_v(self):
        for vi in xrange(self.n_item):
            left = np.dot(self.U.T * self.C[:, vi], self.U) + self.lambda_v * np.identity(self.n_topic)

            self.V[vi, :] = numpy.linalg.solve(left, np.dot(self.U.T * self.C[:, vi],
                                                            self.R[:, vi]) + self.lambda_v * self.theta[vi, :])

    def do_m_step(self):
        self.beta = self.phi_sum / self.phi_sum.sum(0)
        self.phi_sum = np.zeros([self.n_voca, self.n_topic]) + self.eta
