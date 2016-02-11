import time

import numpy as np
from six.moves import xrange
from scipy.special import gammaln, psi

from .formatted_logger import formatted_logger

eps = 1e-3

logger = formatted_logger('vbLDA')

def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return psi(alpha) - psi(np.sum(alpha))
    return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]


class vbLDA:
    """
    Latent dirichlet allocation,
    Blei, David M and Ng, Andrew Y and Jordan, Michael I, 2003

    Latent Dirichlet allocation with mean field variational inference
    """

    def __init__(self, n_doc, n_voca, n_topic, alpha=0.1, beta=0.01, is_compute_bound=True):
        self.n_voca = n_voca
        self.n_topic = n_topic
        self.n_doc = n_doc
        self.alpha = alpha
        self.beta = beta

        self._lambda = 1 * np.random.gamma(100., 1. / 100, (self.n_topic, self.n_voca))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        self.gamma_iter = 5
        self.gamma = 1 * np.random.gamma(100., 1. / 100, (self.n_doc, self.n_topic))

        self.is_compute_bound = is_compute_bound

    def fit(self, doc_ids, doc_cnt, max_iter=100):

        for iter in xrange(max_iter):
            tic = time.time()
            _, bound = self.do_m_step(doc_ids, doc_cnt)

            if self.verbose:
                logger.info('[ITER] %d,\telapsed time:%.2f,\tELBO:%.2f', iter, time.time() - tic, bound)

    def do_e_step(self, doc_ids, doc_cnt):
        """
        compute approximate topic distribution of each document and each word
        """
        Elogtheta = dirichlet_expectation(self.gamma)
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros(self._lambda.shape)

        for d in range(0, self.n_doc):
            ids = doc_ids[d]
            cnt = np.array(doc_cnt[d])
            if np.sum(cnt) == 0:
                continue

            gammad = self.gamma[d, :]

            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

            for iter in xrange(self.gamma_iter):
                lastgamma = gammad

                gammad = self.alpha + expElogthetad * np.dot(cnt / phinorm, expElogbetad.T)

                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

                meanchange = np.mean(abs(gammad - lastgamma))

                if (meanchange < eps):
                    break

            self.gamma[d, :] = gammad
            sstats[:, ids] += np.outer(expElogthetad.T, cnt / phinorm)

        sstats = sstats * self._expElogbeta

        return (self.gamma, sstats)

    def do_m_step(self, doc_ids, doc_cnt):
        """
        estimate topic distribution based on computed approx. topic distribution
        """
        (gamma, sstats) = self.do_e_step(doc_ids, doc_cnt)

        self._lambda = self.beta + sstats
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)

        bound = 0
        if self.is_compute_bound:
            bound = self.approx_bound(doc_ids, doc_cnt, gamma)

        return gamma, bound

    def approx_bound(self, doc_ids, doc_cnt, gamma):
        """
        Compute lower bound of the corpus
        """

        score = 0
        Elogtheta = dirichlet_expectation(gamma)

        # E[log p(docs | theta, beta)]
        for d in range(0, self.n_doc):
            ids = doc_ids[d]
            cts = np.array(doc_cnt[d])
            phinorm = np.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
            score += np.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self.alpha - gamma) * Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self.alpha))
        score += sum(gammaln(self.alpha * self.n_topic) - gammaln(np.sum(gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + np.sum((self.beta - self._lambda) * self._Elogbeta)
        score = score + np.sum(gammaln(self._lambda) - gammaln(self.beta))
        score = score + np.sum(gammaln(self.beta * self.n_voca) - gammaln(np.sum(self._lambda, 1)))

        return score
