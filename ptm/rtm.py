from __future__ import print_function
import time

import numpy as np
from scipy.special import gammaln, psi
from six.moves import xrange

from .utils import write_top_words
from .formatted_logger import formatted_logger

eps = 1e-20

logger = formatted_logger('RelationalTopicModel', 'info')


class RelationalTopicModel:
    """ implementation of relational topic model by Chang and Blei (2009)
    I implemented the exponential link probability function in here

    Attributes
    ----------
    eta: ndarray, shape (n_topic)
        coefficient of exponential function
    rho: int
        pseudo number of negative example
    """

    def __init__(self, n_topic, n_doc, n_voca, alpha=0.1, rho=1000, **kwargs):
        self.n_doc = n_doc
        self.n_topic = n_topic
        self.n_voca = n_voca

        self.alpha = alpha

        self.gamma = np.random.gamma(100., 1. / 100, [self.n_doc, self.n_topic])
        self.beta = np.random.dirichlet([5] * self.n_voca, self.n_topic)

        self.nu = 0
        self.eta = np.random.normal(0., 1, self.n_topic)

        self.phi = list()
        self.pi = np.zeros([self.n_doc, self.n_topic])

        self.rho = rho

        self.verbose = kwargs.pop('verbose', True)

        logger.info('Initialize RTM: num_voca:%d, num_topic:%d, num_doc:%d' % (self.n_voca, self.n_topic, self.n_doc))

    def fit(self, doc_ids, doc_cnt, doc_links, max_iter=100):
        for di in xrange(self.n_doc):
            unique_word = len(doc_ids[di])
            cnt = doc_cnt[di]
            self.phi.append(np.random.dirichlet([10] * self.n_topic, unique_word).T)  # list of KxW
            self.pi[di, :] = np.sum(cnt * self.phi[di], 1) / np.sum(cnt * self.phi[di])

        for iter in xrange(max_iter):
            tic = time.time()
            self.variation_update(doc_ids, doc_cnt, doc_links)
            self.parameter_estimation(doc_links)
            if self.verbose:
                elbo = self.compute_elbo(doc_ids, doc_cnt, doc_links)
                logger.info('[ITER] %3d,\tElapsed time: %.3f\tELBO: %.3f', iter, time.time()-tic, elbo)

    def compute_elbo(self, doc_ids, doc_cnt, doc_links):
        """ compute evidence lower bound for trained model
        """
        elbo = 0

        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:, np.newaxis]  # D x K
        log_beta = np.log(self.beta + eps)

        for di in xrange(self.n_doc):
            words = doc_ids[di]
            cnt = doc_cnt[di]

            elbo += np.sum(cnt * (self.phi[di] * log_beta[:, words]))  # E_q[log p(w_{d,n}|\beta,z_{d,n})]
            elbo += np.sum((self.alpha - 1.) * e_log_theta[di, :])  # E_q[log p(\theta_d | alpha)]
            elbo += np.sum(self.phi[di].T * e_log_theta[di, :])  # E_q[log p(z_{d,n}|\theta_d)]

            elbo += -gammaln(np.sum(self.gamma[di, :])) + np.sum(gammaln(self.gamma[di, :])) \
                    - np.sum((self.gamma[di, :] - 1.) * (e_log_theta[di, :]))  # - E_q[log q(theta|gamma)]
            elbo += - np.sum(cnt * self.phi[di] * np.log(self.phi[di]))  # - E_q[log q(z|phi)]

            for adi in doc_links[di]:
                elbo += np.dot(self.eta,
                               self.pi[di] * self.pi[adi]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]

        return elbo

    def variation_update(self, doc_ids, doc_cnt, doc_links):
        # update phi, gamma
        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:, np.newaxis]

        new_beta = np.zeros([self.n_topic, self.n_voca])

        for di in xrange(self.n_doc):
            words = doc_ids[di]
            cnt = doc_cnt[di]
            doc_len = np.sum(cnt)

            new_phi = np.log(self.beta[:, words] + eps) + e_log_theta[di, :][:, np.newaxis]

            gradient = np.zeros(self.n_topic)
            for ai in doc_links[di]:
                gradient += self.eta * self.pi[ai, :] / doc_len

            new_phi += gradient[:, np.newaxis]
            new_phi = np.exp(new_phi)
            new_phi = new_phi / np.sum(new_phi, 0)

            self.phi[di] = new_phi

            self.pi[di, :] = np.sum(cnt * self.phi[di], 1) / np.sum(cnt * self.phi[di])
            self.gamma[di, :] = np.sum(cnt * self.phi[di], 1) + self.alpha
            new_beta[:, words] += (cnt * self.phi[di])

        self.beta = new_beta / np.sum(new_beta, 1)[:, np.newaxis]

    def parameter_estimation(self, doc_links):
        # update eta, nu
        pi_sum = np.zeros(self.n_topic)

        num_links = 0.

        for di in xrange(self.n_doc):
            for adi in doc_links[di]:
                pi_sum += self.pi[di, :] * self.pi[adi, :]
                num_links += 1

        num_links /= 2.  # divide by 2 for bidirectional edge
        pi_sum /= 2.

        pi_alpha = np.zeros(self.n_topic) + self.alpha / (self.alpha * self.n_topic) * self.alpha / (self.alpha * self.n_topic)

        self.nu = np.log(num_links - np.sum(pi_sum)) - np.log(
            self.rho * (self.n_topic - 1) / self.n_topic + num_links - np.sum(pi_sum))
        self.eta = np.log(pi_sum) - np.log(pi_sum + self.rho * pi_alpha) - self.nu

    def save_model(self, output_directory, vocab=None):
        import os
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        np.savetxt(output_directory + '/eta.txt', self.eta, delimiter='\t')
        np.savetxt(output_directory + '/beta.txt', self.beta, delimiter='\t')
        np.savetxt(output_directory + '/gamma.txt', self.gamma, delimiter='\t')
        with open(output_directory + '/nu.txt', 'w') as f:
            f.write('%f\n' % self.nu)

        if vocab is not None:
            write_top_words(self.beta, vocab, output_directory + '/top_words.csv')

