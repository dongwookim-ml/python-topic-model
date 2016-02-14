from __future__ import print_function
import time

import numpy as np
from scipy.special import gammaln
from six.moves import xrange

from .base import BaseGibbsParamTopicModel
from .formatted_logger import formatted_logger
from .utils import sampling_from_dist

logger = formatted_logger('GibbsLDA')


class GibbsLDA(BaseGibbsParamTopicModel):
    """
    Latent dirichlet allocation,
    Blei, David M and Ng, Andrew Y and Jordan, Michael I, 2003
    
    Latent Dirichlet allocation with collapsed Gibbs sampling

    Attributes
    ----------
    topic_assignment:
        list of topic assignment for each word token

    """

    def __init__(self, n_doc, n_voca, n_topic, alpha=0.1, beta=0.01, **kwargs):
        super(GibbsLDA, self).__init__(n_doc=n_doc, n_voca=n_voca, n_topic=n_topic, alpha=alpha, beta=beta, **kwargs)

    def random_init(self, docs):
        """

        Parameters
        ----------
        docs: list, size=n_doc

        """
        for di in range(len(docs)):
            doc = docs[di]
            topics = np.random.randint(self.n_topic, size=len(doc))
            self.topic_assignment.append(topics)

            for wi in range(len(doc)):
                topic = topics[wi]
                word = doc[wi]
                self.TW[topic, word] += 1
                self.sum_T[topic] += 1
                self.DT[di, topic] += 1

    def fit(self, docs, max_iter=100):
        """ Gibbs sampling for LDA

        Parameters
        ----------
        docs
        max_iter: int
            maximum number of Gibbs sampling iteration

        """
        self.random_init(docs)

        for iteration in xrange(max_iter):
            prev = time.clock()

            for di in xrange(len(docs)):
                doc = docs[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = self.topic_assignment[di][wi]

                    self.TW[old_topic, word] -= 1
                    self.sum_T[old_topic] -= 1
                    self.DT[di, old_topic] -= 1

                    # compute conditional probability of a topic of current word wi
                    prob = (self.TW[:, word] / self.sum_T) * (self.DT[di, :])

                    new_topic = sampling_from_dist(prob)

                    self.topic_assignment[di][wi] = new_topic
                    self.TW[new_topic, word] += 1
                    self.sum_T[new_topic] += 1
                    self.DT[di, new_topic] += 1

            if self.verbose:
                logger.info('[ITER] %d,\telapsed time:%.2f,\tlog_likelihood:%.2f', iteration, time.clock() - prev, self.log_likelihood(docs))

    def log_likelihood(self, docs):
        """
        likelihood function
        """
        ll = len(docs) * gammaln(self.alpha * self.n_topic)
        ll -= len(docs) * self.n_topic * gammaln(self.alpha)
        ll += self.n_topic * gammaln(self.beta * self.n_voca)
        ll -= self.n_topic * self.n_voca * gammaln(self.beta)

        for di in xrange(len(docs)):
            ll += gammaln(self.DT[di, :]).sum() - gammaln(self.DT[di, :].sum())
        for ki in xrange(self.n_topic):
            ll += gammaln(self.TW[ki, :]).sum() - gammaln(self.TW[ki, :].sum())

        return ll
