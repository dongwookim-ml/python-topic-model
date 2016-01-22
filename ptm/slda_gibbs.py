import numpy as np
import time
from scipy.special import gammaln
from numpy.linalg import solve

def sampling_from_dist(prob):
    thr = prob.sum() * np.random.rand()
    new_topic=0
    tmp = prob[new_topic]
    while tmp < thr:
        new_topic += 1
        tmp += prob[new_topic]
    return new_topic

"""
supervised topic model with normal distribution
based on stochastic EM (gibbs-EM)
"""
class sLDA:
    def __init__(self, numTopic, numWord, alpha=0.1, beta=0.01):
        self.K = numTopic
        self.W = numWord

        #hyper-parameters
        self.alpha = alpha
        self.beta = beta

        self.eta = np.random.normal(scale=5,size=self.K)
        self.sigma = 1

    def random_init(self, docs, responses):
        """
        Random initialization of topics
        """
        print 'random init'
        #summary statistics

        self.WK = np.zeros([self.W,self.K]) + self.beta
        self.sumK = np.zeros([self.K]) + self.beta * self.W
        self.doc_topic_sum = np.zeros([len(docs), self.K]) + self.alpha

        #topic assignments
        self.doc_topics = list()

        for di in xrange(len(docs)):
            doc = docs[di]
            topics = np.random.randint(self.K, size = len(doc))
            self.doc_topics.append(topics)

            for wi in xrange(len(doc)):
                topic = topics[wi]
                word = doc[wi]
                self.WK[word,topic] += 1
                self.sumK[topic] +=1
                self.doc_topic_sum[di, topic] += 1

        print 'done'

    def stochasticEM(self, max_iter, docs, responses):
        """
        gibbs sampling
        """
        prev = time.clock()

        for iteration in xrange(max_iter):
            prev = time.clock()

            for di in xrange(len(docs)):
                doc = docs[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = self.doc_topics[di][wi]

                    self.WK[word,old_topic] -= 1
                    self.sumK[old_topic] -= 1
                    self.doc_topic_sum[di,old_topic] -= 1

                    z_bar = np.zeros([self.K,self.K]) + self.doc_topic_sum[di,:] + np.identity(self.K)
                    # this seems more straightforward than z_bar/z_bar.sum(1)
                    z_bar /= self.doc_topic_sum[di,:].sum() + 1

                    #update
                    prob = (self.WK[word, :])/(self.sumK[:]) * (self.doc_topic_sum[di,:]) * np.exp(np.negative((responses[di] - np.dot(z_bar,self.eta))**2)/2/self.sigma)

                    new_topic = sampling_from_dist(prob)

                    self.doc_topics[di][wi] = new_topic
                    self.WK[word,new_topic] += 1
                    self.sumK[new_topic] += 1
                    self.doc_topic_sum[di,new_topic] += 1

            #estimate parameters
            z_bar = self.doc_topic_sum / self.doc_topic_sum.sum(1)[:,np.newaxis] # DxK
            self.eta = solve(np.dot(z_bar.T,z_bar), np.dot(z_bar, responses) )

            #compute mean absolute error
            mae = np.abs(responses - np.dot(z_bar, self.eta)).sum()
            print 'iteration',iteration, time.clock()-prev, mae, self.loglikelihood(docs)

    def heldoutSampling(self, max_iter, heldout):
        h_doc_topics = list()
        h_doc_topic_sum = np.zeros([len(heldout), self.K]) + self.alpha

        for di in xrange(len(heldout)):
            doc = heldout[di]
            topics = np.random.randint(self.K, size = len(doc))
            h_doc_topics.append(topics)

            for wi in xrange(len(doc)):
                topic = topics[wi]
                word = doc[wi]
                h_doc_topic_sum[di, topic] += 1

        for iteration in xrange(max_iter):
            for di in xrange(len(heldout)):
                doc = heldout[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = h_doc_topics[di][wi]

                    h_doc_topic_sum[di,old_topic] -= 1

                    #update
                    prob = (self.WK[word, :])/(self.sumK[:]) * (self.doc_topic_sum[di,:])

                    new_topic = sampling_from_dist(prob)

                    h_doc_topics[di][wi] = new_topic
                    h_doc_topic_sum[di,new_topic] += 1

        return h_doc_topic_sum

    def loglikelihood(self, docs):
        """
        likelihood function
        does not contain normal distribution part
        """
        ll = 0

        ll += len(docs) * gammaln(self.alpha*self.K)
        ll -= len(docs) * self.K * gammaln(self.alpha)
        ll += self.K * gammaln(self.beta*self.W)
        ll -= self.K * self.W * gammaln(self.beta) 

        for di in xrange(len(docs)):
            ll += gammaln(self.doc_topic_sum[di,:]).sum() - gammaln(self.doc_topic_sum[di,:].sum())
        for ki in xrange(self.K):
            ll += gammaln(self.WK[:,ki]).sum() - gammaln(self.WK[:,ki].sum())

        return ll

