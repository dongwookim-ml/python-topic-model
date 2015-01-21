import numpy as np
import time
from scipy.special import gammaln

def sampling_from_dist(prob):
    thr = prob.sum() * np.random.rand()
    new_topic=0
    tmp = prob[new_topic]
    while tmp < thr:
        new_topic += 1
        tmp += prob[new_topic]
    return new_topic

class gibbsLDA:
    """
    Latent dirichlet allocation,
    Blei, David M and Ng, Andrew Y and Jordan, Michael I, 2003
    
    Latent Dirichlet allocation with collapsed Gibbs sampling
    """

    def __init__(self, numTopic, numWord, alpha=0.1, beta=0.01):
        self.K = numTopic
        self.W = numWord

        #hyper-parameters
        self.alpha = alpha
        self.beta = beta

    def random_init(self, docs):
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

    def gibbs_sampling(self, max_iter, docs):
        """
        Argument:
        max_iter:
        docs:
        """
        prev = time.clock()

        #print 'start Gibbs sampling'
        for iteration in xrange(max_iter):

            #print iteration, time.clock() - prev, self.loglikelihood(docs)
            prev = time.clock()

            for di in xrange(len(docs)):
                doc = docs[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = self.doc_topics[di][wi]

                    self.WK[word,old_topic] -= 1
                    self.sumK[old_topic] -= 1
                    self.doc_topic_sum[di,old_topic] -= 1

                    #update
                    prob = (self.WK[word, :])/(self.sumK[:]) * (self.doc_topic_sum[di,:])

                    new_topic = sampling_from_dist(prob)

                    self.doc_topics[di][wi] = new_topic
                    self.WK[word,new_topic] += 1
                    self.sumK[new_topic] += 1
                    self.doc_topic_sum[di,new_topic] += 1

        #print 'sampling done'

    def loglikelihood(self, docs):
        """
        likelihood function
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


if __name__ == '__main__':
    #test
    docs = [[0,1,2,3,3,4,3,4,5], [2,3,3,5,6,7,8,3,8,9,5]]
    
    model = gibbsLDA(2, 10)
    model.random_init(docs)
    model.gibbs_sampling(100,docs)
