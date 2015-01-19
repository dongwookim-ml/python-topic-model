import numpy as np 
from scipy.special import gammaln, psi

eps = 1e-10

class rtm:
    """ implementation of relational topic model by Chang and Blei (2009)
    I implemented the exponential link probability function in here
    """

    def __init__(self, num_topic, num_doc, num_voca, doc_ids, doc_cnt, doc_links, rho):
        self.D = num_doc 
        self.K = num_topic
        self.V = num_voca

        self.alpha = 1.

        self.gamma = np.random.gamma(100., 1./100, [self.D, self.K])
        self.beta = np.random.dirichlet([5]*self.V, self.K)

        self.nu = 0
        self.eta = np.random.normal(0.,1, self.K)

        self.phi = list()
        self.pi = np.zeros([self.D, self.K])

        for di in xrange(self.D):
            unique_word = len(doc_ids[di])
            cnt = doc_cnt[di]
            self.phi.append(np.random.dirichlet([10]*self.K, unique_word).T) # K x W
            self.pi[di,:] = np.sum(cnt*self.phi[di],1)/np.sum(cnt*self.phi[di])

        self.doc_ids = doc_ids
        self.doc_cnt = doc_cnt
        self.doc_links = doc_links
        self.rho = rho  #regularization parameter

    def posterior_inference(self, max_iter):
        for iter in xrange(max_iter):
            self.variation_update()
            self.parameter_estimation()

    def variation_update(self):
        #update phi, gamma
        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:,np.newaxis]

        new_beta = np.zeros([self.K, self.V])

        for di in xrange(self.D):
            words = self.doc_ids[di]
            cnt = self.doc_cnt[di]
            doc_len = np.sum(cnt)

            new_phi = np.log(self.beta[:,words]) + e_log_theta[di,:][:,np.newaxis]

            gradient = np.zeros(self.K)
            for adi in self.doc_links[di]:
                gradient += self.eta * self.pi[adi,:] / doc_len

            new_phi += gradient[:,np.newaxis]
            new_phi = np.exp(new_phi)
            new_phi = new_phi/np.sum(new_phi,0)

            self.phi[di] = new_phi

            self.pi[di,:] = np.sum(cnt * self.phi[di],1)/np.sum(cnt * self.phi[di])
            self.gamma[di,:] = np.sum(cnt * self.phi[di], 1) + self.alpha
            new_beta[:, words] += (cnt * self.phi[di])

        self.beta = new_beta / np.sum(new_beta, 1)[:,np.newaxis]

    def parameter_estimation(self):
        #update eta, nu
        pi_sum = np.zeros(self.K)

        num_links = 0.

        for di in xrange(self.D):
            for adi in self.doc_links[di]:
                pi_sum += self.pi[di,:]*self.pi[adi,:]
                num_links += 1

        pi_alpha = np.zeros(self.K) + self.alpha/(self.alpha*self.K)*self.alpha/(self.alpha*self.K)

        self.nu = np.log(num_links-np.sum(pi_sum)) - np.log(self.rho*(self.K-1)/self.K + num_links - np.sum(pi_sum))
        self.eta = np.log(pi_sum) - np.log(pi_sum + self.rho * pi_alpha) - self.nu 
