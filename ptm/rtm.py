import numpy as np
import utils
from scipy.special import gammaln, psi
from formatted_logger import formatted_logger

eps = 1e-20

log = formatted_logger('RTM', 'info')

class rtm:
    """ implementation of relational topic model by Chang and Blei (2009)
    I implemented the exponential link probability function in here
    """

    def __init__(self, num_topic, num_doc, num_voca, doc_ids, doc_cnt, doc_links, rho):
        self.D = num_doc 
        self.K = num_topic
        self.V = num_voca

        self.alpha = .1

        self.gamma = np.random.gamma(100., 1./100, [self.D, self.K])
        self.beta = np.random.dirichlet([5]*self.V, self.K)

        self.nu = 0
        self.eta = np.random.normal(0.,1, self.K)

        self.phi = list()
        self.pi = np.zeros([self.D, self.K])

        for di in xrange(self.D):
            unique_word = len(doc_ids[di])
            cnt = doc_cnt[di]
            self.phi.append(np.random.dirichlet([10]*self.K, unique_word).T)    # list of KxW
            self.pi[di,:] = np.sum(cnt*self.phi[di],1)/np.sum(cnt*self.phi[di])

        self.doc_ids = doc_ids
        self.doc_cnt = doc_cnt
        self.doc_links = doc_links
        self.rho = rho  #regularization parameter

        log.info('Initialize RTM: num_voca:%d, num_topic:%d, num_doc:%d' % (self.V,self.K,self.D))

    def posterior_inference(self, max_iter):
        for iter in xrange(max_iter):
            self.variation_update()
            self.parameter_estimation()
            log.info('%d iter: ELBO = %.3f' % (iter, self.compute_elbo()))

    def compute_elbo(self):
        """ compute evidence lower bound for trained model
        """
        elbo = 0

        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:,np.newaxis] # D x K
        log_beta = np.log(self.beta+eps)

        for di in xrange(self.D):
            words = self.doc_ids[di]
            cnt = self.doc_cnt[di]
            
            elbo += np.sum(cnt * (self.phi[di] * log_beta[:,words])) # E_q[log p(w_{d,n}|\beta,z_{d,n})]
            elbo += np.sum((self.alpha - 1.)*e_log_theta[di,:]) # E_q[log p(\theta_d | alpha)]
            elbo += np.sum(self.phi[di].T * e_log_theta[di,:])  # E_q[log p(z_{d,n}|\theta_d)]

            elbo += -gammaln(np.sum(self.gamma[di,:])) + np.sum(gammaln(self.gamma[di,:])) \
                - np.sum((self.gamma[di,:] - 1.)*(e_log_theta[di,:]))   # - E_q[log q(theta|gamma)]
            elbo += - np.sum(cnt * self.phi[di] * np.log(self.phi[di])) # - E_q[log q(z|phi)]

            for adi in self.doc_links[di]:
                elbo += np.dot(self.eta, self.pi[di]*self.pi[adi]) + self.nu # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]

        return elbo

    def variation_update(self):
        #update phi, gamma
        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:,np.newaxis]

        new_beta = np.zeros([self.K, self.V])

        for di in xrange(self.D):
            words = self.doc_ids[di]
            cnt = self.doc_cnt[di]
            doc_len = np.sum(cnt)

            new_phi = np.log(self.beta[:,words]+eps) + e_log_theta[di,:][:,np.newaxis]

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

        num_links /= 2. # divide by 2 for bidirectional edge
        pi_sum /= 2.

        pi_alpha = np.zeros(self.K) + self.alpha/(self.alpha*self.K)*self.alpha/(self.alpha*self.K)

        self.nu = np.log(num_links-np.sum(pi_sum)) - np.log(self.rho*(self.K-1)/self.K + num_links - np.sum(pi_sum))
        self.eta = np.log(pi_sum) - np.log(pi_sum + self.rho * pi_alpha) - self.nu 

    def save_model(self, output_directory, vocab=None):
        import os
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        np.savetxt(output_directory+'/eta.txt', self.eta, delimiter='\t')
        np.savetxt(output_directory+'/beta.txt', self.beta, delimiter='\t')
        np.savetxt(output_directory+'/gamma.txt',self.gamma,delimiter='\t')
        with open(output_directory+'/nu.txt', 'w') as f:
            f.write('%f\n'%self.nu)

        if vocab != None:
            utils.write_top_words(self.beta, vocab, output_directory+'/top_words.csv')

def main():
    rho = 1
    num_topic = 2
    num_voca = 6
    num_doc = 2
    doc_ids = [[0,1,4],[2,3,5]]
    doc_cnt = [[2,2,3],[1,3,1]]
    doc_links = [[1],[0]]   #bidirectional link
    model = rtm(num_topic, num_doc, num_voca, doc_ids, doc_cnt, doc_links, rho)
    model.posterior_inference(10)

if __name__ == '__main__':
    main()