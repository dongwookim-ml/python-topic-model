import time
import numpy as np
import numpy.linalg
import simplex_projection
import scipy.optimize

e = 1e-100
error_diff = 10

class CTM:
    """
    Correlated topic models,
    Blei, David and Lafferty, John,
    2006
    """
    
    def __init__(self, topic_size, voca_size, user_size, item_size, doc_item, doc_cnt, ratings=None):
        self.lambda_u = 0.01
        self.lambda_v = 0.01
        self.alpha = 1
        self.eta = 0.01
        self.a = 1
        self.b = 0.01

        self.topic_size = topic_size
        self.voca_size = voca_size
        self.user_size = user_size
        self.item_size = item_size

        #U = user_topic matrix, U x K
        self.U = np.random.multivariate_normal(np.zeros(topic_size), np.identity(topic_size)*(1./self.lambda_u), size=self.user_size)
        #V = item(doc)_topic matrix, V x K
        self.V = np.random.multivariate_normal(np.zeros(topic_size), np.identity(topic_size)*(1./self.lambda_u), size=self.item_size)
        self.theta = np.random.random([item_size,topic_size])
        self.theta = self.theta/self.theta.sum(1)[:,np.newaxis] #normalize
        self.beta = np.random.random([voca_size,topic_size])
        self.beta = self.beta/self.beta.sum(0) #normalize

        self.doc_item = doc_item
        self.doc_cnt = doc_cnt

        self.C = np.zeros([user_size, item_size]) + self.b
        self.R = np.zeros([user_size, item_size]) #user_size x item_size

        if ratings:
            for di in xrange(len(ratings)):
                rate = ratings[di]
                for user in rate:
                    self.C[user,di] += self.a - self.b
                    self.R[user,di] = 1

        self.phi_sum = np.zeros([voca_size, topic_size]) + self.eta
        
    def learning_fixed_theta(self, max_iter):
        old_err = 0
        for iteration in xrange(max_iter):
            prev = time.clock()
            self.update_u()
            self.update_v()
            err = self.sqr_error()
            print 'Iteration-', iteration,  time.clock() - prev, err
            if abs(old_err - err) < error_diff:
                break

    #reconstructing matrix for prediction
    def predict_item(self):
        return np.dot(self.U, self.V.T)
        
    #reconstruction error    
    def sqr_error(self):    
        err = (self.R - self.predict_item())**2
        err = err.sum()

        return err

    def do_e_step(self):
        self.update_u()
        self.update_v()
        self.update_theta()

    def update_theta(self):
        def func(x, v, phi, beta, lambda_v):
            return 0.5 * lambda_v * np.dot((v-x).T, v-x) - np.sum(np.sum(phi * ( np.log(x*beta) - np.log(phi) )))
        
        for vi in xrange(self.item_size):
            W = np.array(self.doc_item[vi])
            word_beta = self.beta[W,:]
            phi = self.theta[vi,:] * word_beta + e     # W x K
            phi = phi/phi.sum(1)[:,np.newaxis]
            result = scipy.optimize.minimize(func, self.theta[vi,:], method='nelder-mead', args=(self.V[vi,:], phi, word_beta, self.lambda_v))
            self.theta[vi,:] = simplex_projection.euclidean_proj_simplex(result.x)
            self.phi_sum[W,:] += np.array(self.doc_cnt[vi])[:,np.newaxis] * phi

    def update_u(self):
        for ui in xrange(self.user_size):
            left = np.dot(self.V.T * self.C[ui,:], self.V) + self.lambda_u * np.identity(self.topic_size)

            self.U[ui,:] = numpy.linalg.solve(left, np.dot(self.V.T * self.C[ui,:],self.R[ui,:]))

    def update_v(self):
        for vi in xrange(self.item_size):
            left = np.dot(self.U.T * self.C[:,vi], self.U) + self.lambda_v * np.identity(self.topic_size)

            self.V[vi,:] = numpy.linalg.solve(left, np.dot(self.U.T * self.C[:,vi],self.R[:,vi] ) + self.lambda_v * self.theta[vi,:])

    def do_m_step(self):
        self.beta = self.phi_sum / self.phi_sum.sum(0)
        self.phi_sum = np.zeros([self.voca_size, self.topic_size]) + self.eta

def main():
    doc_word = [[0,1,2,4,5], [2,3,5,6,7,8,9]]
    doc_cnt = [[1,2,3,2,1], [3,4,5,1,2,3,4]]
    rate_user = [[0,1,2],[2,3]]
    model = CTM(3, 10, 4, 2, doc_word, doc_cnt, rate_user)
    model.learning(10)

if __name__ == '__main__':
    main()
