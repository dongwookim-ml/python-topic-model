import numpy as np
import time
import utils
from scipy.special import gammaln, psi

eps = 1e-100

class hdsp:
    """
    hierarchical dirichlet scaling process (hdsp)

    """

    def __init__(self, K, N, J, dir_prior=0.5):
        self.K = K          # number of topics
        self.N = N          # vocabulary size
        self.J = J          # num labels
        self.V = np.zeros(self.K)

        #for even p
        self.V[0] = 1./self.K
        for k in xrange(1,K-1):
            self.V[k] = (1./self.K)/np.prod(1.-self.V[:k])
        self.V[self.K-1] = 1.

        self.p = self.getP(self.V)
        self.alpha = 5.
        self.alpha_1 = 1    #prior for alpha
        self.alpha_2 = 1e-3    #prior for alpha
        self.beta = 5.
        self.beta_1 = 1
        self.beta_2 = 1e-3
        self.dir_prior = dir_prior
        self.gamma = np.random.gamma(shape=1, scale=1, size=[self.N, self.K]) + self.dir_prior
        self.c_a_max_step = 10
        
        self.is_plot = False
        self.is_verbose = True
        self.is_compute_lb = True
        self.ll_diff_frac = 1e-3

    def runVariationalEM(self, max_iter, corpus, directory, prediction=True, logger=None):

        if self.is_plot:
            import matplotlib.pyplot as plt
            plt.ion()
        lbs = list()

        curr = time.clock()
        
        best_micro = 0

        for iter in xrange(max_iter):
            lb = 0
            lb += self.update_C(corpus)
            lb += self.update_Z(corpus)
            lb += self.newton_W(corpus)
            lb += self.update_V(corpus)
            self.update_alpha()
            self.update_beta(corpus)
            if corpus.heldout_ids != None and prediction:
                acc, acc_pn, conf  = self.heldout_prediction(corpus)
                micro, macro, f1, prc, rcl = utils.get_f1_from_confusion(conf)

                mmsum = micro+macro

                print '%d iter, %d topics, %.2f time, %.2f lower_bound %.3f accuracy %.3f accuracy_pn' % (iter, self.K, time.clock()-curr, lb, micro, macro)
                if mmsum > best_micro:
                    self.save_result(directory+'_best', corpus, prediction)
                    best_micro = mmsum

            elif corpus.heldout_ids != None and not prediction:
                perp = self.heldout_perplexity(corpus)
                print '%d iter, %d topics, %.2f time, %.2f lower_bound %.3f perplexity' % (iter, self.K, time.clock()-curr, lb, perp)
                if logger:
                    logger.write('%d,%d,%f,%f,%f,%f\n'%(iter, self.K, self.dir_prior, time.clock()-curr, lb, perp))
                
            elif corpus.heldout_ids == None:
                print '%d iter, %d topics, %.2f time, %.2f lower_bound' % (iter, self.K, time.clock()-curr, lb)    
                
            if iter > 0:
                lbs.append(lb)
                if self.is_plot:
                    plt.close
                    plt.plot(lbs)
                    plt.draw()

            if iter > 30:
                if (abs(lbs[-1] - lbs[-2])/abs(lbs[-2])) < self.ll_diff_frac :
                    break

        return lbs

    def getStickLeft(self, V):
        stl = np.ones(self.K)
        stl[1:] = np.cumprod(1.-V)[:-1]
        return stl

    def getP(self, V):
        one_v = np.ones(self.K)
        one_v[1:] = (1.-V)[:-1]
        p = V * np.cumprod(one_v)
        return p

    #update per word v.d. phi (c denoted by z in the icml paper)
    def update_C(self, corpus):

        corpus.phi_doc = np.zeros([corpus.M, self.K])
        psiGamma = psi(self.gamma)
        gammaSum = np.sum(self.gamma,0)
        psiGammaSum = psi(np.sum(self.gamma, 0))
        lnZ = psi(corpus.A) - np.log(corpus.B)
        Z = corpus.A/corpus.B

        #entropy of q(eta)
        lb = 0
        if(self.is_compute_lb):
            lb += -np.sum(gammaln(gammaSum)) + np.sum(gammaln(self.gamma)) - np.sum((self.gamma - 1)*(psiGamma - psiGammaSum))
            #expectation of eta over variational q(eta)
            lb += self.K * gammaln(self.dir_prior*self.N) - self.K * self.N * gammaln(self.dir_prior) - np.sum((self.dir_prior-1)*(psiGamma-psiGammaSum))


        self.gamma = np.zeros([self.N, self.K]) + self.dir_prior     #multinomial topic distribution prior

        for m in xrange(corpus.M):
            ids = corpus.word_ids[m]
            cnt = corpus.word_cnt[m]

            # C = len(ids) x K
            E_ln_eta = psiGamma[ids,:] - psiGammaSum
            C = np.exp(E_ln_eta + lnZ[m,:])
            C = C/np.sum(C,1)[:,np.newaxis]

            self.gamma[ids,:] += cnt[:,np.newaxis] * C
            corpus.phi_doc[m,:] = np.sum(cnt[:,np.newaxis] * C,0)

            #expectation of p(X) over variational q
            lb += np.sum(cnt[:,np.newaxis] * C * E_ln_eta)
            #entropy of q(C)
            lb -= np.sum(cnt[:,np.newaxis] * C * np.log(C+eps))
            #expectation of p(C) over variational q
            lb += np.sum(cnt[:,np.newaxis] * C * (lnZ[m,:] - np.log(np.sum(Z[m,:]))) )

        if self.is_verbose:
            print 'p(x,c)-q(c) %f' %lb

        return lb

    #update variational gamma prior a and b for Z_mk  (z denoted by \pi in the icml paper)
    def update_Z(self, corpus):
        lb = 0
        bp = self.beta*self.p

        corpus.A = bp + corpus.phi_doc
        # taylor approximation on E[\sum lnZ]
        xi = np.sum(corpus.A/corpus.B, 1)
        E_exp_wr = np.exp(np.dot(corpus.R, corpus.w))
        E_wr = np.dot(corpus.R,corpus.w)        # M x K

        corpus.B = E_exp_wr + (corpus.Nm / xi)[:,np.newaxis]

        # expectation of p(Z)            
        lb += np.sum(-bp * E_wr + (bp-1)*(psi(corpus.A)-np.log(corpus.B)) - E_exp_wr*(corpus.A/corpus.B) - gammaln(bp))

        # entropy of q(Z)
        lb -= np.sum(corpus.A*np.log(corpus.B) + (corpus.A-1)*(psi(corpus.A) - np.log(corpus.B)) - corpus.A - gammaln(corpus.A))
        if self.is_verbose:
            print 'p(z)-q(z) %f' %lb
        return lb

    def newton_W(self, corpus):
        lb = 0
        bp = self.beta * self.p
        Z = corpus.A/corpus.B
        lnZ = psi(corpus.A)-np.log(corpus.B)       

        for ki in np.random.permutation(corpus.K):

            E_exp_wr = np.exp(np.dot(corpus.R, corpus.w))       # M x K
            E_wr = np.dot(corpus.R,corpus.w)        # M x K

            det_w = np.zeros([self.J])
            H = np.zeros([self.J,self.J])

            new_second = corpus.R*(E_exp_wr[:,ki][:,np.newaxis])*(Z[:,ki][:,np.newaxis])      # M x J
            det_w = np.sum(bp[ki]*corpus.R - new_second, 0) - corpus.w[:,ki]      # with normal prior mean 0 and variance 1

            H = - np.dot(new_second.T, corpus.R) - np.identity(self.J) # - identity for normal
            # for ji in xrange(corpus.J):
            #     H[:,ji] = np.sum(- corpus.R * new_second[:,ji][:,np.newaxis], 0)

                # second = corpus.R[:,ji]*E_exp_wr[:,ki]*Z[:,ki]     # M-dim
                # det_w[ji] = np.sum(bp[ki]*corpus.R[:,ji] - second) # - 2.0 * corpus.w[ji,ki] # normal prior

                # for ji2 in xrange(corpus.J):
                #     H[ji2,ji] = np.sum(- corpus.R[:,ji2] * corpus.R[:,ji] * E_exp_wr[:,ki]*Z[:,ki])


            invH = np.linalg.inv(H)

            corpus.w[:,ki] = corpus.w[:,ki] - np.dot(invH, det_w)

        E_exp_wr = np.exp(np.dot(corpus.R, corpus.w))       # M x K
        E_wr = np.dot(corpus.R,corpus.w)        # M x K
        
        lb = np.sum(-bp * E_wr + (bp-1)*(lnZ) - E_exp_wr*(Z))
        
        if self.is_verbose:
            print 'p(w)-q(w) %f, max %f, min %f' % (lb, np.max(corpus.w), np.min(corpus.w))

        return lb


    #coordinate ascent for w_jk
    def update_W(self, corpus):
        lb = 0
        bp = self.beta * self.p
        Z = corpus.A/corpus.B
        lnZ = psi(corpus.A)-np.log(corpus.B)

        for iter in xrange(10):
            E_exp_wr = np.exp(np.dot(corpus.R, corpus.w))       # M x K
            E_wr = np.dot(corpus.R,corpus.w)        # M x K

            old_lb = np.sum(-bp * E_wr  - E_exp_wr*(Z))

            del_w = np.zeros([corpus.J, self.K])
            for ji in xrange(corpus.J):
                for ki in xrange(corpus.K):
                    del_w[ji,ki] = np.sum(bp[ki]*corpus.R[:,ji] - corpus.R[:,ji]*E_exp_wr[:,ki]*Z[:,ki])

            stepsize = 1.0/np.max(np.abs(del_w))
            steps = np.logspace(-10,0)

            ll = list()

            for si in xrange(len(steps)):
                step = steps[si]
                new_w = corpus.w + step * stepsize * del_w
                E_exp_wr = np.exp(np.dot(corpus.R, new_w))
                E_wr = np.dot(corpus.R,new_w)        # M x K
                new_lb = np.sum(-bp * E_wr - E_exp_wr*(Z))
                if np.isnan(new_lb):
                    break
                ll.append(new_lb)

            ll = np.array(ll)
            idx = ll.argsort()[::-1][0]
            
            corpus.w = corpus.w + steps[idx]*stepsize*del_w

            print '\t%d w old new diff %f \t %f \t %f \t%f \t%f \t%f' %(iter, (ll[idx] - old_lb), stepsize, np.max(np.abs(del_w)), np.max(np.abs(del_w))*stepsize, np.max(corpus.w), np.min(corpus.w) )
            if np.abs(ll[idx] - old_lb) < 0.1:
                break

        lb = np.sum(-bp * E_wr + (bp-1)*(lnZ) - E_exp_wr*(Z))
        
        if self.is_verbose:
            print 'p(w)-q(w) %f' % lb

        return lb

    #coordinate ascent for V
    def update_V(self, corpus):
        lb = 0 

        sumLnZ = np.sum(psi(corpus.A) - np.log(corpus.B), 0)        # K dim

        tmp = np.dot(corpus.R, corpus.w)  # M x K
        sum_r_w = np.sum(tmp, 0)
        assert len(sum_r_w) == self.K

        for i in xrange(self.c_a_max_step):
            one_V = 1-self.V
            stickLeft = self.getStickLeft(self.V)       # prod(1-V_(dim-1))
            p = self.V * stickLeft

            psiV = psi(self.beta * p)

            vVec = - self.beta*stickLeft*sum_r_w + self.beta*stickLeft*sumLnZ - corpus.M*self.beta*stickLeft*psiV;

            for k in xrange(self.K):
                tmp1 = self.beta*sum(sum_r_w[k+1:]*p[k+1:]/one_V[k]);
                tmp2 = self.beta*sum(sumLnZ[k+1:]*p[k+1:]/one_V[k]);
                tmp3 = corpus.M*self.beta*sum(psiV[k+1:]*p[k+1:]/one_V[k]);
                vVec[k] = vVec[k] + tmp1 - tmp2;
                vVec[k] = vVec[k] + tmp3;
                vVec[k] = vVec[k] 
            vVec[:self.K-2] -= (self.alpha-1)/one_V[:self.K-2];
            vVec[self.K-1] = 0;
            step_stick = self.getstepSTICK(self.V,vVec,sum_r_w,sumLnZ,self.beta,self.alpha,corpus.M);
            self.V = self.V + step_stick*vVec;
            self.p = self.getP(self.V)

        lb += self.K*gammaln(self.alpha+1) - self.K*gammaln(self.alpha) + np.sum((self.alpha-1)*np.log(1-self.V[:self.K-1]))
        if self.is_verbose:
            print 'p(V)-q(V) %f' % lb
        return lb

    def update_alpha(self):
        old = (self.K-1) * gammaln(self.alpha + 1) - (self.K-1) * gammaln(self.alpha) + np.sum(self.alpha*(1-self.V[:-1])) + self.alpha_1*np.log(self.alpha_2) + (self.alpha_1 - 1)*np.log(self.alpha) - self.alpha_2 * self.alpha - gammaln(self.alpha_1)

        self.alpha = (self.K + self.alpha_1 -2)/(self.alpha_2 - np.sum(np.log(1-self.V[:-1]+eps)))

        new = (self.K-1) * gammaln(self.alpha + 1) - (self.K-1) * gammaln(self.alpha) + np.sum(self.alpha*(1-self.V[:-1]))  + self.alpha_1*np.log(self.alpha_2) + (self.alpha_1 - 1)*np.log(self.alpha) - self.alpha_2 * self.alpha - gammaln(self.alpha_1)

        if self.is_verbose:
            print 'new alpha = %.2f, %.2f' % (self.alpha, (new-old))

    def update_beta(self, corpus):
        E_wr = np.dot(corpus.R, corpus.w)       #M x K
        lnZ = psi(corpus.A) - np.log(corpus.B)

        first = self.p * E_wr

        # since beta does not change a lot, this way is more efficient
        candidate = np.linspace(-1, 1, 31)
        f = np.zeros(len(candidate))
        for i in xrange(len(candidate)):
            step = candidate[i]
            new_beta = self.beta + self.beta*step
            if new_beta < 0:
                f[i] = -np.inf
            else:
                bp = new_beta * self.p
                f[i] = np.sum(new_beta * first) + np.sum((bp - 1) * lnZ) - np.sum(corpus.M * gammaln(bp))
        best_idx = f.argsort()[-1]
        maxstep = candidate[best_idx]
        self.beta += self.beta*maxstep

        if self.is_verbose:
            print 'new beta = %.2f, %.2f' % (self.beta, candidate[best_idx])


    # get stick length to update the gradient
    def getstepSTICK(self,curr,grad,sumMu,sumlnZ,beta,alpha,M):
        _curr = curr[:len(curr)-1]
        _grad = grad[:len(curr)-1]
        _curr = _curr[_grad != 0]
        _grad = _grad[_grad != 0]

        step_zero = -_curr/_grad
        step_one = (1-_curr)/_grad
        min_zero = 1
        min_one = 1
        if(np.sum(step_zero>=0) > 0):
            min_zero = min(step_zero[step_zero>=0])
        if(np.sum(step_one>=0) > 0):
            min_one = min(step_one[step_one>=0])
        max_step = min([min_zero,min_one])

        if max_step > 0:
            step_check_vec = np.array([.01, .125, .25, .375, .5, .625, .75, .875 ])*max_step;
        else:
            step_check_vec = list();

        f = np.zeros(len(step_check_vec));
        for ite in xrange(len(step_check_vec)):
            step_check = step_check_vec[ite];
            vec_check = curr + step_check*grad;
            p = self.getP(vec_check)
            f[ite] = -np.sum(beta*p*sumMu) - M*np.sum(gammaln(beta*p)) + np.sum((beta*p-1)*sumlnZ) + (alpha-1.)*np.sum(np.log(1.-vec_check[:-1]+eps))

        if len(f) != 0:
            b = f.argsort()[-1]
            step = step_check_vec[b]
        else:
            step = 0;

        if b == 0:
            rho = .5;
            bool = 1;
            fold = f[b];
            while bool:
                step = rho*step;
                vec_check = curr + step*grad;
                tmp = np.zeros(vec_check.size)
                tmp[1:] = vec_check[:-1]
                p = vec_check * np.cumprod(1-tmp)
                fnew =  -np.sum(beta*p*sumMu) - M*np.sum(gammaln(beta*p)) + np.sum((beta*p-1)*sumlnZ) + (alpha-1.)*np.sum(np.log(1.-vec_check[:-1]+eps))
                if fnew > fold:
                    fold = fnew
                else:
                    bool = 0
            step = step/rho
        return step

    def write_top_words(self, corpus, filepath):
        with open(filepath, 'w') as f:
            posterior_topic_count = np.sum(self.gamma, 0)
            topic_rank = posterior_topic_count.argsort()[::-1]

            for ti in topic_rank:
                top_words = corpus.vocab[self.gamma[:,ti].argsort()[::-1][:20]]
                f.write( '%d,%f' % (ti, self.p[ti]) )
                for word in top_words:
                    f.write(',' + word)
                f.write('\n')

    def write_label_top_words(self, corpus, filepath):
        
        bp = self.beta * self.p

        with open(filepath, 'w') as f, open(filepath+'all.csv', 'w') as f2:
            mean = corpus.w
            for li in xrange(corpus.J):
                for ki in xrange(corpus.K):
                    top_words = corpus.vocab[self.gamma[:,ki].argsort()[::-1][:20]]
                    f2.write('%s,%d,%f' % (corpus.label_names[li].replace(',',' '), ki, mean[li,ki]*bp[ki]))
                    for word in top_words:
                        f2.write(',' + word)
                    f2.write('\n')
                min_topic = mean[li,:].argsort()[0]
                max_topic = mean[li,:].argsort()[-1]
                top_words = corpus.vocab[self.gamma[:,min_topic].argsort()[::-1][:20]]
                f.write('min,%s,%f'%(corpus.label_names[li].replace(',',' '), mean[li,min_topic] ))
                for word in top_words:
                    f.write(',' + word)
                f.write('\n')

                f.write('max,%s,%f'%(corpus.label_names[li].replace(',',' '), mean[li,max_topic] ))
                top_words = corpus.vocab[self.gamma[:,max_topic].argsort()[::-1][:20]]
                for word in top_words:
                    f.write(',' + word)
                f.write('\n')

    def save_result(self, folder, corpus, prediction):
        import os, cPickle
        if not os.path.exists(folder):
            os.mkdir(folder)
        np.savetxt(folder+'/final_w.csv', corpus.w, delimiter=',')
        np.savetxt(folder+'/final_V.csv', self.V, delimiter=',')
        np.savetxt(folder+'/gamma.csv', self.gamma, delimiter=',')
        np.savetxt(folder+'/A.csv',corpus.A, delimiter=',')
        np.savetxt(folder+'/B.csv',corpus.A, delimiter=',')
        self.write_top_words(corpus, folder + '/final_top_words.csv')
        self.write_label_top_words(corpus, folder + '/final_label_top_words.csv')
        #cPickle.dump([self,corpus], open(folder+'/model_corpus.pkl','w'))
        if prediction:
            acc, pn_acc, conf = self.heldout_prediction(corpus)
            micro, macro, f1, prc, rcl = utils.get_f1_from_confusion(conf)        
            with open(folder+'/acc.txt', 'w') as f:
                f.write('%f\n'%acc)
                f.write('%f\n'%pn_acc)
                f.write('%f\n'%micro)
                f.write('%f\n'%macro)
            np.savetxt(folder+'/conf.csv', conf, delimiter=',')

    def heldout_perplexity(self, corpus):
        num_hdoc = len(corpus.heldout_ids)
        topic = self.gamma/np.sum(self.gamma, 0)

        mean = corpus.w
        bp = self.beta * self.p

        perp = 0
        cnt_sum = 0

        wr = np.dot(corpus.heldout_responses, corpus.w) # m x k

        for di in xrange(num_hdoc):
            doc = corpus.heldout_ids[di]
            cnt = corpus.heldout_cnt[di]
            Z = np.zeros(self.K)

            Z = bp / np.exp(wr[di,:])
            Z /= np.sum(Z)

            if np.sum(cnt) != 0:
                perp -= np.sum(np.log(np.dot(topic[doc,:], Z) + eps) * cnt)
                cnt_sum += np.sum(cnt)

        return np.exp(perp/cnt_sum)

class hdsp_corpus:
    def __init__(self, vocab, word_ids, word_cnt, K, labels, label_names = None, heldout_ids = None, heldout_cnt = None, heldout_responses = None):
        # type of word_ids[0] and word_cnt[0] is np.array
        if type(vocab) == type(list()):
            self.vocab = np.array(vocab)
        self.word_ids = word_ids
        self.word_cnt = word_cnt
        self.R = labels        # M x J matrix
        self.K = K      #num topics
        self.N = len(vocab)             #num voca
        self.M = len(word_ids)          #num documents
        self.J = labels.shape[1]
        self.A = np.random.gamma(shape=1, scale=1, size=[self.M,self.K])
        self.B = np.random.gamma(shape=1, scale=1, size=[self.M,self.K])
        self.w = np.zeros([self.J, self.K])
        self.r_j = np.sum(self.R, 0)
        self.label_names = label_names

        self.heldout_ids = heldout_ids
        self.heldout_cnt = heldout_cnt
        self.heldout_responses = heldout_responses

        self.Nm = np.zeros(self.M)
        for i in xrange(self.M):
            self.Nm[i] = np.sum(word_cnt[i])
