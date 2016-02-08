import numpy as np
import time
from scipy.special import gammaln, psi

eps = 1e-100


class HDSP:
    """
    hierarchical dirichlet scaling process (hdsp)
    Dongwoo Kim, Alice Oh, 2014

    """

    def __init__(self, n_topic, n_voca, n_label, dir_prior=0.5):
        self.n_topic = n_topic
        self.n_voca = n_voca  # vocabulary size
        self.n_label = n_label  # num labels
        self.V = np.zeros(self.n_topic)

        # for even p
        self.V[0] = 1. / self.n_topic
        for k in xrange(1, n_topic - 1):
            self.V[k] = (1. / self.n_topic) / np.prod(1. - self.V[:k])
        self.V[self.n_topic - 1] = 1.

        self.p = self.getP(self.V)
        self.alpha = 5.
        self.alpha_1 = 1  # prior for alpha
        self.alpha_2 = 1e-3  # prior for alpha
        self.beta = 5.
        self.beta_1 = 1
        self.beta_2 = 1e-3
        self.dir_prior = dir_prior
        self.gamma = np.random.gamma(shape=1, scale=1, size=[self.n_voca, self.n_topic]) + self.dir_prior
        self.c_a_max_step = 3

        self.is_plot = True
        self.is_hdp = False
        self.is_verbose = True
        self.hdp_init_step = 10
        self.ll_diff_frac = 1e-3

    def runVariationalEM(self, max_iter, corpus, isHDP=False):

        if self.is_plot:
            import matplotlib.pyplot as plt
            plt.ion()
        lbs = list()

        for iter in xrange(max_iter):
            lb = 0
            curr = time.clock()
            lb += self.update_C(corpus)
            lb += self.update_Z(corpus, iter)
            if iter >= self.hdp_init_step:
                lb += self.update_W(corpus)
            lb += self.update_V(corpus)
            self.update_alpha()
            self.update_beta(corpus)
            if self.is_verbose and corpus.heldout_ids != None:
                perp = self.heldout_perplexity(corpus, isHDP)
                print('no init %d iter, %d topics, %.2f time, %.2f lower_bound %.3f heldout_perp' % (
                    iter, self.n_topic, time.clock() - curr, lb, perp))
            if self.is_verbose and corpus.heldout_ids == None:
                print('no init %d iter, %d topics, %.2f time, %.2f lower_bound' % (
                    iter, self.n_topic, time.clock() - curr, lb))

            if iter > 0:
                lbs.append(lb)
                if self.is_plot:
                    plt.plot(lbs)
                    plt.draw()

            if iter > 30:
                if (abs(lbs[-1] - lbs[-2]) / abs(lbs[-2])) < self.ll_diff_frac:
                    break

        return lbs

    def getStickLeft(self, V):
        stl = np.ones(self.n_topic)
        stl[1:] = np.cumprod(1. - V)[:-1]
        return stl

    def getP(self, V):
        one_v = np.ones(self.n_topic)
        one_v[1:] = (1. - V)[:-1]
        p = V * np.cumprod(one_v)
        return p

    # update per word v.d. phi
    def update_C(self, corpus):

        corpus.phi_doc = np.zeros([corpus.M, self.n_topic])
        psiGamma = psi(self.gamma)
        gammaSum = np.sum(self.gamma, 0)
        psiGammaSum = psi(np.sum(self.gamma, 0))
        lnZ = psi(corpus.A) - np.log(corpus.B)
        Z = corpus.A / corpus.B

        # entropy of q(eta)
        lb = 0
        if (self.is_compute_lb):
            lb += -np.sum(gammaln(gammaSum)) + np.sum(gammaln(self.gamma)) - np.sum(
                (self.gamma - 1) * (psiGamma - psiGammaSum))
            # expectation of eta over variational q(eta)
            lb += self.n_topic * gammaln(self.dir_prior * self.n_voca) - self.n_topic * self.n_voca * gammaln(self.dir_prior) - np.sum(
                (self.dir_prior - 1) * (psiGamma - psiGammaSum))

        self.gamma = np.zeros([self.n_voca, self.n_topic]) + self.dir_prior  # multinomial topic distribution prior

        for m in xrange(corpus.M):
            ids = corpus.word_ids[m]
            cnt = corpus.word_cnt[m]

            # C = len(ids) x K
            E_ln_eta = psiGamma[ids, :] - psiGammaSum
            C = np.exp(E_ln_eta + lnZ[m, :])
            C = C / np.sum(C, 1)[:, np.newaxis]

            self.gamma[ids, :] += cnt[:, np.newaxis] * C
            corpus.phi_doc[m, :] = np.sum(cnt[:, np.newaxis] * C, 0)

            # expectation of p(X) over variational q
            lb += np.sum(cnt[:, np.newaxis] * C * E_ln_eta)
            # entropy of q(C)
            lb -= np.sum(cnt[:, np.newaxis] * C * np.log(C + eps))
            # expectation of p(C) over variational q
            lb += np.sum(cnt[:, np.newaxis] * C * (lnZ[m, :] - np.log(np.sum(Z[m, :]))))

        if self.is_verbose:
            print('p(x,c)-q(c) %f' % lb)

        return lb

    # update variational gamma prior a and b for Z_mk
    def update_Z(self, corpus, iter):
        lb = 0
        bp = self.beta * self.p

        corpus.A = bp + corpus.phi_doc
        # taylor approximation on E[\sum lnZ]
        xi = np.sum(corpus.A / corpus.B, 1)
        E_inv_w = np.zeros([corpus.M, corpus.K])
        ln_E_w = np.zeros([corpus.M, corpus.K])
        for mi in xrange(corpus.M):
            E_inv_w[mi, :] = np.prod((corpus.w_A / corpus.w_B)[corpus.R[mi, :] == 1, :], 0)
            ln_E_w[mi, :] = np.sum((np.log(corpus.w_B) - psi(corpus.w_A)) * corpus.R[mi, :][:, np.newaxis], 0)

        if iter < self.hdp_init_step:
            corpus.B = 1. + (corpus.Nm / xi)[:, np.newaxis]
        else:
            corpus.B = E_inv_w + (corpus.Nm / xi)[:, np.newaxis]

        # expectation of p(Z)            
        lb += np.sum(
            -bp * ln_E_w + (bp - 1) * (psi(corpus.A) - np.log(corpus.B)) - E_inv_w * (corpus.A / corpus.B) - gammaln(
                bp))

        # entropy of q(Z)
        lb -= np.sum(
            corpus.A * np.log(corpus.B) + (corpus.A - 1) * (psi(corpus.A) - np.log(corpus.B)) - corpus.A - gammaln(
                corpus.A))
        if self.is_verbose:
            print('p(z)-q(z) %f' % lb)
        return lb

    # coordinate ascent for w_jk
    def update_W(self, corpus):
        lb = 0
        bp = self.beta * self.p

        corpus.w_A = np.outer(corpus.r_j, bp) + corpus.w_a_prior
        E_z = corpus.A / corpus.B

        # use precomputation
        pre_E_inv_w = np.zeros([corpus.M, corpus.K])
        for mi in xrange(corpus.M):
            pre_E_inv_w[mi, :] = np.prod((corpus.w_A / corpus.w_B)[corpus.R[mi, :] == 1, :], 0)

        for i in xrange(self.c_a_max_step):
            for j in np.random.permutation(corpus.J):
                # for j in xrange(corpus.J):
                oldAdivB = corpus.w_A[j, :] / corpus.w_B[j, :]
                pre_E_inv_w[corpus.R[:, j] == 1, :] /= oldAdivB

                newB = (corpus.w_b_prior + np.sum(corpus.R[:, j][:, np.newaxis] * pre_E_inv_w * E_z, 0))
                # recovery
                pre_E_inv_w[corpus.R[:, j] == 1, :] *= oldAdivB
                # update
                pre_E_inv_w[corpus.R[:, j] == 1, :] *= corpus.w_B[j, :]
                # if j==0:
                #     newB *= (corpus.w_A[0,0]-1.)/newB[0]
                corpus.w_B[j, :] = newB
                pre_E_inv_w[corpus.R[:, j] == 1, :] /= corpus.w_B[j, :]

        # E[log P(Z) + log P(W) - log Q(W)]
        lb += np.sum(
            corpus.w_a_prior * np.log(corpus.w_b_prior) - gammaln(corpus.w_a_prior) - (corpus.w_a_prior + 1) * (
            np.log(corpus.w_B) - psi(corpus.w_A)) - corpus.w_b_prior * (corpus.w_A / corpus.w_B))
        lb -= np.sum(corpus.w_A * np.log(corpus.w_B) - gammaln(corpus.w_A) - (corpus.w_A + 1) * (
        np.log(corpus.w_B) - psi(corpus.w_A)) - corpus.w_B * (corpus.w_A / corpus.w_B))
        if self.is_verbose:
            mean = np.mean(corpus.w_B / (corpus.w_A - 1))
            print('W mean %f' % mean)
            print('p(w)-q(w) %f' % lb)
        return lb

    # coordinate ascent for V
    def update_V(self, corpus):
        lb = 0

        # bp = self.beta * self.p
        # lnZm = psi(corpus.A) - np.log(corpus.B)

        sumLnZ = np.sum(psi(corpus.A) - np.log(corpus.B), 0)  # K dim

        E_ln_w = np.log(corpus.w_B) - psi(corpus.w_A)
        E_w_inv = corpus.w_A / corpus.w_B
        sum_r_w = np.zeros(self.n_topic)
        E_w_inv_doc_prod = np.zeros([corpus.M, self.n_topic])
        sum_E_w_inv = np.zeros(self.n_topic)

        for mi in xrange(corpus.M):
            sum_r_w += np.sum(E_ln_w[corpus.R[mi, :] == 1, :], 0)
            sum_E_w_inv += np.prod(E_w_inv[corpus.R[mi, :] == 1, :], 0)
            E_w_inv_doc_prod[mi, :] = np.prod(E_w_inv[corpus.R[mi, :] == 1, :], 0)

        # old_ll = -np.sum(sum_r_w * bp) + np.sum((bp-1)*lnZm) - np.sum(corpus.M*gammaln(bp)) + np.sum((self.alpha-1)*np.log(1-self.V[:self.K-1]))

        for i in xrange(self.c_a_max_step):
            one_V = 1 - self.V
            stickLeft = self.getStickLeft(self.V)  # prod(1-V_(dim-1))
            p = self.V * stickLeft

            psiV = psi(self.beta * p)

            vVec = - self.beta * stickLeft * sum_r_w + self.beta * stickLeft * sumLnZ - corpus.M * self.beta * stickLeft * psiV;

            for k in xrange(self.n_topic):
                tmp1 = self.beta * sum(sum_r_w[k + 1:] * p[k + 1:] / one_V[k]);
                tmp2 = self.beta * sum(sumLnZ[k + 1:] * p[k + 1:] / one_V[k]);
                tmp3 = corpus.M * self.beta * sum(psiV[k + 1:] * p[k + 1:] / one_V[k]);
                vVec[k] = vVec[k] + tmp1 - tmp2;
                vVec[k] = vVec[k] + tmp3;
                vVec[k] = vVec[k]
            vVec[:self.n_topic - 2] -= (self.alpha - 1) / one_V[:self.n_topic - 2];
            vVec[self.n_topic - 1] = 0;
            step_stick = self.getstepSTICK(self.V, vVec, sum_r_w, sumLnZ, self.beta, self.alpha, corpus.M);
            self.V = self.V + step_stick * vVec;
            self.p = self.getP(self.V)

        # bp = self.beta * self.p
        # new_ll = -np.sum(sum_r_w * bp) + np.sum((bp-1)*lnZm) - np.sum(corpus.M*gammaln(bp)) + np.sum((self.alpha-1)*np.log(1-self.V[:self.K-1]))

        lb += self.n_topic * gammaln(self.alpha + 1) - self.n_topic * gammaln(self.alpha) + np.sum(
            (self.alpha - 1) * np.log(1 - self.V[:self.n_topic - 1]))
        if self.is_verbose:
            print('p(V)-q(V) %f' % lb)
        return lb

    def update_alpha(self):
        old = (self.n_topic - 1) * gammaln(self.alpha + 1) - (self.n_topic - 1) * gammaln(self.alpha) + np.sum(
            self.alpha * (1 - self.V[:-1])) + self.alpha_1 * np.log(self.alpha_2) + (self.alpha_1 - 1) * np.log(
            self.alpha) - self.alpha_2 * self.alpha - gammaln(self.alpha_1)

        self.alpha = (self.n_topic + self.alpha_1 - 2) / (self.alpha_2 - np.sum(np.log(1 - self.V[:-1] + eps)))

        new = (self.n_topic - 1) * gammaln(self.alpha + 1) - (self.n_topic - 1) * gammaln(self.alpha) + np.sum(
            self.alpha * (1 - self.V[:-1])) + self.alpha_1 * np.log(self.alpha_2) + (self.alpha_1 - 1) * np.log(
            self.alpha) - self.alpha_2 * self.alpha - gammaln(self.alpha_1)

        if self.is_verbose:
            print('new alpha = %.2f, %.2f' % (self.alpha, (new - old)))

    def update_beta(self, corpus):
        ElogW = np.log(corpus.w_B) - psi(corpus.w_A)
        lnZ = psi(corpus.A) - np.log(corpus.B)

        first = np.zeros([corpus.M, self.n_topic])
        for mi in xrange(corpus.M):
            first[mi, :] = - self.p * np.sum(ElogW[corpus.R[mi, :] == 1, :], 0)
        # first_sum = np.sum(first)
        # second = np.sum(lnZ * self.p)

        # for i in xrange(1):
        #     last = - corpus.M * np.sum(self.p*psi(self.beta * self.p))
        #     gradient = first_sum + second + last
        #     gradient /= corpus.M * np.sum(self.p * self.p * psi(self.beta*self.p))
        #     step = self.getstepBeta(gradient, self.beta, first, lnZ, self.p, corpus)
        #     self.beta += step*gradient

        # since beta does not change a lot, this way is more efficient
        candidate = np.linspace(-1, 1, 31)
        f = np.zeros(len(candidate))
        for i in xrange(len(candidate)):
            step = candidate[i]
            new_beta = self.beta + self.beta * step
            if new_beta < 0:
                f[i] = -np.inf
            else:
                bp = new_beta * self.p
                f[i] = np.sum(new_beta * first) + np.sum(bp * lnZ) - np.sum(corpus.M * gammaln(bp))
        best_idx = f.argsort()[-1]
        maxstep = candidate[best_idx]
        self.beta += self.beta * maxstep

        if self.is_verbose:
            print('new beta = %.2f, %.2f' % (self.beta, candidate[best_idx]))

    def getstepBeta(self, beta, betaVec, first, lnZ, p, corpus):
        maxstep = 1

        if betaVec < 0:
            maxstep = -beta / betaVec

        if maxstep > 1:
            maxstep = 1

        # Get stepsize checkpoints
        if maxstep > 0:
            step_check_vec = np.array([0, 0.01, .125, .35, .5, .685, .875]) * maxstep
        else:
            step_check_vec = 0

        # Calculate objective for each stepsize
        f = np.zeros(len(step_check_vec))
        for ite in xrange(len(step_check_vec)):
            step_check = step_check_vec[ite]
            beta_check = beta + step_check * betaVec
            bp = beta_check * p
            f[ite] = np.sum(beta_check * first) + np.sum(bp * lnZ) - np.sum(corpus.M * gammaln(bp))

        # Pick best stepsize
        if len(f) != 0:
            b = f.argsort()[-1]
            step = step_check_vec[b]
        else:
            step = 0
        return step

    # get stick length to update the gradient
    def getstepSTICK(self, curr, grad, sumMu, sumlnZ, beta, alpha, M):
        _curr = curr[:len(curr) - 1]
        _grad = grad[:len(curr) - 1]
        _curr = _curr[_grad != 0]
        _grad = _grad[_grad != 0]

        step_zero = -_curr / _grad
        step_one = (1 - _curr) / _grad
        min_zero = 1
        min_one = 1
        if (np.sum(step_zero > 0) > 0):
            min_zero = min(step_zero[step_zero > 0])
        if (np.sum(step_one > 0) > 0):
            min_one = min(step_one[step_one > 0])
        max_step = min([min_zero, min_one]);

        if max_step > 0:
            step_check_vec = np.array([.01, .125, .25, .375, .5, .625, .75, .875]) * max_step;
        else:
            step_check_vec = list();

        f = np.zeros(len(step_check_vec));
        for ite in xrange(len(step_check_vec)):
            step_check = step_check_vec[ite];
            vec_check = curr + step_check * grad;
            p = self.getP(vec_check)
            f[ite] = -np.sum(beta * p * sumMu) - M * np.sum(gammaln(beta * p)) + np.sum((beta * p - 1) * sumlnZ) + (
                                                                                                                   alpha - 1.) * np.sum(
                np.log(1. - vec_check[:-1] + eps))

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
                step = rho * step;
                vec_check = curr + step * grad;
                tmp = np.zeros(vec_check.size)
                tmp[1:] = vec_check[:-1]
                p = vec_check * np.cumprod(1 - tmp)
                fnew = -np.sum(beta * p * sumMu) - M * np.sum(gammaln(beta * p)) + np.sum((beta * p - 1) * sumlnZ) + (
                                                                                                                     alpha - 1.) * np.sum(
                    np.log(1. - vec_check[:-1] + eps))
                if fnew > fold:
                    fold = fnew
                else:
                    bool = 0
            step = step / rho
        return step

    def write_top_words(self, corpus, filepath):
        with open(filepath, 'w') as f:
            for ti in xrange(corpus.K):
                top_words = corpus.vocab[self.gamma[:, ti].argsort()[::-1][:20]]
                f.write('%d,%f' % (ti, self.p[ti]))
                for word in top_words:
                    f.write(',' + word)
                f.write('\n')

    def write_label_top_words(self, corpus, filepath):
        bp = self.beta * self.p
        with open(filepath, 'w') as f:
            with open(filepath + 'all.csv', 'w') as f2:
                mean = corpus.w_B / (corpus.w_A - 1. + eps)
                for li in xrange(corpus.J):
                    for ki in xrange(corpus.K):
                        top_words = corpus.vocab[self.gamma[:, ki].argsort()[::-1][:20]]
                        f2.write('%s,%d,%f' % (corpus.label_names[li].replace(',', ' '), ki, mean[li, ki] * bp[ki]))
                        for word in top_words:
                            f2.write(',' + word)
                        f2.write('\n')
                    min_topic = mean[li, :].argsort()[0]
                    max_topic = mean[li, :].argsort()[-1]
                    top_words = corpus.vocab[self.gamma[:, min_topic].argsort()[::-1][:20]]
                    f.write('min,%s,%f' % (corpus.label_names[li].replace(',', ' '), mean[li, min_topic]))
                    for word in top_words:
                        f.write(',' + word)
                    f.write('\n')

                    f.write('max,%s,%f' % (corpus.label_names[li].replace(',', ' '), mean[li, max_topic]))
                    top_words = corpus.vocab[self.gamma[:, max_topic].argsort()[::-1][:20]]
                    for word in top_words:
                        f.write(',' + word)
                    f.write('\n')

    def save_result(self, folder, corpus):
        import os, cPickle
        if not os.path.exists(folder):
            os.mkdir(folder)
        np.savetxt(folder + '/final_w_a.csv', corpus.w_A, delimiter=',')
        np.savetxt(folder + '/final_w_b.csv', corpus.w_B, delimiter=',')
        np.savetxt(folder + '/final_V.csv', self.V, delimiter=',')
        np.savetxt(folder + '/gamma.csv', self.gamma, delimiter=',')
        self.write_top_words(corpus, folder + '/final_top_words.csv')
        self.write_label_top_words(corpus, folder + '/final_label_top_words.csv')
        cPickle.dump([self, corpus], open(folder + '/model_corpus.pkl', 'w'))

    def heldout_perplexity(self, corpus, isHDP):
        num_hdoc = len(corpus.heldout_ids)
        topic = self.gamma / np.sum(self.gamma, 0)
        if isHDP:
            mean = np.ones(corpus.w_B.shape)
        else:
            mean = corpus.w_B / (corpus.w_A - 1)
        bp = self.beta * self.p

        perp = 0
        cnt_sum = 0

        for di in xrange(num_hdoc):
            doc = corpus.heldout_ids[di]
            cnt = corpus.heldout_cnt[di]
            Z = np.zeros(self.n_topic)

            if isHDP:
                w = np.ones(self.n_topic)
            else:
                w = np.prod(mean[corpus.heldout_responses[di, :] == 1, :], 0)

            Z = bp * w
            # for k in xrange(self.K):
            #     Z[k] = np.random.gamma(bp[k], w[k])

            Z /= np.sum(Z)

            if np.sum(cnt) != 0:
                perp -= np.sum(np.log(np.dot(topic[doc, :], Z) + eps) * cnt)
                cnt_sum += np.sum(cnt)

        return np.exp(perp / cnt_sum)

    def in_document_heldout_LL(self, corpus):
        assert len(corpus.heldout_ids) == len(corpus.word_ids)

        topic = self.gamma / np.sum(self.gamma, 0)  # NxK
        LL = 0.
        cnt_sum = 0.
        Z = corpus.A / corpus.B  # MxK
        Z /= np.sum(Z, 1)[:, np.newaxis]
        for di in xrange(corpus.M):
            doc = corpus.heldout_ids[di]
            cnt = corpus.heldout_cnt[di]
            if np.sum(cnt) != 0:
                LL += np.sum(np.log(np.dot(topic[doc, :], Z[di, :]) + eps) * cnt)
                cnt_sum += np.sum(cnt)
        return LL / cnt_sum


class hdsp_corpus:
    def __init__(self, vocab, word_ids, word_cnt, K, labels, label_names=None, heldout_ids=None, heldout_cnt=None,
                 heldout_responses=None):
        # type of word_ids[0] and word_cnt[0] is np.array
        if type(vocab) == type(list()):
            self.vocab = np.array(vocab)
        self.word_ids = word_ids
        self.word_cnt = word_cnt
        self.R = labels  # M x J matrix
        self.K = K  # num topics
        self.N = len(vocab)  # num voca
        self.M = len(word_ids)  # num documents
        self.J = labels.shape[1]
        self.A = np.random.gamma(shape=1, scale=1, size=[self.M, self.K])
        self.B = np.random.gamma(shape=1, scale=1, size=[self.M, self.K])
        self.w_a_prior = 2
        self.w_b_prior = 1
        self.w_A = np.zeros([self.J, self.K]) + self.w_a_prior
        self.w_B = np.zeros([self.J, self.K]) + self.w_b_prior
        self.r_j = np.sum(self.R, 0)
        self.label_names = label_names

        self.heldout_ids = heldout_ids
        self.heldout_cnt = heldout_cnt
        self.heldout_responses = heldout_responses

        self.Nm = np.zeros(self.M)
        for i in xrange(self.M):
            self.Nm[i] = np.sum(word_cnt[i])
