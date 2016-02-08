import numpy as np
import time
from scipy.special import gammaln, psi

eps = 1e-100


class Corpus:
    def __init__(self, vocab, word_ids, word_cnt, n_topic):
        self.vocab = np.array(vocab)
        self.word_ids = word_ids
        self.word_cnt = word_cnt
        self.n_topic = n_topic  # num topics
        self.n_voca = len(vocab)
        self.n_doc = len(word_ids)
        self.A = np.random.gamma(shape=1, scale=1, size=[self.n_doc, self.n_topic])
        self.B = np.random.gamma(shape=1, scale=1, size=[self.n_doc, self.n_topic])
        self.mu = np.zeros([self.n_doc, self.n_topic])
        self.sigma = np.ones([self.n_doc, self.n_topic])

        self.Nm = np.zeros(self.n_doc)
        for i in xrange(self.n_doc):
            self.Nm[i] = np.sum(word_cnt[i])

class DILN:
    """
    The Discrete Infinite Logistic Normal Distribution (DILN),
    Paisley, John and Wang, Chong and Blei, David, 2011

    Attributes
    ----------
    n_topic: int
        number of truncated topics for variational inference
    n_voca: int
        vocabulary size
    """

    def __init__(self, n_topic, n_voca):
        self.n_topic = n_topic
        self.n_voca = n_voca  # vocabulary size
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
        self.dir_prior = 0.5
        self.mean = np.zeros(self.n_topic)
        self.Kern = np.identity(self.n_topic)
        self.invKern = np.linalg.inv(self.Kern)
        self.gamma = np.random.gamma(shape=1, scale=1, size=[self.n_voca, self.n_topic]) + self.dir_prior
        self.c_a_max_step = 5
        self.is_compute_lb = True

    def fit(self, corpus, max_iter=100):
        """ Run variational EM to fit the model

        Parameters
        ----------
        max_iter: int
            maximum number of iterations
        corpus:

        Returns
        -------

        """

        lbs = list()

        for iter in xrange(max_iter):
            lb = 0
            curr = time.clock()
            lb += self.update_C(corpus, False)
            lb += self.update_Z(corpus)
            lb += self.update_W(corpus)
            lb += self.update_V(corpus)
            # self.update_alpha()
            # self.update_beta(corpus)
            self.update_mean_Kernel(corpus)
            print('%d iter, %.2f time, %.2f lower_bound' % (iter, time.clock() - curr, lb))

            if iter > 3:
                lbs.append(lb)
                if iter > 5:
                    if (abs(lbs[-1] - lbs[-2]) / abs(lbs[-2])) < 1e-5:
                        break
                    if (lbs[-1] < lbs[-2]):
                        break

    def update_mean_Kernel(self, corpus):
        self.mean = np.mean(corpus.mu, 0)
        self.Kern = (np.dot((corpus.mu - self.mean).T, (corpus.mu - self.mean)) + np.diag(
            np.sum(corpus.sigma, 0))) / corpus.M

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
    def update_C(self, corpus, is_heldout):

        corpus.phi_doc = np.zeros([corpus.M, self.n_topic])
        psiGamma = psi(self.gamma)
        gammaSum = np.sum(self.gamma, 0)
        psiGammaSum = psi(np.sum(self.gamma, 0))
        lnZ = psi(corpus.A) - np.log(corpus.B)
        Z = corpus.A / corpus.B

        lb = 0
        if (self.is_compute_lb):
            # expectation of p(eta) over variational q(eta)
            l1 = self.n_topic * gammaln(self.dir_prior * self.n_voca) - self.n_topic * self.n_voca * gammaln(self.dir_prior) - np.sum(
                (self.dir_prior - 1) * (psiGamma - psiGammaSum))
            lb += l1
            # entropy of q(eta)
            l2 = np.sum(gammaln(gammaSum)) - np.sum(gammaln(self.gamma)) + np.sum(
                (self.gamma - 1) * (psiGamma - psiGammaSum))
            lb -= l2

        if not is_heldout:
            self.gamma = np.zeros([self.n_voca, self.n_topic]) + self.dir_prior  # multinomial topic distribution prior

        for m in xrange(corpus.M):
            ids = corpus.word_ids[m]
            cnt = corpus.word_cnt[m]

            # C = len(ids) x K
            E_ln_eta = psiGamma[ids, :] - psiGammaSum
            C = np.exp(E_ln_eta + lnZ[m, :])
            C = C / np.sum(C, 1)[:, np.newaxis]

            if not is_heldout:
                self.gamma[ids, :] += cnt[:, np.newaxis] * C
            corpus.phi_doc[m, :] = np.sum(cnt[:, np.newaxis] * C, 0)

            if (self.is_compute_lb):
                # expectation of p(X) over variational q
                lb += np.sum(cnt[:, np.newaxis] * C * E_ln_eta)
                # expectation of p(C) over variational q
                l1 = np.sum(cnt[:, np.newaxis] * C * (lnZ[m, :] - np.log(np.sum(Z[m, :]))))
                lb += l1
                # entropy of q(C)
                l2 = np.sum(cnt[:, np.newaxis] * C * np.log(C + eps))
                lb -= l2

        # print ' E[p(eta,C,X)]-E[q(eta,C)] = %f' % lb

        return lb

    # update variational gamma prior a and b for Z_mk
    def update_Z(self, corpus):
        lb = 0
        bp = self.beta * self.p
        # for m in xrange(corpus.M):
        #     xi = np.sum(corpus.A[m,:]/corpus.B[m,:])
        #     corpus.A[m,:] = bp + corpus.phi_doc[m,:]
        #     corpus.B[m,:] = np.exp(-corpus.mu[m,:] + 0.5*corpus.sigma[m,:]) + sum(corpus.word_cnt[m]) / xi

        xi = np.sum(corpus.A / corpus.B, 1)  # m dim
        corpus.A = bp + corpus.phi_doc
        corpus.B = np.exp(-corpus.mu + 0.5 * corpus.sigma) + (corpus.Nm / xi)[:, np.newaxis]

        if (self.is_compute_lb):
            # expectation of p(Z)
            E_ln_Z = psi(corpus.A) - np.log(corpus.B)
            l1 = np.sum(-bp * corpus.mu) + np.sum((bp - 1) * (E_ln_Z)) - np.sum(
                np.exp((-corpus.mu + 0.5 * corpus.sigma)) * corpus.A / corpus.B) - corpus.M * np.sum(gammaln(bp))
            lb += l1
            # entropy of q(Z)
            l2 = np.sum(corpus.A * np.log(corpus.B)) + np.sum((corpus.A - 1) * (E_ln_Z)) - np.sum(corpus.A) - np.sum(
                gammaln(corpus.A))
            lb -= l2
            # print ' E[p(Z)]-E[q(Z)] = %f' % lb

        return lb

    # coordinate ascent for w_mk
    def update_W(self, corpus):
        lb = 0
        bp = self.beta * self.p

        # oldlb = -np.sum(bp*corpus.mu) + np.sum((bp - 1)*(psi(corpus.A) - np.log(corpus.B))) - np.sum(np.exp(-corpus.mu + 0.5*corpus.sigma) * corpus.A/corpus.B)
        # oldlb += -0.5*np.sum(np.diag(np.dot(np.dot(corpus.mu-self.mean, self.invKern), (corpus.mu-self.mean).T ))) - 0.5 * np.sum(np.diag(self.Kern) * corpus.sigma) + 0.5 * np.sum(np.log(corpus.sigma))

        adivb = corpus.A / corpus.B
        for m in xrange(corpus.M):
            gradMU = - bp + (adivb[m, :]) * np.exp(-corpus.mu[m, :] + 0.5 * corpus.sigma[m, :]) - np.dot(self.invKern, (
            corpus.mu[m, :] - self.mean))
            gradV = -0.5 * (adivb[m, :]) * np.exp(-corpus.mu[m, :] + 0.5 * corpus.sigma[m, :]) - .5 * np.diag(
                self.invKern) + .5 / corpus.sigma[m, :]
            stepsize = self.getstepMUV(corpus.mu[m, :], corpus.sigma[m, :], gradMU, gradV, bp, adivb[m, :], self.mean,
                                       self.invKern)
            corpus.mu[m, :] += stepsize * gradMU;
            gradV *= stepsize
            gradV[gradV > 200] = 200
            corpus.sigma[m, :] += gradV;

        self.mean = np.mean(corpus.mu, 0)
        self.Kern = (np.dot((corpus.mu - self.mean).T, corpus.mu - self.mean) + np.diag(
            np.sum(corpus.sigma, 0))) / corpus.M
        self.invKern = np.linalg.inv(self.Kern)

        if self.is_compute_lb:
            # lb += -np.sum(bp*corpus.mu) + np.sum((bp - 1)*(psi(corpus.A) - np.log(corpus.B))) - np.sum(np.exp(-corpus.mu + 0.5*corpus.sigma) * corpus.A/corpus.B)
            # expectation of p(w) given variational parameter
            l1 = -0.5 * np.sum(
                np.diag(np.dot(np.dot(corpus.mu - self.mean, self.invKern), (corpus.mu - self.mean).T))) - 0.5 * np.sum(
                np.diag(self.Kern) * corpus.sigma)
            lb += l1
            # entropy of q(w)
            l2 = -0.5 * np.sum(np.log(corpus.sigma))
            lb -= l2
            # print ' E[p(w)]-E[q(w)] = %f' % lb

        return lb

    # coordinate ascent for V
    def update_V(self, corpus):
        lb = 0
        # bp = self.beta*self.p
        # old_ll = self.K*gammaln(self.alpha+1) - self.K*gammaln(self.alpha) + np.sum((self.alpha-1)*np.log(1-self.V[:self.K-1]))
        # old_ll += np.sum(-bp*corpus.mu) 
        # old_ll += np.sum((bp-1)*(psi(corpus.A) - np.log(corpus.B))) 
        # old_ll -= np.sum(gammaln(bp))

        for i in xrange(self.c_a_max_step):
            one_V = 1 - self.V
            sumMu = np.sum(corpus.mu, 0)  # K dim
            sumLnZ = np.sum(psi(corpus.A) - np.log(corpus.B), 0)  # K dim
            stickLeft = self.getStickLeft(self.V)  # prod(1-V_(dim-1))
            p = self.V * stickLeft

            psiV = psi(self.beta * p)

            vVec = - self.beta * stickLeft * sumMu + self.beta * stickLeft * sumLnZ - corpus.M * self.beta * stickLeft * psiV;

            for k in xrange(self.n_topic):
                tmp1 = self.beta * sum(sumMu[k + 1:] * p[k + 1:] / one_V[k]);
                tmp2 = self.beta * sum(sumLnZ[k + 1:] * p[k + 1:] / one_V[k]);
                tmp3 = corpus.M * self.beta * sum(psiV[k + 1:] * p[k + 1:] / one_V[k]);
                vVec[k] = vVec[k] + tmp1 - tmp2;
                vVec[k] = vVec[k] + tmp3;
                vVec[k] = vVec[k]
            vVec[:self.n_topic - 2] -= (self.alpha - 1) / one_V[:self.n_topic - 2];
            vVec[self.n_topic - 1] = 0;
            step_stick = self.getstepSTICK(self.V, vVec, sumMu, sumLnZ, self.beta, self.alpha, corpus.M);
            self.V = self.V + step_stick * vVec;
            self.p = self.getP(self.V)

        # bp = self.beta*self.p
        # new_ll = self.K*gammaln(self.alpha+1) - self.K*gammaln(self.alpha) + np.sum((self.alpha-1)*np.log(1-self.V[:self.K-1]))
        # new_ll += np.sum(-bp*corpus.mu) 
        # new_ll += np.sum((bp-1)*(psi(corpus.A) - np.log(corpus.B))) 
        # new_ll -= np.sum(gammaln(bp))

        if self.is_compute_lb:
            # expectation of p(V)
            lb += (self.n_topic - 1) * gammaln(self.alpha + 1) - (self.n_topic - 1) * gammaln(self.alpha) + np.sum(
                (self.alpha - 1) * np.log(1 - self.V[:-1]))
            # print ' E[p(V)]-E[q(V)] = %f' % lb

        # print '%f diff     %f' % (new_ll - old_ll, lb)
        return lb

    def update_alpha(self):
        self.alpha = (eps + self.n_topic - 2) / (self.alpha_2 - np.sum(np.log(1 - self.V[:-1] + eps)))

    def update_beta(self, corpus):
        lb = 0
        return lb

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
            step_check_vec = np.array([0., .01, .125, .25, .375, .5, .625, .75, .875]) * max_step;
        else:
            step_check_vec = list();

        f = np.zeros(len(step_check_vec));
        for ite in xrange(len(step_check_vec)):
            step_check = step_check_vec[ite];
            vec_check = curr + step_check * grad;
            p = self.getP(vec_check)
            f[ite] = -np.sum(beta * p * sumMu) - M * np.sum(gammaln(beta * p)) + np.sum((beta * p - 1) * sumlnZ)\
                     + (alpha - 1.) * np.sum(np.log(1. - vec_check[:-1] + eps))

        if len(f) != 0:
            b = f.argsort()[-1]
            step = step_check_vec[b]
        else:
            step = 0;

        if b == 1:
            rho = .5;
            bool = 1;
            fold = f[b];
            while bool:
                step = rho * step;
                vec_check = curr + step * grad;
                tmp = np.zeros(vec_check.size)
                tmp[1:] = vec_check[:-1]
                p = vec_check * np.cumprod(1 - tmp)
                fnew = -np.sum(beta * p * sumMu) - M * np.sum(gammaln(beta * p)) + np.sum((beta * p - 1) * sumlnZ) \
                       + (alpha - 1.) * np.sum(np.log(1. - vec_check[:-1] + eps))
                if fnew > fold:
                    fold = fnew
                else:
                    bool = 0
            step = step / rho
        return step

    # get stick length to update the gradient
    def getstepMUV(self, currMu, currV, vecMu, vecV, bp, AdivB, u, invKern):
        stepsize = 0
        steps = -currV[vecV != 0] / vecV[vecV != 0]
        isbound = np.sum(steps > 0) > 0
        maxstep2 = 0
        if np.sum(steps[steps > 0]) > 0:
            maxstep2 = np.min(steps[steps > 0]);
        if np.sum(steps >= 0) > 0:
            maxstep = min(steps[steps > 0])
            maxstep = min([maxstep, 1])
        else:
            maxstep = 1

        if maxstep > 0:
            step_check_vec = np.array([0., .01, .125, .25, .375, .5, .625, .75, .875]) * maxstep;
        else:
            step_check_vec = list();

        f = np.zeros(len(step_check_vec));
        for ite in xrange(len(step_check_vec)):
            step_check = step_check_vec[ite]
            mu_check = currMu + step_check * vecMu;
            v_check = currV + step_check * vecV;
            v_check[v_check > 200] = 200

            f[ite] = - np.sum(mu_check * bp) - np.sum(AdivB * np.exp(-mu_check + .5 * v_check)) - .5 * np.dot(
                (mu_check - u), np.dot(invKern, (mu_check - u))) - np.dot(.5 * np.diag(invKern), v_check) + .5 * np.sum(
                np.log(v_check + eps))

        if len(f) != 0:
            b = f.argsort()[-1];
            stepsize = step_check_vec[b];
        else:
            stepsize = 0

        if b == len(step_check_vec):
            rho = 1.5
            bool = 1
            fold = f(b);
            while bool:
                stepsize = rho * stepsize
                if isbound:
                    if stepsize > maxstep2:
                        bool = 0;
                        break
                mu_check = currMu + stepsize * vecMu
                v_check = currV + stepsize * vecV;
                v_check[v_check > 200] = 200
                fnew = - np.sum(mu_check * bp) - np.sum(AdivB * np.exp(-mu_check + .5 * v_check)) - .5 * np.dot(
                    (mu_check - u), np.dot(invKern, (mu_check - u))) - np.dot(.5 * np.diag(invKern),
                                                                              v_check) + .5 * np.sum(
                    np.log(v_check + eps))
                if fnew > fold:
                    fold = fnew
                else:
                    bool = 0
            stepsize = stepsize / rho

        if b == 1:
            rho = .5;
            bool = 1;
            fold = f[b];
            while bool:
                stepsize = rho * stepsize;
                mu_check = currMu + stepsize * vecMu;
                v_check = currV + stepsize * vecV;
                v_check[v_check > 200] = 200
                fnew = - np.sum(mu_check * bp) - np.sum(AdivB * np.exp(-mu_check + .5 * v_check)) - .5 * np.dot(
                    (mu_check - u), np.dot(invKern, (mu_check - u))) - np.dot(.5 * np.diag(invKern),
                    v_check) + .5 * np.sum(np.log(v_check + eps))
                if fnew > fold:
                    fold = fnew
                else:
                    bool = 0
            stepsize = stepsize / rho

        return stepsize

    def write_top_words(self, corpus, filepath):
        with open(filepath, 'w') as f:
            for ti in xrange(corpus.K):
                top_words = corpus.vocab[self.gamma[:, ti].argsort()[::-1][:20]]
                f.write('%d,%f' % (ti, self.p[ti]))
                for word in top_words:
                    f.write(',' + word)
                f.write('\n')

    def write_corr_topics(self, corpus, filepath, thr=-1e100):
        with open(filepath, 'w') as f:
            Kern = self.Kern

            for ti in xrange(self.n_topic - 1):
                for ki in xrange(ti + 1, self.n_topic):
                    if Kern[ti, ki] > thr:
                        f.write(str(ti) + ',' + str(ki) + ',' + str(Kern[ti, ki]))
                        top = corpus.vocab[self.gamma[:, ti].argsort()[::-1][:5]]
                        for word in top:
                            f.write(',' + word)
                        top = corpus.vocab[self.gamma[:, ki].argsort()[::-1][:5]]
                        f.write(',|')
                        for word in top:
                            f.write(',' + word)
                        f.write('\n')

    def save_result(self, folder, corpus):
        import os, cPickle
        if not os.path.exists(folder):
            os.mkdir(folder)
        np.savetxt(folder + '/final_mu.csv', corpus.mu, delimiter=',')
        np.savetxt(folder + '/final_sigma.csv', corpus.sigma, delimiter=',')
        np.savetxt(folder + '/final_mean.csv', self.mean, delimiter=',')
        np.savetxt(folder + '/final_K.csv', self.Kern, delimiter=',')
        np.savetxt(folder + '/final_V.csv', self.V, delimiter=',')
        self.write_top_words(corpus, folder + '/final_top_words.csv')
        self.write_corr_topics(corpus, folder + '/final_corr_topics.csv')
        cPickle.dump(self, open(folder + '/model.pkl', 'w'))

