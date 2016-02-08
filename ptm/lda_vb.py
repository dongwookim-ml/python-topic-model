import numpy as np
from scipy.special import gammaln, psi

eps = 0.01


def dirichlet_expectation(alpha):
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(np.sum(alpha)))
    return (psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


class vbLDA:
    """
    Latent dirichlet allocation,
    Blei, David M and Ng, Andrew Y and Jordan, Michael I, 2003
    
    Latent Dirichlet allocation with mean field variational inference
    """

    def __init__(self, vocab, K, wordids, wordcts, alpha=0.1, eta=0.01):
        """
        Arguments: 
        vocab: Dictionary mapping from words to integer ids.
        K:
        wordids:
        wordcts:
        alpha:
        eta:
        """
        self.vocab = vocab
        self.n_voca = len(vocab)
        self.n_topic = K
        self.n_doc = len(wordids)
        self.alpha = alpha
        self.eta = eta

        self._wordids = wordids
        self._wordcts = wordcts
        self._lambda = 1 * np.random.gamma(100., 1. / 100, (self.n_topic, self.n_voca))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        self.gamma_iter = 5
        self.gamma = 1 * np.random.gamma(100., 1. / 100, (self.n_doc, self.n_topic))

    def do_e_step(self):
        """
        compute approximate topic distribution of each document and each word
        """
        # self.gamma = 1*np.random.gamma(100.,1./100, (self._D, self._K))
        # random initialize gamma
        Elogtheta = dirichlet_expectation(self.gamma)
        expElogtheta = np.exp(Elogtheta)

        # sufficient statistics to update lambda
        sstats = np.zeros(self._lambda.shape)

        for d in range(0, self.n_doc):
            ids = self._wordids[d]
            cts = np.array(self._wordcts[d])
            gammad = self.gamma[d, :]

            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

            for it in xrange(self.gamma_iter):
                lastgamma = gammad

                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)

                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

                meanchange = np.mean(abs(gammad - lastgamma))

                if (meanchange < eps):
                    break

            self.gamma[d, :] = gammad
            sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return (self.gamma, sstats)

    def do_m_step(self, isComputeBound=True):
        """
        estimate topic distribution based on computed approx. topic distribution
        """
        (gamma, sstats) = self.do_e_step()

        self._lambda = self.eta + sstats
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)

        bound = 0
        if isComputeBound:
            bound = self.approx_bound(gamma)

        return (gamma, bound)

    def approx_bound(self, gamma):
        """
        Compute lower bound of the corpus
        """

        score = 0
        Elogtheta = dirichlet_expectation(gamma)

        # E[log p(docs | theta, beta)]
        for d in range(0, self.n_doc):
            ids = self._wordids[d]
            cts = np.array(self._wordcts[d])
            phinorm = np.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
            score += np.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self.alpha - gamma) * Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self.alpha))
        score += sum(gammaln(self.alpha * self.n_topic) - gammaln(np.sum(gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + np.sum((self.eta - self._lambda) * self._Elogbeta)
        score = score + np.sum(gammaln(self._lambda) - gammaln(self.eta))
        score = score + np.sum(gammaln(self.eta * self.n_voca) - gammaln(np.sum(self._lambda, 1)))

        return (score)


if __name__ == '__main__':
    # test
    import nltk_corpus

    voca, wordids, wordcnt = nltk_corpus.get_reuters_cnt_ids(num_doc=100, max_voca=1000)

    max_iter = 50
    num_topic = 5

    model = vbLDA(voca, num_topic, wordids, wordcnt)
    for i in xrange(max_iter):
        (gamma, bound) = model.do_m_step()
        print(i, bound)
