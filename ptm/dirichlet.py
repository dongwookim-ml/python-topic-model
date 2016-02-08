import numpy as np
from scipy.stats import dirichlet
from scipy.special import psi, polygamma, gammaln

eps = 1e-100
max_iter = 10
converge_criteria = 0.001


def parameter_estimation(theta, old_alpha):
    """Estimating a dirichlet parameter given a set of multinomial parameters.
    
    Parameters
    ----------

    theta: a set of multinomial, N x K matrix (N = # of observation, K = dimension of dirichlet)
    old_alpha: initial guess on the dirichlet parameter (K-dim)
    """

    log_p_bar = np.mean(np.log(theta), 0)  # sufficient statistics

    for j in xrange(max_iter):
        digamma_alpha = psi(np.sum(old_alpha)) + log_p_bar
        old_alpha = np.exp(digamma_alpha) + 0.5
        old_alpha[old_alpha < 0.6] = 1.0 / (- digamma_alpha[old_alpha < 0.6] + psi(1.))

        for i in xrange(max_iter):
            new_alpha = old_alpha - (psi(old_alpha) - digamma_alpha) / (polygamma(1, old_alpha))
            old_alpha = new_alpha

    return new_alpha


def collapsed_parameter_estimation(z, alpha):
    """Estimating a dirichlet parameter in the collapsed sampling environment of Dir-Mult

    Parameters
    ----------

    z: assignment count, N x K matrix
    alpha:  initial guess on the dirichlet parameter (K-dim)
    """

    N, K = z.shape

    z_sum = np.sum(z, 1)

    converge = False

    cur_iter = 0

    max_iter = 30
    while not converge or (cur_iter <= max_iter):

        random_order = np.random.permutation(K)

        for ti in random_order:
            alpha_sum = np.sum(alpha)
            numerator = 0
            denominator = 0

            for ni in xrange(N):
                numerator += psi(z[ni, ti] + alpha[ti]) - psi(alpha[ti])
                denominator += psi(z_sum[ni] + alpha_sum) - psi(alpha_sum)

            old_val = alpha[ti]
            alpha[ti] = alpha[ti] * (numerator / denominator)

            if alpha[ti] <= 0:
                alpha[ti] = old_val

            new_sum = np.sum(alpha)

            if np.abs(alpha_sum - new_sum) < converge_criteria:
                converge = True
                break

        cur_iter += 1

    return alpha


def test_parameter_estimation():
    N = 100  # number of observations
    K = 50  # dimension of Dirichlet

    _alpha = np.random.gamma(1, 1) * np.random.dirichlet([1.] * K)  # ground truth alpha

    obs = np.random.dirichlet(_alpha, size=N) + eps  # draw N samples from Dir(_alpha)
    obs /= np.sum(obs, 1)[:, np.newaxis]  # renormalize for added eps

    initial_alpha = np.ones(K)  # first guess on alpha

    # estimating
    est_alpha = parameter_estimation(obs, initial_alpha)

    g_ll = 0  # log-likelihood with ground truth parameter
    b_ll = 0  # log-likelihood with initial guess of alpha
    ll = 0  # log-likelihood with estimated parameter
    for i in xrange(N):
        g_ll += dirichlet.logpdf(obs[i], _alpha)
        b_ll += dirichlet.logpdf(obs[i], initial_alpha)
        ll += dirichlet.logpdf(obs[i], est_alpha)

    print('Test with parameter estimation')
    print('likelihood p(obs|_alpha) = %.3f' % g_ll)
    print('likelihood p(obs|initial_alpha) = %.3f' % b_ll)
    print('likelihood p(obs|estimate_alpha) = %.3f' % ll)
    print('likelihood difference = %.3f' % (g_ll - ll))


def test_collapsed_parameter_estimation():
    N = 100  # number of observations
    K = 50  # dimension of Dirichlet
    W = 200  # number of multinomial trials for each observation

    _alpha = np.random.gamma(1, 1) * np.random.dirichlet([1.] * K)  # ground truth alpha

    obs = np.zeros([N, K])
    for i in xrange(N):
        theta = np.random.dirichlet(_alpha)
        obs[i] = np.random.multinomial(W, theta)

    alpha = np.ones(K)
    alpha = collapsed_parameter_estimation(obs, alpha)

    g_ll = 0
    ll = 0
    for i in xrange(N):
        g_ll += gammaln(np.sum(_alpha)) - np.sum(gammaln(_alpha)) \
                + np.sum(gammaln(obs[i] + _alpha)) - gammaln(np.sum(obs[i] + _alpha))
        ll += gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) \
              + np.sum(gammaln(obs[i] + alpha)) - gammaln(np.sum(obs[i] + alpha))

    print('Test with collapsed sampling optimizer')
    print('likelihood p(obs|_alpha) = %.3f' % g_ll)
    print('likelihood p(obs|alpha) = %.3f' % ll)
    print('likelihood difference = %.3f' % (g_ll - ll))


if __name__ == '__main__':
    test_parameter_estimation()
    test_collapsed_parameter_estimation()
