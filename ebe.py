"""Implementation of the efficient empirical Berry-Esseen bounds of
Austern and Mackey, Efficient Concentration with Gaussian Approximation.
"""
from math import sqrt, log
from scipy.optimize import minimize_scalar
from scipy.stats import kstwo, ksone
from ttictoc import tic,toc
import os
import pickle
from efficient import quantile_bound
from bounds import asymptotic_quantile, hoeffding_quantile
from sd import EBUB, EBLB
import numpy as np
import multiprocess

# Directory for storing eb_sig_est results
eb_sig_est_dir = "eb_sig_est"
os.makedirs(eb_sig_est_dir, exist_ok=True)

def eb_sig_est(delta, W, R=1, verbose=False):
    """Returns upper and lower estimates for the population standard deviation
    sig of n independent and identically distributed random variables W[i].

    The estimates (sig_lower, sig_upper) satisfy sig_lower <= sig <= sig_upper
    with probability at least 1-delta and are based on EBLB and EBUB.
    """
    tic()
    # Normalize variables to [0,1] as expected by EBUB and EBLB
    if R != 1:  W = W / R
    n = W.shape[0]
    half_delta = delta / 2
    ebUB = EBUB(max_rounds=n, alpha=half_delta)
    one_minus_proportion = 1/(1+log(n))
    proportion = 1 - one_minus_proportion
    ebLB = EBLB(max_rounds=n, alpha=proportion*half_delta,
                tilde_alpha=one_minus_proportion*half_delta, tilde_CS=True)
    # Compute upper and lower bounds sequentially as a function of the sample
    # sample sequence
    for Wi in W:
        ebUB(Wi)
        ebLB(Wi)
    sig_lower = ebLB.get_center_minus_radius()
    sig_upper = ebUB.get_center_plus_radius()
    if R != 1: 
        # Scale back to original range
        sig_lower = sig_lower * R
        sig_upper = sig_upper * R
    elapsed = toc()
    if verbose:
        print(f"elapsed: {elapsed}s, [sig_lower,sig_upper]=({sig_lower},{sig_upper})\n")
    return (sig_lower, sig_upper)

def beta_sample(n, a=1., b=1., R=1, seed=0):
    """Returns n independent and identically distributed sample points from
    a Beta(a,b) * R distribution using np.random.default_rng(seed).
    """
    rng_gen = np.random.default_rng(seed)
    W = rng_gen.beta(a, b, size=n) * R
    return W

def ebe(delta, n, sig_ci, p=None, R=1, two_sided=True, verbose=False, cache=None):
    """Empirical Berry-Esseen quantile bound for an iid sum

    Returns q s.t. P(|S| > q) <= delta (if two_sided) or P(S > q) <= delta
    (otherwise) for a sum S = sqrt(n) (1/n) sum_{i=1}^n W[i] - E[W[i]]
    of independent and identically distributed W[i] in [0, R].

    Specifically, returns
      sup_{sig in [sig_lower(prob), sig_upper(prob)]}
        quantile_bound(delta-prob, n, p=p, sig=sig, R)
    where the true population standard deviation sig lies in 
    [sig_lower(prob), sig_upper(prob)] with probability at least 1-prob,
    and prob = delta * asymptotic_quantile(delta,sig=1,two_sided=two_sided)/sqrt_n.

    Args:
      delta - Upper tail probability
      n - Number of sample points W[i]
      sig_ci - Function that takes in a failure probability prob and returns a tuple
        (sig_lower, sig_upper) that satisfies sig_lower <= sig <= sig_upper
        with probability at least 1-prob for the population standard deviation sig.
      p - None (to return minimum bound over all p) or integer Wasserstein
        distance power >= 1
      R - Boundedness parameter
      two_sided - Provide two-sided quantile bound?
      verbose - Display detailed progress information?
      cache - If not None, this dictionary (indexed by candidate sig values)
        of dictionaries (indexed by p) will be queried for
        precomputed values before computing any gsn_wass(n, p, sig, R) values;
        if not present, values will be added to this dictionary
    """
    if cache is None:
        # Initialize empty cache dictionary
        cache = {}

    tic()
    # Assign portion of failure probability to sigma confidence interval
    sqrt_n = sqrt(n)
    prob = delta * asymptotic_quantile(delta,sig=1,two_sided=two_sided)/sqrt_n
    delta_minus_prob = delta - prob

    # Define collection of bounds parameterized by prob in (0,delta)
    half_R = R/2

    # Compute baseline deterministic quantile bound
    hoeff_q = hoeffding_quantile(delta_minus_prob,R,two_sided=two_sided)
    # Compute KS-based quantile bound
    c_prime =  (
        R * kstwo.ppf(1-delta_minus_prob,n) if two_sided else R * ksone.ppf(1-delta_minus_prob,n))
    baseline_bound = min(sqrt_n*c_prime, hoeff_q)

    # Compute 1-prob confidence interval for sigma and empirical standard
    # deviation sighat
    sig_lower, sig_upper = sig_ci(prob)

    if verbose:
        print(f"prob={prob}: [sig_lower, sig_upper] = [{sig_lower}, {sig_upper}]")

    sig_upper = min(sig_upper, half_R)
    sig_lower = min(half_R, max(0., sig_lower))

    # Maximize quantile_bound(delta-prob) over [sig_lower,sig_upper]
    # Terminate optimization early if any bound is worse than alternative
    stop_early = [False]
    def tomin(sig):
        if stop_early[0]:
            return -baseline_bound
        # Reuse any cached gsn_wass results for this candidate
        # sigma value
        if sig not in cache:
            cache[sig] = {}
        q_bd = quantile_bound(delta-prob,n,p=p,sig=sig,R=R,
                              two_sided=two_sided, cache=cache[sig])
        if q_bd >= baseline_bound:
            stop_early[0] = True
        return -q_bd

    mins=minimize_scalar(tomin, bounds=[sig_lower,sig_upper],
                         method='bounded')
    B_bound=-mins.fun
    bound_prob = min(B_bound, baseline_bound)

    if False:
        from matplotlib import pyplot as plt
        plt.rcParams['text.usetex'] = True
        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)

        # Plot log bound as a function of k
        prob_vals = np.linspace(0, delta)
        plt.plot(prob_vals, [log(bound(prob,verbose=False)) for prob in prob_vals])
        plt.ylabel(r'Log bound on $1-\delta$ quantile')
        plt.xlabel(r'Failure probability')
        plt.title(fr'$\delta$ = {delta}, n = {n}, ' r'$\hat{\sigma}$'
                  f' = {np.std(W, ddof=0)}, R = {R}')
        plt.tight_layout()
        plt.show()

    elapsed = toc()
    if verbose:
        print(f"elapsed: {elapsed}s, prob={prob}, bound(prob)={bound_prob}\n")
    return bound_prob

def ebe_beta(delta, n, a=1, b=1, seed=0, recompute=False, p=None, R=1, 
             two_sided=True, verbose=False, cache=None):
    """Empirical Berry-Esseen quantile bound for a sum of variables W[i]
    sampled independently from a Beta(a,b) * R distribution using 
    np.random.default_rng(seed).
    
    Returns q s.t. P(|S| > q) <= delta (if two_sided) or P(S > q) <= delta
    (otherwise) for any sum S = sqrt(n) (1/n) sum_{i=1}^n W[i] - E[W[i]]
    with each independent and identically distributed W[i] in [0, R]
    based on the observed statistics sighat.

    Specifically, returns
      sup_{sig in [sig_lower(prob), sig_upper(prob)]}
        quantile_bound(delta-prob, n, p=p, sig=sig, R)
    where sig in [sig_lower(prob), sig_upper(prob)] with probability at least 1-prob,
    and prob = delta * asymptotic_quantile(delta,sig=1,two_sided=two_sided)/sqrt_n.

    Args:
      delta - Upper tail probability
      n - Number of datapoints W_i
      a - Beta(a,b) distribution shape parameter >= 1
      b - Beta(a,b) distribution shape parameter >= 1
      seed - Random seed for sampling W_i
      recompute - If True, will recompute the sigma confidence intervals
        instead of loading previously computed intervals from disk
      p - None (to return minimum bound over all p) or integer Wasserstein
        distance power >= 1
      R - Boundedness parameter
      two_sided - Provide two-sided quantile bound?
      verbose - Display detailed progress information?
      cache - If not None, this dictionary (indexed by candidate sig values)
        of dictionaries (indexed by p) will be queried for
        precomputed values before computing any gsn_wass(n, p, sig, R) values;
        if not present, values will be added to this dictionary
    """
    # Sample iid sequence
    W = beta_sample(n, a=a, b=b, R=R, seed=seed)

    # Define sigma confidence interval function
    def sig_ci(prob):
        # Check if results have already been saved to disk for these settings
        filename = os.path.join(eb_sig_est_dir,
                                f"delta{prob}-n{n}-a{a}-b{b}-R{R}-seed{seed}.pkl")
        tic()
        if (not recompute) and os.path.isfile(filename):
            with open(filename, "rb") as f:
                res = pickle.load(f)
        else:
            # Otherwise compute from scratch
            res = eb_sig_est(prob, W, R=R, verbose=False)
            # Save result to disk
            with open(filename, "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = toc()
        if verbose:
            print(f"elapsed: {elapsed}s, [sig_lower,sig_upper]=[{res[0]},{res[1]}]\n")
        return res
    
    return ebe(delta, n, sig_ci, p=p, R=R, two_sided=two_sided,
               verbose=verbose, cache=cache)

def ebe_ci(W, delta, R=1, verbose=False):
    """Empirical Berry-Esseen confidence interval for the mean

    Given independent and identically distributed W[i] in [0, R],
    returns an empirical Berry-Esseen confidence interval for mu = E[W[i]]:
    (mu_lower, mu_upper) s.t. P(mu_lower <= mu <= mu_upper) <= delta.

    Args:
      W - array of sample points
      delta - Significance level 
      R - Boundedness parameter
      verbose - Display detailed progress information?
    """
    n = W.shape[0]
    # Define sigma confidence interval function
    def sig_ci(prob):
        tic()
        res = eb_sig_est(prob, W, R=R, verbose=False)
        elapsed = toc()
        if verbose:
            print(f"elapsed: {elapsed}s, [sig_lower,sig_upper]=[{res[0]},{res[1]}]\n")
        return res
    # Get two-sided quantile bound for S_n = sqrt(n)(mean(W) - mu) divided by sqrt(n)
    tic()
    cache = {} # Dictionary for storing cached gsn_wass results
    q = ebe(delta, n, sig_ci, R=R, two_sided=True,
            verbose=verbose, cache=cache) / sqrt(n)
    # Convert into confidence interval
    Wbar = np.mean(W)
    mu_lower = max(0, Wbar - q)
    mu_upper = min(R, Wbar + q)
    elapsed = toc()
    if verbose:
        print(f"elapsed: {elapsed}s, [mu_lower,Wbar,mu_upper]=[{mu_lower},{Wbar},{mu_upper}]\n")
    return (mu_lower, mu_upper)

def ebe_ci_seq(W, ns, delta, R=1, parallel=True, verbose=False):
    """Sequence of empirical Berry-Esseen confidence intervals for the mean

    Given independent and identically distributed W[i] in [0, R],
    returns arrays of lower and upper confidence empirical Berry-Esseen 
    confidence bounds mu_lower and mu_upper for mu = E[W[i]] such that 
    P(mu_lower[j] <= mu <= mu_upper[j]) <= delta
    based on the first ns[j] sample points W[0:ns[j]].

    Args:
      W - array of sample points
      ns - array of sample sizes at which to compute confidence intervals
      delta - Significance level 
      R - Boundedness parameter
      parallel - Compute confidence intervals in parallel?
      verbose - Display detailed progress information?
    """
    def ci_fn(W):
        return ebe_ci(W, delta, R=R, verbose=verbose)

    return get_ci_seq(W, ci_fn, ns, parallel=parallel)

def get_ci_seq(x, ci_fn, times, parallel=False):
    """
    Get sequence of confidence intervals

    Source: https://github.com/gostevehoward/confseq/src/confseq/misc.py

    Parameters
    ----------
    x, array-like
        The vector of observations between 0 and 1.

    ci_fn, univariate function
        A function which takes an array-like of bounded numbers `x`
        and outputs a tuple `(l, u)` of lower and upper confidence
        intervals. Note that `l` and `u` are scalars (not vectors).

    times, array-like of positive integers
        Times at which to compute the confidence interval.

    parallel, boolean
        Should this function be parallelized?

    Returns
    -------
    l, array-like of [0, 1]-valued reals
        Lower confidence intervals

    u, array-like of [0, 1]-valued reals
        Upper confidence intervals
    """
    x = np.array(x)

    l = np.repeat(0.0, len(times))
    u = np.repeat(1.0, len(times))

    if parallel:
        n_cores = len(os.sched_getaffinity(0)) ###multiprocess.cpu_count()
        print("Using " + str(n_cores) + " cores")
        with multiprocess.Pool(n_cores) as p:
            result = np.array(p.map(lambda time: ci_fn(x[0:time]), times))
        l, u = result[:, 0], result[:, 1]
    else:
        for i in np.arange(0, len(times)):
            time = times[i]
            x_t = x[0:time]
            l[i], u[i] = ci_fn(x_t)

    return l, u