#
# Helper functions and constants
#
import numpy as np
from math import exp, sqrt, pi
from scipy.integrate import quad
from numpy.polynomial.hermite import hermval
import os
from sympy.functions.combinatorial.numbers import stirling, binomial

#
# Mathematical constants
#
rt_2_pi = sqrt(2*pi)
rt_pi = sqrt(pi)
half_pi = pi/2
rt_2 = sqrt(2)

#
# Precompute norms of Hermite polynomials
#
def compute_hermite_norm(n, p):
    """Compute the p-th norm of the n-th Hermite polynomial."""
    two_to_minus_n_over_2 = 2**(-n/2)
    list_arg = [0]*n + [1]
    def integrand(x):
        # This evaluates the n-th Hermite polynomial at x
        Hn_x = two_to_minus_n_over_2 * hermval(x/rt_2, list_arg)
        return np.abs(Hn_x)**p * exp(-x**2/2)/rt_2_pi
    integral, _ = quad(integrand, -np.inf, np.inf)
    return integral**(1/p)

def compute_norms_array(k, ps):
    """Return a 2D array with dimensions (k, len(ps)).
    Each entry (i,j) is the p-th norm of the i+2-nd Hermite polynomial for
    p=ps[j]."""
    norms_array = np.empty((k, len(ps)))

    for n in range(k):
        for j, p in enumerate(ps):
            norms_array[n, j] = compute_hermite_norm(n+2, p)
    return norms_array

# Precompute norms of 2nd through k+1-st Hermite polynomials
k = 11
ps = [2,3,4,5,6,7]
H_norm = compute_norms_array(k, ps)

#
# Helper functions
#
def savefig(fig, file_name, fig_dir="figs"):
    """Saves figure into specified file inside of fig_dir inside"""
    os.makedirs(fig_dir, exist_ok=True)
    # Trim whitespace around the image
    fig.savefig(os.path.join(fig_dir, file_name),bbox_inches='tight',
                pad_inches = 0.05)

def binom_moment(mom=1,n=1,prob=0.5,stirlings=None,binomials=None):
    """Returns the uncentered mom-th moment of a binomial distribution with
    count n and success probability prob

    Args:
      mom - integer moment >= 1
      n - integer count parameter >= 1
      prob - success probability in [0,1]
      stirlings - If not None, this dictionary, indexed by k, will be
        queried for precomputed values before computing any stirling(mom,k)
        values; if not present, values will be added to this dictionary
      binomials - If not None, this dictionary, indexed by k, will be queried
        for precomputed values before computing any binomial(n,k) values;
        if not present, values will be added to this dictionary
    """
    # Initialize empty dictionaries if necessary
    if stirlings is None:
        stirlings = {}
    if binomials is None:
        binomials = {}
    moment = 0.
    factorial = 1
    for k in range(mom+1):
        # Get needed stirling and binomial number
        # Transform sympy.core.numbers.Float into float
        if k not in stirlings:
            stirlings[k] = float(stirling(mom,k))
        if k not in binomials:
            binomials[k] = float(binomial(n,k))
        moment += binomials[k]*factorial*stirlings[k]*prob**k
        # Update factorial
        factorial *= (k+1)
    return moment



