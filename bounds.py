#
# Standard tail and quantile bounds
#
from math import pi, sqrt, exp, log, inf
from scipy.stats import norm
from scipy.optimize import bisect
from bentkus import bentkus, QFunc
from numpy import floor, square
import numpy as np

#
# Standard tail bounds
#
norm_sf = norm.sf
def asymptotic_tail(t, sig=1, two_sided=True):
    """Returns P(|G| >= t) (two-sided) or P(G >= t) (one-sided) for
    G normally distributed with mean zero and variance sig^2"""
    delta = norm_sf(t/sig)
    if two_sided: delta *= 2
    return delta

def hoeffding_tail(t, R=1, two_sided=True):
    """Returns Hoeffding upper bound on P(|S| > t) or P(S > t)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - EW_i
    with each independent and identically distributed W_i in [0, R]."""
    # Source: Theorem 1 (2.3) of Hoeffding, Wassily (1963).
    # "Probability inequalities for sums of bounded random variables"
    delta = exp( - ((t/R)**2) *2 )
    if two_sided: delta *= 2
    return delta

def bernstein_tail(t, n, sig=1, R=1, two_sided=True):
    """Returns Bernstein upper bound on P(|S| > t) or P(S > t)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i] with each
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Each |W_i - E[W_i]| <= R_s almost surely
    R_s = deviation_bound(sig, R)
    # Source: Cor. 2.11 of Boucheron, Lugosi, and Massart.
    # "Concentration Inequalities A Nonasymptotic Theory of Independence"
    delta = exp( - (t**2) / 2 / ((sig ** 2) + R_s * t / 3 / sqrt(n)) )
    if two_sided: delta *= 2
    return delta

def berry_esseen_penalty(n, sig=1, R=1, two_sided=True):
    """Returns Berry-Esseen two-sided or one-sided bound on the uniform distance
    between the Normal(0, sig^2) distribution function and the distribution
    function of any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i] with each
    independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # The Berry-Esseen penalty is in terms of the ratio E[|W_1 - EW_1|^3]/sig^3
    # <= any almost sure bound Rstilde on the deviation |W_1 - EW_1|/sig
    Rstilde = deviation_bound(sig, R)/sig

    # Source: Shevtsova, Irina (2011). "On the absolute constants in the
    # Berry Esseen type inequalities for identically distributed summands"
    penalty = min(.3328 * ( Rstilde + .429 ),
                  .33554 * ( Rstilde + .415 )) / sqrt(n)
    return 2*penalty if two_sided else penalty

def berry_esseen_tail(t, n, sig=1, R=1, two_sided=True):
    """Returns Berry-Esseen upper bound on P(|S| > t) or P(S > t)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    return (asymptotic_tail(t, sig=sig, two_sided=two_sided) +
            berry_esseen_penalty(n, sig=sig, R=R, two_sided=two_sided))

def nonuniform_berry_esseen_penalty(n, sig=1, R=1, two_sided=True):
    """Returns coefficient of the non-uniform Berry-Esseen two-sided or
    one-sided bound on the uniform distance between the Normal(0, sig^2)
    distribution function and the distribution function of any sum
    S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # The Berry-Esseen penalty is in terms of the ratio E[|W_1 - EW_1|^3]/sig^3
    # <= any almost sure bound Rstilde on the deviation |W_1 - EW_1|/sig
    Rstilde = deviation_bound(sig, R)/sig

    # Source: p. 54 of Shevtsova, Irina (2017). "On the absolute constants in
    # Nagaev–Bikelis-type inequalities."
    penalty = min(Rstilde * 17.36, Rstilde * 15.70 + 0.646) / sqrt(n)
    return 2*penalty if two_sided else penalty

def nonuniform_berry_esseen_tail(t, n, sig=1, R=1, two_sided=True):
    """Returns non-uniform Berry-Esseen upper bound on P(|S| > t) or P(S > t)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Source: p. 54 of Shevtsova, Irina (2017). "On the absolute constants in
    # Nagaev–Bikelis-type inequalities."
    return (asymptotic_tail(t, sig=sig, two_sided=two_sided) +
            nonuniform_berry_esseen_penalty(n,sig=sig,R=R,two_sided=two_sided)
            / (1+abs(t/sig))**3)

def unidentical_berry_esseen_tail(t, n, sig=1, third=1, two_sided=True):
    """Returns unidentical Berry-Esseen upper bound on P(|S| > t) or P(S > t)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n Y_i
    with independent Y_i satisfying E[Y_i] = 0, (1/n) sum_i Var(Y_i) >= sig^2,
    and (1/n) sum_i E|Y_i|^3 <= third
    """
    # Source: Thm. 1 of Shevtsova, Irina (2012). "An Improvement of Convergence
    # Rate Estimates in the Lyapunov Theorem."
    delta = (asymptotic_tail(t, sig=sig, two_sided=False) +
             .56 * third / (sig ** 3) / sqrt(n))
    if two_sided: delta *= 2
    return delta

def unidentical_bernstein_tail(t, n, sig=1, R=1, two_sided=True):
    """Returns Bernstein upper bound on P(|S| > t) or P(S > t)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n Y_i
    with independent Y_i satisfying E[Y_i] = 0, (1/n) sum_i Var(Y_i) <= sig^2,
    and |Y_i| <= R almost surely."""
    # Source: Cor. 2.11 of Boucheron, Lugosi, and Massart.
    # "Concentration Inequalities A Nonasymptotic Theory of Independence"
    delta = exp( - (t**2) / 2 / ((sig ** 2) + R * t / 3 / sqrt(n)) )
    if two_sided: delta *= 2
    return delta

def bentkus_tail(t, n, sig=1, R=1, two_sided=True):
    """Returns Bentkus upper bound on P(|S| > t) or P(S > t)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Source: Theorem 2.1 of Bentkus et al. (2006), 
    # On domination of tail probabilities of (super) martingales: explicit bounds
    
    # Each |W_i - E[W_i]| <= R_s almost surely
    R_s = deviation_bound(sig, R)
    q_func = QFunc(n=n, A=sig, B=R_s)
    delta = q_func.P2(q_func.nb + sqrt(n)*t*(1-q_func.bias)/R_s)
    if two_sided: delta *= 2
    return delta

norm_logsf = norm.logsf
def feller_tail(t, n, sig=1, R=1, two_sided=True):
    """Returns min of 1 and Feller upper bound on P(|S| > t) or P(S > t)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Source: Theorem 1 of Feller (1943),
    # GENERALIZATION OF A PROBABILITY LIMIT THEOREM OF CRAMER
    
    # Each |W_i - E[W_i]| <= R_s almost surely
    R_s = deviation_bound(sig, R)
    c_R_sig = R_s / sig
    u = t/sig
    sqrt_n = sqrt(n)
    denom = (sqrt_n / (12*c_R_sig)) - u
    denom_leq_0 = denom <= 0
    # if denom <= 0:
    #     # Feller bound is infinite
    #     # Return maximum possible tail bound
    #     return 1
    cubic_term = (u**3) / (14 * np.maximum(denom,denom_leq_0))
    exp_arg = norm_logsf(u) + cubic_term
    # if exp_arg >= 0:
    #     # Feller bound is larger than 1
    #     # Return maximum possible tail bound
    #     return 1
    exp_arg2 = cubic_term -(u**2)/2 + log(9 * c_R_sig / sqrt_n)
    # if exp_arg2 >= 0:
    #     # Feller bound is larger than 1
    #     # Return maximum possible tail bound
    #     return 1
    # delta = exp(exp_arg) + exp(exp_arg2)
    delta = np.where(denom_leq_0 | (exp_arg >= 0) | (exp_arg2 >= 0), 
                     1., np.exp(exp_arg) + np.exp(exp_arg2))
    if two_sided: delta *= 2
    return np.minimum(delta, 1)
    # return min(delta, 1)

#
# Standard quantile bounds
#
def deviation_bound(sig, R=1):
    """Returns almost sure bound on |W - EW| when W in [0,R] with
    Var(W) = sig^2"""
    # Source: Austern and Mackey,
    # "Efficient Concentration with Gaussian Approximation"
    return 0.5*(R+sqrt(R**2-4*sig**2))

norm_ppf = norm.ppf
def asymptotic_quantile(delta, sig=1, two_sided=True):
    """Returns Normal(0, sig^2) two-sided (1-delta/2) or one-sided (1-delta)
    quantile"""
    return sig * norm_ppf(min(1,(1-delta/2) if two_sided else (1-delta)))

def hoeffding_quantile(delta, R=1, two_sided=True):
    """Returns Hoeffding two-sided (1-delta/2) or one-sided (1-delta) quantile
    bound for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - EW_i
    with each independent and identically distributed W_i in [0, R]."""
    # Source: Theorem 1 (2.3) of Hoeffding, Wassily (1963).
    # "Probability inequalities for sums of bounded random variables"
    log_delta_term = log(2/delta) if two_sided else log(1/delta)
    return R * sqrt(log_delta_term/2)

def bernstein_quantile(delta, n, sig=1, R=1, two_sided=True):
    """Returns Bernstein two-sided (1-delta/2) or one-sided (1-delta) quantile
    bound for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i] with each
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Each |W_i - E[W_i]| <= R_s almost surely
    R_s = deviation_bound(sig, R)
    # Source: Theorem 2.10 of Boucheron, Lugosi, and Massart.
    # "Concentration Inequalities A Nonasymptotic Theory of Independence"
    log_delta_term = log(2/delta) if two_sided else log(1/delta)
    return (R_s/3 * log_delta_term / sqrt(n) ) + sig * sqrt(2 * log_delta_term)

def berry_esseen_quantile(delta, n, sig=1, R=1, two_sided=True):
    """Returns Berry-Esseen two-sided (1-delta/2) or one-sided (1-delta)
    quantile bound for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    return asymptotic_quantile(
        max(0, delta -
            berry_esseen_penalty(n, sig=sig, R=R, two_sided=two_sided)),
        sig=sig, two_sided=two_sided)

def berry_esseen_penalty_rw(n, sig=1, R=1, two_sided=True):
    """Returns Berry-Esseen two-sided or one-sided bound employed by Romano
    and Wolf on the uniform distance between the standard normal distribution
    function and the distribution function of any sum
    sqrt(n) (1/n) sum_{i=1}^n (W_i - E[W_i])/sig with each
    independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # The Berry-Esseen penalty is in terms of the ratio E[|W_1 - EW_1|^3]/sig^3
    # <= R/sig
    R_over_sig = R/sig
    # Source: Shevtsova, Irina (2011). "On the absolute constants in the
    # Berry Esseen type inequalities for identically distributed summands"
    penalty = min(.3328 * ( R_over_sig + .429 ),
                  .33554 * ( R_over_sig + .415 )) / sqrt(n)
    return 2*penalty if two_sided else penalty

def berry_esseen_quantile_rw(delta, n, sig=1, R=1, two_sided=True):
    """Returns Berry-Esseen two-sided (1-delta/2) or one-sided (1-delta)
    quantile bound employed by Romano and Wolf for any sum
    S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    return asymptotic_quantile(
        max(0, delta -
            berry_esseen_penalty_rw(n, sig=sig, R=R, two_sided=two_sided)),
        sig=sig, two_sided=two_sided)

def nonuniform_berry_esseen_quantile(delta,n,sig=1,R=1,two_sided=True):
    """Returns non-uniform Berry-Esseen two-sided (1-delta/2) or one-sided
    (1-delta) quantile bound for any sum
    S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Find quantile bound by inverting tail bound
    def invert_nonuniform_berry_esseen_tail(delta, tmin, tmax):
        """Returns value of t in [tmin, tmax] for which
        nonuniform_berry_esseen_tail(t,n,sig,R) = delta"""
        g = lambda t: (
            nonuniform_berry_esseen_tail(t,n,sig=sig,R=R,two_sided=two_sided)
            - delta)
        # Use the bisect method to find the root of g within the interval
        # [tmin, tmax]
        return bisect(g, tmin, tmax)

    # Quantile bound will be no smaller than asymptotic quantile
    tmin=asymptotic_quantile(delta, sig=sig, two_sided=two_sided)
    # Since P(G >= t) = P(Z >= t/sig) <= P(|Z|+1 >= 1+t/sig)
    # <= E[(|Z|+1)^3] / (1+|t|/sig)^3 = (4+5*sqrt(2/pi)) / (1+|t|/sig)^3
    # for G ~ N(0, sig^2) and Z ~ N(0, 1), quantile bound will be no larger
    # than the following
    gsn_coeff = 4+5*sqrt(2/pi)
    coeff = (nonuniform_berry_esseen_penalty(n,sig=sig,R=R,two_sided=two_sided)
        + gsn_coeff)
    tmax= sig * ((coeff/delta)**(1./3) - 1)
    return invert_nonuniform_berry_esseen_tail(delta, tmin, tmax)

def emp_bernstein_quantile(delta, n, sighat=1, R=1, two_sided=True):
    """Returns empirical Bernstein two-sided (1-delta/2) or one-sided (1-delta)
    quantile bound for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    empirical variance sighat^2 = (1/n) sum_i (W_i - (1/n)sum_j W_j)^2."""
    # Source: Thm. 4 of Maurer and Pontil,
    # "Empirical Bernstein Bounds and Sample Variance Penalization"
    # Thm. 4 operates on [0,1] so need to multiply boundedness term by R
    log_delta_term = log(4/delta) if two_sided else log(2/delta)
    return ( (7*R/3 * log_delta_term * sqrt(n) / (n-1) ) +
            sighat * sqrt( 2 * log_delta_term * n/(n-1) ) )

def bentkus_quantile(delta,n,sig=1,R=1,two_sided=True):
    """Returns Bentkus two-sided (1-delta/2) or one-sided (1-delta)
    quantile bound for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Source: Theorem 2.1 of Bentkus et al. (2006), 
    # On domination of tail probabilities of (super) martingales: explicit bounds
    
    # Each |W_i - E[W_i]| <= R_s almost surely
    R_s = deviation_bound(sig, R)
    # bentkus returns the quantile bound for sqrt(n) S_n, so divide by sqrt(n)
    return bentkus(n, delta/2 if two_sided else delta, sig, R_s)/sqrt(n)

def kz_sig_est(delta, W, L=0, U=1):
    """Upper confidence bound for the standard deviation of bounded independent variables

    Given independent W[i] in [L, U], returns sig_upper satisfying 
    P(sig <= sig_upper) >= 1-delta for sig the standard deviation of each W[i].

    Source: Equation (33) of
    Kuchibhotla and Zheng, Near-Optimal Confidence Sequences for Bounded Random Variables
    
    Args:
        delta: significance level
        W: An array of bounded independent random variables
        L: Lower bound for the random variables
        U: Upper bound for the random variables
    """
    n = W.size
    half_n = n / 2.0
    floor_half_n = floor(half_n)
    V_n = sum(square(W[1::2]-W[0::2])) / 2
    g = sqrt(n/2) * (U - L) * norm_ppf(1 - 2*delta / exp(2))
    return (g + sqrt(g**2 + 4 * V_n * floor_half_n)) / (2 * floor_half_n)

def emp_bentkus_quantile(delta,W,R=1,two_sided=True):
    """Returns empirical Bentkus two-sided (1-delta/2) or one-sided (1-delta)
    quantile bound for the sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Use fraction of significance level for sigma estimation and remainder for
    # mean estimation
    delta_1 = 2*delta/3
    delta_2 = delta - delta_1
    sig_upper = kz_sig_est(delta_2, W, L=0, U=R)

    # bentkus returns the quantile bound for sqrt(n) S_n, so divide by sqrt(n)
    n = W.size
    return bentkus(n, delta_1/2 if two_sided else delta_1, sig_upper, R)/sqrt(n)

def feller_quantile(delta,n,sig=1,R=1,two_sided=True):
    """Returns Feller two-sided (1-delta/2) or one-sided
    (1-delta) quantile bound for any sum
    S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2."""
    # Each |W_i - E[W_i]| <= R_s almost surely
    R_s = deviation_bound(sig, R)
    # |S_n| <= sqrt_n * R_s almost surely
    sqrt_n = sqrt(n)
    default_bound = sqrt_n * R_s
    # Feller tail bound is only valid for t <= sqrt_n * (sig ** 2) / (12 * R_s)
    tmax = sqrt_n * (sig**2) / (12 * R_s)
    # Quantile bound will be no smaller than asymptotic quantile
    tmin=asymptotic_quantile(delta, sig=sig, two_sided=two_sided)

    # Find quantile bound by inverting tail bound
    g = lambda t: (
        feller_tail(t,n,sig=sig,R=R,two_sided=two_sided) - delta
        )
    # First find the smallest value of this function
    ts = np.linspace(tmin, tmax, 1000000)
    gs = g(ts) 
    imin = np.argmin(gs)
    tmin = ts[imin]
    gmin = gs[imin]
    print(f"gmin={gmin} at t={tmin}", flush=True)
    # Error tolerance
    tol = 1e-6
    if gmin > tol:
        # The target level delta cannot be achieved by this tail bound
        # Return default almost sure bound
        return default_bound
    elif gmin >= 0 and gmin <= tol:
        # Closest value is sufficiently close
        iminabs = np.argmin(np.abs(gs))
        tminabs = ts[iminabs]
        print(f"gminabs={gs[iminabs]} at t={tminabs}", flush=True)
        return tminabs
    else:
        # Find the smallest nonnegative value
        iminpos = np.argmin(np.where(gs >= 0, gs, inf))
        closest_above = ts[iminpos]
        gminpos = gs[iminpos]
        # Find the smallest magnitude negative value
        imaxneg = np.argmax(np.where(gs < 0, gs, -inf))
        closest_below = ts[imaxneg]
        gmaxneg = gs[imaxneg]
        print(f"(gminpos,gmaxneg) = ({gminpos},{gmaxneg}) at t=({closest_above},{closest_below})", flush=True)
        # Search for delta between brackets
        if closest_below > closest_above:
            smaller_closest = closest_above
            larger_closest = closest_below
            if gminpos <= tol:
                # Smaller solution is sufficiently close
                return smaller_closest
        else:
            smaller_closest = closest_below
            larger_closest = closest_above
            if gmaxneg >= -tol:
                # Smaller solution is sufficiently close
                return smaller_closest
        tstar = bisect(g, smaller_closest, larger_closest)
        print(f"bisect g(t)={g(tstar)} at t={tstar}", flush=True)
        return tstar