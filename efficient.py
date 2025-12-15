"""Implementation of the efficient concentration inequalities of
Austern and Mackey, Efficient Concentration with Gaussian Approximation.
"""
import numpy as np
from math import pi, sqrt, exp, log, inf, factorial, ceil, asin
from math import e as euler
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import norm, binom
from scipy.special import gammaln, gamma
from scipy.optimize import bisect
from ttictoc import tic,toc

from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from helpers import (
    rt_2, rt_pi, half_pi, rt_2_pi,
    H_norm, binom_moment
)
from bounds import (
    deviation_bound, 
    asymptotic_quantile, berry_esseen_quantile, hoeffding_quantile, 
    bernstein_quantile, bentkus_quantile, nonuniform_berry_esseen_quantile,
    hoeffding_tail, unidentical_bernstein_tail, 
    unidentical_berry_esseen_tail, berry_esseen_tail
)

#
# Wasserstein distance bound
#
def gsn_wass(n, p, sig=1, R=1, Kp=6, verbose=False):
    """Bound on Lp Wasserstein distance between an iid sum and a Gaussian
    with matching mean and variance

    Returns bound on the Lp Wasserstein distance between a zero mean Gaussian
    distribution with variance sig^2 and any sum
    S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2.

    Args:
      n - Number of datapoints W_i
      p - Wasserstein distance power >= 1
      sig - Standard deviation parameter
      R - Boundedness parameter
      Kp - Positive integer truncation parameter in the bound of Austern and
        Mackey (can typically be left at its default value)
      verbose - If True will display minimize_scalar convergence messages
    """
    if sig <= 0: return 0
    if p == 1:
        return deviation_bound(sig,R=R) / sqrt(n)

    p_minus_1 = p-1
    sqrt_p_minus_1 = sqrt(p_minus_1)
    Rtilde = R / sig
    Rtilde_sqd = Rtilde * Rtilde
    Rtilde_to_2_over_p = Rtilde**(2/p)
    Rsmaller=Rtilde-0.5*(Rtilde-sqrt(Rtilde**2-4))
    Rsmaller_sqd = Rsmaller * Rsmaller
    Rsmaller_sqd_minus_1 = Rsmaller_sqd - 1
    p_plus_2 = p+2
    one_over_p = 1/p
    n_to_1_over_p = n ** one_over_p
    sqrt_n = sqrt(n)
    two_sqrt_n = 2 * sqrt_n

    A = sqrt(p_plus_2 * euler / 2) * ( (2 * euler) ** one_over_p )
    Atilde = p_plus_2 * n_to_1_over_p / (two_sqrt_n * Rtilde_to_2_over_p)
    Astar = Rtilde_to_2_over_p * Atilde
    tildeU = (2**one_over_p*Rtilde*Atilde+rt_2*A)

    ceil_p = ceil(p)
    two_over_Rtilde_sqd = 2/Rtilde_sqd

    # Prepare dictionaries to store binomial coefficients for
    # computing binomial moments
    binomials = {}
    if True:
        bin_mom = binom_moment(mom=ceil_p, n=n, prob=two_over_Rtilde_sqd,
                              binomials=binomials)
    else:
        bin_mom = binom.moment(ceil_p, n, two_over_Rtilde_sqd)
    moment_bin = Rtilde_sqd / n * bin_mom**(1/ceil_p)

    # Prepare dictionaries to store Stirling numbers for
    # computing binomial moments
    stirlings = {}
    ceil_p_over_2 = ceil(p/2)
    if True:
        bin_mom = binom_moment(mom=ceil_p_over_2, n=n, prob=two_over_Rtilde_sqd,
                               stirlings=stirlings, binomials=binomials)
    else:
        bin_mom = binom.moment(ceil_p_over_2, n, two_over_Rtilde_sqd)
    moment_bin_symm = sqrt(Rtilde_sqd / n * bin_mom**(1/ceil_p_over_2))

    #for $k=2$
    if Rsmaller>1:
        if True:
            bin_mom = binom_moment(mom=ceil_p_over_2, n=n,
                                  prob=2*Rsmaller_sqd_minus_1/Rsmaller_sqd**2,
                                  stirlings=stirlings, binomials=binomials)
        else:
            bin_mom = binom.moment(ceil_p_over_2, n,
                                  2*Rsmaller_sqd_minus_1/Rsmaller_sqd**2)
        moment_bin_symm_2=(Rsmaller_sqd / n * sqrt(bin_mom**(1/ceil_p_over_2)))

    std_gsn_p_norm = rt_2 * exp( (gammaln((p+1)/2) - log(pi)/2) / p )

    term2p=max(Rsmaller_sqd_minus_1**(1-one_over_p),
        ((Rsmaller_sqd_minus_1**p + Rsmaller_sqd_minus_1)/Rsmaller_sqd)**one_over_p)
    # the different terms candidate for 1/2\|\sum_i X_i^2-1\|_p
    common3a = min(sqrt_p_minus_1,std_gsn_p_norm )
    r3a = common3a * (term2p*Astar + sqrt(Rsmaller_sqd_minus_1) * A ) / two_sqrt_n
    c3a = common3a * sqrt_p_minus_1 * term2p  / two_sqrt_n

    gamma_factor = (gamma((p+1)/2)/rt_pi)**one_over_p*rt_2
    d3a = (gamma_factor*Rsmaller**(2*(1-2/p))*(Rsmaller_sqd_minus_1)**one_over_p /
           two_sqrt_n)

    if p>=4 and Rsmaller>1:
        d3a=min(2**(-one_over_p)*moment_bin_symm_2*gamma_factor/two_sqrt_n, d3a)

    c3a=min(d3a,c3a)

    c_M = Rtilde_sqd / n


    # Baseline upper bound based on the triangle inequality:
    # E[|S-G|^p]^{1/p} <= E[|S|^p]^{1/p} + E[|G|^p]^{1/p}
    # <= sqrt(p-1) E[|Y_1|^p]^{1/p} + E[|G|^p]^{1/p}
    # (by the Marcinkiewicz-Zygmund inequality)
    # <= sqrt(p-1) E[ R^{p-2} |Y_1|^2]^{1/p} + E[|G|^p]^{1/p}
    # <= sqrt(p-1) R * (sig/R)^{2/p} + E[|G|^p]^{1/p}
    # = sqrt(p-1) R / Rtilde^{2/p} + E[|G|^p]^{1/p}
    # triangle_bound = E[|S-G|^p]^{1/p} / sig
    triangle_bound = (sqrt_p_minus_1 * Rtilde / Rtilde_to_2_over_p
                      + std_gsn_p_norm)
    if verbose:
        print(f"triangle bound: {triangle_bound}")

    # Note that we must have k > c_M to ensure M is well defined and finite.
    # For an upper bound on k, choose a value larger than the
    # analytically-derived choice (2/(p+2)) from Austern and Mackey
    k_max = 1

    if verbose:
        print(f"k range: ({c_M}, {k_max}]")

    # If the bound range is empty, return triangle bound
    if k_max <= c_M:
        return sig*triangle_bound

    # Define collection of bounds on p-Wasserstein distance parameterized by k
    one_over_n = 1/n
    Rtilde_over_sqrt_n = Rtilde / sqrt_n
    # Precompute constants needed for bound
    # fl_coeff
    fl_coeff = p_minus_1*Rtilde_sqd/2
    # f_coeff: (B_{p,n}/2) /sqrt(n)
    f_coeff = moment_bin/sqrt_n/2
    Kp_fourth_root = Kp**(1/4)
    two_Kp_plus_one = 2*Kp+1
    f_Kp_factor = Kp_fourth_root/(2*(Kp+1)*sqrt(two_Kp_plus_one))
    # f3_coeff
    Cp = gamma_factor*Rtilde**(1-2/p)*2**(1/p)
    if p>=4:
        Cp = min(moment_bin_symm*gamma_factor,Cp)
    factor = min(Cp,tildeU)
    f3_coeff = factor * one_over_n /2
    f3_Kp_factor = Kp_fourth_root/two_Kp_plus_one
    # g58_coeff
    pi_exp_factor = (pi)**(1/4)*(exp(19/300))
    renorm = pi_exp_factor * 2 * f_Kp_factor * sqrt_p_minus_1
    g58_coeff = f_coeff*renorm/2
    # g68_coeff
    g68_coeff = f3_coeff * pi_exp_factor * f3_Kp_factor
    def bound(k):
        # Bound is only finite for k > c_M
        if k <= c_M:
            return inf

        M_sqd = 1 - c_M / k
        M = sqrt(M_sqd)
        r2_bound = std_gsn_p_norm*(half_pi-asin(M))

        sqrt_k = sqrt(k)
        term_3 = 1 - Rtilde_over_sqrt_n / sqrt_k
        r3_bound = r3a * term_3
        c3_bound = c3a * term_3

        C = k * p_minus_1
        Rtilde_sqd_over_k = Rtilde_sqd/k
        k_over_Rtilde_sqd = k/Rtilde_sqd

        #bounding for the even terms
        def f(i,x):
            """Function defining the summand
            (Rtilde^{2i}) *
              (||H_{2i+1}||_p/(2i+2)!
               - 2^{-i} exp(19/300)pi^{1/4} sqrt(p-1)^{2i+1}/
               (2(K_p +1)sqrt(2K_p+1) i! ) * (1/x-1/n)^i/sqrt(x)."""
            # Note: For greater efficiency, delta could be precomputed for each
            # i,p value, in the same way H_norm is precomputed
            if p<7:
                delta=H_norm[2*i-1,p-2]/(factorial(2*i+2))
            else:
                delta=sqrt_p_minus_1**(2*i+1)/((2*i+2)*sqrt(factorial(2*i+1)))
            delta -= (f_Kp_factor*2**(-i)*pi_exp_factor*p_minus_1**(i+1/2)
                /factorial(i))
            v=(Rtilde_sqd)**i

            uu=((1/x-1/n)**i)*delta*v/sqrt(x)
            return uu

        def g5(x):
            """Function to sum the values of f for i between 1 and Kp"""
            total = 0
            for i in range(1, Kp):
                total += f(i,x)
            return total

        int_g5 = quad(g5, Rtilde_sqd_over_k, n)[0] * f_coeff

        def g58(x):
            """This corresponds to the integral for terms k >= 4 that are even.
            This is computed by upper bounding the Hermite polynomial and lower
            bounding sqrt(k!) and relating the terms to an exponential."""
            fl= fl_coeff*(x-one_over_n)
            return (exp(fl)-1)/(x**(3/2))
        int_g58 = quad(g58, one_over_n, k_over_Rtilde_sqd)[0] * g58_coeff
        r4_bound = int_g5+int_g58
        c4_bound = r4_bound

        #bounding for the odd terms
        def f3(i,x):
            """Function defining the summand for odd K_p <= k >= 3."""
            v=(Rtilde_sqd)**i
            if p<7:
                delta=H_norm[2*i-2,p-2]/(factorial(2*i+1))
            else:
                delta=sqrt_p_minus_1**(2*i)/((2*i+1)*sqrt(factorial(2*i)))
            delta -=f3_Kp_factor*2**(-i)*pi_exp_factor*p_minus_1**i/factorial(i)
            uu=(1/x-one_over_n)**(i-1/2)*delta*v/sqrt(x)
            return uu

        def g6(x):
            """Function to sum the values of f3 for i between 1 and Kp-1"""
            total = 0
            for i in range(1, Kp):
                total += f3(i,x)
            return total

        int_g6 = quad(g6, Rtilde_sqd_over_k, n)[0] * f3_coeff

        def g68(x):
            """This corresponds to the integral for odd terms k >= 3.
            This is computed by upper bounding the Hermite polynomial and
            lower bounding sqrt(k!) and relating the terms to an exponential."""
            fl = fl_coeff*(x-one_over_n)
            return (exp(fl)-1)/(x**1.5*sqrt(x-one_over_n))
        int_g68 = quad(g68, one_over_n, k_over_Rtilde_sqd)[0] * g68_coeff
        r5_bound = int_g6+int_g68
        c5_bound = r5_bound

        #multiplicative term
        if p==2:
            scaling_term=1
        else:
            scaling_term=1/M

        return (scaling_term*(r2_bound
                +
                min(c3_bound, r3_bound)
                +
                min(c4_bound, r4_bound)
                +
                min(c5_bound, r5_bound)))

    if verbose:
        # Plot log bound as a function of k
        ks = np.geomspace(c_M, k_max)
        plt.plot(ks, [log(sig*bound(k)) for k in ks])
        plt.ylabel(r'Log bound on p-Wasserstein distance')
        plt.xlabel(r'$k$')
        plt.title(f'n = {n}, p = {p}, sig = {sig}, R = {R}')
        plt.tight_layout()
        plt.show()

    # Find minimal value of bound(k) over [c_M, k_max]
    minimum = minimize_scalar(bound, bounds=[c_M, k_max],
                              method='bounded', options={"disp":verbose})
    if verbose:
        print(sig*minimum.fun)
    val=sig*min(triangle_bound, minimum.fun)
    return val

#
# Zero-bias tail bounds
#
def Sprime_tail(t, n, sig=1, R=1, two_sided=True, verbose=False):
    """Tail bound for an auxiliary sum S'

    Returns delta such that P(|S'| > t) <= delta (if two_sided) or
    P(S' > t) <= delta (otherwise) for any sum
    S' = sqrt(n) (1/n) [sum_{i=1}^{n-1} (W_i - E[W_i]) + Y_{n}']
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2, and
    Y_{n}' = Y_{n}^* + U(Y_{n} - Y_{n}^*) for independent
    (U, Y_n^*) with U ~ Unif([0,1]) and Y_n^* having the
    zero-bias distribution of (W_n-EW_n).

    Specifically, returns
      delta = min(1, delta_hoeff((u)_+), delta_bern((u)_+), delta_be(u))
    for u = t-R_s/(4sqrt(n)) and R_s = (R+sqrt(R^2-4sig^2))/2
    where delta_{hoeff,bern,be} are respectively Hoeffding, Bernstein, and
    Berry-Esseen tail bounds for unidentically distributed variable sums.

    Args:
      t - Positive number
      n - Number of datapoints X_i
      sig - Standard deviation parameter; Var(X_i) = sig^2/n
      R - Boundedness parameter; |X_i| <= sqrt(n) R
      two_sided - return two-sided tail bounds?
      verbose - If True will print diagnostic messages
    """
    sqrt_n = sqrt(n)
    sig_sqd = sig**2
    R_sqd = R**2
    R_s = (R+sqrt(R_sqd-4*sig_sqd))/2
    u = t-R_s/4/sqrt_n
    # Upper bound for average variance
    R_s_sqd = R_s**2
    v_up_sqd = sig_sqd+(R_s_sqd-6*sig_sqd)/(9*n)
    # Square root of lower bound for average variance
    v_low = sig*sqrt(1-89./144./n)
    R_s_mod = (R+sqrt(R_sqd-220*sig_sqd/144))/2
    delta_be = unidentical_berry_esseen_tail(
        u, n,
        sig=v_low,
        third=R_s*v_up_sqd+min(R_s/4,R_s_mod-R_s)*(sig_sqd+R_s_sqd/3)/3/n,
        two_sided=two_sided)
    if u <= 0:
        return min(1, delta_be)
    delta_hoeff = hoeffding_tail(u, R=R, two_sided=two_sided)
    delta_bern = unidentical_bernstein_tail(
        u, n, sig=sqrt(v_up_sqd), R=R_s,
        two_sided=two_sided)
    if verbose:
        print(f"be={delta_be}, hoeff={delta_hoeff}, bern={delta_bern}")
    return min(1, delta_hoeff, delta_bern, delta_be)

def zero_tail_iid(t, n, sig=1, R=1, two_sided=True, verbose=False):
    """Zero bias tail bound for an iid sum that exploits specific iid structure

    Returns delta such that P(|S| > t) <= delta (if two_sided) or
    P(S > t) <= delta (otherwise) for any sum
    S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2.

    Specifically, returns for u = (sqrt(n)/sqrt(n+1))(t-R_s/sqrt(n))_+/sig,
      delta = min(1, delta_2)
    where
      delta_2 = (sig sqrt{n+1})/(sig sqrt{n+1}+R diff[u]) Phi^c(u)
        + (R)/(sig sqrt{n+1}+R diff[u])
        x inf_{lam in [0,1]} [h_u(lam u) + (h_u(u) - h_u(lam u))
        x delta_{aux,2}(lam sig u sqrt(n)/sqrt(n+1))],
      delta_{aux,2}(t) = Sprime_tail(t, n+1, sig, R),
      h_u(w) = (w + (1+w^2) sqrt(2pi)exp(w^2/2)Phi(w)) Phi^c(u), and
      diff[u] = h_u(u)-(2/(u+sqrt(u^2+8/pi)-8u/(pi(u+sqrt(u^2+8/pi))^2))) Phi(u)

    Args:
      t - Positive number
      n - Number of datapoints W_i
      sig - Standard deviation parameter; Var(W_i) = sig^2
      R - Boundedness parameter
      two_sided - return two-sided tail bounds?
      verbose - If True will display diagnostic messages
    """
    R_s=0.5*(R+sqrt(R**2-4*sig**2))
    sqrt_n = sqrt(n)
    sqrt_n_plus_1 = sqrt(n+1)
    sqrt_n_ratio = sqrt_n / sqrt_n_plus_1
    u = sqrt_n_ratio * (t-R_s/sqrt_n)/sig

    # Bound reduces to 1 for non-positive u
    if u <= 0:
        return 1.

    # Helper function
    log_Phic_u = norm.logsf(u)
    Phic_u = exp(log_Phic_u)
    norm_cdf = norm.cdf
    def h_u(w):
        w_sqd = w**2
        return (w * Phic_u + (1+w_sqd) * rt_2_pi * norm_cdf(w) *
                exp(log_Phic_u + w_sqd/2))

    # Define collection of bounds parameterized by lam in [0,1]
    h_u_u = h_u(u)
    sig_sqrt_n_ratio = sig * sqrt_n_ratio
    def bound(lam, verbose=verbose):
        lam_u = lam*u
        h_u_lam_u = h_u(lam_u)
        # Compute one-sided bound here and account for two-sidedness at the end
        return (h_u_lam_u + (h_u_u - h_u_lam_u) *
                Sprime_tail(lam_u * sig_sqrt_n_ratio, n+1, sig=sig, R=R,
                            verbose=False, two_sided=False))
    # Find minimal value of bound(lam) over [0,1]
    tic()
    minimum = minimize_scalar(bound, bounds=[0, 1], method='bounded',
                              options={"disp":verbose})
    elapsed = toc()
    if verbose:
        print(f"elapsed: {elapsed}s")
        print(f"minimum={minimum.fun}, lam={minimum.x}\n")
    if False:
        # Plot bound as a function of lam
        lams = np.linspace(0, 1)
        plt.plot(lams, [bound(lam) for lam in lams])
        plt.ylabel(r'Zero-bias lambda term')
        plt.xlabel(r'$\lambda$')
        plt.title(f't = {t}, u ={u}, n = {n}, sig = {sig}, R = {R}')
        plt.tight_layout()
        plt.show()
    # Return complete bound
    sig_sqrt_n_plus_1 = sig * sqrt_n_plus_1
    delta = (sig_sqrt_n_plus_1 * Phic_u
             + R * minimum.fun)/(sig_sqrt_n_plus_1 + R * u)
    if two_sided: delta *= 2
    return delta

def zero_tail_general(t, n, sig=1, R=1, two_sided=True, verbose=False):
    """Zero bias tail bound for an iid sum derived from unidentical sum tail
    bound

    Returns delta such that P(|S| > t) <= delta (if two_sided) or
    P(S > t) <= delta (otherwise) for any sum
    S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2.

    Specifically, returns for u = (t-R/sqrt(n))_+/sig,
      delta = min(1, delta_1)
    where
      delta_1 = (sig sqrt{n})/(sig sqrt{n}+R diff[u]) Phi^c(u)
        + R/(sig sqrt{n}+R diff[u])
        x inf_{lam in [0,1]} [h_u(lam u) +
                              (h_u(u) - h_u(lam u)) delta_{aux,1}(lam sig u)]
      delta_{aux,1}(t) = Sprime_tail(t, n, sig, R)

    Args:
      t - Positive number
      n - Number of datapoints W_i
      sig - Standard deviation parameter; Var(W_i) = sig^2
      R - Boundedness parameter
      two_sided - return two-sided tail bounds?
      verbose - If True will display diagnostic messages
    """

    sqrt_n = sqrt(n)
    u = (t-R/sqrt_n)/sig

    # Bound reduces to 1 for non-positive u
    if u <= 0:
        return 1.

    # Helper function
    log_Phic_u = norm.logsf(u)
    Phic_u = exp(log_Phic_u)
    norm_cdf = norm.cdf
    def h_u(w):
        w_sqd = w**2
        return (w * Phic_u +
                (1+w_sqd) * rt_2_pi * norm_cdf(w) * exp(log_Phic_u + w_sqd/2))

    # Define collection of bounds parameterized by lam in [0,1]
    h_u_u = h_u(u)
    def bound(lam, verbose=verbose):
        lam_u = lam*u
        h_u_lam_u = h_u(lam_u)
        # Compute one-sided bound here and account for two-sidedness at the end
        return (h_u_lam_u + (h_u_u - h_u_lam_u) *
                Sprime_tail(lam_u * sig, n, sig=sig, R=R, verbose=False,
                            two_sided=False))
    # Find minimal value of bound(lam) over [0,1]
    tic()
    minimum = minimize_scalar(bound, bounds=[0, 1], method='bounded',
                              options={"disp":verbose})
    elapsed = toc()
    if verbose:
        print(f"elapsed: {elapsed}s")
        print(f"minimum={minimum.fun}, lam={minimum.x}\n")
    if False:
        # Plot bound as a function of lam
        lams = np.linspace(0, 1)
        plt.plot(lams, [bound(lam) for lam in lams])
        plt.ylabel(r'Zero-bias lambda term')
        plt.xlabel(r'$\lambda$')
        plt.title(f't = {t}, u ={u}, n = {n}, sig = {sig}, R = {R}')
        plt.tight_layout()
        plt.show()
    # Return complete bound
    sig_sqrt_n = sig * sqrt_n
    delta = (sig_sqrt_n * Phic_u + R * minimum.fun) /(sig_sqrt_n + R * u)
    if two_sided: delta *= 2
    return delta

def zero_tail(t, n, sig=1, R=1, two_sided=True, verbose=False):
    """Zero bias tail bound for an iid sum

    Returns minimum of zero_tail_iid and zero_tail_general.

    Args:
      t - Positive number
      n - Number of datapoints W_i
      sig - Standard deviation parameter; Var(W_i) = sig^2
      R - Boundedness parameter
      two_sided - return two-sided tail bounds?
      verbose - If True will display diagnostic messages
    """
    return min(zero_tail_iid(t, n, sig=sig, R=R, two_sided=two_sided,
                             verbose=verbose),
               zero_tail_general(t, n, sig=sig, R=R, two_sided=two_sided,
                             verbose=verbose))


#
# Efficient tail bound
#
def tail_bound(t, n, p=None, sig=1, R=1, two_sided=True, verbose=False, cache=None):
    """Tail bound for an iid sum based on gsn_wass

    Returns delta such that P(|S| > t) <= delta (if two_sided) or
    P(S > t) <= delta (otherwise) for any sum
    S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2.

    Specifically, returns, if p is None,
      delta = min(inf_{rho in (0,1), p >= 2} delta_{rho,p}, delta_{Bernstein})
    or, if p is not None,
      delta = min(inf_{rho in (0,1)} delta_{rho,p}, delta_{Bernstein})
    where, if two-sided,
      delta_{rho,p} = CCDF(rho t / sig) + gsn_wass(n, p, sig, R)^p/(t(1-rho))^p,
      CCDF(s) = P(Z >= s) = 1 - CDF(s) for Z a standard Gaussian,
      and delta_{Bernstein} = exp( -(t^2) / 2 / ((sig^2) + R t / 3 / sqrt(n)) ),
    and, if one-sided,
      delta_{rho,p} = CCDF(rho t/sig)+gsn_wass(n, p, sig, R)^p/(1+(t(1-rho))^p)
      and delta_{Bernstein} = 2 exp(-(t^2) / 2 / ((sig^2) + R t / 3 / sqrt(n))).

    Args:
      t - Positive number
      n - Number of datapoints Y_i
      sig - Standard deviation parameter; Var(Y_i) = sig^2
      R - Boundedness parameter
      p - None (to return minimum bound over all p) or integer Wasserstein
        distance power >= 1
      two_sided - return two-sided tail bounds?
      verbose - If True will display minimize_scalar convergence messages
      cache - If not None, this dictionary, indexed by p, will be queried for
        precomputed values before computing any gsn_wass(n, p, sig, R) values;
        if not present, values will be added to this dictionary
    """

    # Bound reduces to 1 for non-positive t
    if t <= 0:
        return 1.

    if cache is None:
        # Initialize empty dictionary
        cache = {}

    # Compute minimum bound over desired range of p values
    if p is None:
        # Consider all p values >= 2
        min_over_p = True
        p_init = 2
    else:
        min_over_p = False
        p_init = p

    # Compute sufficient statistic t/sig
    t_over_sig = t/sig

    # Define collection of bounds parameterized by rho in [0,1]
    def bound(rho, verbose=verbose):
        if rho == 1:
            min_wass_term = 1
            p = None
        else:
            one_minus_rho_times_t = (1-rho)*t
            # Optimize p-Wasserstein component of bound over p
            min_wass_term = 1
            p = p_init
            while True:
                # Look up p-Wasserstein distance
                if p not in cache:
                    cache[p] = gsn_wass(n, p, sig=sig, R=R)
                W_ratio = cache[p] / one_minus_rho_times_t

                W_ratio_to_p = W_ratio ** p
                if (p == 2) and not two_sided:
                    # Use Cantelli's inequality [1 / ( 1 + t^p /(gamma * W)^p )]
                    # for one-sided when p = 2
                    wass_term = W_ratio_to_p / (W_ratio_to_p + 1)
                else:
                    # Otherwise use Chebyshev's inequality
                    wass_term = W_ratio_to_p
                min_wass_term = min(min_wass_term, wass_term)

                # Terminate if only a single p value was requested, if wass_term
                # begins to increase, or if gamma_W_over_t_to_p >= 1 (since all
                # bounds based on larger p will be no smaller)
                if (not min_over_p or (min_wass_term < wass_term) or
                    (W_ratio_to_p >= 1)):
                    break

                # Otherwise, consider the next value of p
                p += 1

        # Compute Gaussian tail probability
        gsn_term = norm.sf(rho * t_over_sig)
        if two_sided:
            gsn_term *= 2
        bound_val = min_wass_term + gsn_term
        if verbose:
            print(f"rho={rho}: bound={bound_val}, wass={min_wass_term}, "
                  f"gsn={gsn_term}, p_last={p}")
        return bound_val

    if verbose:
        # Plot log bound as a function of k
        rho_vals = np.linspace(0, 1)
        plt.plot(rho_vals, [log(bound(rho,verbose=False)) for rho in rho_vals])
        plt.ylabel(r'Log bound on $P(S \geq t)$')
        plt.xlabel(r'$\rho$')
        plt.title(fr't = {t}, n = {n}, p = {p}, $\sigma$ = {sig}, R = {R}')
        plt.tight_layout()
        plt.show()

    # Find minimal value of bound(rho) over [0,1]
    tic()
    minimum = minimize_scalar(bound, bounds=[0, 1], method='bounded',
                              options={"disp":verbose})
    elapsed = toc()
    if verbose:
        print(f"elapsed: {elapsed}s")
    # Compare with baseline tail bounds
    bern_delta = bernstein_tail(t, n, sig=sig, R=R, two_sided=two_sided)
    be_delta = berry_esseen_tail(t, n, sig=sig, R=R, two_sided=two_sided)
    nbe_delta = nonuniform_berry_esseen_tail(t, n, sig=sig, R=R,
                                             two_sided=two_sided)
    hoeff_delta = hoeffding_tail(t, R=R, two_sided=two_sided)
    zero_delta = zero_tail(t, n, sig=sig, R=R, two_sided=two_sided)
    if verbose:
        print(f"minimum={minimum.fun}, rho={minimum.x}, "
              f"bernstein={bern_delta}, BE={be_delta}, NBE={nbe_delta}, "
              f"hoeffding={hoeff_delta}, zero={zero_delta}\n")
    return min(minimum.fun, bern_delta, be_delta, nbe_delta, hoeff_delta,
               zero_delta)


#
# Efficient quantile bound
#
def zero_quantile(delta, n, qmax, sig=1, R=1, two_sided=True):
    """Zero bias quantile bound for an iid sum

    Returns minimum of qmax and zero bias quantile q
    s.t. P(|S| > q) <= delta (if two_sided) or P(S > q) <= delta (otherwise)
    for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2.

    The zero bias quantile q is computed by inverting zero_tail.

    Args:
      delta - Upper tail probability
      n - Number of datapoints W_i
      qmax - Returns minimum of qmax and zero
      sig - Standard deviation parameter
      R - Boundedness parameter
      two_sided - Provide two-sided quantile bound?
    """
    # if zero_tail(qmax) >= delta, then zero bias quantile >= qmax
    if zero_tail(qmax,n,sig=sig,R=R,two_sided=two_sided) >= delta:
        return qmax

    # Find quantile bound by inverting tail bound
    def invert_zero_tail(delta, qmin, qmax):
        """Returns value of q in [qmin, qmax] for which
        zero_tail(q,n,sig,R) = delta"""
        g = lambda q: (
            zero_tail(q,n,sig=sig,R=R,two_sided=two_sided)
            - delta)
        # Use the bisect method to find the root of g within the interval
        # [qmin, qmax]
        return bisect(g, qmin, qmax)

    # Quantile bound will be no smaller than asymptotic quantile
    qmin=asymptotic_quantile(delta, sig=sig, two_sided=two_sided)
    return invert_zero_tail(delta, qmin, qmax)

def compute_p_norm(t, p):
    """Returns (2 * int_t^inf |x|^p phi(x) dx)^{1/p} for t >= 0 and
    phi the standard normal pdf"""
    def integrand(x, p):
        return (abs(x) ** p) * norm.pdf(x)
    # Calculate the integral for the region Z >= t
    integral_pos = quad(integrand, t, np.inf, args=(p))[0]

    # Double to account for the region Z <= -t
    p_norm = 2*integral_pos
    return p_norm**(1/p)

def quantile_bound(delta, n, p=None, sig=1, R=1, two_sided=True, verbose=False,
                   cache=None):
    """Quantile bound for an iid sum

    Returns q s.t. P(|S| > q) <= delta (if two_sided) or P(S > q) <= delta
    (otherwise) for any sum S = sqrt(n) (1/n) sum_{i=1}^n W_i - E[W_i]
    with each independent and identically distributed W_i in [0, R] with
    Var(W_i) = sig^2.

    Specifically, returns, if p is None,
      q = min( inf_{a in (0, delta), p >= 1} q_{a,p}, q_{baseline} )
    or, if p is not None,
      q = min( inf_{a in (0, delta)} q_{a,p}, q_{baseline} )
    where, if not two_sided,
      q_{a,p} = gsn_wass(n, p, sig, R)(1/a - indic{p==2})^{1/p}
              + sig GsnInvCDF(1-(delta-a)),
      GsnInvCDF(s) satisfies P(Z <= GsnInvCDF(s)) = s for Z a standard Gaussian,
    and, if two_sided,
      q_{a,p} = gsn_wass(n, p, sig, R)/a^{1/p} + sig GsnInvCDF(1-(delta-a)/2)
    and q_{baseline} is the best of the corresponding delta Bernstein,
    Berry Esseen, non-uniform Berry Esseen, Hoeffding, and zero bias quantiles.

    Args:
      delta - Upper tail probability
      n - Number of datapoints W_i
      p - None (to return minimum bound over all p) or integer Wasserstein
        distance power >= 1
      sig - Standard deviation parameter
      R - Boundedness parameter
      two_sided - Provide two-sided quantile bound?
      verbose - If True will display minimize_scalar convergence messages
      cache - If not None, this dictionary, indexed by p, will be queried for
        precomputed values before computing any gsn_wass(n, p, sig, R) values;
        if not present, values will be added to this dictionary
    """
    if sig == 0:
        return 0

    if cache is None:
        # Initialize empty dictionary
        cache = {}

    # Compute minimum bound over desired range of p values
    if p is None:
        # Consider all p values >= 1
        min_over_p = True
        p_init = 1
    else:
        min_over_p = False
        p_init = p

    # First, compute the Wasserstein bound
    # min_p (1/delta)**(1/p)* [cache[p]+\|Z I(Z >= Phi^{-1}(delta))\|_p]
    wass_q = inf
    one_over_delta = 1/delta
    p = p_init
    asymp_q = asymptotic_quantile(delta,sig=sig)
    while True:
        # Look up the p-Wasserstein distance between sum S and Gaussian with
        # matching mean and variance
        if p not in cache:
            cache[p] = gsn_wass(n, p, sig=sig, R=R, verbose=verbose)

        wass_term = (sig*compute_p_norm(asymp_q, p)+cache[p])*one_over_delta**(1/p)
        wass_q = min(wass_q, wass_term)

        # Terminate if only a single p value was requested or bound for the
        # most recent p was greater than prior bound
        if (not min_over_p or wass_term > wass_q):
            break

        # Otherwise, consider the next value of p
        p += 1

    # Next, define a collection of alternative Wasserstein bounds parameterized
    # by a in (0,delta)
    def bound(a, verbose=verbose):
        # Bound is only finite for a in (0,delta)
        if a <= 0 or a >= delta:
            return inf

        p = p_init
        a_term = 1/a
        min_wass_term = inf
        while True:
            # Look up the p-Wasserstein distance between sum S and Gaussian with
            # matching mean and variance
            if p not in cache:
                cache[p] = gsn_wass(n, p, sig=sig, R=R, verbose=verbose)

            if not two_sided and (p == 2):
                # Use Cantelli's inequality
                wass_term = cache[p] * ((a_term-1) ** (1/p))
            else:
                # Use Chebyshev's inequality
                wass_term = cache[p] * (a_term ** (1/p))

            min_wass_term = min(min_wass_term, wass_term)
            # Terminate if only a single p value was requested or bound for the
            # most recent p was greater than prior bound
            if (not min_over_p or wass_term > min_wass_term):
                break

            # Otherwise, consider the next value of p
            p += 1

        gsn_term = asymptotic_quantile(delta-a, sig=sig, two_sided=two_sided)

        bound_val = min_wass_term + gsn_term
        if verbose:
            print(f"a={a}: bound={bound_val}, wass={min_wass_term}, "
                  f"gsn={gsn_term}, p_last={p}")
        return bound_val


    if verbose:
        # Plot log bound as a function of a
        a_vals = np.linspace(0, delta)
        plt.plot(a_vals, [log(bound(a,verbose=False)) for a in a_vals])
        plt.ylabel(r'Log bound on $1-\delta$ quantile')
        plt.xlabel(r'$a$')
        plt.title(fr'$\delta$ = {delta}, n = {n}, $\sigma$ = {sig}, R = {R}')
        plt.tight_layout()
        plt.show()

    # Find minimal value of bound(a) over (0,delta)
    tic()
    minimum = minimize_scalar(bound, bounds=[0, delta], method='bounded',
                              options={"disp":verbose})
    elapsed = toc()
    if verbose:
        print(f"elapsed: {elapsed}s")
    # Compare with baseline quantile bounds
    bern_q = bernstein_quantile(delta, n, sig=sig, R=R, two_sided=two_sided)
    be_q = berry_esseen_quantile(delta, n, sig=sig, R=R, two_sided=two_sided)
    nbe_q = nonuniform_berry_esseen_quantile(delta, n, sig=sig, R=R,
                                             two_sided=two_sided)
    hoeff_q = hoeffding_quantile(delta, R=R, two_sided=two_sided)
    bentkus_q = bentkus_quantile(delta, n, sig=sig, R=R, two_sided=two_sided)
    min_q = min(minimum.fun, wass_q, bern_q, be_q, nbe_q, hoeff_q, bentkus_q)
    zero_q = zero_quantile(delta, n, min_q, sig=sig, R=R, two_sided=two_sided)
    if verbose:
        print(f"minimum={minimum.fun}, a={minimum.x}, wass={wass_q}, "
              f"bernstein={bern_q}, BE={be_q}, NBE={nbe_q}, "
              f"bentkus={bentkus_q}, "
              f"hoeffding={hoeff_q}, zero={zero_q}\n")

    return min(min_q, zero_q)