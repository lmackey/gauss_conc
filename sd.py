"""Empirical Bernstein standard deviation confidence bounds
Source: Section 4.4 of Martinez-Taboada and Ramdas, 
Sharp empirical Bernstein bounds for the variance of bounded random variables
https://github.com/DMartinezT/emp_bernstein_variance
"""
import numpy as np

# Auxiliary functions
def psi_e(lam):
    return -lam - np.log(1 - lam)

def psi_p(lam):
    return np.exp(lam) - lam - 1

class EBUB:
    """Empirical Bernstein upper confidence bounds for population standard deviation.
    """
    def __init__(self, max_rounds, alpha = 0.05, c1 = 0.5, c2 = 0.25**2, c3=0.25, c4=0.5, CS=False):
        self.max_rounds = max_rounds
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

        self.sumvarhat = self.c2
        self.summuhat = self.c3
        self.tilde_summuhat = self.c4

        self.aux = np.sqrt(2*np.log(1 / self.alpha))
        self.part1 = 0
        self.part2 = np.log(1/self.alpha)

        self.sum_lambdas = 0
        self.sum_bernstein_center = 0
        self.CS = CS
        self.t = 0



    def __call__(self, Y):

        X = (Y - self.tilde_summuhat/(self.t+1))**2
        radius = (X - self.summuhat/(self.t+1))**2

        if self.CS:
            self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * np.log(self.t+2)), self.c1)
        else:
            self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * self.max_rounds / (self.t+1)), self.c1)
        self.sum_lambdas += self.lambda_b

        # Center
        self.sum_bernstein_center += self.lambda_b * X
        self.bernstein_center = self.sum_bernstein_center / self.sum_lambdas
        # Radius
        self.part1 += radius*psi_e(self.lambda_b)
        self.bernstein_radius = (self.part1+self.part2)/self.sum_lambdas

        # Update auxiliary variables
        self.summuhat += X
        self.sumvarhat += radius
        self.tilde_summuhat += Y
        self.t += 1

    def get_center_plus_radius(self):
        return np.sqrt(self.bernstein_center + self.bernstein_radius)

    def get_center(self):
        return np.sqrt(self.bernstein_center)

class EBLB:
    """Empirical Bernstein lower confidence bounds for population standard deviation.
    """
    def __init__(self, max_rounds, alpha = 0.05, tilde_alpha = 0.05,
                 c1 = 0.5, c2 = 0.0625, c3=0.25, c4=0.5, c5=2,
                 CS=False, tilde_CS=False):
        self.max_rounds = max_rounds
        self.CS = CS
        self.tilde_CS = tilde_CS
        self.alpha = alpha
        self.tilde_alpha = tilde_alpha

        self.c1 = c1
        self.c5 = c5
        self.sumvarhat = c2
        self.tilde_sumvarhat = c3
        self.tilde_summuhat = c4

        self.t = 0


        self.aux = np.sqrt(2*np.log(1 / self.alpha))
        self.part1 = 0
        self.part2 = np.log(1/self.alpha)
        self.sum_lambdas = 0

        self.bernstein_center = None
        self.bernstein_radius = None
        self.sum_bernstein_center = 0


        self.tilde_aux = np.sqrt(2*np.log(2 / self.tilde_alpha))

        self.tilde_sum_bernstein_center = 0
        self.tilde_bernstein_center = 0
        self.sum_psi_tilde_lambdas = 0
        self.sum_tilde_lambdas = 0

        self.dt = None
        self.rt = None
        self.tilde_at = None
        self.tilde_bt = None
        self.tilde_ct = None
        self.sum_at = 0
        self.sum_bt = 0
        self.sum_ct = 0
        self.at = None
        self.bt = None
        self.ct = None


    def __call__(self, Y):

        # Empirical Bernstein for the variance
        X = (Y - self.tilde_bernstein_center)**2
        radius = (X - self.tilde_sumvarhat/(self.t+1))**2

        varhat = self.tilde_sumvarhat / (self.t+1) # radius
        if self.sum_tilde_lambdas > 0:
            threshold = (np.log(2/self.tilde_alpha) + varhat * self.sum_psi_tilde_lambdas)/self.sum_tilde_lambdas
            condition_upsilon = threshold < 1
            if condition_upsilon:
                if self.CS:
                    self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * np.log(self.t+2)), self.c1)
                else:
                    self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * self.max_rounds / (self.t+1)), self.c1)
            else:
                self.lambda_b = 0
        else:
            self.lambda_b = 0
        self.sum_lambdas += self.lambda_b

        # Center
        self.sum_bernstein_center += self.lambda_b * X
        # Radius
        self.part1 += radius*psi_e(self.lambda_b)
        # Both
        if self.sum_lambdas > 0:
            self.bernstein_center = self.sum_bernstein_center / self.sum_lambdas
            self.bernstein_radius = (self.part1+self.part2)/self.sum_lambdas

        # Extra term due to variance overestimation
        if self.lambda_b > 0:

            self.tilde_at = self.sum_psi_tilde_lambdas**2 / self.sum_tilde_lambdas**2
            self.tilde_bt = 2*np.log(2/self.tilde_alpha)*self.sum_psi_tilde_lambdas / self.sum_tilde_lambdas**2
            self.tilde_ct = np.log(2/self.tilde_alpha)**2 / self.sum_tilde_lambdas**2

            self.dt = self.bernstein_center
            self.rt = self.bernstein_radius

            self.sum_at += self.tilde_at * self.lambda_b
            self.at = self.sum_at / self.sum_lambdas
            self.sum_bt += self.tilde_bt * self.lambda_b
            self.bt = 1 + self.sum_bt / self.sum_lambdas
            self.sum_ct += self.tilde_ct * self.lambda_b
            self.ct = self.sum_ct / self.sum_lambdas


        # Update empirical Bennett
        # Tilde lambdas
        tilde_radius = (Y - self.tilde_summuhat/(self.t+1))**2
        if self.tilde_CS:
            self.tilde_lambda_b = np.minimum(self.tilde_aux / np.sqrt(self.tilde_sumvarhat * np.log(self.t+2) ), self.c5)
        else:
            self.tilde_lambda_b = np.minimum(self.tilde_aux / np.sqrt(self.tilde_sumvarhat * self.max_rounds / (self.t+1)), self.c5)
        self.sum_tilde_lambdas += self.tilde_lambda_b
        self.sum_psi_tilde_lambdas += psi_p(self.tilde_lambda_b)
        # Tilde center
        self.tilde_sum_bernstein_center += self.tilde_lambda_b * Y
        self.tilde_bernstein_center = self.tilde_sum_bernstein_center / self.sum_tilde_lambdas

        # Update auxiliary variables
        self.sumvarhat += radius
        self.tilde_summuhat += Y
        self.tilde_sumvarhat += tilde_radius
        self.t += 1

    def get_center_minus_radius(self):
        c = self.dt - self.ct - self.rt
        if c < 0:
            return 0
        else:
            sol = (-self.bt + np.sqrt(self.bt**2 + 4*self.at*c)) / (2*self.at)
            return np.sqrt(sol)

    def get_center(self):
        return np.sqrt(self.bernstein_center)