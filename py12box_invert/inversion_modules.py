import numpy as np
import pymc3 as pm
import pandas as pd
from py12box_invert.core import global_mf, hemis_mf, annual_means


class Inverse_method:
    """Inverse method modules

    Each method must have a corresponding {method}_posterior function that allows
    posterior mole fraction and emissions matrices to be calculated
    """

    def rigby14(self):
        # Check that self.matrices has correct inputs

        # Run inversion

        # Store outputs in self.mod_posterior
        
        pass

    def rigby14_posterior(self):
        pass

    def analytical_gaussian(self):
        """
        TODO: Update this docstring
        Perform a Gaussian analytical inversion assuming linearity.
        The inferred value 'x_hat' is the difference in emissions from the reference run.
        
            Parameters:
                y (array)         : Deviation from a priori emissions
                H (array)         : Sensitivity matrix
                x_a (array)       : A priori deviation (defaults to zeros)
                R (square matrix) : Model-measurement covariance matrix
                P_inv (square matrix) : Emissions inverse covariance matrix
            Returns:
                x_hat (array): Posterior mean difference from a priori
                P_hat (array): Posterior covariance matrix
        """
        R_inv = np.linalg.inv(self.mat.R)
        self.mat.x_hat = np.linalg.inv(self.mat.H.T @ R_inv @ self.mat.H + self.mat.P_inv) @ (self.mat.H.T @ R_inv @ self.mat.y + self.mat.P_inv @ self.mat.x_a)
        self.mat.P_hat = np.linalg.inv(self.mat.H.T @ R_inv @ self.mat.H + self.mat.P_inv)
    
    def analytical_gaussian_posterior(self):
        """Method to process posterior mole fractions and emissions
        """
        # Posterior mole fraction
        y_hat = self.sensitivity.sensitivity @ self.mat.x_hat + self.mod_prior.mf[:, :4].flatten()
        self.mod_posterior.mf = y_hat.reshape(int(len(y_hat)/4), 4)

        # Posterior emissions
        freq_months = self.sensitivity.freq_months
        self.mod_posterior.emissions = self.mod_prior.emissions.copy()
        for ti in range(int(self.mod_prior.emissions.shape[0]/freq_months)):
            for bi in range(4):
                self.mod_posterior.emissions[ti*freq_months:(ti+1)*freq_months, bi] += self.mat.x_hat[4*ti + bi]    


# def analytical_gaussian(y, H, x_a, R, P_inv, sensitivity, mf_ref, emis_ref, freq, time):
#     """
#     Do an analytical Gaussian inversion, assuming conjugacy (Gaussian likelihood and prior)
#     """

#     #Do inversion
#     x_hat, P_hat = inversion_analytical(y, H, x_a, R, P_inv)
    
#     #Calculate global and hemispheric mole fraction
#     xmf_out, xmf_sd_out = global_mf(sensitivity, x_hat, P_hat, mf_ref)
#     xmf_N_out, xmf_N_sd_out, xmf_S_out, xmf_S_sd_out = hemis_mf(sensitivity, x_hat, P_hat, mf_ref)
#     index = np.round(time[::12]).astype(int)
#     model_mf = pd.DataFrame(index=index, \
#                             data={"Global_mf": xmf_out, "Global_mf_sd": xmf_sd_out, \
#                                     "N_mf":xmf_N_out, "N_mf_sd":xmf_N_sd_out, \
#                                     "S_mf":xmf_S_out, "S_mf_sd":xmf_S_sd_out})
    
#     #Calculate annual emissions
#     x_out, x_sd_out = annual_means(x_hat, P_hat, emis_ref, freq=freq)
#     model_emis = pd.DataFrame(index=index, data={"Global_emissions": x_out, \
#                                                               "Global_emissions_sd": x_sd_out})  
    
#     return model_emis, model_mf


def NUTS_expRW1(H, x_a, R, y, emis_ref, sensitivity, time, nit=10000, tune=None, burn=None, freq="yearly"):
    """
    Use MCMC to infer emissions.
    Uses a random walk model of order 1, such that y = Hx + e,
    where x = exp(z) and (z_i - z_j) ~ N(0, tau^-1), 
    assuming constant spacing in time.
    The model-measurement error includes an unknown model error 
    term (s), currently such that e ~ N(0, ε^2 + s^2) and s~LN(1,1).
    
    """    
    H_split = []
    for bx in range(4):
        H_split = H_split + [H[:,bx::4]]

    if not tune:
        tune = int(nit*0.2)
    if not burn:
        burn = int(nit*0.1)

    with pm.Model() as model:
        T0 = pm.Exponential("t", lam=2) #pm.Uniform("t0", lower = 0.01, upper=100)
        #X0 = pm.GaussianRandomWalk("x0", sigma=T0, shape=int(len(x_a)/4))
        #X1 = pm.GaussianRandomWalk("x1", sigma=T0, shape=int(len(x_a)/4))
        #X2 = pm.GaussianRandomWalk("x2", sigma=T0, shape=int(len(x_a)/4))
        #X3 = pm.GaussianRandomWalk("x3", sigma=T0, shape=int(len(x_a)/4))
        X0 = RW2("x0", sigma=T0, shape=int(len(x_a)/4))
        X1 = RW2("x1", sigma=T0, shape=int(len(x_a)/4))
        X2 = RW2("x2", sigma=T0, shape=int(len(x_a)/4))
        X3 = RW2("x3", sigma=T0, shape=int(len(x_a)/4))
        sig = pm.Lognormal("s", mu = 1, sd = 1)
        mu = pm.math.dot(H_split[0],pm.math.exp(X0)) + \
             pm.math.dot(H_split[1],pm.math.exp(X1)) + \
             pm.math.dot(H_split[2],pm.math.exp(X2)) + \
             pm.math.dot(H_split[3],pm.math.exp(X3)) 
        sd = np.sqrt(np.diag(R) + sig**2)
        Y = pm.Normal('y', mu = mu, sd=sd, observed=y, shape = len(y))
        trace = pm.sample(nit, tune=int(tune), chains=1,
                            progressbar=False, target_accept=0.9) 
        outs = []
        for xi in ["x0","x1","x2","x3"]:
            outs = outs + [trace.get_values(xi, burn=burn)[0:int((nit)-burn)]]

    outs_exp=np.exp(np.array(outs))  
    repfreq={"monthly":1, "quarterly":3, "yearly":12}
    x_mnth_trace = np.repeat(np.sum(outs_exp, axis=0),repfreq[freq], axis=1).T + \
                    np.expand_dims(np.sum(emis_ref,axis=1), axis=1)
    x_yr_trace = x_mnth_trace.reshape(-1, 12, x_mnth_trace.shape[1]).mean(1)
    x_out = np.mean(x_yr_trace, axis=1)
    x_out_hpd_95 = pm.stats.hpd(x_yr_trace.T, 0.95)
    x_out_hpd_68 = pm.stats.hpd(x_yr_trace.T, 0.68)
    
    #if freq == "yearly":
        #x_hat = np.mean(outs_exp,axis=1)
        #x_mnth = np.sum(np.repeat(x_hat.T,12, axis=0)+ emis_ref, axis=1)
        #x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
        #x_out_hpd_68 = pm.stats.hpd(np.sum(outs_exp,axis=0), 0.68)
        #x_out_hpd_95 = pm.stats.hpd(np.sum(outs_exp,axis=0), 0.95)
    #elif freq == "monthly":
    #    print("Nope")
    #elif freq == "quarterly":
    
    
    #x_mnth = np.sum(np.repeat(x_hat.T,12, axis=0)+ emis_ref, axis=1)
    #x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
    
    mf_trace = np.zeros((sensitivity.shape[0], nit-burn))
    for bx in range(4):
        mf_trace += sensitivity[:,bx::4] @ outs_exp[bx,:,:].T
    mf_N_trace = (mf_trace[0::4] + mf_trace[1::4])/2.
    mf_S_trace = (mf_trace[2::4] + mf_trace[3::4])/2.
    mf_G_trace = (mf_N_trace + mf_S_trace)/2.
    xmf_out = np.mean(mf_G_trace, axis=1)
    xmf_N_out = np.mean(mf_N_trace, axis=1)
    xmf_S_out = np.mean(mf_S_trace, axis=1)
    xmf_out_hpd_68 = pm.stats.hpd(mf_G_trace.T, 0.68)
    xmf_out_hpd_95 = pm.stats.hpd(mf_G_trace.T, 0.95)
    xmf_N_hpd_68 = pm.stats.hpd(mf_N_trace.T, 0.68)
    xmf_N_hpd_95 = pm.stats.hpd(mf_N_trace.T, 0.95)
    xmf_S_hpd_68 = pm.stats.hpd(mf_N_trace.T, 0.68)
    xmf_S_hpd_95 = pm.stats.hpd(mf_N_trace.T, 0.95)
    
    index_emis = np.round(time[::12]).astype(int)
    model_emis = pd.DataFrame(index=index_emis, data={"Global_emissions": x_out, \
                                                "Global_emissions_16": x_out_hpd_68[:,0], \
                                                "Global_emissions_84": x_out_hpd_68[:,1], \
                                                "Global_emissions_2.5": x_out_hpd_95[:,0], \
                                                "Global_emissions_97.5": x_out_hpd_95[:,1]}) 
    model_mf = pd.DataFrame(index=time, \
                            data={"Global_mf": xmf_out, \
                                  "Global_mf_16": xmf_out_hpd_68[:,0], \
                                  "Global_mf_84": xmf_out_hpd_68[:,1], \
                                  "Global_mf_2.5": xmf_out_hpd_95[:,0], \
                                  "Global_mf_97.5": xmf_out_hpd_95[:,1], \
                                  "N_mf": xmf_N_out, \
                                  "N_mf_16": xmf_N_hpd_68[:,0], \
                                  "N_mf_84": xmf_N_hpd_68[:,1], \
                                  "N_mf_2.5": xmf_N_hpd_95[:,0], \
                                  "N_mf_97.5": xmf_N_hpd_95[:,1], \
                                  "S_mf": xmf_S_out, \
                                  "S_mf_16": xmf_S_hpd_68[:,0], \
                                  "S_mf_84": xmf_S_hpd_68[:,1], \
                                  "S_mf_2.5": xmf_S_hpd_95[:,0], \
                                  "s_mf_97.5": xmf_S_hpd_95[:,1]})
    
    
    return model_emis, model_mf


#### This need to be tidied up – currenty it won't be compatible with
#### newer non-theano releases. 
from scipy import stats
import theano.tensor as tt
from theano import scan

from pymc3.util import get_variable_name
from pymc3.distributions.continuous import get_tau_sigma, Normal, Flat
from pymc3.distributions.shape_utils import to_tuple
from pymc3.distributions import multivariate
from pymc3.distributions import distribution

class RW2(distribution.Continuous):
    """Random Walk of order 2 with Normal innovations

    Parameters
    ----------
    mu: tensor
        innovation drift, defaults to 0.0
        For vector valued mu, first dimension must match shape of the random walk, and
        the first element will be discarded (since there is no innovation in the first timestep)
    sigma : tensor
        sigma > 0, innovation standard deviation (only required if tau is not specified)
        For vector valued sigma, first dimension must match shape of the random walk, and
        the first element will be discarded (since there is no innovation in the first timestep)
    tau : tensor
        tau > 0, innovation precision (only required if sigma is not specified)
        For vector valued tau, first dimension must match shape of the random walk, and
        the first element will be discarded (since there is no innovation in the first timestep)
    init : distribution
        distribution for initial value (Defaults to Flat())
    """

    def __init__(self, tau=None, init=Flat.dist(), sigma=None, mu=0.,
                 sd=None, *args, **kwargs):
        kwargs.setdefault('shape', 1)
        super().__init__(*args, **kwargs)
        if sum(self.shape) == 0:
            raise TypeError("RW2 must be supplied a non-zero shape argument!")
        if sd is not None:
            sigma = sd
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.tau = tt.as_tensor_variable(tau)
        sigma = tt.as_tensor_variable(sigma)
        self.sigma = self.sd = sigma
        self.mu = tt.as_tensor_variable(mu)
        self.init = init
        self.mean = tt.as_tensor_variable(0.)

    def _mu_and_sigma(self, mu, sigma):
        """Helper to get mu and sigma if they are high dimensional."""
        if sigma.ndim > 0:
            sigma = sigma[2:]
        if mu.ndim > 0:
            mu = mu[2:]
        return mu, sigma

    def logp(self, x):
        """
        Calculate log-probability of Gaussian Random Walk distribution at specified value.

        Parameters
        ----------
        x : numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        if x.ndim > 0:
            x_im1 = x[1:-1]
            x_im2 = x[:-2]
            x_i = x[2:]
            mu, sigma = self._mu_and_sigma(self.mu, self.sigma)
            innov_like = Normal.dist(mu=2*x_im1 - x_im2 + mu, sigma=sigma).logp(x_i)
            return self.init.logp(x[0]) + tt.sum(innov_like)
        return self.init.logp(x)