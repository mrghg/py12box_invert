import numpy as np
import pandas as pd
from py12box import startup, core,  model
from tqdm import tqdm
from py12box_invert import utils
from pathlib import Path
import pymc3 as pm


def fwd_model_inputs(project_path, species):
    """
    Get inputs to 12-box model
    
        Parameters:
            project_path (pathlib path): Path to project
            species (str)              : Species name
        Returns:
            Box model class
    """
    mod = model.Model(species, project_path)
    return mod #time, ic, emissions, mol_mass, lifetime, F, temperature, cl, oh, oh_a, oh_er


def flux_sensitivity(project_path, species, ic0=None, freq="monthly"):
    """
    Derive linear yearly flux sensitivities
    
        Parameters:
            project_path (pathlib path): Path to project
            species (str)              : Species name
            ic0 (array)                : Initial conditions for 4 surface boxes
            freq (str, optional)       : Frequency to infer ("monthly", "quarterly", "yearly")
                                         Default is monthly.
        Returns:
            Linear sensitivity to emissions
            Reference mole fraction
            a priori emissions
            time of emissions

    """
    freq = freq.lower()
    if freq not in ["monthly", "quarterly", "yearly"]:
        raise Exception('Frequency must be "monthly", "quarterly" or "yearly"')
    meanfreq={"monthly":1, "quarterly":3, "yearly":12}
    mod = fwd_model_inputs(project_path, species)
    #emis0 = np.nanmean(mod.emissions.flatten().reshape(-1, meanfreq[freq]), axis=1)
    emis0 = mod.emissions.T.reshape(-1,meanfreq[freq]).mean(1).reshape(4,-1).T.flatten()
    if ic0 is not None:
        if len(ic0) == 4:
            ic = utils.approx_initial_conditions(species, project_path, ic0)
            mod.ic = ic
        else:
            print("ic0 does not have only 4 surface boxes. Ignoring.")

    emissions = np.zeros_like(mod.emissions)
    mod.emissions = emissions
    mod.run()
    mf_ref = mod.mf
    #mod.run()
    #mf_ref = mod.mf
    #emissions = mod.emissions
    
    if len(mod.time) % 12:
        raise Exception("Emissions must contain whole years")
        
    if freq=="yearly":
        nyear = int(len(emissions)/12)
        sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nyear*4)))

        for mi in tqdm(range(nyear)):
            for bi in range(4):

                emissions_perturbed = emissions.copy()
                emissions_perturbed[mi*12:(12*mi+12), bi] +=1

                mod.emissions = emissions_perturbed
                mod.run(verbose=False)
                mf_perturbed = mod.mf
                sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.

    elif freq == "monthly":
        nmonth = len(emissions)
        sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nmonth*4)))

        for mi in tqdm(range(nmonth)):
            for bi in range(4):

                emissions_perturbed = emissions.copy()
                emissions_perturbed[mi, bi] +=1

                mod.emissions = emissions_perturbed
                mod.run(verbose=False)
                mf_perturbed = mod.mf
                sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.

    elif freq == "quarterly":
        nquart = int(len(emissions)/3.)
        sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nquart*4)))

        for mi in tqdm(range(nquart)):
            for bi in range(4):

                emissions_perturbed = emissions.copy()
                emissions_perturbed[mi*3:(3*mi+3), bi] +=1

                mod.emissions = emissions_perturbed
                mod.run(verbose=False)
                mf_perturbed = mod.mf
                sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.
                
    return sensitivity, mf_ref[:,:4].flatten(), emissions, mod.time, emis0


def inversion_analytical(y, H, x_a, R, P_inv):
    """
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
    R_inv = np.linalg.inv(R)
    x_hat = np.linalg.inv(H.T @ R_inv @ H + P_inv) @ (H.T @ R_inv @ y + P_inv @ x_a)
    P_hat = np.linalg.inv(H.T @ R_inv @ H + P_inv)
    return x_hat, P_hat

def annual_means(x_hat, P_hat, emis_ref,  freq="monthly"):
    """
    Derive annual mean emissions from inversion output
    
        Parameters:
            x_hat (array)       : Posterior mean difference from a priori
            P_hat (array)       : Posterior covariance matrix
            emis_ref (array)    : Reference emissions
            freq (str, optional): Frequency to infer ("monthly", "quarterly", "yearly")
                                         Default is monthly.
        Returns:
            x_out (array)   : Annual emissions,
            x_sd_out (array): 1 std deviation uncertain in annual emissions
    """
    
    if freq == "yearly":
        x_mnth = np.sum(np.repeat(x_hat.reshape(int(len(x_hat)/4),4),12, axis=0)+ emis_ref, axis=1)
        x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
        x_sd_out = np.zeros(len(x_out))
        j = 0
        for i in range(0,len(x_hat),4):
            x_sd_out[j] = np.sqrt(np.sum(P_hat[i:(i+4),i:(i+4)]))
            j+=1
    elif freq == "monthly":
        x_mnth = np.sum(x_hat.reshape(int(len(x_hat)/4),4)+ emis_ref, axis=1)
        x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
        x_sd_out = np.zeros(len(x_out))
        j = 0
        for i in range(0,len(x_hat),48):
            x_sd_out[j] = np.sqrt(np.sum(P_hat[i:(i+48),i:(i+48)])/12)
            j +=1
    elif freq == "quarterly":
        x_mnth = np.sum(np.repeat(x_hat.reshape(int(len(x_hat)/4),4),3, axis=0)+ emis_ref, axis=1)
        x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
        x_sd_out = np.zeros(len(x_out))
        j = 0
        for i in range(0,len(x_hat),16):
            x_sd_out[j] = np.sqrt(np.sum(P_hat[i:(i+12),i:(i+12)])/3.)
            j+=1
    
    return x_out, x_sd_out


def global_mf(sensitivity, x_hat, P_hat, mf_ref):
    """
        Calculates linear predictor global mole fractions.
    
        Parameters:
            sensitivity (array): Array of sensitivity to emissions
            x_hat (array)      : Posterior mean difference from a priori
            P_hat (array)      : Posterior covariance matrix
            mf_array (array)   : Reference mole fraction
        Returns:
            xmf_out (array)   : Global mole fraction
            xmf_sd_out (array): Global mole fraction std dev
    """
    
    xmf_hat = sensitivity @ x_hat
    xmf_mnth = np.mean(xmf_hat.reshape(int(len(xmf_hat)/4),4)+ mf_ref.reshape(int(len(xmf_hat)/4),4), axis=1)
    xmf_out = np.mean(xmf_mnth.reshape(-1, 12), axis=1)
    xmf_sd_out = np.zeros(len(xmf_out))
    j = 0
    xmf_Cov = sensitivity @ P_hat @ sensitivity.T
    for i in range(0,len(xmf_hat),48):
        xmf_sd_out[j] = np.sqrt(np.sum(xmf_Cov[i:(i+48),i:(i+48)])/12)
        j +=1
        
    return xmf_out, xmf_sd_out

def hemis_mf(sensitivity, x_hat, P_hat, mf_ref):

    """
    Calculates linear predictor hemispheric mole fractions.
        Parameters:
            sensitivity (array): Array of sensitivity to emissions
            x_hat (array)      : Posterior mean difference from a priori
            P_hat (array)      : Posterior covariance matrix
            mf_array (array)   : Reference mole fraction
        Returns:
            xmf_N_out (array)   : N Hemisphere mole fraction
            xmf_N_sd_out (array): N Hemisphere mole fraction std dev
            xmf_S_out (array)   : S Hemisphere mole fraction
            xmf_S_sd_out (array): S Hemisphere mole fraction std dev
    """

    xmf_hat = sensitivity @ x_hat
    xmf_N_mnth = np.mean(xmf_hat.reshape(int(len(xmf_hat)/4),4)[:,:1] + \
                         mf_ref.reshape(int(len(xmf_hat)/4),4)[:,:1], axis=1)
    xmf_S_mnth = np.mean(xmf_hat.reshape(int(len(xmf_hat)/4),4)[:,2:] + \
                         mf_ref.reshape(int(len(xmf_hat)/4),4)[:,2:], axis=1)
    xmf_N_out = np.mean(xmf_N_mnth.reshape(-1, 12), axis=1)
    xmf_S_out = np.mean(xmf_S_mnth.reshape(-1, 12), axis=1)
    xmf_N_sd_out = np.zeros(len(xmf_N_out))
    xmf_S_sd_out = np.zeros(len(xmf_S_out))
    xmf_Cov = sensitivity @ P_hat @ sensitivity.T
    N_ind = np.arange(len(xmf_hat))
    S_ind = np.arange(len(xmf_hat))

    for i in range(2):
        N_ind = np.delete(N_ind, np.arange(2,len(N_ind),4-i))
        S_ind = np.delete(S_ind, np.arange(0,len(S_ind),4-i))
    xmf_N_Cov = xmf_Cov[tuple(np.meshgrid(N_ind,N_ind))]
    xmf_S_Cov = xmf_Cov[tuple(np.meshgrid(N_ind,N_ind))]
    j = 0
    for i in range(0,len(xmf_N_sd_out),24):
        xmf_N_sd_out[j] = np.sqrt(np.sum(xmf_N_Cov[i:(i+24),i:(i+24)])/12)
        xmf_S_sd_out[j] = np.sqrt(np.sum(xmf_S_Cov[i:(i+24),i:(i+24)])/12)
        j +=1
    
    return xmf_N_out, xmf_N_sd_out, xmf_S_out, xmf_S_sd_out

def inversion_matrices(obs, sensitivity, mf_ref, obs_sd, P_sd, emis0):
    """
    Drop sensitivities to no observations and set up matrices
    Prior uncertainty on emissions defaults to 100.
    
        Parameters:
            obs (array)          : Array of boxed observations
            sensitivity (array)  : Array of sensitivity to emissions
            mf_ref (array)       : Array of reference run mole fraction
            obs_sd (array)       : Observation error
            P_sd (array, options): Std dev uncertainty in % of a priori emissions 
                                   Defaults to 100% and min. of 1 Gg/box/yr
            
         Returns:
             H (array)         : Sensitivity matrix
             y (array)         : Deviation from a priori emissions
             R (square matrix) : Model-measurement covariance matrix
             P (square matrix) : Emissions covariance matrix
             x_a (array)       : A priori deviation (defaults to zeros)
    """

    wh_obs = np.isfinite(obs)
    
    H = sensitivity[wh_obs,:]
    
    y = obs[wh_obs] - mf_ref[wh_obs]
    
    R = np.diag(obs_sd[wh_obs]**2)
    
    P_inv = np.linalg.inv(np.diag(P_sd**2))
    
    x_a = emis0
    
    return H, y, R, P_inv, x_a


def run_inversion(project_path, species, obs_path=None, ic0=None, 
                  emissions_sd=1., freq="monthly", MCMC=False,
                  nit=10000, tune=None, burn=None):
    """
    Run inversion for 12-box model to estimate monthly means from 4 surface boxes.
    
        Parameters:
            project_path (pathlib path)  : Path to project
            species (str)                : Species name
            obs_path (pathlib path)
            ic0 (array)                  : Initial conditions for 4 surface boxes
            emissions_sd (array, options): Std dev uncertainty in % of a priori emissions 
                                           Defaults to 100%, and minimum of 10 Gg/box/yr
            freq (str, optional)         : Frequency to infer ("monthly", "quarterly", "yearly")
                                           Default is monthly.
            MCMC (bool, optional)        : Set to True toUse MCMC to derive emissions estimates. 
                                           Only currently works with freq="yearly". False uses analytical
                                           inversion.
            nit (int, optional)          : Number of steps in MCMC sampling.
            burn (int, optional)         : Number of steps to burn in MCMC sampling (defaults to 10%).
            tune (int, optional)         : Number of tuning stepins in MCMC sampling (defaults to 20%).
        Returns:
            model_mf (dataframe)         : Dataframe of inferred hemispheric and global mole fraction.
            model_emis (dataframe)       : Dataframe of inferred global emissions.
    """

    #Get obs
    if not obs_path and not (project_path / f"{species}_obs.csv").exists():
        raise Exception("No obs file given.")
    elif (project_path / f"{species}_obs.csv").exists():
        obs_path = project_path / f"{species}_obs.csv"
    
    obstime, mf_box, mf_var_box =  utils.obs_read(obs_path)
#    obstime = utils.decimal_date(obsdf.index)

    #Box obs
#    mf_box, mf_var_box = utils.obs_box(obsdf)
    
    #Get sensitivities
    sensitivity, mf_ref, emis_ref, time, emis0 = flux_sensitivity(project_path, species, ic0=ic0, freq=freq)

    # Pad obs to senstivity
    obs, obs_sd = utils.pad_obs(mf_box, mf_var_box, time, obstime)
    
    #Get matrices for inversion
    P_sd = emissions_sd*emis0
    P_sd[P_sd < 10.] = 10.
    H, y, R, P_inv, x_a = inversion_matrices(obs, sensitivity, mf_ref, obs_sd, P_sd, emis0)
    if MCMC:
        model_emis, model_mf = NUTS_expRW1(H, x_a, R, y, emis_ref, sensitivity, time,
                                           nit=nit, tune=tune, burn=burn, freq=freq)
    else:
        model_emis, model_mf = analytical_gaussian(y, H, x_a, R, P_inv, sensitivity, mf_ref,
                                                   emis_ref, freq, time) 
    
    return model_emis, model_mf


def analytical_gaussian(y, H, x_a, R, P_inv, sensitivity, mf_ref, emis_ref, freq, time):
    """
    Do an analytical Gaussian inversion, assuming conjugacy (Gaussian likelihood and prior)
    """

    #Do inversion
    x_hat, P_hat = inversion_analytical(y, H, x_a, R, P_inv)
    
    #Calculate global and hemispheric mole fraction
    xmf_out, xmf_sd_out = global_mf(sensitivity, x_hat, P_hat, mf_ref)
    xmf_N_out, xmf_N_sd_out, xmf_S_out, xmf_S_sd_out = hemis_mf(sensitivity, x_hat, P_hat, mf_ref)
    index = np.round(time[::12]).astype(int)
    model_mf = pd.DataFrame(index=index, \
                            data={"Global_mf": xmf_out, "Global_mf_sd": xmf_sd_out, \
                                    "N_mf":xmf_N_out, "N_mf_sd":xmf_N_sd_out, \
                                    "S_mf":xmf_S_out, "S_mf_sd":xmf_S_sd_out})
    
    #Calculate annual emissions
    x_out, x_sd_out = annual_means(x_hat, P_hat, emis_ref, freq=freq)
    model_emis = pd.DataFrame(index=index, data={"Global_emissions": x_out, \
                                                              "Global_emissions_sd": x_sd_out})  
    
    return model_emis, model_mf


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
