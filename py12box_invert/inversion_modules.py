import numpy as np
import pymc3 as pm
import pandas as pd
import json
from scipy.optimize import minimize
from py12box_invert.core import global_mf, hemis_mf, annual_means
from py12box_invert.kz_filter import kz_filter
from py12box_invert.invert import Store_model

def posterior_emissions(emissions_prior, freq_months, x_hat, P_hat):
    """Calculate linear adjustment to mole fraction

    Parameters
    ----------
    freq_moths : int
        Number of months in each emissions aggregation period (e.g. monthly, seasonal, annual)
    emissions_prior : ndarray
        A priori emissions estimate
    x_hat : ndarray
        Posterior emissions adjustment (Gg/yr)
    P_hat : ndarray
        Posterior emissions uncertainty covariance (Gg/yr)

    Returns 
    -------
    ndarray : 2d, n_months/freq_months x n_box
        Array of posterior emissions
    ndarray : 2d, n_months/freq_months x n_box
        Array of posterior emissions uncertainties (1-sigma)
    """

    emissions = emissions_prior.copy()
    emissions_sd = np.zeros_like(emissions)
    for ti in range(int(emissions.shape[0]/freq_months)):
        for bi in range(4):
            emissions[ti*freq_months:(ti+1)*freq_months, bi] += x_hat[4*ti + bi]
            emissions_sd[ti*freq_months:(ti+1)*freq_months, bi] = np.sqrt(P_hat[4*ti + bi, 4*ti + bi])

    return emissions, emissions_sd


def posterior_mf(mf_prior, sensitivity, x_hat, P_hat, calc_uncertainty=True):
    """Calculate linear adjustment to mole fraction array

    Parameters
    ----------
    mf_prior : ndarray
        A priori mole fractions
    sensitivity : ndarray
        Sensitivity matrix (not filtered for missing observations)
    x_hat : ndarray
        Posterior adjustment to emissions (Gg/yr)
    P_hat : ndarray
        Uncertainty covariance in posterior emissions (Gg/yr)
    calc_uncertainty : Boolean, default=True
        Set to false to return zeros instead of calculating uncertainty
    
    Returns 
    -------
    ndarray : 2d, n_months x n_box
        Array of posterior mole fractions
    ndarray : 2d, n_months x n_box
        Array of mole fraction uncertainties (1-sigma)
    """

    def mf_reshape(y):
        return y.reshape(int(len(y)/4), 4)

    # Note that this is labelled y_posterior, but dimension is larger than y
    #  because of obs that have been removed from y (sensitivity is used, rather than H)
    y_posterior = sensitivity @ x_hat
    if calc_uncertainty:
        R_posterior = sensitivity @ P_hat @ sensitivity.T
    else:
        R_posterior = np.zeros((len(y_posterior), len(y_posterior)))

    # Return arrays reshaped to match mf format (time, box)
    return mf_reshape(y_posterior) + mf_prior[:, :4], mf_reshape(np.sqrt(np.diag(R_posterior)))


def calc_lifetime_uncertainty(lifetime_fractional_error, steady_state_lifetime, burden, emissions):
    """Lifetime uncertainty function following Rigby et al., 2014

    Parameters
    ----------
    lifetime_fractional_error : float
        Relative uncertainty in 1/lifetime
    steady_state_lifetime : float
        Steady state lifetime in years
    burden : ndarray
        Burden in each atmospheric box at each timestep
    emissions : ndarray
        Emissions in four surface boxes at each timestep

    Returns
    -------
    ndarray
        Lifetime uncertainty function (Gg/yr)
    """

    global_burden = burden.sum(axis=1)

    # Scale the burden by absolute emissions in each box
    global_emissions = np.abs(emissions).sum(axis=1)
    burden_scaled = np.vstack([global_burden * np.abs(emissions[:, bi])/global_emissions for bi in range(4)]).T

    #TODO: Make unit conversion more robust throughout. Does this work if emissions are in Tg/yr?
    lifetime_uncertainty = burden_scaled * lifetime_fractional_error / steady_state_lifetime / 1e9

    return lifetime_uncertainty


class Inverse_method:
    """Inverse method modules

    Each method must have a corresponding {method}_posterior function that allows
    posterior mole fraction and emissions matrices to be calculated
    """

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
        self.mat.x_hat = np.linalg.inv(self.mat.H.T @ R_inv @ self.mat.H + self.mat.P_inv) @ \
                        (self.mat.H.T @ R_inv @ self.mat.y + self.mat.P_inv @ self.mat.x_a)
        self.mat.P_hat = np.linalg.inv(self.mat.H.T @ R_inv @ self.mat.H + self.mat.P_inv)
    
    
    def analytical_gaussian_posterior(self):
        """Method to process posterior mole fractions and emissions
        """

        # Posterior emissions
        #self.mod_posterior.emissions, self.mod_posterior.emissions_sd = posterior_emissions(self.mod_prior.emissions,
        emissions, emissions_sd = posterior_emissions(self.mod_prior.emissions,
                                                        self.sensitivity.freq_months,
                                                        self.mat.x_hat,
                                                        self.mat.P_hat)

        # Posterior mole fraction
        #self.mod_posterior.mf, self.mod_posterior.mf_sd = posterior_mf(self.mod_prior.mf,
        mf, mf_sd = posterior_mf(self.mod_prior.mf,
                                    self.sensitivity.sensitivity,
                                    self.mat.x_hat,
                                    self.mat.P_hat)

        # Rerun forward model with optimized emissions
        self.mod.emissions = emissions.copy()
        self.mod.run(verbose=False)

        # Check that mole fraction agrees with sensitivity*emissions
        if not np.allclose(self.mod.mf[:, :4], mf, rtol=0.001):
            raise Exception("Optimized model run doesn't match linear prediction")

        # Store posterior model
        self.mod_posterior = Store_model(self.mod)

        # Add in uncertainties
        self.mod_posterior.emissions_sd = emissions_sd
        self.mod_posterior.mf_sd = mf_sd


    def analytical_gaussian_posterior_ensemble(self,
                                                n_sample=1000,
                                                scale_error=0.,
                                                lifetime_error=0.,
                                                transport_error=0.01):
        """Create an ensemble of posterior emissions and mole fractions

        Parameters
        ----------
        n_sample : int, optional
            Number of samples of posterior distribution, by default 1000
        scale_error : float, optional
            Fractional scale uncertainty (e.g., 0.02 for 2%), by default 0.
        lifetime_error : float, optional
            Fractional uncertainty in 1/lifetime, by default 0.
        transport_error : float, optional
            Fractional uncertainty due to model transport, by default 0.01

        Returns
        -------
        ndarray (n_months, n_boxes, n_samples)
            Posterior emissions ensemble
        ndarray (n_months, n_boxes, n_samples)
            Posterior mole fraction ensemble
        """

        emissions_ensemble = np.zeros(self.mod_prior.emissions.shape + (n_sample,))

        L_hat = np.linalg.cholesky(self.mat.P_hat)

        emissions_lifetime_uncertainty = calc_lifetime_uncertainty(lifetime_error,
                                                                self.mod.steady_state_lifetime,
                                                                self.mod_posterior.burden,
                                                                self.mod_posterior.emissions)

        emissions_ensemble = []
        mf_ensemble = []

        for i in range(n_sample):
            x_sample = self.mat.x_hat + L_hat @ np.random.normal(size=self.mat.x_hat.size)

            mf_sample, _ = posterior_mf(self.mod_prior.mf,
                                        self.sensitivity.sensitivity,
                                        x_sample,
                                        self.mat.P_hat,
                                        calc_uncertainty=False)
            
            emissions_sample, _ = posterior_emissions(self.mod_prior.emissions,
                                                    self.sensitivity.freq_months,
                                                    x_sample,
                                                    self.mat.P_hat)

            # Scale uncertainty
            mf_sample *= (1. + np.random.normal() * scale_error)
            emissions_sample *= (1. + np.random.normal() * scale_error)

            # Transport uncertainty
            emissions_sample *= (1. + np.random.normal() * transport_error)

            # Lifetime uncertainty
            emissions_sample += np.random.normal() * emissions_lifetime_uncertainty

            emissions_ensemble.append(emissions_sample)
            mf_ensemble.append(mf_sample)

        return np.dstack(emissions_ensemble), np.dstack(mf_ensemble)

    
    def rigby14(self):
        """Emissions growth-constrainted method of Rigby et al., 2011 and 2014
        """
        
        # Difference operator
        nx = len(self.mat.x_a)
        D = np.zeros((nx, nx))
        for xi in range(nx):
            if (xi % (nx/4)) != nx/4 - 1:
                D[xi, xi] = -1.
                D[xi, xi+1] = 1.

        H = self.mat.H.copy()

        R_inv = np.linalg.inv(self.mat.R)
        self.mat.P_hat = np.linalg.inv(H.T @ R_inv @ H + D.T @ self.mat.P_inv @ D)
        self.mat.x_hat = self.mat.P_hat @ (H.T @ R_inv @ self.mat.y + D.T @ self.mat.P_inv @ self.mat.x_a)
        
        
    def rigby14_posterior(self):
        """
        The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()
        
    def rigby14_annualemissions(self, lifetime_err_path=None):
        """
        The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_annualemissions(lifetime_err_path)
        
    def rigby14_annualmf(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualmf()
     
    def rigby14_growthrate(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_growthrate()


    def iterative_rigby14(self):
        """
        As rigby14 but optimises x_hat such that all values are >=0. 
        Assumes linearity in the uncertainty.
        """
        H = self.mat.H.copy()
        R_inv = np.linalg.inv(self.mat.R)
        # Difference operator
        nx = len(self.mat.x_a)
        D = np.zeros((nx, nx))
        for xi in range(nx):
            if (xi % (nx/4)) != nx/4 - 1:
                D[xi, xi] = -1.
                D[xi, xi+1] = 1.
        #Lower limit
        freq_months = self.sensitivity.freq_months
        apriori = self.mod_prior.emissions.copy()
        llim = np.zeros(int(len(apriori.ravel())/(freq_months)))
        for bi in range(4):
            llim[bi::4] = -1*np.mean(apriori[:,bi].reshape(-1, freq_months), axis=1)
        bounds = tuple((ll, None) for ll in llim)
        #Cost function to minimise
        def cost(x):
            return (self.mat.y-H@x).T @ R_inv @ (self.mat.y-H@x) + (D@x-self.mat.x_a).T @ self.mat.P_inv @ (D@x-self.mat.x_a)
        cst = 1e18
        #Initialise with 5 different starting values to avoid local minima
        for i in range(5):
            xout0 = minimize(cost, llim-llim*np.random.random(size=len(llim)), method='L-BFGS-B', 
                            bounds=bounds)
            if xout0.fun < cst:
                cst = xout0.fun
                xout = xout0
        self.mat.x_hat = xout.x
        self.mat.P_hat = np.linalg.inv(H.T @ R_inv @ H + D.T @ self.mat.P_inv @ D)

    def iterative_rigby14_posterior(self):
        """The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()
        
    def iterative_rigby14_annualemissions(self, lifetime_err_path=None):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualemissions(lifetime_err_path)
        
    def iterative_rigby14_annualmf(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualmf()

    def iterative_rigby14_growthrate(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_growthrate()


    def empirical_bayes(self):
        """
        As iterative_rigby14 but optimises an iid model uncertainty. 
        i.e. there is an additional model error added to the measurement error, which is 
        assumed to be the same at all times and in all boxes.
        It currently also assumes that the R matrix is diagonal to speed up having to do a matrix
        inversion at each step.
        """
        print("Caution: R matrix is assumed to be diagonal.")
        H = self.mat.H.copy()
        R = self.mat.R
        # Difference operator
        nx = len(self.mat.x_a)
        ny = len(self.mat.y)
        D = np.zeros((nx, nx))
        for xi in range(nx):
            if (xi % (nx/4)) != nx/4 - 1:
                D[xi, xi] = -1.
                D[xi, xi+1] = 1.
        #Lower limit
        freq_months = self.sensitivity.freq_months
        apriori = self.mod_prior.emissions.copy()
        llim = np.zeros(int(len(apriori.ravel())/(freq_months)))
        for bi in range(4):
            llim[bi::4] = -1*np.mean(apriori[:,bi].reshape(-1, freq_months), axis=1)
        llim = np.append(llim,[1e-20])
        bounds = tuple((ll, None) for ll in llim)
        #Cost function to minimise
        def cost(x):
            R_inv =1./(np.diag(R) + x[-1])
            return np.sum((self.mat.y-H@x[:-1])**2 / R_inv) + \
                   (D@x[:-1]-self.mat.x_a).T @ self.mat.P_inv @ (D@x[:-1]-self.mat.x_a) + \
                   np.sum(np.log(R_inv))
        cst = 1e18
        #Initialise with 5 different starting values to avoid local minima
        for i in range(5):
            init0 = llim-llim*np.random.random(size=len(llim))
            init0[-1] = np.random.random()*2.
            xout0 = minimize(cost, init0, method='L-BFGS-B', 
                            bounds=bounds)
            if xout0.fun < cst:
                cst = xout0.fun
                xout = xout0
        self.mat.x_hat = xout.x[:-1]
        self.mat.P_hat = np.linalg.inv(H.T @ np.linalg.inv(R + np.diag(np.repeat(xout.x[-1], ny))) @ H + \
                                       D.T @ self.mat.P_inv @ D)

    def empirical_bayes_posterior(self):
        """The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()
        
    def empirical_bayes_annualemissions(self, lifetime_err_path=None):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualemissions(lifetime_err_path)
        
    def empirical_bayes_annualmf(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualmf()
        
    def empirical_bayes_growthrate(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_growthrate()
        

    # def lifetime_var(self, lifetime_err_path):
    #     """
    #     Emissions uncertainty variance added due to lifetime uncertainty.
    #     1/Lifetime 1sd uncertainties (in %) from Rigby et al 2014 / SPARC 2013
    #     sig_emissions = B*sig_1/lifetime
    #     Assumed to be independent year-to-year.
        
    #     Inputs:
    #         lifetime_err_path : str, optional
    #             Path to the JSON file containing the inverse lifetime error as a decimal 
    #             fraction (e.g. "inverse_lifetime_error":[{"CFC-11:0.1"}] for 10%
    #             uncertainty in the lifetime). The error is then added to the posterior
    #             emissions uncertainty. Ignored if set to None (default).
    #     """
    #     with open(lifetime_err_path) as f:
    #         lt_err_data = json.load(f)
    #     lifetime_uncertainty = lt_err_data["inverse_lifetime_error"]
    #     if self.mod.species not in lifetime_uncertainty.keys():
    #         raise Exception("No lifetime uncertainty for species {}".format(self.mod.species))
    #     n_months = int(12./self.sensitivity.freq_months)
    #     burden_sd = np.zeros_like(self.mod_posterior.annualemissionssd)
    #     for i in range(len(self.mod_posterior.annualemissionssd)):
    #         burden_sd[i] = self.mod.burden.sum(axis=1)[(i*n_months):((i+1)*n_months)].mean()/1e9
            
    #     return (1/self.mod.steady_state_lifetime*lifetime_uncertainty[self.mod.species]*burden_sd)**2

    # def modeltransport_var(self):
    #     """
    #     Transport error translates to 1% of emissions uncertainty.
    #     """
    #     return self.mat.P_hat*0.01**2

    # def analytical_gaussian_annualemissions(self, lifetime_err_path=None):
    #     """
    #     Calculate annual total annual emissions and its uncertaity
    #     Inputs:
    #         lifetime_err_path : str, optional
    #             Path to the JSON file containing the inverse lifetime error as a decimal 
    #             fraction (e.g. "inverse_lifetime_error":[{"CFC-11:0.1"}] for 10%
    #             uncertainty in the lifetime). The error is then added to the posterior
    #             emissions uncertainty. Ignored if set to None (default).
    #     """

    #     # Annual emissions and uncertainty from posterior covariance
    #     n_months = int(12./self.sensitivity.freq_months)
    #     self.mod_posterior.annual_emissions = np.add.reduceat(self.mod_posterior.emissions.sum(axis=1), 
    #                                                           np.arange(0, len(self.mod_posterior.emissions.sum(axis=1)), 12))/12
    #     self.mod_posterior.annual_emissions_sd = np.zeros_like(self.mod_posterior.annual_emissions)

    #     # Add in model and lifetime uncertainty
    #     R_expand = self.mat.P_hat + self.modeltransport_var()
    #     for i in np.arange(len(self.mod_posterior.annual_emissions_sd)):
    #         if lifetime_err_path:
    #             self.mod_posterior.annual_emissions_sd[i] = np.sqrt(R_expand[(i*n_months*4):((i+1)*n_months*4),(i*n_months*4):((i+1)*n_months*4)].sum()/(n_months)**2 + \
    #                                                   self.lifetime_var(lifetime_err_path)[i])
    #         else:
    #             self.mod_posterior.annual_emissions_sd[i] = np.sqrt(R_expand[(i*n_months*4):((i+1)*n_months*4),(i*n_months*4):((i+1)*n_months*4)].sum()/(n_months)**2) 

            
    # def analytical_gaussian_annualmf(self):
    #     """
    #     Calculate annual global and semi-hemispheric mf with uncertianties
    #     """
    #     # Calculate annual mf in each semi-hemisphere and globally with uncertainties
    #     self.mod_posterior.annualmf = np.add.reduceat(self.mod_posterior.mf,
    #                                                   np.arange(0,len(self.mod.time), 12), axis=0)/12.
    #     self.mod_posterior.annualglobalmf = self.mod_posterior.annualmf.copy().mean(axis=1)
    #     R_hat = (self.sensitivity.sensitivity @ self.mat.P_hat @ self.sensitivity.sensitivity.T)*1.02**2 #2% calibration error

    #     self.mod_posterior.annualmfsd = np.zeros_like(self.mod_posterior.annualmf)
    #     self.mod_posterior.annualglobalmfsd = np.zeros_like(self.mod_posterior.annualglobalmf)
    #     for i in range(len(self.mod_posterior.annualglobalmf)):
    #         self.mod_posterior.annualglobalmfsd[i] = np.sqrt(R_hat[i*48:(i+1)*48].sum()/48**2)
    #     for bi in range(4):
    #         R_hat_bx = R_hat[bi::4,bi::4]
    #         for i in range(self.mod_posterior.annualglobalmf.shape[0]):
    #             self.mod_posterior.annualmfsd[i,bi] = np.sqrt(R_hat_bx[i*12:(i+1)*12].sum()/12**2)
        

    # def analytical_gaussian_growthrate(self, nsample=1000):
    #     """
    #     Calculate growth rates using KZ filter.
    #     Monte Carlo sample mfs from posterior predicted covariance.
        
    #     Inputs:
    #         nsample:
    #             Number of Monte Carlo samples
    #     """
    #     L_hat = np.linalg.cholesky(self.mat.P_hat)
    #     sample = self.sensitivity.sensitivity @ L_hat @ np.random.normal(size=(self.sensitivity.sensitivity.shape[1], nsample))
    #     gr_array = np.zeros((len(self.mod_posterior.mf[6:-6,0]), 4, nsample))
    #     gr_annual_array = np.zeros((len(self.mod_posterior.mf[12:-12:12,0]), 4, nsample))
    #     gr = np.zeros((len(self.mod_posterior.mf[6:-6,0]), 4))
    #     gr_annual = np.zeros((len(self.mod_posterior.mf[12:-12:12,0]), 4))

    #     # Just x12 to convert from per months to per year
    #     for si in range(nsample):
    #         for bx in range(4):
    #             gr_diff = np.diff(self.mod_posterior.mf[:,bx]+sample[bx::4,si])
    #             gr_array[:,bx,si] = kz_filter(gr_diff, 12, 1)*12
    #             gr_annual_array[:,bx,si] = np.mean(gr_array[6:-6,bx,si].reshape(-1, 12), axis=1)
    #     gr_sd = np.std(gr_array, axis=2)
    #     gr_annual_sd  = np.std(gr_annual_array, axis=2)
    #     for bx in range(4):
    #         gr_diff = np.diff(self.mod_posterior.mf[:,bx])
    #         gr[:,bx] = kz_filter(gr_diff, 12, 1)*12
    #         gr_annual[:,bx] = np.mean(gr[6:-6, bx].reshape(-1, 12), axis=1)

    #     self.growth_rate.time = self.mod.time[6:-6]
    #     self.growth_rate.mf = gr
    #     self.growth_rate.mfsd = gr_sd
    #     self.growth_rate.annualmf = gr_annual
    #     self.growth_rate.annualmfsd = gr_annual_sd        
        
