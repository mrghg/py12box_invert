import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.optimize import minimize
from py12box_invert.core import global_mf, hemis_mf, annual_means


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
        # Posterior mole fraction
        y_hat = self.sensitivity.sensitivity @ self.mat.x_hat + self.mod_prior.mf[:, :4].flatten()
        self.mod_posterior.mf = y_hat.reshape(int(len(y_hat)/4), 4)
        R_hat = self.sensitivity.sensitivity @ self.mat.P_hat @ self.sensitivity.sensitivity.T
        self.mod_posterior.mfsd = np.sqrt(np.diag(R_hat)).reshape(int(len(y_hat)/4), 4)
        
        # Posterior emissions
        freq_months = self.sensitivity.freq_months
        self.mod_posterior.emissions = self.mod_prior.emissions.copy()
        self.mod_posterior.emissionssd = np.zeros_like(self.mod_posterior.emissions)
        for ti in range(int(self.mod_prior.emissions.shape[0]/freq_months)):
            for bi in range(4):
                self.mod_posterior.emissions[ti*freq_months:(ti+1)*freq_months, bi] += self.mat.x_hat[4*ti + bi] 
                self.mod_posterior.emissionssd[ti*freq_months:(ti+1)*freq_months, bi] = np.sqrt(self.mat.P_hat[4*ti + bi, 4*ti + bi])
    
    def analytical_gaussian_annualemissions(self):
        """
        Calculate annual total annual emissions and its uncertaity
        """        
        n_months = int(12./self.sensitivity.freq_months)
        self.mod_posterior.annualemissions = np.add.reduceat(self.mod_posterior.emissions.sum(axis=1), 
                                                             np.arange(0, len(self.mod_posterior.emissions.sum(axis=1)), 12))/12
        self.mod_posterior.annualemissionssd = np.zeros_like(self.mod_posterior.annualemissions)
        R_expand = self.mat.P_hat + self.modeltransport_var()
        for i in np.arange(len(self.mod_posterior.annualemissionssd)):
            #self.mod_posterior.annualemissionssd[i] = np.sqrt(self.mat.P_hat[(i*n_months*4):((i+1)*n_months*4),(i*n_months*4):((i+1)*n_months*4)].sum()/(n_months)**2)
            self.mod_posterior.annualemissionssd[i] = np.sqrt(R_expand[(i*n_months*4):((i+1)*n_months*4),(i*n_months*4):((i+1)*n_months*4)].sum()/(n_months)**2 + \
                                                      self.lifetime_var()[i])


            
    def analytical_gaussian_annualmf(self):
        """
        Calculate annual global and hemispheric mf with uncertianties
        """
        # Calculate annual mf in each semi-hemisphere and globally with uncertainties
        self.mod_posterior.annualmf = np.add.reduceat(self.mod_posterior.mf, np.arange(0,len(self.mod.time), 12), axis=0)/12.
        self.mod_posterior.annualglobalmf = self.mod_posterior.annualmf.copy().mean(axis=1)
        R_hat = self.sensitivity.sensitivity @ self.mat.P_hat @ self.sensitivity.sensitivity.T
        self.mod_posterior.annualmfsd = np.zeros_like(self.mod_posterior.annualmf)
        self.mod_posterior.annualglobalmfsd = np.zeros_like(self.mod_posterior.annualglobalmf)
        for i in range(len(self.mod_posterior.annualglobalmf)):
            self.mod_posterior.annualglobalmfsd[i] = np.sqrt(R_hat[i*48:(i+1)*48].sum()/48**2)
        for bi in range(4):
            R_hat_bx = R_hat[bi::4,bi::4]
            for i in range(self.mod_posterior.annualglobalmf.shape[0]):
                self.mod_posterior.annualmfsd[i,bi] = np.sqrt(R_hat_bx[i*12:(i+1)*12].sum()/12**2)
        
        
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
        
        
    def rigby14_posterior(self):
        """The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()
        
    def rigby14_annualemissions(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualemissions()
        
    def rigby14_annualmf(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualmf()
    
    def iterative_rigby14_posterior(self):
        """The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()
        
    def iterative_rigby14_annualemissions(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualemissions()
        
    def iterative_rigby14_annualmf(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualmf()

    def empirical_bayes_posterior(self):
        """The same as a standard analytical Gaussian
        """

        self.analytical_gaussian_posterior()
        
    def empirical_bayes_annualemissions(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualemissions()
        
    def empirical_bayes_annualmf(self):
        """
        The same as a standard analytical Gaussian
        """
        
        self.analytical_gaussian_annualmf()
        
#     def add_emissions_uncertainty(self):
#         """
#         Follow the approach of Rigby et al 2014 where uncertainty is added post hoc 
#         to the emissions and mole fraction. This uncertainty is due to model error 
#         and lifetime uncertainty. A Monte Carlo approach is taken for mf.
#         """
    def lifetime_var(self):
        """
        Emissions uncertainty stddev added due to lifetime uncertainty.
        1/Lifetime 1sd uncertainties (in %) from Rigby et al 2014 / SPARC 2013
        sig_emissions = B*sig_1/lifetime
        """
        lifetime_uncertainty = {"CFC-11" : 0.10, "CFC-12": 0.20,"CFC-13" : 0.20, "CFC-113" : 0.20,
                                "CFC-114" : 0.20,"CFC-115" : 0.20,"CCl4" : 0.20,"CH3CCl3" : 0.14,
                                "HCFC-22" : 0.16,"HCFC-141b" : 0.15,"HCFC-142b" : 0.20,"HCFC-124" : 0.20,
                                "HFC-23" : 0.21,"HFC-32" : 0.17,"HFC-125" : 0.17,"HFC-134a" : 0.18,
                                "HFC-143a" : 0.19,"HFC-152a" : 0.15,"HFC-227ea" : 0.21,"HFC-236fa" : 0.21,
                                "HFC-245fa" : 0.22,"HFC-365mfc" : 0.21,"SF6" : 0.0,"CF4" : 0.0,"C2F6" : 0.0,
                                "C3F8" : 0.0,"NF3" : 0.0}
        n_months = int(12./self.sensitivity.freq_months)
        burden_mnth_sd = self.mod.burden.sum(axis=1)
        burden_sd = np.zeros_like(self.mod_posterior.annualemissionssd)
        for i in np.arange(len(self.mod_posterior.annualemissionssd)):
            burden_sd[i] = burden_mnth_sd[(i*n_months):((i+1)*n_months)].mean()/1e9

        return (1/self.mod.steady_state_lifetime*lifetime_uncertainty[self.mod.species]*burden_sd)**2

    def modeltransport_var(self):
        """
        Transport error translates to 1% of emissions uncertainty.
        """
        return self.mat.P_hat*0.01**2       

        #return lifetime_sd() + modeltransport_sd()