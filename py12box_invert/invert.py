import numpy as np
from copy import deepcopy
from tqdm import tqdm
from bisect import bisect
from math import ceil
from multiprocessing import Pool

from py12box_invert.obs import Obs
from py12box.model import Model
#from py12box.core import flux_sensitivity


class Prior_model:
    """Class to store some of the parameters from the prior model

    """
    def __init__(self, mod):

        # Model inputs
        self.time = mod.time.copy()
        self.emissions = mod.emissions.copy()
        self.ic = mod.ic.copy()
        self.lifetime = mod.lifetime.copy()
        self.steady_state_lifetime = mod.steady_state_lifetime.copy()
        
        # Model outputs
        self.mf = mod.mf.copy()


class Matrices:
    """Empty class to store inversion matrices
    """
    pass


class Invert:

    def __init__(self, project_path, species,
                        obs_path=None, 
                        start_year = None,
                        end_year = None,
                        ic0=None, 
                        emissions_sd=1., freq="monthly", MCMC=False,
                        nit=10000, tune=None, burn=None):
        

        # Get obs
        if not obs_path and not (project_path / f"{species}_obs.csv").exists():
            raise Exception("No obs file given.")
        elif (project_path / f"{species}_obs.csv").exists():
            obs_path = project_path / f"{species}_obs.csv"
        
        self.obs = Obs(obs_path, start_year=start_year)

        # Get model inputs
        self.mod = Model(species, project_path, start_year=start_year)

        # Align model and obs, and change start/end dates, if needed
        if start_year:
            self.change_start_year(start_year)
        else:
            # Align to obs dataset.
            self.change_start_year(int(self.obs.time[0]))

        if end_year:
            self.change_end_year(end_year)
        else:
            #Align to obs dataset
            self.change_end_year(int(self.obs.time[-1])+1)

        # Reference run
        print("Model reference run...")
        self.mod.run()

        # Store some inputs and outputs from prior model
        # TODO: Note that use of the change_start_date or Change_end_date methods
        # may cause the prior model to become mis-aligned. 
        # To align, need to re-run model and then Prior_model step.
        # Check if this is a problem.
        self.mod_prior = Prior_model(self.mod)

        # Area to store inversion matrices
        self.mat = Matrices()


    def change_start_year(self, start_year):
        """Simple wrapper for changing start year for obs and model

        Parameters
        ----------
        start_year : flt
            New start year
        """

        self.obs.change_start_year(start_year)
        self.mod.change_start_year(start_year)
        #TODO: Add sensitivity?


    def change_end_year(self, end_year):
        """Simple wrapper for changing end year for obs and model

        Parameters
        ----------
        end_year : flt
            New end year
        """

        self.obs.change_end_year(end_year)
        self.mod.change_end_year(end_year)
        #TODO: Add sensitivity?


    def run_sensitivity(self, freq="monthly"):
        """
        Derive linear yearly flux sensitivities
        
            Parameters:
                freq (str, optional)       : Frequency to infer ("monthly", "quarterly", "yearly")
                                            Default is monthly.
            Attributes:
                Linear sensitivity to emissions
                Note that self.sensitivity can be reshaped using:
                np.reshape(sens, (int(sens.shape[0]/4), 4, int(sens.shape[1]/4), 4))
                where the dimensions are then [mf_time, mf_box, flux_time, flux_box]

        """

        if not np.allclose(self.mod.time, self.mod_prior.time):
            raise Exception('''Prior model has become mis-aligned with model,
                                probably because start or end dates have been changed.
                                Before calculating sensitivity, re-run model and store Prior_model''')

        if not np.allclose(self.mod.time, self.obs.time):
            raise Exception('''Model has become mis-aligned with obs,
                                probably because start or end dates have been changed for one or the other.
                                ''')

        freq = freq.lower()
        if freq not in ["monthly", "quarterly", "yearly"]:
            raise Exception('Frequency must be "monthly", "quarterly" or "yearly"')
        
        freq_months={"monthly":1, "quarterly":3, "yearly":12}[freq]
        
        nmonths = len(self.mod_prior.time)

        if nmonths % 12:
            raise Exception("Emissions must contain whole years")
        
        # number of discrete periods over which emissions sensitivity is calculated
        nsens = int(nmonths/freq_months)

        # empty sensitivity matrix. Factors of 4 are for surface boxes 
        # (mole fraction in rows, emissions in columns)
        self.sensitivity = np.zeros((nmonths*4, nsens*4))

        def sensitivity_section(mod, mod_prior, nsens_section, t0):

            sens = np.zeros((len(mod.mf[:, :4].flatten()), nsens_section*4))

            for ti in range(nsens_section):
                for bi in range(4):

                    # Perturb emissions uniformly throughout specified time period
                    emissions_perturbed = mod_prior.emissions.copy()
                    emissions_perturbed[(t0 + ti)*freq_months:freq_months*(t0+ti+1), bi] += 1.
                    mod.emissions = emissions_perturbed.copy()

                    # Run perturbed model
                    mod.run(verbose=False)
                    
                    # Store sensitivity column
                    sens[:, 4*ti + bi] = (mod.mf[:,:4].flatten() - mod_prior.mf[:,:4].flatten()) / 1.

            return sens

        nthreads=12
        nsens_section = ceil(nsens/nthreads)

        #with Pool(processes=nthreads) as pool:
        for thread in range(nthreads):
            
            if nsens_section * (thread + 1) > nsens:
                nsens_out = nsens - (nsens_section * thread)
            else:
                nsens_out = nsens_section
            
            if nsens_out > 0:
                self.sensitivity[:, thread*nsens_section*4 : thread*nsens_section*4 + nsens_out*4] = \
                    sensitivity_section(self.mod,
                                        self.mod_prior,
                                        nsens_out,
                                        nsens_section*thread)

        # for ti in tqdm(range(nsens)):
        #     for bi in range(4):

        #         # Perturb emissions uniformly throughout specified time period
        #         emissions_perturbed = self.mod_prior.emissions.copy()
        #         emissions_perturbed[ti*freq_months:freq_months*(ti+1), bi] += 1
        #         self.mod.emissions = emissions_perturbed.copy()

        #         # Run perturbed model
        #         self.mod.run(verbose=False)
                
        #         # Store sensitivity column
        #         self.sensitivity[:, 4*ti + bi] = (self.mod.mf[:,:4].flatten() - self.mod_prior.mf[:,:4].flatten()) / 1.


    def create_matrices(self, sigma_P=None):

        wh_obs = np.isfinite(self.obs.mf.flatten())
        nx = self.sensitivity.shape[1]

        self.mat.H = self.sensitivity[wh_obs,:]

        # Flatten model outputs, noting that all 12 boxes are output (compared to 4 for obs)
        self.mat.y = self.obs.mf[:, :4].flatten()[wh_obs] - self.mod_prior.mf[:, :4].flatten()[wh_obs]

        #TODO: Functions to choose uncertainty estimation method
        self.mat.R = np.diag(self.obs.mf_uncertainty.flatten()[wh_obs]**2)
    
        #TODO: Function to choose emissions uncertainty method
        # this is just a placeholder
        self.mat.P_inv = np.linalg.inv(np.diag(np.ones(nx)*sigma_P**2))
    
        self.mat.x_a = np.zeros(nx)
