import numpy as np
from copy import deepcopy
from tqdm import tqdm
from bisect import bisect

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

        #TODO: Somewhere around here, align times for model and obs arrays

        # Get model inputs
        self.mod = Model(species, project_path, start_year=start_year)

        # Reference run
        print("Model reference run...")
        self.mod.run()

        # Store some inputs and outputs from prior model
        self.mod_prior = Prior_model(self.mod)


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

        for ti in tqdm(range(nsens)):
            for bi in range(4):

                # Perturb emissions uniformly throughout specified time period
                emissions_perturbed = self.mod_prior.emissions.copy()
                emissions_perturbed[ti*freq_months:freq_months*(ti+1), bi] += 1
                self.mod.emissions = emissions_perturbed.copy()

                # Run perturbed model
                self.mod.run(verbose=False)
                
                # Store sensitivity column
                self.sensitivity[:, 4*ti + bi] = (self.mod.mf[:,:4].flatten() - self.mod_prior.mf[:,:4].flatten()) / 1.


    # # # Pad obs to senstivity
    # # obs, obs_sd = utils.pad_obs(mf_box, mf_var_box, time, obstime)

        