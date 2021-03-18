import numpy as np
from copy import deepcopy
from tqdm import tqdm

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
                        obs_path=None, ic0=None, 
                        emissions_sd=1., freq="monthly", MCMC=False,
                        nit=10000, tune=None, burn=None):
        

        # Get obs
        if not obs_path and not (project_path / f"{species}_obs.csv").exists():
            raise Exception("No obs file given.")
        elif (project_path / f"{species}_obs.csv").exists():
            obs_path = project_path / f"{species}_obs.csv"
        
        self.obs = Obs(obs_path)
        
        # Get model inputs
        self.mod = Model(species, project_path)

        # Reference run
        print("Model reference run...")
        self.mod.run()

        # Store some inputs and outputs from prior model
        self.mod_prior = Prior_model(self.mod)


    def calc_sensitivity(self, freq="monthly"):
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
        
        #emis0 = self.mod.emissions.T.reshape(-1,meanfreq[freq]).mean(1).reshape(4,-1).T.flatten()

        # emissions = np.zeros_like(self.mod_prior.emissions)
                
        ntime = len(self.mod_prior.time)

        if ntime % 12:
            raise Exception("Emissions must contain whole years")
            
        if freq == "yearly":
            nyear = int(ntime/12)
            self.sensitivity = np.zeros((ntime*4, nyear*4))

            for mi in tqdm(range(nyear)):
                for bi in range(4):

                    emissions_perturbed = self.mod_prior.emissions.copy()
                    emissions_perturbed[mi*12:(12*mi+12), bi] +=1

                    self.mod.emissions = emissions_perturbed.copy()
                    self.mod.run(verbose=False)
                    
                    self.sensitivity[:, 4*mi + bi] = (self.mod.mf[:,:4].flatten() - self.mod_prior.mf[:,:4].flatten()) / 1.

    #     elif freq == "monthly":
    #         nmonth = len(emissions)
    #         sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nmonth*4)))

    #         for mi in tqdm(range(nmonth)):
    #             for bi in range(4):

    #                 emissions_perturbed = emissions.copy()
    #                 emissions_perturbed[mi, bi] +=1

    #                 mod.emissions = emissions_perturbed
    #                 mod.run(verbose=False)
    #                 mf_perturbed = mod.mf
    #                 sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.

    #     elif freq == "quarterly":
    #         nquart = int(len(emissions)/3.)
    #         sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nquart*4)))

    #         for mi in tqdm(range(nquart)):
    #             for bi in range(4):

    #                 emissions_perturbed = emissions.copy()
    #                 emissions_perturbed[mi*3:(3*mi+3), bi] +=1

    #                 mod.emissions = emissions_perturbed
    #                 mod.run(verbose=False)
    #                 mf_perturbed = mod.mf
    #                 sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.
                
    # return sensitivity, mf_ref[:,:4].flatten(), emissions, mod.time, emis0



    # # # Pad obs to senstivity
    # # obs, obs_sd = utils.pad_obs(mf_box, mf_var_box, time, obstime)

        