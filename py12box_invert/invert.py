import numpy as np
from math import ceil
from multiprocessing import Pool

from py12box_invert.obs import Obs
from py12box_invert.plot_altair import Plot
from py12box_invert.inversion_modules import Inverse_method
from py12box_invert.utils import Store_model, aggregate_outputs, sensitivity_section
from py12box.model import Model, core

#from py12box.core import flux_sensitivity

class Matrices:
    """Empty class to store inversion matrices
    """
    pass

class Sensitivity:
    """Empty class to store sensitivity
    """
    pass

class Outputs:
    """Empty class to store outputs
    """
    pass

# class Growth_rate:
#     """Empty class to store growth rates
#     """
#     pass

# class Species:
#     """Empty class to store species info
#     """
#     pass


class Invert(Inverse_method, Plot):

    def __init__(self, project_path, species,
                        obs_path=None, 
                        start_year = None,
                        end_year = None,
                        method = "rigby14",
                        ic_years = 3):
        """Set up 12-box model inversion class

        Parameters
        ----------
        project_path : path-like object
            Path to directory containing prior emissions and observations
        species : str
            Species name
        obs_path : path-like object, optional
            Path to obs-file if not in project path, by default None
        start_year : flt, optional
            First year to start the inversion, by default None
        end_year : flt, optional
            Final year to run inversion up until
            (i.e., if you want to run through 2000, end_year=2001.), by default None
        method : str, optional
            Inverse method to choose. Must be in inversion_modules, by default "rigby14"
        ic_years : int, optional
            Number of years to pad the inversion before the first observation
            in order to allow for some spinup/discard years.
            Note that the emissions file must cover the implied period
            (i.e. (year of first obs - ic_years) onwards), by default 3

        Raises
        ------
        FileNotFoundError
            If obs files not found
        """
        # Store name of species
        self.species = species
        
        # Get obs
        if not obs_path and not (project_path / f"{species}_obs.csv").exists():
            raise FileNotFoundError("No obs file given.")
        elif (project_path / f"{species}_obs.csv").exists():
            obs_path = project_path / f"{species}_obs.csv"
        
        self.obs = Obs(obs_path)

        # Get model inputs
        self.mod = Model(species, project_path)

        # Align model and obs, and change start/end dates, if needed
        if ic_years and start_year:
            raise Exception("Can't have both a start_year and ic_year")

        if ic_years:
            start_year = int(self.obs.time[np.where(np.isfinite(self.obs.mf))[0][0]]) - \
                ic_years

        if start_year:
            self.change_start_year(start_year)
        else:
            # Align to obs dataset by default
            self.change_start_year(int(self.obs.time[0]))

        if end_year:
            self.change_end_year(end_year)
        else:
            #Align to obs dataset by default
            self.change_end_year(int(self.obs.time[-1])+1)
            end_year = self.obs.time[-1]

        # Reference run
        print("Model reference run...")
        self.mod.run()

        # Store some inputs and outputs from prior model
        # TODO: Note that use of the change_start_date or Change_end_date methods
        # may cause the prior model to become mis-aligned. 
        # To align, need to re-run model and then Store_model step.
        # Check if this is a problem.
        self.mod_prior = Store_model(self.mod)

        # Area to store inversion matrices
        self.mat = Matrices()

        # Area to store sensitivity
        self.sensitivity = Sensitivity()
        
        # #Area to store growth rate
        # self.growth_rate = Growth_rate()

        # Attach inverse method
        self.run_inversion = getattr(self, method)
        
        # Attach methods to process posterior
        self.posterior = getattr(self, f"{method}_posterior")
        self.posterior_ensemble = getattr(self, f"{method}_posterior_ensemble")

        self.outputs = Outputs()

        # # Calculate annual emissions and mf with uncertainties
        # self.annualmf = getattr(self, f"{method}_annualmf")
        # self.annualemissions = getattr(self, f"{method}_annualemissions")
        
        # # Calculate mf growth rate
        # self.growthrate = getattr(self, f"{method}_growthrate")


    def run_spinup(self, nyears=5):
        """Spin model up

        Spin up starts from values in initial conditions array (or zero if not specified)

        Model is spin up for some number of years,
        repeating the first year of emissions each time

        Parameters
        ----------
        nyears : int, optional
            number of years spinup, by default 5
        """

        # Run model repeatedly for first year
        print(f"Spinning up for {nyears} years...")

        for yi in range(nyears):
            self.mod.run(nsteps=15*12, verbose=False)
            self.mod.ic = self.mod.mf_restart[11, :]
    
        self.mod.run(verbose=False)

        print("... done")


    def run_initial_conditions(self):
        """Estimate initial conditions from prior fluxes

        Use the prior fluxes and the model to predict the initial condition
        N years before the first observation

        """

        # Find first timestep with finite obs
        wh = np.where(np.isfinite(self.obs.mf))
        first_obs_ti = wh[0][0]
        first_obs_bi = wh[1][wh[0] == first_obs_ti] # this tells us which boxes were finite at this timestep

        # Block average emissions for comparison with sensitivity matrix
        x_a = self.mod.emissions.reshape(int(self.mod.emissions.shape[0]/self.sensitivity.freq_months), 
                self.sensitivity.freq_months,
                self.mod.emissions.shape[1]).mean(axis=1)
        x_a = x_a.flatten()

        # Work out the mole fractions due to these emissions
        mf_prior = (self.sensitivity.sensitivity @ x_a).reshape(self.obs.mf.shape)

        # If we go back in time, work out how far mf should have been from first obs
        # according to the prior emissions
        mf_offset = np.mean(mf_prior[first_obs_ti, first_obs_bi])
        
        # Work out how far initial conditions are from first obs
        ic_offset = np.mean(self.obs.mf[first_obs_ti, first_obs_bi] - \
                self.mod.ic[first_obs_bi])

        # Adjust initial conditions to reflect these offsets
        self.mod.ic += ic_offset - mf_offset

        # If any values are below zero, reset
        self.mod.ic[self.mod.ic < 1e-12] = 1e-12

        # re-run model from these initial conditions
        self.mod.run(verbose=False)

        # Need to overwrite prior model
        self.mod_prior = Store_model(self.mod)


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
        self.mod.change_end_year(end_year-1/12)
        #TODO: Add sensitivity?


    def run_sensitivity(self, freq="monthly", nthreads=12):
        """
        Derive linear yearly flux sensitivities
        
            Parameters:
                freq (str, optional) : Frequency to infer ("monthly", "quarterly", "yearly")
                                    Default is monthly.
                nthreads (int, optional) : Number of threads to run over
            Attributes:
                Linear sensitivity to emissions
                Note that self.sensitivity can be reshaped using:
                np.reshape(sens, (int(sens.shape[0]/4), 4, int(sens.shape[1]/4), 4))
                where the dimensions are then [mf_time, mf_box, flux_time, flux_box]

        """

        if not np.allclose(self.mod.time, self.mod_prior.time):
            raise Exception('''Prior model has become mis-aligned with model,
                                probably because start or end dates have been changed.
                                Before calculating sensitivity, re-run model and store Store_model''')

        if not np.allclose(self.mod.time, self.obs.time):
            raise Exception('''Model has become mis-aligned with obs,
                                probably because start or end dates have been changed for one or the other.
                                ''')

        freq = freq.lower()
        if freq not in ["monthly", "quarterly", "yearly"]:
            raise Exception('Frequency must be "monthly", "quarterly" or "yearly"')
        
        freq_months={"monthly":1, "quarterly":3, "yearly":12}[freq]
        self.sensitivity.freq_months = freq_months

        nmonths = len(self.mod_prior.time)

        if nmonths % 12:
            raise Exception("Emissions must contain whole years")
        
        # number of discrete periods over which emissions sensitivity is calculated
        nsens = int(nmonths/freq_months)

        # empty sensitivity matrix. Factors of 4 are for surface boxes
        # (mole fraction in rows, emissions in columns)
        self.sensitivity.sensitivity = np.zeros((nmonths*4, nsens*4))

        nsens_section = ceil(nsens/nthreads)
        nsens_out = [nsens_section if nsens_section * (t + 1) < nsens else nsens - (nsens_section * t) for t in range(nthreads)]

        print(f"Calculating sensitivity on {nthreads} threads...")

        with Pool(processes=nthreads) as pool:

            results = []

            for thread in range(nthreads):
                
                results.append(pool.apply_async(sensitivity_section, 
                                                args=(nsens_out[thread],
                                                      nsens_section*thread,
                                                      freq_months, self.mod_prior.mf,
                                                      self.mod.ic, self.mod_prior.emissions, self.mod.mol_mass, self.mod.lifetime,
                                                      self.mod.F, self.mod.temperature, self.mod.oh, self.mod.cl,
                                                      self.mod.oh_a, self.mod.oh_er, self.mod.mass)))

            for thread in range(nthreads):

                if nsens_out[thread] > 0:
                    self.sensitivity.sensitivity[:, thread*nsens_section*4 : thread*nsens_section*4 + nsens_out[thread]*4] = \
                        results[thread].get()

        print("... done")


    def create_matrices(self, sigma_P=None):
        """Set up matrices for inversion

        Parameters
        ----------
        sigma_P : flt, optional
            Placeholder flux uncertainty in Gg/yr, by default None
        """

        self.mat.wh_obs = np.isfinite(self.obs.mf.flatten())
        nx = self.sensitivity.sensitivity.shape[1]

        self.mat.H = self.sensitivity.sensitivity[self.mat.wh_obs,:]

        # Flatten model outputs, noting that all 12 boxes are output (compared to 4 for obs)
        self.mat.y = self.obs.mf[:, :4].flatten()[self.mat.wh_obs] - self.mod_prior.mf[:, :4].flatten()[self.mat.wh_obs]

        #TODO: Functions to choose uncertainty estimation method
        self.mat.R = np.diag(self.obs.mf_uncertainty.flatten()[self.mat.wh_obs]**2)
    
        #TODO: Function to choose emissions uncertainty method
        # this is just a placeholder
        self.mat.P_inv = np.linalg.inv(np.diag(np.ones(nx)*sigma_P**2))
    
        self.mat.x_a = np.zeros(nx)


    def process_outputs(self, uncertainty="1-sigma"):
        """Generate a set of outputs based on posterior solution

        Parameters
        ----------
        uncertainty : str, optional
            Uncertainty measure, by default "1-sigma"
            Can be "N-sigma", "N-percent" (not implemented yet)
            where N is an integer
        """

        emissions_ensemble, \
        mf_ensemble = self.posterior_ensemble(n_sample=1000,
                                            scale_error=0.,
                                            lifetime_error=0.,
                                            transport_error=0.01)

        self.outputs.emissions_global_annual = aggregate_outputs(self.mod_posterior.emissions,
                                                        emissions_ensemble,
                                                        period="annual",
                                                        globe=True,
                                                        uncertainty=uncertainty)

        self.outputs.emissions_annual = aggregate_outputs(self.mod_posterior.emissions,
                                                        emissions_ensemble,
                                                        period="annual",
                                                        globe=False,
                                                        uncertainty=uncertainty)
        
        self.outputs.emissions = aggregate_outputs(self.mod_posterior.emissions,
                                                        emissions_ensemble,
                                                        period="monthly",
                                                        globe=False,
                                                        uncertainty=uncertainty)

