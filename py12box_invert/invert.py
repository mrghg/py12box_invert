import numpy as np
from math import ceil
from multiprocessing import Pool
import pickle
import warnings
import pandas as pd

from py12box_invert.obs import Obs
from py12box_invert.inversion_modules import Inverse_method
from py12box_invert.utils import Store_model, aggregate_outputs, decimal_to_pandas, smooth_outputs
from py12box.model import Model, core


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


class Invert(Inverse_method):

    def __init__(self, project_path, species,
                        obs_path=None, 
                        start_year = None,
                        end_year = None,
                        method = "rigby14",
                        sensitivity_freq = "yearly",
                        spinup_years = 9,
                        ic_years = 3,
                        n_threads = 12):
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
        sensitivity_freq : str, optional
            Frequency of emissions sensitivity ("monthly", "quarterly", "yearly"),
            default is yearly.
        spinup_years : int, optional
            Number of years to spin up the model
        ic_years : int, optional
            Number of years to pad the inversion before the first observation
            in order to allow for some spinup/discard years.
            Note that the emissions file must cover the implied period
            (i.e. (year of first obs - ic_years) onwards), by default 3
        n_threads : int, optional
            Number of threads for sensitivity calculation

        Raises
        ------
        FileNotFoundError
            If obs files not found
        """

        print(f"Setting up inversion for {species}\n")

        # Some housekeeping
        ####################################################

        # Store name of species
        self.species = species

        # Area to store inversion matrices
        self.mat = Matrices()

        # Area to store sensitivity
        self.sensitivity = Sensitivity()

        # Area to store outputs
        self.outputs = Outputs()

        # Attach inverse method
        self.inversion = getattr(self, method)
        
        # Attach methods to process posterior
        self.posterior = getattr(self, f"{method}_posterior")
        self.posterior_ensemble = getattr(self, f"{method}_posterior_ensemble")

        # Get obs
        ##################################################
        if not obs_path and not (project_path / f"{species}_obs.csv").exists():
            raise FileNotFoundError("No obs file given.")
        elif (project_path / f"{species}_obs.csv").exists():
            obs_path = project_path / f"{species}_obs.csv"
        
        self.obs = Obs(obs_path)

        # Initialise model
        #################################################
        self.mod = Model(species, project_path)

        # Align model and obs, and change start/end dates, if needed
        if ic_years and start_year:
            #raise Exception("Can't have both a start_year and ic_year")
            warnings.warn("Can't have both a start_year and ic_year\n Setting ic_years to None.")
            ic_years = None

        # If initial condition years have been set, move start year back
        if ic_years:
            start_year = int(self.obs.time[np.where(np.isfinite(self.obs.mf))[0][0]]) - \
                ic_years

        # Change start year, if needed
        if start_year:
            self.change_start_year(start_year)
        else:
            # Align to obs dataset by default
            self.change_start_year(int(self.obs.time[0]))

        # Change end year, if needed
        if end_year:
            self.change_end_year(end_year)
        else:
            #Align to obs dataset by default
            self.change_end_year(int(self.obs.time[-1]))
            end_year = self.obs.time[-1]

        # Reference run, this will likely change, but need something to kick things off
        self.mod.run(verbose=False)

        # Store some inputs and outputs from prior model
        # TODO: Note that use of the change_start_date or Change_end_date methods
        # may cause the prior model to become mis-aligned. 
        # To align, need to re-run model and then Store_model step.
        # Check if this is a problem.
        self.mod_prior = Store_model(self.mod)

        # Calculate sensitivity
        self.run_sensitivity(freq=sensitivity_freq,
                                nthreads=n_threads)

        # To spin up and estimate initial conditions, iterate between the two a few times
        print(f"Spinning up for {spinup_years} years and estimating initial conditions...")
        for i in range(3):
            # Spinup, if needed
            if spinup_years > 0:
                self.run_spinup(nyears=int(spinup_years/3))

            # Calculate initial conditions.
            # This needs to happen after the sensitivity calculation
            self.run_initial_conditions()
        print("... done")


    def run_inversion(self, prior_flux_uncertainty,
                            n_sample=1000,
                            scale_error=0.,
                            lifetime_error=0.,
                            transport_error=0.01,
                            output_uncertainty="1-sigma"):
        """Run inversion and process outputs

        Parameters
        ----------
        prior_flux_uncertainty : list
            Flux uncertainty in each box in Gg/yr
        scale_error : flt, optional
            Fractional uncertainty in calibration scale (e.g. 0.01 = 1%)
        lifetime_error : flt, optional
            Fractional uncertainty due to 1/lifetime (see Rigby, et al., 2014)
        transport_error : flt, optional
            Uncertainty due to model transport. By default, 0.01
        output_uncertainty : str, optional
            Output uncertainty measure, by default "1-sigma"
            Can be "N-sigma", "N-percent" (not implemented yet)
            where N is an integer
        """
        
        # Set up matrices
        self.create_matrices(sigma_P=prior_flux_uncertainty)

        print("Run inversion...")
        self.inversion()
        self.posterior()
        print("... done")

        print("Calculating outputs...")
        self.process_outputs(n_sample=n_sample,
                            scale_error=scale_error,
                            lifetime_error=lifetime_error,
                            transport_error=transport_error,
                            uncertainty=output_uncertainty)
        print("... done\n")


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
        for yi in range(nyears):
            self.mod.run(nsteps=15*12, verbose=False)
            self.mod.ic = self.mod.mf_restart[11, :]
    
        self.mod.run(verbose=False)

        # Need to overwrite prior model
        self.mod_prior = Store_model(self.mod)


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
        self.mod.change_end_year(end_year)
        self.obs.change_end_year(end_year)
        #TODO: Add sensitivity?

    def run_sensitivity(self, freq="yearly", nthreads=12):
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

        print(f"Calculating flux sensitivity on {nthreads} threads...")

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

        #TODO: Functions to choose obs uncertainty estimation method
        self.mat.R = np.diag(self.obs.mf_uncertainty.flatten()[self.mat.wh_obs]**2)
        if np.isnan(self.mat.R).any():
            raise Exception("Missing measurement errors for some observations.")
            
        #TODO: Function to choose emissions uncertainty method
        # this is just a placeholder
        nsens = int(len(self.mod_prior.time)/self.sensitivity.freq_months)

        if len(sigma_P) != 4:
            raise NotImplementedError("Currently, you must specify the uncertainty in each surface box")
        
        P_diag = np.zeros(nsens*4)
        for ti in range(nsens):
            for bi in range(4):
                P_diag[ti*4 + bi] = sigma_P[bi]

        # Check if there are any zero uncertainties
        if sum(P_diag == 0) > 0:
            raise Exception("P_sigma contains zeros")

        self.mat.P_inv = np.linalg.inv(np.diag(P_diag**2))
    
        # Prior parameters vector
        self.mat.x_a = np.zeros(nx)


    def process_outputs(self, n_sample=1000,
                            scale_error=0.,
                            lifetime_error=0.,
                            transport_error=0.01,
                            uncertainty="1-sigma"):
        """Generate a set of outputs based on posterior solution

        Parameters
        ----------
        uncertainty : str, optional
            Uncertainty measure, by default "1-sigma"
            Can be "N-sigma", "N-percent" (not implemented yet)
            where N is an integer
        """

        print("... calculating posterior ensembles")
        emissions_ensemble, \
        mf_ensemble = self.posterior_ensemble(n_sample=n_sample,
                                            scale_error=scale_error,
                                            lifetime_error=lifetime_error,
                                            transport_error=transport_error)

        # Ensemble without systematic uncertainties
        emissions_ensemble_nosys, \
        mf_ensemble_nosys = self.posterior_ensemble(n_sample=n_sample,
                                            scale_error=0.,
                                            lifetime_error=0.,
                                            transport_error=0.)

        # Store some basic info
        self.outputs.species = self.species
        self.outputs.uncertainty = uncertainty
        # Find first timestep with finite obs
        wh = np.where(np.isfinite(self.obs.mf))
        self.outputs.first_year = int(self.obs.time[wh[0][0]])

        # Store observations
        self.outputs.mf = (self.obs.time, self.obs.mf, self.obs.mf_uncertainty)

        # Store lifetime
        for s in self.mod.__dict__.keys():
            if "steady_state_lifetime" in s:
                setattr(self.outputs, s, getattr(self.mod, s))

        # Store aggregated quantities
        self.outputs.mf_model = aggregate_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.mf,
                                                        mf_ensemble,
                                                        period="monthly",
                                                        uncertainty=uncertainty)

        self.outputs.mf_global_annual = aggregate_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.mf[:, :4],
                                                        mf_ensemble,
                                                        period="annual",
                                                        globe="mean",
                                                        uncertainty=uncertainty)

        self.outputs.mf_global_annual_jan = aggregate_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.mf[:, :4],
                                                        mf_ensemble,
                                                        period="annual-jan",
                                                        globe="mean",
                                                        uncertainty=uncertainty)

        self.outputs.mf_global_growth = smooth_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.mf[:, :4],
                                                        mf_ensemble,
                                                        globe="mean",
                                                        growth=True,
                                                        uncertainty=uncertainty)

        self.outputs.mf_growth = smooth_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.mf[:, :4],
                                                        mf_ensemble,
                                                        growth=True,
                                                        uncertainty=uncertainty)

        self.outputs.emissions_global_annual = aggregate_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.emissions,
                                                        emissions_ensemble,
                                                        period="annual",
                                                        globe="sum",
                                                        uncertainty=uncertainty)

        self.outputs.emissions_global_annual_nosys = aggregate_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.emissions,
                                                        emissions_ensemble_nosys,
                                                        period="annual",
                                                        globe="sum",
                                                        uncertainty=uncertainty)

        self.outputs.emissions_annual = aggregate_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.emissions,
                                                        emissions_ensemble,
                                                        period="annual",
                                                        globe="none",
                                                        uncertainty=uncertainty)

        self.outputs.emissions_annual_nosys = aggregate_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.emissions,
                                                        emissions_ensemble_nosys,
                                                        period="annual",
                                                        globe="none",
                                                        uncertainty=uncertainty)

        self.outputs.emissions = aggregate_outputs(self.mod_posterior.time,
                                                        self.mod_posterior.emissions,
                                                        emissions_ensemble,
                                                        period="monthly",
                                                        globe="none",
                                                        uncertainty=uncertainty)


    def save(self, output_filepath):
        """Save outputs

        Save the contents of Invert.outputs

        Parameters
        ----------
        output_filepath : path object
            Output filepath
        """

        pickle.dump(self.outputs, open(output_filepath, "wb"))
        
    def to_csv(self, output_directory, project_name=""):
        """
        Write outputs to csv
        
        Parameters
        ----------
        output_directory : path object
            Output directory
        """

        def time_df(outvars, var_name):
            # Create a dataframe with time columns in it
            decimal_date = outvars[var_name][0]
            # Adding a time offset to correct annoying issue with converting from 1/12 month
            timestamps = decimal_to_pandas(decimal_date, offset_days=10)
            df = pd.DataFrame({"Year": [t.year for t in timestamps],
                            "Month": [t.month for t in timestamps],
                            "Decimal_date": np.round(decimal_date, decimals=2)})
            return df

        def write_comment_string(var_name, outvars):
            # Write metadata
            if "mf" in var_name:
                units = "ppt"
            elif "emissions" in var_name:
                units = "Gg/yr"
            else:
                raise NotImplementedError(f"Variable {var_name} currently has no units set")
                units = ""
            comment_string = f"# {long_names[var_name]} for {outvars['species']} \n"
            comment_string += "# Outputs from AGAGE 12-box model \n"
            comment_string += "# Time stamps are the centre of the averaging period \n"
            comment_string += "#            ||90-30N| 30-0N| 0-30S | 30-90S\n"
            comment_string += "#            ||=============================\n"
            comment_string += "# 1000-500hPa|| Box0 | Box1 | Box2  | Box3"
            comment_string += "#  500-200hPa|| Box4 | Box5 | Box6  | Box7"
            comment_string += "#  200-   hPa|| Box8 | Box9 | Box10 | Box11"
            comment_string += "# Contact Matt Rigby or Luke Western (University of Bristol) \n"
            comment_string += "# matt.rigby@bristol.ac.uk/luke.western@bristol.ac.uk \n"
            comment_string += f"# File created {str(pd.to_datetime('today', utc=True))} \n"
            comment_string += f"# Units: {units} \n"

            return comment_string

        def global_df(outvars, var_name, uncertainty_string="1-sigma"):
            # Make dataframe for global outputs
            column_name = long_names[var_name].replace(" ", "_")
            df = time_df(outvars, var_name)
            df[column_name] = outvars[var_name][1]
            df[f"{column_name}_{uncertainty_string}"] = outvars[var_name][2]

            return df

        def box_df(outvars, var_name, uncertainty_string="1-sigma"):
            # Make dataframe for per box outputs
            column_name = long_names[var_name].replace(" ", "_")
            dlist = [d for d in outvars[var_name]]
            nout_box = outvars[var_name][1].shape[1]
            nout_box_sd = outvars[var_name][2].shape[1]
            columns = [f"{column_name}_box{box}" for box in range(nout_box)] + \
                      [f"{column_name}_{uncertainty_string}_box{box}" for box in range(nout_box_sd)]
            data = np.concatenate(dlist[1:], axis=1)
            data_df = pd.DataFrame(data=data, columns=columns)
            return pd.concat([time_df(outvars, var_name), data_df], axis=1)


        long_names = {'mf':'Semihemispheric mole fractions', 
          'mf_model' : 'Semihemispheric modelled mole fractions', 
          'mf_global_annual' : 'Global annual mole fraction', 
          'mf_global_annual_jan' : 'Global annual mole fraction Jan', 
          'mf_global_growth' : 'Global mole fraction growth rate', 
          'mf_growth' : 'Semihemispheric mole fraction growth rate', 
          'emissions_global_annual' : 'Global annual emissions', 
          'emissions_global_annual_nosys' : 'Global annual emissions with no systematic uncertainty', 
          'emissions_annual' : 'Semihemispheric annual emissions', 
          'emissions_annual_nosys' :  'Semihemispheric annual emissions with no systematic uncertainty', 
          'emissions' : 'Semihemispheric monthly emissions'}

        if project_name != "":
            project_str = project_name + "_"
        else:
            project_str = ""

        outvars_keys = list(self.outputs.__dict__.keys())
        outvars = self.outputs.__dict__
        
        # Loop over outputs and write
        for var_name in outvars_keys:
            if var_name in long_names.keys():
                comment_string = write_comment_string(var_name, outvars)
                if "global" in var_name:
                    data_df = global_df(outvars, var_name, uncertainty_string=self.outputs.uncertainty)
                else:
                    data_df = box_df(outvars, var_name, uncertainty_string=self.outputs.uncertainty)

                # Cut out spinup years
                data_df = data_df[data_df["Decimal_date"] >= self.outputs.first_year]

                # Output file name
                fname = f"{project_str}{outvars['species']}_{long_names[var_name].replace(' ','_')}.csv"

                # Write
                with open(output_directory / fname, 'w') as fout:
                    fout.write(comment_string)
                    data_df.to_csv(fout, index=False)
        

def sensitivity_section(nsens_section, t0, freq_months, mf_ref,
                        ic, emissions, mol_mass, lifetime,
                        F, temperature, oh, cl, oh_a, oh_er, mass):
    """Calculate some section of the sensitivity matrix 

    Parameters
    ----------
    nsens_section : int
        Number of perturbations to carry out
    t0 : int
        Position of perturbation
    freq_months : int
        Number of months to perturb each time (e.g. monthly, quarterly, annually)
    mf_ref : ndarray
        Reference run mole fraction

    Returns
    -------
    ndarray
        Section of sensitivty matrix
    """

    if nsens_section > 0:

        sens = np.zeros((len(mf_ref[:, :4].flatten()), nsens_section*4))

        for ti in range(nsens_section):
            for bi in range(4):

                # Perturb emissions uniformly throughout specified time period
                emissions_perturbed = emissions.copy()
                emissions_perturbed[(t0 + ti)*freq_months:freq_months*(t0+ti+1), bi] += 1.

                # Run perturbed model
                mf_out, mf_restart, burden_out, q_out, losses, global_lifetimes = \
                        core.model(ic, emissions_perturbed, mol_mass, lifetime,
                                    F, temperature, oh, cl,
                                    arr_oh=np.array([oh_a, oh_er]),
                                    mass=mass)
                
                # Store sensitivity column
                sens[:, 4*ti + bi] = (mf_out[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.

        return sens
    
    else:

        return None