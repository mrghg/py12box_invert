import xarray as xr
import pandas as pd
import numpy as np
from py12box import startup, core
from py12box_invert import core as invcore
import matplotlib.pyplot as plt
from pathlib import Path


class Store_model:
    """Class to store some of the parameters from a model run

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
        self.burden = mod.burden.copy()


def approx_initial_conditions(species, project_path, ic0):
    """
    Spin up the model to approximate initial conditions

        Parameters:
            species (str)              : Species name
            project_path (pathlib path): Path to project
            ic0 (array)                : Initial conditions for 4 surface boxes
        Returns:
            Array of approximates initial conditions for all boxess
    """
    
    if len(ic0) != 4:
        raise("Initial conditions must be 4 elements (surface boxes, ordered N - S)")
    
    mod = invcore.fwd_model_inputs(project_path, species)
    mod.run(verbose=False)
    c_month = mod.mf
    
    #Take final spun up value and scale each semi-hemisphere to surface boxes.
    return c_month[-1,:] * np.tile(ic0[:4]/c_month[-1,:4],3)


def decimal_date(date):
    '''
    Calculate decimal date from pandas DatetimeIndex
    
        Parameters:
            date (pandas DatetimeIndex): Dates to convert
        Returns:
            Array of decimal dates
    '''

    if not isinstance(date, pd.DatetimeIndex):
        raise Exception("Expecting pandas.DatetimeIndex")

    days_in_year = np.array([[365., 366.][int(ly)] for ly in date.is_leap_year])
    
    return (date.year + (date.dayofyear-1.)/days_in_year + date.hour/24.).to_numpy()


def round_date(date):
    """Regularise decimal date so that each month is exactly 1/12 year

    Parameters
    ----------
    date : flt
        Decimal date

    Returns
    -------
    flt
        Decimal date, rounded to nearest 1/12 year
    """

    return np.round(date*12)/12.


def adjust_emissions_time(time):
    """
    Emissions date is not calendar months but 1/12th year fractions
    Adjust to align with calendar decimal years
        
        Parameters:
            time (array): Decimal dates to adjust
        Returns:
            Array of decimal dates
    
    """
    time_start = str(int(time[0]))+"-"+str(int(np.round((time[0]-int(time[0])+1./12.)*12.))).zfill(2)
    time_end = str(int(time[-1]))+"-"+str(int(np.round((time[-1]-int(time[-1])+1./12.)*12.))).zfill(2)
    dt_time = pd.date_range(time_start,time_end, freq="MS")
    return decimal_date(dt_time)


def pad_obs(mf_box, mf_var_box, time, obstime):
    """
    Pad the obs data with NaNs to cover period covered by input emissions

        Parameters:
            mf_box (array)    : Array of boxed observations
            mf_var_box (array): Array of boxed mole fraction variability
            time (array)      : Decimal dates for emissions
            obstime (array)   : Decimal dates for obs
            
         Returns:
             obs (array)   : padded observations
             obs_sd (array): padded observation error
    
    """
    emis_time = adjust_emissions_time(time)
    obs_df = pd.DataFrame(index=obstime, data=mf_box.T,columns=["0","1","2","3"]).reindex(emis_time)
    obs_sd_df = pd.DataFrame(index=obstime, data=mf_var_box.T,columns=["0","1","2","3"]).reindex(emis_time)
    obs = obs_df.values.flatten()
    obs_sd = obs_sd_df.values.flatten()
    return obs, obs_sd


def plot_emissions(model_emissions, species, savepath=None, MCMC=False):
    """
    Plot emissions from analytical inversion
    """
    plt.figure()
    plt.plot(model_emissions.index, model_emissions.Global_emissions)
    if MCMC:
        plt.fill_between(model_emissions.index, 
                         model_emissions.Global_emissions_16,
                         model_emissions.Global_emissions_84, alpha=0.5)
        upy = np.max(model_emissions.Global_emissions_84)
    else:
        plt.fill_between(model_emissions.index,  
                         model_emissions.Global_emissions-model_emissions.Global_emissions_sd,
                         model_emissions.Global_emissions+model_emissions.Global_emissions_sd, alpha=0.5)
        upy = np.max(model_emissions.Global_emissions+model_emissions.Global_emissions_sd)
    plt.xlabel("Date")
    plt.xlim(1980,2020)
    plt.ylim(0, np.ceil(upy))
    plt.ylabel(species+" emissions Gg/yr")
    if savepath:
        savedir = savepath / "emissions" / "plots"
        if not savedir.exists():
            savedir.mkdir(parents=True)
        for ext in [".pdf"]:
            savename = species+"_emissions_"+str(pd.to_datetime("today"))[:10]+ext
            plt.savefig( savedir / savename, dpi=200)
    plt.close()


def plot_mf(model_mf, species, savepath=None):
    """
    Plot Global and hemispheric mole fractions, from analytical inversion
    """
    plt.figure()
    plt.plot(model_mf.index, model_mf.Global_mf, label="Global")
    plt.plot(model_mf.index, model_mf.N_mf, label="N-Hemisphere")
    plt.plot(model_mf.index, model_mf.S_mf, label="S-Hemisphere")
    plt.legend()
    plt.ylabel("ppt")
    plt.xlabel("Date")
    plt.xlim(1980,2020)
    if savepath:
        savedir = savepath / "molefraction" / "plots"
        if not savedir.exists():
            savedir.mkdir(parents=True)
        for ext in [".pdf"]:
            savename = species+"_mf_"+str(pd.to_datetime("today"))[:10]+ext
            plt.savefig( savedir / savename , dpi=200)
    plt.close()


def mf_to_csv(savepath, model_mf, species, comment=None):
    """
    Write Global and hemispheric mole fractions to csv
    """
    savedir = savepath / "molefraction" / "csv"
    if not savedir.exists():
        savedir.mkdir(parents=True)
    savename = species+"_mf_"+str(pd.to_datetime("today"))[:10]+".csv"
    f = open(savedir / savename, 'w')
    if comment is not None:
        if "\n" not in comment[-4:]:
            comment += "\n"
        f.write(comment)
    model_mf.to_csv(f)
    f.close()


def emissions_to_csv(savepath, model_emissions, species, comment=None):
    """
    Write emissions to csv 
    
    """
    savedir = savepath / "emissions" / "csv"
    if not savedir.exists():
        savedir.mkdir(parents=True)
    savename = species+"_emissions_"+str(pd.to_datetime("today"))[:10]+".csv"
    f = open(savedir / savename, 'w')
    if comment is not None:
        if "\n" not in comment[-4:]:
            comment += "\n"
        f.write(comment)
    model_emissions.to_csv(f)
    f.close()
