import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import calendar

from py12box import startup, core
from py12box_invert import core as invcore
import matplotlib.pyplot as plt
from pathlib import Path
from py12box_invert.kz_filter import kz_filter


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


def aggregate_outputs(time, mean, ensemble,
                    period="annual",
                    globe="none",
                    uncertainty="1-sigma"):
    """Aggregate a set of outputs to produce means with uncertainties

    Parameters
    ----------
    time : ndarray
        Decimal date
    mean : ndarray
        Mean posterior values (e.g. mole fraction in each box)
    ensemble : ndarray, n_months x n_box x n_samples
        Monte Carlo ensemble of posterior values
    period : str, optional
        Time period over which to aggregate ("annual", "seasonal"), by default "annual"
    globe : str, optional
        Calculate global "sum" or "mean", by default "none"
    uncertainty : str, optional
        Type of uncertainty to output ("1-sigma", "95-percentile"), by default "1-sigma"

    Returns
    -------
    ndarray
        Mean aggregated quantity
    ndarray
        Uncertainty in aggregated quantity

    Raises
    ------
    NotImplementedError
        Currently, only 1-sigma uncertainties can be output
    """

    # Add a time aggregation dimension
    _mean = np.expand_dims(mean, axis=1)
    _ensemble = np.expand_dims(ensemble, axis=1)

    meanshape = list(_mean.shape)
    enshape = list(_ensemble.shape)

    # Time aggregation reshape (note additional dimension was added to both above)
    if period == "annual":
        enshape[0] = int(enshape[0]/12)
        enshape[1] = 12
        meanshape[0] = int(meanshape[0]/12)
        meanshape[1] = 12
    elif period == "seasonal":
        enshape[0] = int(enshape[0]/3)
        enshape[1] = 3
        meanshape[0] = int(meanshape[0]/3)
        meanshape[1] = 3

    _time = np.reshape(time, meanshape[:2])
    _mean = np.reshape(_mean, meanshape)
    _ensemble = np.reshape(_ensemble, enshape)

    # Global sum/mean
    if globe == "sum":
        _mean = _mean.sum(axis=2)
        _ensemble = _ensemble.sum(axis=2)
    elif globe == "mean":
        _mean = _mean.mean(axis=2)
        _ensemble = _ensemble.mean(axis=2)

    # Do time averaging
    _time = _time.mean(axis=1)
    _mean = _mean.mean(axis=1)
    _ensemble = _ensemble.mean(axis=1)

    if "sigma" in uncertainty:
        _uncertainty = _ensemble.std(axis=-1)
        # Can have n-sigma output
        _uncertainty *= float(uncertainty[0])
    elif "percentile" in uncertainty:
        raise NotImplementedError("Need to add percentiles")

    return _time, _mean, _uncertainty


def smooth_outputs(time, mean, ensemble,
                    globe="none",
                    growth=False,
                    kz_params=(9, 4),
                    uncertainty="1-sigma"
                    ):
    """Aggregate a set of outputs to produce means with uncertainties

    Parameters
    ----------
    time : ndarray
        Decimal date
    mean : ndarray
        Mean posterior values (e.g. mole fraction in each box)
    ensemble : ndarray, n_months x n_box x n_samples
        Monte Carlo ensemble of posterior values
    globe : str, optional
        Calculate global "sum" or "mean", by default "none"
    growth : boolean, optional
        Calculate growth rate, by default False
    kz_params : tuple, optional
        Parameters for K-Z filter (window, order), by default (9, 4), corresponding
        to a 9-month window and 4 smoother passes, which leads to approximately
        an 18 month filter (window * sqrt(order))
    uncertainty : str, optional
        Type of uncertainty to output ("1-sigma", "95-percentile"), by default "1-sigma"

    Returns
    -------
    ndarray
        Smoothed time
    ndarray
        Smoothed mean
    ndarray
        Uncertainty in smoothed quantity

    Raises
    ------
    NotImplementedError
        Currently, only 1-sigma uncertainties can be output
    """

    # Global sum
    if globe == "sum":
        _mean = mean.sum(axis=1)
        _ensemble = ensemble.sum(axis=1)
        # Add axis back in to keep indices consistent below
        _mean = np.expand_dims(_mean, axis=1)
        _ensemble = np.expand_dims(_ensemble, axis=1)
    elif globe == "mean":
        _mean = mean.mean(axis=1)
        _ensemble = ensemble.mean(axis=1)
        # Add axis back in to keep indices consistent below
        _mean = np.expand_dims(_mean, axis=1)
        _ensemble = np.expand_dims(_ensemble, axis=1)
    else:
        _mean = mean.copy()
        _ensemble = ensemble.copy()

    # Calculate monthly growth rate
    #  factor of 12 is to transform to annual growth
    if growth:
        _time = (time[1:] + time[:-1])/2.
        _mean = (_mean[1:,:] - _mean[:-1,:])*12.
        _ensemble = (_ensemble[1:,:,:] - _ensemble[:-1,:,:])*12.
    else:
        _time = time.copy()

    # Do smoothing
    _time = kz_filter(_time, kz_params[0], kz_params[1])
    
    # Arrays to store output
    _out_mean = np.empty((_time.shape[0], _mean.shape[-1]))
    _out_ensemble = np.empty((_time.shape[0], _mean.shape[-1], _ensemble.shape[-1]))

    for bi in range(_mean.shape[-1]):
        _out_mean[:, bi] = kz_filter(_mean[:, bi], kz_params[0], kz_params[1])
        for i in range(ensemble.shape[-1]):
            _out_ensemble[:, bi, i] = kz_filter(_ensemble[:, bi, i], kz_params[0], kz_params[1])

    if "sigma" in uncertainty:
        _uncertainty = _out_ensemble.std(axis=-1)
        # Can have n-sigma output
        _uncertainty *= float(uncertainty[0])
    else:
        raise NotImplementedError("Only sigma uncertainty at the moment")

    return _time, np.squeeze(_out_mean), np.squeeze(_uncertainty)





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


def decimal_to_pandas(dec):

    dates = []
    for f in dec:
        year = int(f)
        yeardatetime = dt.datetime(year, 1, 1)
        daysPerYear = 365 + calendar.leapdays(year, year+1)
        dates.append(pd.Timestamp(yeardatetime + dt.timedelta(days = daysPerYear*(f - year))))

    return dates



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
