import pandas as pd
import numpy as np
import datetime as dt
import calendar

#from py12box_invert import core as invcore
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


def decimal_to_pandas(dec, offset_days=0):
    """
    Convert decimal date to pandas datetime
    
        Parameters:
            dec (float): Decimal dates to convert
        Returns:
            List of pandas datetimes
    """

    dates = []
    for f in dec:
        year = int(f)
        yeardatetime = dt.datetime(year, 1, 1)
        daysPerYear = 365 + calendar.leapdays(year, year+1)
        days=int(daysPerYear*(f - year))
        dates.append(pd.Timestamp(yeardatetime + dt.timedelta(days=days) + dt.timedelta(days=offset_days)))

    return dates


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

            