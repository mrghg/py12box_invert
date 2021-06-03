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


def decimal_to_pandas(dec):
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
        dates.append(pd.Timestamp(yeardatetime + dt.timedelta(days = daysPerYear*(f - year))))

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


def pickle_to_csv(saved_output_path, output_directory):
    """
    Write the pickled outputs to csv file.
    
    Parameters:
        saved_output_path (str):
            the path to the pickled file
        output_directory (pathlib path):
            The output directory to save new csv files
    """
    long_names = {'mf':'Semihemispheric mole fractions', 
      'mf_model' : 'Semihemispheric modelled mole fractions', 
      'mf_global_annual' : 'Global annual mole fraction', 
      'mf_global_growth' : 'Global mole fraction growth rate', 
      'mf_growth' : 'Semihemispheric mole fraction growth rate', 
      'emissions_global_annual' : 'Global annual emissions', 
      'emissions_global_annual_nosys' : 'Global annual emissions with no systematic uncertainty', 
      'emissions_annual' : 'Semihemispheric annual emissions', 
      'emissions_annual_nosys' :  'Semihemispheric annual emissions with no systematic uncertainty', 
      'emissions' : 'Semihemispheric monthly emissions'}
    
    def write_comment_string(var_name, outvars):
        
        if "mf" in var_name:
            units = "ppt"
        elif "emissions" in var_name:
            units = "Gg/yr"
        else:
            raise NotImplementedError("This variable currently has no units set")
            units = ""
            
        comment_string = f"# {long_names[var_name]} for {outvars['species']} \n"
        comment_string += "# Outputs from AGAGE 12-box model \n"
        comment_string += "# Contact Matt Rigby or Luke Western (University of Bristol) \n"
        comment_string += "# matt.rigby@bristol.ac.uk/luke.western@bristol.ac.uk \n"
        comment_string += f"# File created {str(pd.to_datetime('today', utc=True))} \n"
        comment_string += f"# Units: {units} \n"
        
        return comment_string
    
    def global_df(outvars, var_name):
        column_name = long_names[var_name].replace(" ", "_")
        df = pd.DataFrame(data=np.array(outvars[var_name]).T, 
                          columns=["Time", column_name, f"{column_name}_sd"])
        return df
    
    def box_df(outvars, var_name):
        column_name = long_names[var_name].replace(" ", "_")
        data_tup = outvars[var_name]
        dlist = [d for d in outvars[var_name]]
        nout_box = outvars[var_name][1].shape[1]
        nout_box_sd = outvars[var_name][2].shape[1]
        columns = ["Time"] + \
                  [f"{column_name}_box{box}" for box in range(nout_box)] + \
                  [f"{column_name}_sd_box{box}" for box in range(nout_box_sd)]
        data = np.concatenate([np.expand_dims(dlist[0], axis=1), np.concatenate(dlist[1:], axis=1)], axis=1)
        df = pd.DataFrame(data=data, columns=columns)
        return df
    
    out=pickle.load(open(saved_output_path, "rb" ))
    outvars_keys = list(out.__dict__.keys())[1:]
    outvars = out.__dict__
    
    for var_name in outvars_keys:
        comment_string = write_comment_string(var_name, outvars)
        if "global" in var_name:
            data_df = global_df(outvars, var_name)
        else:
            data_df = box_df(outvars, var_name)
        
        with open(output_directory / f"{outvars['species']}_{long_names[var_name].replace(' ','_')}_agage.csv", 'w') as fout:
            fout.write(comment_string)
            data_df.to_csv(fout, index=False)
            