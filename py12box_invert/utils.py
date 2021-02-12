import xarray as xr
import pandas as pd
import numpy as np
from py12box import startup, core
from py12box_invert import core as invcore


def obs_read(species, project_path):
    """
    Read csv file containing monthly mean observations at each site

        Parameters:
            species (str)              : Species name
            project_path (pathlib path): Path to project
        Returns:
            Pandas data frame
    """

    df = pd.read_csv(project_path / species / f"obs_{species}.csv", header=[0, 1, 2],
                     skipinitialspace=True, index_col=0,
                     parse_dates=[0])

    return df

def obs_box(obsdf):
    """
    Box up obs data into surface boxes
    
        Parameters:
            obsdf (dataframe): Pandas dataframe containing obs data
        Returns:
            mf_box (array)    : Array of boxed obs
            mf_var_box (array): Array of boxed mole fraction variability
    
    Q: Should I weight these by latitude?!
    Q: Currently more than one measurement in box then uncertatinty is higher – not really true.
    """
    mf_box = np.zeros((4, len(obsdf.index))) 
    mf_var_box = np.zeros((4, len(obsdf.index))) 
    for sb in range(4):
        mf_box[sb,:] = obsdf.xs("mf",level="var", axis=1).xs(str(sb),level="box", axis=1).mean(axis=1, skipna=True)
        mf_var_box[sb,:] = obsdf.xs("mf_variability",level="var", axis=1).xs(str(sb),level="box", axis=1).apply(np.square).apply(np.nansum, axis=1).apply(np.sqrt)
    return mf_box, mf_var_box

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
    
    days_in_year = np.array([[365., 366.][int(ly)] for ly in date.is_leap_year])
    
    return (date.year + (date.dayofyear-1.)/days_in_year + date.hour/24.).to_numpy()

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


def pad_obs(mf_box, mf_var_box, time,obstime):
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

def inversion_matrices(obs, sensitivity, mf_ref, obs_sd, P_sd=None):
    """
    Drop sensitivities to no observations and set up matrices
    Prior uncertainty on emissions defaults to 100.
    
        Parameters:
            obs (array)          : Array of boxed observations
            sensitivity (array)  : Array of sensitivity to emissions
            mf_ref (array)       : Array of reference run mole fraction
            obs_sd (array)       : Observation error
            P_sd (array, options): Std dev uncertainty in a priori emissions 
                                   If not given then defaults to 100 Gg/box/yr
            
         Returns:
             H (array)         : Sensitivity matrix
             y (array)         : Deviation from a priori emissions
             R (square matrix) : Model-measurement covariance matrix
             P (square matrix) : Emissions covariance matrix
             x_a (array)       : A priori deviation (defaults to zeros)
    """
    wh_obs = np.isfinite(obs)
    H = sensitivity[wh_obs,:]
    y = obs[wh_obs] - mf_ref[wh_obs]
    R = np.diag(obs_sd[wh_obs]**2)
    if not P_sd:
        # If no emissions std is given default to 100 Gg/yr
        print("Prior emissions uncertainty defaulting to 100 Gg/box/yr")
        P_sd = np.ones(H.shape[1])*100
    P = np.diag(P_sd**2)
    x_a = np.zeros(H.shape[1])
    return H, y, R, P, x_a