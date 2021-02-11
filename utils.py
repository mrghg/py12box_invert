from acrg_obs import get_single_site
import xarray as xr
import pandas as pd
#import py12box
import numpy as np
from py12box.py12box import startup, core
from py12box_invert import core as invcore

def monthly_baselines(site, species, box):
    """
    Retrieve observations and return a dataframe of monthly baselines

    Parameters
    ----------
    site : str
        Site code
    species: str
        Species name
    box: int
        12-box model box number (0 - 11)
    """

    #TODO: This function needs moving into a different repo, as it requires ACRG repo
    
    dataset = xr.concat(get_single_site(site, species),
                        dim="time").sortby("time")

    df_mf = dataset.mf.to_pandas()
    df_mf_baseline = pd.DataFrame(df_mf.resample("MS").quantile(0.1), columns=[(site, box, "mf"), ])

    df_repeatability = dataset.mf_repeatability.to_pandas()
    df_repeatability = pd.DataFrame(df_repeatability.resample("MS").mean(),
                                    columns=[(site, box, "mf_repeatability"), ])

    df_variability = pd.DataFrame(df_mf.resample("MS").std(), columns=[(site, box, "mf_variability"), ])

    return pd.concat([df_mf_baseline, df_repeatability, df_variability], axis=1, sort=True)


def obs_write(species, project_path, case, sites=None):
    """
    Write csv file containing monthly mean observations at each site

    Parameters
    ----------
    species : str
        Species string
    project_path: pathlib path
        Path to project
    case: str
        Case folder within project
    """

    if sites is None:
        sites = {0: ["MHD", "ZEP", "THD", "JFJ"],
                 1: ["RPB"],
                 2: ["SMO"],
                 3: ["CGO"]}

    data = []

    for box in sites:
        for site in sites[box]:
            df_single_site = monthly_baselines(site, species, box)
            data.append(df_single_site)

    df = pd.concat(data, axis=1, sort=True)
    df.index.name = None
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["site", "box", "var"])

    df.to_csv(project_path / case / f"obs_{species}.csv")


def obs_read(species, project_path, case):
    """
    Read csv file containing monthly mean observations at each site

    Parameters
    ----------
    species : str
        Species string
    project_path: pathlib path
        Path to project
    case: str
        Case folder within project

    """

    df = pd.read_csv(project_path / case / f"obs_{species}.csv", header=[0, 1, 2],
                     skipinitialspace=True, index_col=0,
                     parse_dates=[0])

    return df

def obs_box(obsdf):
    """
    Box up obs data into surface boxes
    Also returns measurement error as root-square-sum of measurment variability
    Q: Should I weight these by latitude?!
    Q: Currently more than one measurement in box then uncertatinty is higher – not really true.
    """
    mf_box = np.zeros((4, len(obsdf.index))) 
    mf_var_box = np.zeros((4, len(obsdf.index))) 
    for sb in range(4):
        mf_box[sb,:] = obsdf.xs("mf",level="var", axis=1).xs(str(sb),level="box", axis=1).mean(axis=1, skipna=True)
        mf_var_box[sb,:] = obsdf.xs("mf_variability",level="var", axis=1).xs(str(sb),level="box", axis=1).apply(np.square).apply(np.nansum, axis=1).apply(np.sqrt)
    return mf_box, mf_var_box

def approx_initial_conditions(species, project_path, case,
                              ic0):
    """
    Spin up the model to approximate initial conditions
    
    Parameters
    ----------
    species : str
        Species string
    project_path: pathlib path
        Path to project
    case: str
        Case folder within project
    ic: list
        Initial mole fraction at four surface boxes
    """
    
    if len(ic0) != 4:
        raise("Initial conditions must be 4 elements (surface boxes, ordered N - S)")
        
    #mol_mass, OH_A, OH_ER = py12box.setup.get_species_parameters(species)
    #time, emissions, ic, lifetime = py12box.setup.get_case_parameters(project_path, case, species)
    #i_t, i_v1, t, v1, OH, Cl, temperature = py12box.setup.get_model_parameters(int(len(time) / 12))
    #F = setup.transport_matrix(i_t, i_v1, t, v1)
    
    mod = invcore.fwd_model_inputs(project_path, case, species)
    mod.run(verbose=False)
    c_month = mod.mf

#     c_month, burden, emissions_out, losses, lifetimes = \
#         core.model(ic=ic, q=emissions,
#                    mol_mass=mol_mass,
#                    lifetime=lifetime,
#                    F=F,
#                    temp=temperature,
#                    cl=Cl, oh=OH,
#                    arr_oh = np.array([OH_A, OH_ER]))
    
    #Take final spun up value and scale each semi-hemisphere to surface boxes.
    return c_month[-1,:] * np.tile(ic0[:4]/c_month[-1,:4],3)

def decimal_date(date):
    '''
    Calculate decimal date from pandas DatetimeIndex
    
    Parameters
    ----------
    date : pandas DatetimeIndex
    
    '''
    
    days_in_year = np.array([[365., 366.][int(ly)] for ly in date.is_leap_year])
    
    return (date.year + (date.dayofyear-1.)/days_in_year + date.hour/24.).to_numpy()

def adjust_emissions_time(time):
    """
    Currently emissions date is not calendar months but 1/12th year fractions
    Adjust to align with calendar decimal years
    """
    time_start = str(int(time[0]))+"-"+str(int(np.round((time[0]-int(time[0])+1./12.)*12.))).zfill(2)
    time_end = str(int(time[-1]))+"-"+str(int(np.round((time[-1]-int(time[-1])+1./12.)*12.))).zfill(2)
    dt_time = pd.date_range(time_start,time_end, freq="MS")
    return decimal_date(dt_time)


def pad_obs(mf_box, mf_var_box, time,obstime):
    """
    Pad the obs data with NaNs to cover period covered by input emissions
    """
    emis_time = adjust_emissions_time(time)
    obs_df = pd.DataFrame(index=obstime, data=mf_box.T,columns=["0","1","2","3"]).reindex(emis_time)
    obs_sd_df = pd.DataFrame(index=obstime, data=mf_var_box.T,columns=["0","1","2","3"]).reindex(emis_time)
    obs = obs_df.values.flatten()
    obs_sd = obs_sd_df.values.flatten()
    return obs, obs_sd

def inversion_matrices(obs, sensitivity, mf_ref, obs_sd, P_sd):
    """
    Drop sensitivities to no observations and set up matrices
    Prior uncertainty on emissions is hardwired to 100.
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