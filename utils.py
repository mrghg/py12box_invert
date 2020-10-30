from acrg.acrg_obs import get_single_site
import xarray as xr
import pandas as pd
import py12box
import numpy as np


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


def approx_initial_conditions(species, project_path, case,
                              ic):
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
    
    if len(ic) != 4:
        raise("Initial conditions must be 4 elements (surface boxes, ordered N - S)")
        
    mol_mass, OH_A, OH_ER = setup.get_species_parameters(species)
    time, emissions, ic, lifetime = setup.get_case_parameters(project_dir, case, species)
    
    
    
    i_t, i_v1, t, v1, OH, Cl, temperature = setup.get_model_parameters(int(len(time) / 12))
    F = setup.transport_matrix(i_t, i_v1, t, v1)

    c_month, burden, emissions_out, losses, lifetimes = \
        core.model(ic=ic, q=emissions,
                   mol_mass=mol_mass,
                   lifetime=lifetime,
                   F=F,
                   temp=temperature,
                   cl=Cl, oh=OH,
                   arr_oh = np.array([OH_A, OH_ER]))
    

def decimal_date(date):
    '''
    Calculate decimal date from pandas DatetimeIndex
    
    Parameters
    ----------
    date : pandas DatetimeIndex
    
    '''
    
    days_in_year = np.array([[365., 366.][int(ly)] for ly in date.is_leap_year])
    
    return (date.year + (date.dayofyear-1.)/days_in_year + date.hour/24.).to_numpy()