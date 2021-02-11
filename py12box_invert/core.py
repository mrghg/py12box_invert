import numpy as np
import pandas as pd
from py12box import startup, core,  model
from tqdm import tqdm
from py12box_invert import utils


def fwd_model_inputs(project_path, species):
    """
    Get inputs to 12-box model
    
        Parameters:
            project_path (pathlib path): Path to project
            species (str)              : Species name
        Returns:
            Box model class
    """
    mod = model.Model(species, project_path)
    return mod #time, ic, emissions, mol_mass, lifetime, F, temperature, cl, oh, oh_a, oh_er


def flux_sensitivity(project_path, species, ic0=None, freq="monthly"):
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
    
    mod = fwd_model_inputs(project_path, species)
    
    if ic0 is not None:
        if len(ic0) == 4:
            ic = utils.approx_initial_conditions(species, project_path, ic0)
            mod.ic = ic
        else:
            print("ic0 does not have only 4 surface boxes. Ignoring.")

    mod.run()
    mf_ref = mod.mf
    emissions = mod.emissions
    
    if len(emissions) % 12:
        raise Exception("Emissions must contain whole years")
        
    if freq=="yearly":
        nyear = int(len(emissions)/12)
        sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nyear*4)))

        for mi in tqdm(range(nyear)):
            for bi in range(4):

                emissions_perturbed = emissions.copy()
                emissions_perturbed[mi*12:(12*mi+12), bi] +=1

                mod.emissions = emissions_perturbed
                mod.run(verbose=False)
                mf_perturbed = mod.mf
                sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.

    elif freq == "monthly":
        nmonth = len(emissions)
        sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nmonth*4)))

        for mi in tqdm(range(nmonth)):
            for bi in range(4):

                emissions_perturbed = emissions.copy()
                emissions_perturbed[mi, bi] +=1

                mod.emissions = emissions_perturbed
                mod.run(verbose=False)
                mf_perturbed = mod.mf
                sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.

    elif freq == "quarterly":
        nquart = int(len(emissions)/4.)
        sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nquart*4)))

        for mi in tqdm(range(nquart)):
            for bi in range(4):

                emissions_perturbed = emissions.copy()
                emissions_perturbed[mi*4:(4*mi+4), bi] +=1

                mod.emissions = emissions_perturbed
                mod.run(verbose=False)
                mf_perturbed = mod.mf
                sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.
                
    return sensitivity, mf_ref[:,:4].flatten(), emissions, mod.time


def inversion_analytical(y, H, x_a, R, P):
    """
    Perform a Gaussian analytical inversion assuming linearity.
    The inferred value 'x_hat' is the difference in emissions from the reference run.
    
        Parameters:
             y (array)         : Deviation from a priori emissions
             H (array)         : Sensitivity matrix
             x_a (array)       : A priori deviation (defaults to zeros)
             R (square matrix) : Model-measurement covariance matrix
             P (square matrix) : Emissions covariance matrix
        Returns:
            x_hat (array): Posterior mean difference from a priori
            P_hat (array): Posterior covariance matrix
    """
    R_inv = np.linalg.inv(R)
    P_inv = np.linalg.inv(P)
    x_hat = np.linalg.inv(H.T @ R_inv @ H + P_inv) @ (H.T @ R_inv @ y + P_inv @ x_a)
    P_hat = np.linalg.inv(H.T @ R_inv @ H + P_inv)
    return x_hat, P_hat


def annual_means(x_hat, P_hat, emis_ref,  freq="monthly"):
    """
    Derive annual mean emissions from inversion output
    
        Parameters:
            x_hat (array)       : Posterior mean difference from a priori
            P_hat (array)       : Posterior covariance matrix
            emis_ref (array)    : Reference emissions
            freq (str, optional): Frequency to infer ("monthly", "quarterly", "yearly")
                                         Default is monthly.
        Returns:
            x_out (array)   : Annual emissions,
            x_sd_out (array): 1 std deviation uncertain in annual emissions
    """
    
    if freq == "yearly":
        x_mnth = np.sum(np.repeat(x_hat.reshape(int(len(x_hat)/4),4),12, axis=0)+ emis_ref, axis=1)
        x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
        x_sd_out = np.zeros(len(x_out))
        j = 0
        for i in range(0,len(x_hat),4):
            x_sd_out[j] = np.sqrt(np.sum(P_hat[i:(i+4),i:(i+4)]))
            j+=1
    elif freq == "monthly":
        x_mnth = np.sum(x_hat.reshape(int(len(x_hat)/4),4)+ emis_ref, axis=1)
        x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
        x_sd_out = np.zeros(len(x_out))
        j = 0
        for i in range(0,len(x_hat),48):
            x_sd_out[j] = np.sqrt(np.sum(P_hat[i:(i+48),i:(i+48)])/12)
            j +=1
    elif freq == "quarterly":
        x_mnth = np.sum(np.repeat(x_hat.reshape(int(len(x_hat)/4),4),4, axis=0)+ emis_ref, axis=1)
        x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
        x_sd_out = np.zeros(len(x_out))
        j = 0
        for i in range(0,len(x_hat),12):
            x_sd_out[j] = np.sqrt(np.sum(P_hat[i:(i+12),i:(i+12)])/4.)
            j+=1
    
    return x_out, x_sd_out


def run_inversion(project_path, species, ic0=None,  emissions_sd=None, freq="monthly"):
    """
    Run inversion for 12-box model to estimate monthly means from 4 surface boxes.
    
        Parameters:
            project_path (pathlib path)  : Path to project
            species (str)                : Species name
            ic0 (array)                  : Initial conditions for 4 surface boxes
            emissions_sd (array, options): Std dev uncertainty in a priori emissions 
                                           If not given then defaults to 100 Gg/box/yr
            freq (str, optional)         : Frequency to infer ("monthly", "quarterly", "yearly")
                                           Default is monthly.
        Returns:
            x_hat (array)   : Posterior mean difference from a priori
            P_hat (array)   : Posterior covariance matrix
            emis_ref (array): Reference emissions
            time (array)    : Decimal dates for emissions
    """
    #Get obs
    obsdf =  utils.obs_read(species, project_path)
    obstime = utils.decimal_date(obsdf.index)
    #Box obs
    mf_box, mf_var_box = utils.obs_box(obsdf)
    #Get sensitivities
    sensitivity, mf_ref, emis_ref, time = flux_sensitivity(project_path, species, ic0=ic0, freq=freq)
    # Pad obs to senstivity
    obs, obs_sd = utils.pad_obs(mf_box, mf_var_box, time, obstime)
    #Get matrices for inversion 
    H, y, R, P, x_a = utils.inversion_matrices(obs, sensitivity, mf_ref, obs_sd, emissions_sd)
    #Do inversion
    x_hat, P_hat = inversion_analytical(y, H, x_a, R, P)
    return x_hat, P_hat, emis_ref, time

