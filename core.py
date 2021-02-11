import numpy as np
import pandas as pd
from py12box.py12box import startup, core,  model
from tqdm import tqdm
from py12box_invert import utils

def fwd_model(ic, emissions, mol_mass, lifetime, F, temperature, cl, oh, oh_a, oh_er):
    """
    Foward model call of box model
    Inputs:
        ic          initial conditions
        emissons    monthly emissions in 4 surface boxes
        mol_mass    molar mass of species
        lifetime    lifetime of species in each box
        F           Transport matrix
        temperature Temperature in each box in each month
        cl          chlorine loss
        oh          oh loss
        oh_a        oh_a loss
        oh_b        oh_b loss
    Returns:
        mf          monthly mole fraction in each box
        burden      burden of species in each box
        losses      loss in each box
        lifetimes   calculates lifetimes in each box
    """
    mf, burden, emissions_out, losses, lifetimes = \
    core.model(ic=ic, q=emissions,
                mol_mass=mol_mass,
                lifetime=lifetime,
                F=F,
                temp=temperature,
                cl=cl, oh=oh,
                arr_oh=np.array([oh_a, oh_er]))
    return mf, burden, emissions_out, losses, lifetimes

def fwd_model_inputs(project_path, case, species):
    """
    Get inputs to 12-box model
    """
    #mol_mass, oh_a, oh_er = setup.get_species_parameters(species)
    #time, emissions, ic, lifetime = setup.get_case_parameters(project_path, case, species)
    #i_t, i_v1, t, v1, oh, cl, temperature = setup.get_model_parameters(int(len(time) / 12))
    #F = setup.transport_matrix(i_t, i_v1, t, v1)
    mod = model.Model(species, project_path)
    return mod #time, ic, emissions, mol_mass, lifetime, F, temperature, cl, oh, oh_a, oh_er

def flux_sensitivity(project_path, case, species, ic0=None):
    """
    Derive linear yearly flux sensitivities
    
    Parameters
    ----------
    project_path
    case
    species
    ic0 (optional): Surface initial condition if estimating ic

    Returns
    -------

    """
#     time, ic, emissions, mol_mass, lifetime, F, temperature, cl, oh, oh_a, oh_er = \
#             fwd_model_inputs(project_path, case, species)
    mod = \
             fwd_model_inputs(project_path, case, species)
    
    if ic0 is not None:
        if len(ic0) == 4:
            ic = utils.approx_initial_conditions(species, project_path, case,
                                  ic0)
            mod.ic = ic
        else:
            print("ic0 does not have only 4 surface boxes. Ignoring.")
        
    
#     mf_ref, burden, emissions_out, losses, lifetimes = \
#             fwd_model(ic, emissions, mol_mass, lifetime, F, temperature, cl, oh, oh_a, oh_er)
    mod.run()
    mf_ref = mod.mf
    emissions = mod.emissions
    
    if len(emissions) % 12:
        raise Exception("Emissions must contain whole years")
    nyear = int(len(emissions)/12)
    sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nyear*4)))
    
    for mi in tqdm(range(nyear)):
        for bi in range(4):

            emissions_perturbed = emissions.copy()
            emissions_perturbed[mi*12:(12*mi+12), bi] +=1
            
            mod.emissions = emissions_perturbed
            mod.run(verbose=False)
            mf_perturbed = mod.mf
            #mf_perturbed, burden, emissions_out, losses, lifetimes = \
            #        fwd_model(ic, emissions_perturbed, mol_mass, lifetime, F, temperature, cl, oh, oh_a, oh_er)

            sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.
    
    return sensitivity, mf_ref[:,:4].flatten(), emissions, mod.time


def inversion_analytical(y, H, x_a, R, P):
    """
    Perform a Gaussian analytical inversion assuming linearity.
    The inferred value 'x_hat' is the difference in emissions from the reference run.
    
    TODO:
        This should really calculate global totals using off-diagonals
    """
    R_inv = np.linalg.inv(R)
    P_inv = np.linalg.inv(P)
    x_hat = np.linalg.inv(H.T @ R_inv @ H + P_inv) @ (H.T @ R_inv @ y + P_inv @ x_a)
    P_hat = np.linalg.inv(H.T @ R_inv @ H + P_inv)
    return x_hat, P_hat

def annual_means(x_hat, P_hat, emis_ref,  freq="yearly"):
    """
    Derive annual mean emissions from inversion output
    """
    
    if freq == "yearly":
        x_mnth = np.sum(np.repeat(x_hat.reshape(int(len(x_hat)/4),4),12, axis=0)+ emis_ref, axis=1)
        x_out = np.mean(x_mnth.reshape(-1, 12), axis=1)
        x_sd_out = np.zeros(len(x_out))
        j = 0
        for i in range(0,len(x_hat),4):
            x_sd_out[j] = np.sqrt(np.sum(P_hat[i:(i+4),i:(i+4)]))
            j+=1
        
        
    return x_out, x_sd_out

def run_inversion(project_path, case, species, ic0=None,  emissions_sd=None):
    """
    Run inversion for 12-box model to estimate monthly means from 4 surface boxes.
    """
    #Get obs
    obsdf =  utils.obs_read(species, project_path, case)
    obstime = utils.decimal_date(obsdf.index)
    #Box obs
    mf_box, mf_var_box = utils.obs_box(obsdf)
    #Get sensitivities
    sensitivity, mf_ref, emis_ref, time = flux_sensitivity(project_path, species, case, ic0=ic0)
    # Pad obs to senstivity
    obs, obs_sd = utils.pad_obs(mf_box, mf_var_box, time, obstime)
    #Get matrices for inversion 
    H, y, R, P, x_a = utils.inversion_matrices(obs, sensitivity, mf_ref, obs_sd, emissions_sd)
    #Do inversion
    x_hat, P_hat = inversion_analytical(y, H, x_a, R, P)
    return x_hat, P_hat, emis_ref, time

if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    H, mf_ref, emis_ref = flux_sensitivity(Path("/home/lw13938/work/py12box-invert/data/example"), "CFC-11", "CFC-11")

    print(H[0, 0])

    plt.plot(H[:, 0])