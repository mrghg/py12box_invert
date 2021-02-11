import numpy as np
import pandas as pd
from py12box import setup, core
from tqdm import tqdm

def flux_sensitivity(project_path, case, species, ic=None, input_dir=None, styear=None):
    """

    Parameters
    ----------
    project_path
    case
    species

    Returns
    -------

    """
    
    mol_mass, oh_a, oh_er = setup.get_species_parameters(species)
    time, emissions, ic0, lifetime = setup.get_case_parameters(project_path, case, species)
    if not ic:
        ic = np.copy(ic0)
    if input_dir:
        styear = int(time[0])
        i_t, i_v1, t, v1, oh, cl, temperature = setup.get_model_parameters(int(len(time) / 12), input_dir=input_dir, styear=styear)
    else:
        i_t, i_v1, t, v1, oh, cl, temperature = setup.get_model_parameters(int(len(time) / 12))
    F = setup.transport_matrix(i_t, i_v1, t, v1)

    mf_ref, burden, emissions_out, losses, lifetimes = \
        core.model(ic=ic, q=emissions,
                   mol_mass=mol_mass,
                   lifetime=lifetime,
                   F=F,
                   temp=temperature,
                   cl=cl, oh=oh,
                   arr_oh=np.array([oh_a, oh_er]))

    
    nyear = int(len(emissions)/12)
    sensitivity = np.zeros((len(mf_ref[:,:4].flatten()), int(nyear*4)))
    
    for mi in tqdm(range(nyear)):
        for bi in range(4):

            emissions_perturbed = emissions.copy()
            emissions_perturbed[mi*12:(12*mi+12), bi] +=1 #*= 2.
            
            mf_perturbed, burden, emissions_out, losses, lifetimes = \
                core.model(ic=ic, q=emissions_perturbed,
                           mol_mass=mol_mass,
                           lifetime=lifetime,
                           F=F,
                           temp=temperature,
                           cl=cl, oh=oh,
                           arr_oh=np.array([oh_a, oh_er]))

            sensitivity[:, 4*mi + bi] = (mf_perturbed[:,:4].flatten() - mf_ref[:,:4].flatten()) / 1.
    
    return sensitivity, mf_ref[:,:4].flatten()


if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    H = flux_sensitivity(Path("/home/chxmr/code/py12box_invert/data/AGAGE"), "CFC-11", "CFC-11")

    print(H[0, 0])

    plt.plot(H[:, 0])