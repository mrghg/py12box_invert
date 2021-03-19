from py12box_invert.paths import Paths
from py12box_invert.obs import Obs
from py12box_invert.invert import Invert
import numpy as np

species = "CFC-11"
project_path = Paths.data / f"example/{species}"

inv = Invert(project_path, species)

def test_sensitivity():

    inv.flux_sensitivity(freq="yearly")
    sens_yr = inv.sensitivity.copy()

    inv.flux_sensitivity(freq="quarterly")
    sens_qu = inv.sensitivity.copy()

    inv.flux_sensitivity(freq="monthly")
    sens_mo = inv.sensitivity.copy()

    # Check number of columns is correct
    assert len(sens_mo[0,:])/12 == len(sens_yr)
    assert len(sens_qu[0,:])/3 == len(sens_yr)

    # Check number of rows are consistent
    assert len(sens_yr[:, 0]) == len(sens_mo[:, 0])
    assert len(sens_yr[:, 0]) == len(sens_qu[:, 0])

    # Check some obvious values:

    # For first flux element, first mole fraction timestep should be non-zero
    assert sens_yr[0, 0] > 0.
    assert sens_qu[0, 0] > 0.
    assert sens_mo[0, 0] > 0.

    # For last flux element, nothing should be seen before until timestep
    assert np.isclose(sens_mo[-5, -1], 0.)
    assert sens_mo[-1, -1] > 0.
    assert np.isclose(sens_yr[-12*4-1, -1], 0.)
    assert sens_yr[-1, -1] > 0.
    assert np.isclose(sens_qu[-3*4-1, -1], 0.)
    assert sens_qu[-1, -1] > 0.
