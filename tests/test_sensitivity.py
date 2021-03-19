from py12box_invert.paths import Paths
from py12box_invert.obs import Obs
from py12box_invert.invert import Invert
import numpy as np


species = "CFC-11"
project_path = Paths.data / f"example/{species}"

inv = Invert(project_path, species)


def test_sensitivity():

    inv.run_sensitivity(freq="yearly")
    sens_yr = inv.sensitivity.copy()

    inv.run_sensitivity(freq="quarterly")
    sens_qu = inv.sensitivity.copy()

    inv.run_sensitivity(freq="monthly")
    sens_mo = inv.sensitivity.copy()

    # Check number of columns is correct
    assert int(len(sens_mo[0,:])/12) == len(sens_yr[0, :])
    assert int(len(sens_qu[0,:])/4) == len(sens_yr[0, :])

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

    # Test that monthly, quarterly, annual sensitivites sum to same values
    sens_yr_boxed = np.reshape(sens_yr, 
                                (int(sens_yr.shape[0]/4), 4, int(sens_yr.shape[1]/4), 4))
    # Create a 5th axis containing a monthly or seasonal mean and then sum over it to annualise
    sens_qu_boxed_averaged = np.reshape(sens_qu,
                                (int(sens_qu.shape[0]/4), 4, int(sens_qu.shape[1]/4/4), 4, 4)).sum(axis=3)
    sens_mo_boxed_averaged = np.reshape(sens_mo,
                                (int(sens_mo.shape[0]/4), 4, int(sens_mo.shape[1]/4/12), 12, 4)).sum(axis=3)
    assert np.allclose(sens_yr_boxed, sens_qu_boxed_averaged)
    assert np.allclose(sens_yr_boxed, sens_mo_boxed_averaged)
