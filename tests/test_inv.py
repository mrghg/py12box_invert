from py12box_invert.paths import Paths
from py12box_invert.obs import Obs
from py12box_invert.invert import Invert, Store_model
import numpy as np


species = "CFC-11"
project_path = Paths.data / f"example/{species}"

inv = Invert(project_path, species, 
             start_year=2000., end_year=2010.,
             method="rigby14",
             ic_years=None)


def test_alignment():

    assert np.allclose(inv.mod.time, inv.obs.time)
    

def test_sensitivity():

    # Update prior model, to reflect changes in start and end date
    inv.mod.run()
    inv.mod_prior = Store_model(inv.mod)

    inv.run_sensitivity(freq="yearly")
    sens_yr = inv.sensitivity.sensitivity.copy()

    inv.run_sensitivity(freq="quarterly")
    sens_qu = inv.sensitivity.sensitivity.copy()

    inv.run_sensitivity(freq="monthly")
    sens_mo = inv.sensitivity.sensitivity.copy()

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

    # For last flux element, nothing should be seen before last timestep
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


def test_matrices():

    inv.run_sensitivity("yearly")

    inv.create_matrices(sigma_P=10.)

    # Check that sensitivity matrix has same number of rows as number of finite obs
    assert inv.mat.H.shape[0] == int(np.isfinite(inv.obs.mf).sum())
    # Check sensitivty matrix has columns corresponding to annual perturbations
    assert inv.mat.H.shape[1] == int(inv.mod.emissions.shape[0]*inv.mod.emissions.shape[1]/12)

    assert inv.mat.y.shape[0] == inv.mat.H.shape[0]
    assert inv.mat.x_a.shape[0] == inv.mat.H.shape[1]

    assert inv.mat.P_inv.shape[0] == inv.mat.P_inv.shape[1]

    # Check all diagonal prior covariances are positive
    assert sum(np.diag(inv.mat.P_inv) > 0.) == inv.mat.x_a.shape[0]

    assert inv.mat.y.shape[0] == inv.mat.R.shape[0]
    assert inv.mat.y.shape[0] == inv.mat.R.shape[1]

    # Check all diagonal measurement covariances are positive
    assert sum(np.diag(inv.mat.R) > 0.) == inv.mat.y.shape[0]


def test_inversion():

    inv.run_inversion()
    
    # Has a posterior solution been found, and does it take reasonable values
    assert np.abs(inv.mat.x_hat.sum()) < 1e6

    # Is posterior covariance positive definite?
    L_hat = np.linalg.cholesky(inv.mat.P_hat)


def test_posterior():

    inv.posterior()
    