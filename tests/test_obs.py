import pandas as pd
from py12box_invert.paths import Paths
from py12box_invert.obs import Obs

def test_read():

    cfc11 = Obs(Paths.data / "example/CFC-11/CFC-11_obs.csv")
    
    assert cfc11.scale == "SIO-05"
    assert cfc11.units == "ppt"
    assert cfc11.mf.shape == (511, 4)
    assert cfc11.mf_uncertainty.shape == (511, 4)
    assert cfc11.mf[0 , 0] == 158.39
