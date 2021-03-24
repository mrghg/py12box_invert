import pandas as pd
import numpy as np
from py12box_invert.paths import Paths
from py12box_invert.obs import Obs

obs_file = Paths.data / "example/CFC-11/CFC-11_obs.csv"
cfc11 = Obs(obs_file)

with open(obs_file, "r") as f:
    lines = f.readlines()

ncomment = len([l for l in lines if l[0] == "#"])
nheader = 2 # There are two header lines
nlines = len(lines) - ncomment - nheader


def test_read():

    assert cfc11.scale == "SIO-05"
    assert cfc11.units == "ppt"
    assert cfc11.mf.shape == (nlines, 4)
    assert cfc11.mf_uncertainty.shape == (nlines, 4)
    assert cfc11.mf[0 , 0] == 158.39


def test_change_start_year():

    initial_len = cfc11.time.shape[0]
    initial_date = cfc11.time[0]

    #First, check padding
    cfc11.change_start_year(1970.)
    assert cfc11.time.shape[0] == int(np.round(initial_len + (initial_date - 1970.)*12))
    assert np.isfinite(cfc11.mf[:12,:]).sum() == 0

    # Now, check trimming
    cfc11.change_start_year(2000.)
    assert cfc11.time.shape[0] == int(np.round((cfc11.time[-1] - 2000.)*12)) + 1
    assert cfc11.time[0] == 2000.
    assert cfc11.time.shape[0] == cfc11.mf.shape[0]
    assert cfc11.time.shape[0] == cfc11.mf_uncertainty.shape[0]
    


def test_change_end_year():

    initial_len = cfc11.time.shape[0]
    initial_end = cfc11.time[-1]

    cfc11.change_end_year(2010.)
    assert cfc11.time.shape[0] == initial_len - int(np.round((initial_end - 2010.)*12)) - 1
    assert np.isclose(cfc11.time[-1], 2010. - 1./12.)
    assert cfc11.time.shape[0] == cfc11.mf.shape[0]
    assert cfc11.time.shape[0] == cfc11.mf_uncertainty.shape[0]
