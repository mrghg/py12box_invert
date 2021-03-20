import pandas as pd
from bisect import bisect
from numpy import arange, hstack, vstack, zeros, nan

from py12box_invert.utils import decimal_date, round_date

class Obs:
    """Class to store observations
    """

    def __init__(self, obs_file, start_year=None):
        """Read obs file

        Parameters
        ----------
        obs_file : str or pathlib.Path
            Filename for obs file
        """

        self.obs_read(obs_file)
        
        if start_year != None:
            self.change_start_year(start_year)


    def obs_read(self, obs_file):
        """Read monthly mean csv file

        Converts time to decimal date, with months rounded to 1/12 year

        Parameters
        ----------
        fname : str
            Path to data file
        """

        def split_and_tidy(s):
            s = s.split(":")[-1]
            s = s.replace(" ", "")
            s = s.replace("\n", "")
            return s

        self.scale = "unknown"
        self.units = "unknown"

        with open(obs_file, "r") as f:
            l = f.readline()
            while l[0] == "#":
                l = f.readline()
                if "SCALE: " in l:
                    self.scale = split_and_tidy(l)
                if "UNITS: " in l:
                    self.units = split_and_tidy(l)

        df = pd.read_csv(obs_file,
                        comment="#", header=[0, 1], index_col=[0])

        self.time = round_date(decimal_date(pd.DatetimeIndex(df.index)))
        self.mf = df.xs("mf", level="var", axis=1).values
        self.mf_uncertainty = df.xs("mf_variability", level="var", axis=1).values


    def change_start_year(self, start_year):
        """Change the start year of the obs class

        If new start year is before first element, will pad with monthly nans

        Parameters
        ----------
        start_year : flt
            New start year
        """

        if float(start_year) > self.time[0]:
            # Trim at new start date
            ti = bisect(self.time, float(start_year)) - 1
            self.time = self.time[ti:]
            self.mf = self.mf[ti:,:]
            self.mf_uncertainty = self.mf_uncertainty[ti:,:]

        elif float(start_year) < self.time[0]:
            # Pad with nans
            new_time = arange(start_year, self.time[0], step=1/12)
            self.time = hstack([new_time, self.time])
            nanarray = zeros((len(new_time), 4))*nan
            self.mf = vstack([nanarray, self.mf])
            self.mf_uncertainty = vstack([nanarray, self.mf_uncertainty])

    def change_end_year(self, end_year):
        """Change end year of the obs class

        Parameters
        ----------
        end_year : flt
            New end year
        """

        if float(end_year) < self.time[-1]:
            # Trim at new end date
            ti = bisect(self.time, float(end_year)) - 1
            self.time = self.time[:ti]
            self.mf = self.mf[:ti, :]
            self.mf_uncertainty = self.mf_uncertainty[:ti, :]
        elif float(end_year) > self.time[-1]:
            # Pad with nans
            new_time = arange(self.time[-1] + 1/12, end_year, step=1/12)
            self.time = hstack([self.time, new_time])
            nanarray = zeros((len(new_time), 4))*nan
            self.mf = vstack([self.mf, nanarray])
            self.mf_uncertainty = vstack([self.mf_uncertainty, nanarray])