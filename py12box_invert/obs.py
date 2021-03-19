import pandas as pd
from bisect import bisect

from py12box_invert.utils import decimal_date

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

        self.time = decimal_date(pd.DatetimeIndex(df.index))
        self.mf = df.xs("mf", level="var", axis=1).values
        self.mf_uncertainty = df.xs("mf_variability", level="var", axis=1).values


    def change_start_year(self, start_year):

        if start_year > self.time[0]:

            ti = bisect(self.time, start_year) - 1
            self.time = self.time[ti:]
            self.mf = self.mf[ti:,:]
            self.mf_uncertainty = self.mf_uncertainty[ti:,:]

        elif start_year < self.time[0]:
            raise Exception("NEED TO SORT OUT PADDING!")
