import pandas as pd
from py12box_invert.utils import decimal_date

class Obs:
    """Class to store observations
    """

    def __init__(self, obs_file):
        """Read obs file

        Parameters
        ----------
        obs_file : str or pathlib.Path
            Filename for obs file
        """

        self.obs_read(obs_file)

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
