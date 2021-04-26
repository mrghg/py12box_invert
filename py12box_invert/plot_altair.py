import altair as alt
from altair.vegalite.v4.schema.channels import Tooltip
import pandas as pd
import numpy as np


box_name = ["30 \u00B0N - 90 \u00B0N", 
            "00 \u00B0N - 30 \u00B0N",
            "30 \u00B0S - 00 \u00B0S",
            "90 \u00B0S - 30 \u00B0S"]

box_color = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']


def dec_to_month(dec_date):
    """Quick function to convert decimal date array to pandas

    WARNING: Assumes each month is just 1/12 of the year

    Parameters
    ----------
    dec_date : flt
        Decimal date

    Returns
    -------
    pd.Datetime
        Pandas datetime object
    """

    nt=len(dec_date)
    dec_date_round = np.round(dec_date, decimals=4)
    date_out = pd.DataFrame({"year": dec_date_round.astype(int),
        "month": np.round(((dec_date_round - dec_date_round.astype(int))*12.)) + 1,
        "day": np.ones(nt)})
    return pd.to_datetime(date_out)


class Plot:
    
    def plot_mf(self):

        selection = alt.selection_multi(fields=['Box'], bind='legend')

        data = pd.concat([pd.DataFrame(
                data={"Date": dec_to_month(self.obs.time),
                    "Box": box_name[bi],
                    "Obs": self.obs.mf[:, bi],
                    "Obs_uncertainty": self.obs.mf_uncertainty[:, bi],
                    "Model": self.mod_posterior.mf[:, bi]}
                ) for bi in range(4)])

        xmin=data[np.isfinite(data["Obs"])]["Date"].min()
        xmax=data[np.isfinite(data["Obs"])]["Date"].max()
        ymin=data["Obs"].min()
        ymax=data["Obs"].max()

        # Calculate some y-ranges for the error bars
        base = alt.Chart(data).transform_calculate(
            ymin="datum.Obs-datum.Obs_uncertainty",
            ymax="datum.Obs+datum.Obs_uncertainty"
        )

        # Plot the observations
        obs_plot = base.mark_point(filled=True, size=5).encode(
                x=alt.X("Date:T",
                    scale=alt.Scale(domain=(xmin, xmax))),
                y=alt.Y("Obs:Q",
                    title="CFC-11 (ppt)",
                    scale=alt.Scale(domain=(ymin, ymax))),
                color="Box:N", 
                opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1)),
                tooltip=["Date", "Obs"]
        )

        # Add error bars
        error_bars = base.mark_errorbar().encode(
                x="Date:T",
                y=alt.Y("ymin:Q", title=""),
                y2="ymax:Q",
                color="Box:N",
                opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1))
        )

        # Model
        mod_plot = base.mark_line(strokeWidth=2).encode(
            x="Date:T",
            y=alt.Y("Model:Q", title=""),
            color="Box:N",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
            )

        alt.layer(obs_plot, error_bars, mod_plot).add_selection(
            selection
            ).properties(
                width=600,
                height=400).interactive().display()