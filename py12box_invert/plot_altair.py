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
        
        x_domain = alt.selection_interval(bind='scales', encodings=['x'])

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
                    title=f"{self.species} (ppt)",
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

        mf_upper = alt.layer(obs_plot, error_bars, mod_plot).add_selection(
            selection
            ).properties(
                width=600,
                height=400).interactive()
        
        #
        # Plot the growth rate data
        #
        data_gr = pd.concat([pd.DataFrame(
        data={"Date": dec_to_month(self.growth_rate.time),
            "Box": box_name[bi],
            "GR": self.growth_rate.mf[:, bi],
            "GR_uncertainty": self.growth_rate.mfsd[:, bi],
            "zero_line": np.zeros(len(self.growth_rate.time))}
        ) for bi in range(4)])
        
        ymin_gr=data_gr["GR"].min()
        ymax_gr=data_gr["GR"].max()

        # Calculate some y-ranges for the error bars
        base_gr = alt.Chart(data_gr).transform_calculate(
            ymin_gr="datum.GR-datum.GR_uncertainty",
            ymax_gr="datum.GR+datum.GR_uncertainty"
        )

        # Plot the zero line for clarity
        gr_zeros = base_gr.mark_line(color="#000000").encode(
                x=alt.X("Date:T"),
                y=alt.Y("zero_line"))

        # Plot the growth rate
        gr_plot = base_gr.mark_line().encode( #mark_point(filled=True, size=5).encode(
                x=alt.X("Date:T",
                    scale=alt.Scale(domain=(xmin, xmax))),
                y=alt.Y("GR:Q",
                    title=f"Growth rate (ppt/yr)",
                    scale=alt.Scale(domain=(ymin_gr, ymax_gr))),
                color="Box:N", 
                opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
                tooltip=["Date", "GR"]
        )

        # Add error shading
        gr_error_bars = base_gr.mark_area().encode(
                x="Date:T",
                y=alt.Y("ymin_gr:Q", title=""),
                y2="ymax_gr:Q",
                color="Box:N",
                opacity=alt.condition(selection, alt.value(0.4), alt.value(0.1)))


        gr_lower = alt.layer(gr_zeros,gr_error_bars,gr_plot).add_selection(
        selection, x_domain
        ).properties(
            width=600,
            height=150).interactive() 


        alt.vconcat(mf_upper, gr_lower).display()  