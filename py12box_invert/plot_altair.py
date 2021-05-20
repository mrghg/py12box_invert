import altair as alt
from altair.vegalite.v4.schema.channels import Tooltip
import pandas as pd
import numpy as np
from py12box_invert.utils import decimal_to_pandas as dec_convert


box_name = ["30 \u00B0N - 90 \u00B0N", 
            "00 \u00B0N - 30 \u00B0N",
            "30 \u00B0S - 00 \u00B0S",
            "90 \u00B0S - 30 \u00B0S"]

box_color = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']


# def dec_to_month(dec_date):
#     """Quick function to convert decimal date array to pandas

#     WARNING: Assumes each month is just 1/12 of the year

#     Parameters
#     ----------
#     dec_date : flt
#         Decimal date

#     Returns
#     -------
#     pd.Datetime
#         Pandas datetime object
#     """

#     nt=len(dec_date)
#     dec_date_round = np.round(dec_date, decimals=4)
#     date_out = pd.DataFrame({"year": dec_date_round.astype(int),
#         "month": np.round(((dec_date_round - dec_date_round.astype(int))*12.)) + 1,
#         "day": np.ones(nt)})
#     return pd.to_datetime(date_out)



class Plot:
    
    def plot_mf(self):

        selection = alt.selection_multi(fields=['Box'], bind='legend')

        data = pd.concat([pd.DataFrame(
                data={"Date": dec_convert(self.obs.time),
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

        mf_upper = alt.layer(obs_plot, error_bars, mod_plot).properties(
                width=600,
                height=400).interactive()
        
        #
        # Plot the growth rate data
        #
        data_gr = pd.concat([pd.DataFrame(
            data={"Date": dec_convert(self.outputs.mf_growth[0]),
                "Box": box_name[bi],
                "GR": self.outputs.mf_growth[1][:, bi],
                "GR_uncertainty": self.outputs.mf_growth[2][:, bi]}
            ) for bi in range(4)])
        

        data_global_gr = pd.DataFrame(
                            data={"Date": dec_convert(self.outputs.mf_global_growth[0]),
                                "GR_global": self.outputs.mf_global_growth[1],
                                "GR_global_uncertainty": self.outputs.mf_global_growth[2]}
        )

        ymin_gr=data_gr["GR"].min()
        ymax_gr=data_gr["GR"].max()

        # Calculate some y-ranges for the error bars
        base_gr = alt.Chart(data_gr).transform_calculate(
            ymin_gr="datum.GR-datum.GR_uncertainty",
            ymax_gr="datum.GR+datum.GR_uncertainty"
        )

        # Plot the zero line for clarity
        overlay = pd.DataFrame({'y': [0]})
        zline = alt.Chart(overlay).mark_rule().encode(y="y")

        # Plot the growth rate
        gr_plot = base_gr.mark_line().encode(
                x=alt.X("Date:T",
                    scale=alt.Scale(domain=(xmin, xmax))),
                y=alt.Y("GR:Q",
                    title=f"Growth rate (ppt/yr)",
                    scale=alt.Scale(domain=(ymin_gr, ymax_gr))),
                color="Box:N",
                opacity=alt.condition(selection, alt.value(0.5), alt.value(0.1)),
#                tooltip=["Date", "GR"]
                )

        gr_global = alt.Chart(data_global_gr).mark_line(color="black").encode(
                x=alt.X("Date:T"),
                y=alt.Y("GR_global:Q")
                )

        #TODO: Add global uncertainty

        # # Add error shading
        # gr_error_bars = base_gr.mark_area().encode(
        #         x="Date:T",
        #         y=alt.Y("ymin_gr:Q", title=""),
        #         y2="ymax_gr:Q",
        #         color="Box:N",
        #         opacity=alt.condition(selection, alt.value(0.3), alt.value(0.1))
        #         )

        gr_lower = alt.layer(gr_plot, gr_global).properties(
            width=600,
            height=150).interactive()

        alt.vconcat(mf_upper, gr_lower).add_selection(
            selection, x_domain
        ).display()
        
        
        
    def plot_emissions(self):
        """
        Plot emissions using altair.
        """

        data = pd.DataFrame(
        data={"Date": dec_convert(self.outputs.emissions_global_annual[0]),
              "Emissions": self.outputs.emissions_global_annual[1]}
        )
        data = data.set_index('Date')

        data_err = pd.DataFrame(
                data={"Date": dec_convert(self.outputs.emissions_global_annual[0]),
                    "Emissions": self.outputs.emissions_global_annual[1],
                    "Emissions_uncertainty":self.outputs.emissions_global_annual[2]}
                )

        base = alt.Chart(data_err).transform_calculate(
            ymin="datum.Emissions-datum.Emissions_uncertainty",
            ymax="datum.Emissions+datum.Emissions_uncertainty"
        )

        source = data.reset_index().melt('Date', var_name='Legend', value_name='y')
        # Create a selection that chooses the nearest point & selects based on x-value
        nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                fields=['Date'], empty='none')

        # The basic line
        line = alt.Chart(source).mark_line().encode(
            x='Date:T',
            y='y:Q',
            color='Legend:N'
        )
        # Transparent selectors across the chart. This is what tells us
        # the x-value of the cursor
        selectors = alt.Chart(source).mark_point().encode(
            x='Date:T',
            opacity=alt.value(0),
        ).add_selection(
            nearest
        )

        # Draw points on the line, and highlight based on selection
        points = line.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        # Draw text labels near the points, and highlight based on selection
        text = line.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, 'y:Q', alt.value(' '))
        )

        # Draw a rule at the location of the selection
        rules = alt.Chart(source).mark_rule(color='gray').encode(
            x='Date:T',
        ).transform_filter(
            nearest
        )

        em_error_bars = base.mark_area(opacity=0.3).encode(
            x="Date:T",
            y=alt.Y("ymin:Q", title=f"{self.species} Emissions (Gg/yr)"),
            y2="ymax:Q",
            )


        # Put the five layers into a chart and bind the data
        alt.layer(
            line, selectors, points, rules, text,em_error_bars
        ).properties(
            width=600, height=300
        ).display()
