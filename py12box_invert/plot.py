import altair as alt
import pandas as pd
import numpy as np

from py12box_invert.utils import decimal_to_pandas as dec_convert


box_name = ["30 \u00B0N - 90 \u00B0N", 
            "00 \u00B0N - 30 \u00B0N",
            "30 \u00B0S - 00 \u00B0S",
            "90 \u00B0S - 30 \u00B0S"]

box_color = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']


def plot_mf(inv):
    """Plot mole fraction

    Parameters
    ----------
    inv : Invert.outputs object
        Outputs class from py12box_invert.Invert
    """

    selection = alt.selection_multi(fields=['Box'], bind='legend')

    data_global = pd.DataFrame(
            data={"Date": dec_convert(inv.mf[0]),
                  "Box": "global",
                  "Obs": inv.mf[1].mean(axis=1),
                  "Obs_uncertainty": np.sqrt(((inv.mf[2])**2).sum(axis=1)) / 4,
                  "Model": inv.mf_model[1][:,:4].mean(axis=1)}
                  )

    data = pd.concat([pd.DataFrame(
            data={"Date": dec_convert(inv.mf[0]),
                "Box": box_name[bi],
                "Obs": inv.mf[1][:, bi],
                "Obs_uncertainty": inv.mf[2][:, bi],
                "Model": inv.mf_model[1][:, bi]}
            ) for bi in range(4)])# + [data_global])

    xmin=data[np.isfinite(data["Obs"])]["Date"].min()
    xmax=data[np.isfinite(data["Obs"])]["Date"].max()
    ymin=data["Obs"].min()
    ymax=data["Obs"].max()

    # Calculate some y-ranges for the error bars
    base = alt.Chart(data).transform_calculate(
        ymin="datum.Obs-datum.Obs_uncertainty",
        ymax="datum.Obs+datum.Obs_uncertainty"
    )

    base_global = alt.Chart(data_global).transform_calculate(
        ymin="datum.Obs-datum.Obs_uncertainty",
        ymax="datum.Obs+datum.Obs_uncertainty"
    )

    # Plot the observations
    obs_plot = base.mark_point(filled=True, size=5).encode(
            x=alt.X("Date:T",
                scale=alt.Scale(domain=(xmin, xmax)),
                axis=alt.Axis(title=None)),
            y=alt.Y("Obs:Q",
                title=f"{inv.species} (ppt)",
                scale=alt.Scale(domain=(ymin, ymax))),
            # color="Box:N", 
            color=alt.Color("Box", sort=box_name, legend=alt.Legend(orient='bottom')),
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1)),
            tooltip=["Box", "Date", "Obs"]
    )

    obs_global = base_global.mark_point(filled=True, size=5, color="black").encode(
            x=alt.X("Date:T"),
            y=alt.Y("Obs:Q"),
            tooltip=["Box", "Date", "Obs"]
            )


    # Add error bars
    error_bars = base.mark_errorbar().encode(
            x="Date:T",
            y=alt.Y("ymin:Q", title=""),
            y2="ymax:Q",
            # color="Box:N",
            color=alt.Color("Box", sort=box_name),
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1))
    )

    error_bars_global = base_global.mark_errorbar(color="black").encode(
            x="Date:T",
            y=alt.Y("ymin:Q", title=""),
            y2="ymax:Q"
    )


    # Model
    mod_plot = base.mark_line(strokeWidth=2).encode(
        x="Date:T",
        y=alt.Y("Model:Q", title=""),
        # color="Box:N",
        color=alt.Color("Box", sort=box_name),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
    )

    mod_plot_global = base_global.mark_line(strokeWidth=2, color="black").encode(
        x="Date:T",
        y=alt.Y("Model:Q", title="")
    )


    mf_upper = alt.layer(obs_plot, obs_global, error_bars, error_bars_global, mod_plot, mod_plot_global).properties(
            height=250,
            width="container"
            ).interactive()

    return mf_upper.add_selection(selection)
    
def plot_gr(inv):
    """Plot growth rate

    Parameters
    ----------
    inv : Invert.outputs object
        Outputs class from py12box_invert.Invert
    """
    
    selection = alt.selection_multi(fields=['Box'], bind='legend')

    data = pd.concat([pd.DataFrame(
            data={"Date": dec_convert(inv.mf[0]),
                "Box": box_name[bi],
                "Obs": inv.mf[1][:, bi],
                "Obs_uncertainty": inv.mf[2][:, bi],
                "Model": inv.mf_model[1][:, bi]}
            ) for bi in range(4)])

    xmin=data[np.isfinite(data["Obs"])]["Date"].min()
    xmax=data[np.isfinite(data["Obs"])]["Date"].max()

    data_gr = pd.concat([pd.DataFrame(
        data={"Date": dec_convert(inv.mf_growth[0]),
            "Box": box_name[bi],
            "GR": inv.mf_growth[1][:, bi],
            "GR_uncertainty": inv.mf_growth[2][:, bi]}
        ) for bi in range(4)])
    

    data_global_gr = pd.DataFrame(
                        data={"Date": dec_convert(inv.mf_global_growth[0]),
                            "GR_global": inv.mf_global_growth[1],
                            "GR_global_uncertainty": inv.mf_global_growth[2]}
    )

    ymin_gr=data_gr["GR"].min()
    ymax_gr=data_gr["GR"].max()

    # Plot the zero line for clarity
    overlay = pd.DataFrame({'y': [0]})
    zline = alt.Chart(overlay).mark_rule().encode(y="y")

    # Plot the growth rate
    gr_plot = alt.Chart(data_gr).mark_line().encode(
            x=alt.X("Date:T",
                scale=alt.Scale(domain=(xmin, xmax)),
                axis=alt.Axis(title=None)),
            y=alt.Y("GR:Q",
                title="Growth rate (ppt/yr)",
                scale=alt.Scale(domain=(ymin_gr, ymax_gr))),
            color=alt.Color("Box", sort=box_name, legend=alt.Legend(orient='bottom')),
            opacity=alt.condition(selection, alt.value(0.6), alt.value(0.1)),
            )

    # Calculate some y-ranges for the error bars
    base_gr = alt.Chart(data_global_gr).transform_calculate(
        ymin_gr="datum.GR_global-datum.GR_global_uncertainty",
        ymax_gr="datum.GR_global+datum.GR_global_uncertainty"
    )

    gr_global = base_gr.mark_line(color="black").encode(
            x=alt.X("Date:T"),
            y=alt.Y("GR_global:Q"),
            tooltip=["Date", "GR_global"]
            )

    # Add error shading
    gr_error_bars = base_gr.mark_area(color="grey").encode(
            x="Date:T",
            y=alt.Y("ymin_gr:Q", title=""),
            y2="ymax_gr:Q",
            opacity=alt.value(0.5)
            )

    gr_lower = alt.layer(gr_plot, gr_global, gr_error_bars).properties(
        height=250,
        width="container"
        ).interactive()

    return gr_lower.add_selection(
                                selection
                                )


def plot_emissions(inv):
    """
    Plot emissions using Altair.
    """

    data = pd.DataFrame(
            data={"Date": dec_convert(inv.emissions_global_annual[0]),
                "Emissions": inv.emissions_global_annual[1],
                "Emissions_uncertainty":inv.emissions_global_annual[2]}
            )

    # Find when the obs start and end
    wh = np.where(np.isfinite(inv.mf[1]))
    xmin=dec_convert([inv.mf[0][wh[0].min()],])[0]
    xmax=dec_convert([inv.mf[0][wh[0].max()],])[0]

    ymax=(data[data["Date"] > xmin]["Emissions"] + data[data["Date"] > xmin]["Emissions_uncertainty"]).max()

    source = data.copy()

    base = alt.Chart(source).transform_calculate(
        ymin="datum.Emissions-datum.Emissions_uncertainty",
        ymax="datum.Emissions+datum.Emissions_uncertainty"
    )

    # The basic line
    #TODO: Tooltip doesn't work. Maybe table needs to be long format?
    emissions = base.mark_line().encode(
        x=alt.X('Date:T',
                scale=alt.Scale(domain=(xmin, xmax)),
                title=None),
        y=alt.Y("Emissions:Q",
                scale=alt.Scale(domain=[0., ymax]),
                title=f"{inv.species} emissions (Gg/yr)"),
        tooltip=["Date", "Emissions"]
    )

    emissions_uncertainty = base.mark_area(opacity=0.3).encode(
        x=alt.X("Date:T",
                scale=alt.Scale(domain=(xmin, xmax))),
        y=alt.Y("ymin:Q", title=""),
        y2="ymax:Q",
    )

    return (emissions_uncertainty + emissions).properties(
                width="container",
                height=250
                ).interactive()


def plot_semihemi_emissions(inv):
    """
    Plot semihemispheric emissions using Altair.
    """

    selection = alt.selection_multi(fields=['Box'], bind='legend')

    data = pd.concat([pd.DataFrame(
            data={"Date": dec_convert(inv.emissions_annual[0]),
                "Box": box_name[bi],
                "Emissions": inv.emissions_annual[1][:, bi],
                "Emissions_uncertainty": inv.emissions_annual[2][:, bi]}
            ) for bi in range(4)])

    # Find when the obs start and end
    wh = np.where(np.isfinite(inv.mf[1]))
    xmin=dec_convert([inv.mf[0][wh[0].min()],])[0]
    xmax=dec_convert([inv.mf[0][wh[0].max()],])[0]

    ymax=(data[data["Date"] > xmin]["Emissions"] + data[data["Date"] > xmin]["Emissions_uncertainty"]).max()

    source = data.copy()

    base = alt.Chart(source).transform_calculate(
        ymin="datum.Emissions-datum.Emissions_uncertainty",
        ymax="datum.Emissions+datum.Emissions_uncertainty"
    )

    # The basic line
    #TODO: Tooltip doesn't work. Maybe table needs to be long format?
    emissions = base.mark_line().encode(
        x=alt.X('Date:T',
                scale=alt.Scale(domain=(xmin, xmax)),
                title=None),
        y=alt.Y("Emissions:Q",
                scale=alt.Scale(domain=[0., ymax]),
                title=f"{inv.species} emissions (Gg/yr)"),
        color=alt.Color("Box", sort=box_name, legend=alt.Legend(orient='bottom')),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1)),
        tooltip=["Date", "Emissions"]
    )

    emissions_uncertainty = base.mark_area(opacity=0.3).encode(
        x=alt.X("Date:T",
                scale=alt.Scale(domain=(xmin, xmax))),
        y=alt.Y("ymin:Q", title=""),
        y2="ymax:Q",
        color=alt.Color("Box", sort=box_name, legend=alt.Legend(orient='bottom')),
        opacity=alt.condition(selection, alt.value(0.3), alt.value(0.1)),
    )

    return (emissions + emissions_uncertainty).properties(
                #width=400, height=400
                width="container",
                height=250
                ).interactive().add_selection(selection)


def plot_residuals(inv):
    """Plot residual of modelled mole fraction and observations

    Parameters
    ----------
    inv : Invert.outputs object
        Outputs class from py12box_invert.Invert
    """

    selection = alt.selection_multi(fields=['Box'], bind='legend')

    data = pd.concat([pd.DataFrame(
            data={"Date": dec_convert(inv.mf[0]),
                "Box": box_name[bi],
                "Obs": inv.mf[1][:, bi],
                "Obs_uncertainty": inv.mf[2][:, bi],
                "Model": inv.mf_model[1][:, bi]}
            ) for bi in range(4)])

    # Find when the obs start and end
    wh = np.where(np.isfinite(inv.mf[1]))
    xmin=dec_convert([inv.mf[0][wh[0].min()],])[0]
    xmax=dec_convert([inv.mf[0][wh[0].max()],])[0]

    diff_data = pd.DataFrame(
                    data={"Date": data["Date"],
                          "Residual":data["Model"] - data["Obs"],
                          "Box": data["Box"]} 
    )

    p = alt.Chart(diff_data).mark_point(filled=True, size=10).encode(
        x=alt.X("Date",
                scale=alt.Scale(domain=(xmin, xmax)),
                title=None),
        y=alt.Y("Residual", title="Residual (model - observations) (ppt)"),
        color=alt.Color("Box", sort=box_name, legend=alt.Legend(orient='bottom')),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1)),
        tooltip=["Date", "Residual"]
    )

    return p.properties(
        width="container",
        height=250
    ).interactive().add_selection(selection)
