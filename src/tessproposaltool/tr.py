import numpy as np
import pandas as pd
import os
import shutil

import matplotlib.pyplot as plt
from tesswcs import WCS, pointings
from tesswcs.utils import footprint

from astroquery.mast import Catalogs
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

from . import get_logger

logger = get_logger()

flags = [
    "extremely_bright",
    "bright",
    "faint",
    "extremely_faint",
    "not_visible",
]

flag_thresholds = {
    "extremely_bright": 4,
    "bright": 8,  # 7.5,
    "faint": 14,
    "extremely_faint": 20,
    "not_visible": 0,
}

Review_Cycle = 9


def _proposal_targets_from_csv(file, proposal):
    target_list = pd.read_csv(file)
    prop_numbs = target_list["prnb"].unique()
    output_columns = [
        "tic_number",
        "ra [ deg ]",
        "dec [ deg ]",
        "pm_ra [ mas/yr ]",
        "pm_dec [ mas/yr ]",
        "tess_mag [ mag ]",
        "target_name",
        "extended",
        "special_handling",
        "cadence_20sec",
    ]
    if proposal in prop_numbs:
        mask = target_list["prnb"] == proposal
        return target_list.loc[mask, output_columns]
    else:
        logger.error("Proposal Number Not Found in Target List File")

def _plot_visibility(sectors, sectors_visible, targets, name):
    # Make a Visibility Plot
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111, projection="mollweide")
    
    # ax.grid(True)
    # Add The Ecliptic Plane
    ecliptic_plane = SkyCoord(
        np.arange(0, 360, 1),
        np.arange(0, 360, 1) * 0,
        unit="deg",
        frame="geocentricmeanecliptic",
    ).transform_to("icrs")
    s = np.argsort(ecliptic_plane.ra.wrap_at(180 * u.deg).rad)
    plt.plot(
        ecliptic_plane.ra.wrap_at(180 * u.deg).rad[s],
        ecliptic_plane.dec.rad[s],
        color="grey",
        zorder=-10,
        alpha=0.5,
        lw=10,
    )
    # Add TESS Sector Outlines
    for sector in sectors:
        ra, dec, roll = pointings[sector - 1][["RA", "Dec", "Roll"]]
        for camera in np.arange(1, 5):
            for ccd in np.arange(1, 5):
                # predict the WCS
                wcs = WCS.predict(ra, dec, roll, camera=camera, ccd=ccd)
                # create world coordinates from a pixel footprint
                c = wcs.pixel_to_world(*footprint().T)
                # Plot each camera/CCD
                # TODO: Couldnt get fill to work, revisit
                ax.scatter(
                    c.ra.wrap_at(180 * u.deg).rad,
                    c.dec.rad,
                    color="k",
                    alpha=0.1,
                    lw=0.2,
                    s=0.1,
                )
    # Add the targets
    tic_vis = {
        "visible": sectors_visible.loc[
            sectors_visible["#sectors"] > 0, "tic"
        ].values,
        "not_visible": sectors_visible.loc[
            sectors_visible["#sectors"] == 0, "tic"
        ].values,
    }
    tic_vis_colors = {
        "visible": "green",
        "not_visible": "red",
    }
    tic_vis_label = {
        "visible": "Visible",
        "not_visible": "Not Visible",
    }
    for tics in ["visible", "not_visible"]:
        mask = targets["ID"].isin(tic_vis[tics])
        c = SkyCoord(
            targets.loc[mask, "ra"].values * u.deg,
            targets.loc[mask, "dec"].values * u.deg,
            pm_ra_cosdec=targets.loc[mask, "pmRA"].values * u.mas / u.yr,
            pm_dec=targets.loc[mask, "pmDEC"].values * u.mas / u.yr,
            equinox="J2000",
        )
        ax.scatter(
            c.ra.wrap_at(180 * u.deg).rad,
            c.dec.rad,
            lw=0.3,
            s=10,
            color=tic_vis_colors[tics],
            edgecolors="k",
            label=tic_vis_label[tics],
        )
    ax.legend()
    ax.set_xlabel("RA")
    ax.set_ylabel("DEC")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(7)
    # ax1=fig.add_subplot(122)
    # ax1.axis('off')
    # ax1.annotate("some Details Here",(0.6,0.75),xycoords='figure fraction')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def _plot_sectors_visible(sectors_visible, cycle, name):
    # Make A Number of Sectors Histogram Plot
    rcParams["font.weight"] = "bold"
    hrange = np.arange(
        0,
        sectors_visible["#sectors"].max()+1,
        1,
    )

    sectors_hist = np.histogram(sectors_visible["#sectors"].values, bins=hrange)
    
    # Make a Visibility Plot
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    
    ax.stairs(sectors_hist[0], edges=sectors_hist[1], fill=True, color='k')

    ax.set_xlabel(f'Number of Cycle {cycle} Sectors Visible', weight="bold")
    ax.set_ylabel("Number of Targets", weight="bold")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def check_visibility(targets, cycle=Review_Cycle, plot=True, proposal=0, t20s=[False]):
    target_coords = SkyCoord(
        targets["ra"].values * u.deg,
        targets["dec"].values * u.deg,
        pm_ra_cosdec=targets["pmRA"].values * u.mas / u.yr,
        pm_dec=targets["pmDEC"].values * u.mas / u.yr,
        equinox="J2000",
    )
    sectors_visible = pd.DataFrame(
        {"tic": targets["ID"], "#sectors": [0] * len(targets["ID"])}
    )
    sectors = pointings["Sector"][pointings["Cycle"] == cycle].data

    for sector in sectors:
        sectors_visible[f"s{sector}"] = 0
        ra, dec, roll = pointings[sector - 1][["RA", "Dec", "Roll"]]
        for camera in np.arange(1, 5):
            for ccd in np.arange(1, 5):
                # predict the WCS
                wcs = WCS.predict(ra, dec, roll, camera=camera, ccd=ccd)
                # check if the target falls inside the CCD
                for tic, target in zip(targets["ID"].values, target_coords):
                    if wcs.footprint_contains(target):
                        sectors_visible.loc[
                            sectors_visible["tic"] == tic, f"s{sector}"
                        ] = 1
    scol = [f"s{s}" for s in sectors]
    for item in sectors_visible["tic"]:
        sectors_visible.loc[sectors_visible["tic"] == item, "#sectors"] = np.sum(
            sectors_visible.loc[sectors_visible["tic"] == item, scol].values
        )


    return sectors_visible
def _plot_mag_hist(target_tic, flag_df,name):
   
    """ These are just plot sketches, TBD Improved
    Current idea is they should be about the width of a printed page and the 
    height we would want in the summary, with the lot on the left and 
    any additional summary information on the right
    """
    # Make A Magnitude Histogram Plot
    rcParams["font.weight"] = "bold"
    hrange = np.arange(
        np.floor(target_tic.Tmag.values.min()),
        np.ceil(target_tic.Tmag.values.max()) + 1,
    )
    mag_hist = np.histogram(target_tic.Tmag.values, bins=hrange)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    ax.stairs(mag_hist[0], edges=mag_hist[1], fill=True)
    color_dict = {
        "extremely_bright": "darkviolet",
        "bright": "lightseagreen",
        "extremely_faint": "firebrick",
        "faint": "orange",
    }

    for f in flags[0:4]:
        if flag_df.loc[f].values[0] > 0:
            if f == "extremely_bright":
                mask = target_tic.Tmag.values < flag_thresholds["extremely_bright"]
            if f == "bright":
                mask = (target_tic.Tmag.values < flag_thresholds["bright"]) & (
                    target_tic.Tmag.values > flag_thresholds["extremely_bright"]
                )
            if f == "extremely_faint":
                mask = target_tic.Tmag.values > flag_thresholds["extremely_faint"]
            if f == "faint":
                mask = (target_tic.Tmag.values > flag_thresholds["faint"]) & (
                    target_tic.Tmag.values < flag_thresholds["extremely_faint"]
                )
            flag_values = target_tic.Tmag.values[mask]
            flag_hist = np.histogram(flag_values, bins=mag_hist[1])
            
            ax.stairs(
                flag_hist[0], edges=mag_hist[1], color=color_dict[f], fill=True
            )

    ax.set_xlabel("TESS Magnitude", weight="bold")
    ax.set_ylabel("Number of Targets", weight="bold")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def _plot_mag_hist_old(target_tic, flag_df,name):
    
    """ These are just plot sketches, TBD Improved
    Current idea is they should be about the width of a printed page and the 
    height we would want in the summary, with the lot on the left and 
    any additional summary information on the right
    """
    # Make A Magnitude Histogram Plot
    rcParams["font.weight"] = "bold"
    hrange = np.arange(
        np.floor(target_tic.Tmag.values.min()),
        np.ceil(target_tic.Tmag.values.max()) + 1,
    )
    mag_hist = np.histogram(target_tic.Tmag.values, bins=hrange)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2))
    # gs=gridspec.GridSpec(1,2,figure=fig)
    # ax1 = plt.subplot(gs[0,0])
    ax[0].stairs(mag_hist[0], edges=mag_hist[1], fill=True)
    color_dict = {
        "extremely_bright": "darkviolet",
        "bright": "lightseagreen",
        "extremely_faint": "firebrick",
        "faint": "orange",
    }
    annotate_offset = 0
    # Set Annotate parameters so we can algorithmically place labels
    an_top = 0.875
    shift_an = 0.15
    align_an_x = 0.55
    align_an_col = 0.7
    astring = f"{np.format_float_positional(np.float16(len(target_tic.Tmag.values)), trim='-')} TOTAL Targets"
    ax[1].annotate(
        astring, (align_an_x, an_top - annotate_offset), xycoords="figure fraction"
    )
    
    annotate_offset = annotate_offset + shift_an
    
    for f in flags[0:4]:
        if flag_df.loc[f].values[0] > 0:
            if f == "extremely_bright":
                mask = target_tic.Tmag.values < flag_thresholds["extremely_bright"]
            if f == "bright":
                mask = (target_tic.Tmag.values < flag_thresholds["bright"]) & (
                    target_tic.Tmag.values > flag_thresholds["extremely_bright"]
                )
            if f == "extremely_faint":
                mask = target_tic.Tmag.values > flag_thresholds["extremely_faint"]
            if f == "faint":
                mask = (target_tic.Tmag.values > flag_thresholds["faint"]) & (
                    target_tic.Tmag.values < flag_thresholds["extremely_faint"]
                )
            flag_values = target_tic.Tmag.values[mask]
            flag_hist = np.histogram(flag_values, bins=mag_hist[1])
            ax[0].stairs(
                flag_hist[0], edges=mag_hist[1], color=color_dict[f], fill=True
            )
        # Add an annotation for legend
        v = np.format_float_positional(
            np.float16(flag_df.loc[f].values[0]), trim="-"
        )
        astring = f"{v} flagged as "
        ax[1].annotate(
            astring,
            (align_an_x, an_top - annotate_offset),
            xycoords="figure fraction",
        )
        ax[1].annotate(
            f"{f}",
            (align_an_col, an_top - annotate_offset),
            xycoords="figure fraction",
            color=color_dict[f],
        )
        annotate_offset = annotate_offset + shift_an
    nflag = flag_df.loc[flags[0:4]].values.sum()
    nnflag = len(target_tic.Tmag.values) - nflag
    v = np.format_float_positional(np.float16(nnflag), trim="-")
    astring = f"{v} Targets NOT flagged"
    ax[1].annotate(
        astring, (align_an_x, an_top - annotate_offset), xycoords="figure fraction"
    )
    ax[0].set_xlabel("TESS Magnitude", weight="bold")
    ax[0].set_ylabel("Number of Targets", weight="bold")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def _target_flags(target_tic,cycle,proposal):
    
    target_flags = {}

    # Check Brightness Distribution
    target_flags["extremely_bright"] = target_tic.Tmag.lt(
        flag_thresholds["extremely_bright"]
    ).sum()
    
    target_flags["bright"] = (
        target_tic.Tmag.lt(flag_thresholds["bright"])
        & target_tic.Tmag.gt(flag_thresholds["extremely_bright"])
    ).sum()

    target_flags["faint"] = (
        target_tic.Tmag.gt(flag_thresholds["faint"])
        & target_tic.Tmag.lt(flag_thresholds["extremely_faint"])
    ).sum()
    target_flags["extremely_faint"] = target_tic.Tmag.gt(
        flag_thresholds["extremely_faint"]
    ).sum()

    # Check Target Visibility
    sv_str = f'C{cycle}_G{proposal}_sectorsvisible.csv'
    if os.path.isfile(sv_str):
        sectors_visible = pd.read_csv(sv_str, index_col=0)
    else:
        sectors_visible = check_visibility(
            target_tic, cycle=cycle, plot=plot, proposal=proposal, t20s=t20s
            )
        sectors_visible.to_csv(sv_str)
    
    not_vis = np.sum(sectors_visible["#sectors"] == 0)
    target_flags["not_visible"] = not_vis * (not_vis > flag_thresholds["not_visible"])

    flag_df = pd.DataFrame({"value": target_flags.values()}, index=target_flags.keys())
    
    return flag_df, sectors_visible

def technical_review(ticinfo, 
                     plot=True, 
                     cycle=Review_Cycle, 
                     proposal=0,
                     targets=None, 
                     write=True,
                     overwrite=True, 
                     ):
    """Inclusions for a complete Technical Review:
    - Are Targets Observable?
    - Are they too bright?
    - Are they too faint?
    - How many sectors are targets visible for?

    Diagnostics:
    - what is their magnitude distribution?
    - what is their spatial distribution?
    - How many sectors are targets visible for?
    - Have they been observed before? Maybe not this """
    
    targtic_str = f'C{cycle}_G{proposal}_target_tic.csv'
    if os.path.isfile(targtic_str):
        target_tic = pd.read_csv(targtic_str, index_col=0, low_memory=False)
    else:
        target_tic = Catalogs.query_criteria(catalog="TIC", ID=ticinfo).to_pandas()
        target_tic.to_csv(targtic_str)
    
    create_120_file = (not os.path.isfile(f"{proposal}_120_flags.csv")) or overwrite
    if(create_120_file):
        flag_df_120, nsectors_120 = _target_flags(target_tic,cycle,proposal)
    else:
        flag_df_120 = pd.read_csv(f"{proposal}_120_flags.csv", index_col=0)
        nsectors_020 = pd.read_csv(f"{proposal}_020_flags.csv", index_col=0)

    
    create_020_file = (not os.path.isfile(f"{proposal}_020_flags.csv")) or overwrite
    if(targets is not None):
        tcut = targets.tic_number.isin(target_tic.ID)
        target_tic = pd.merge(target_tic, targets[tcut], how='outer', left_on='ID', right_on='tic_number')
        t20s = target_tic.cadence_20sec.values == 'Y'
    else:
        t20s=[False]

    flag_df_020 = None
    if(create_020_file):
        flag_df_020, nsectors_020 = _target_flags(target_tic[t20s],cycle,proposal)
    else:
        flag_df_020 = pd.read_csv(f"{proposal}_020_flags.csv", index_col=0)
        nsectors_020 = pd.read_csv(f"{proposal}_020_flags.csv", index_col=0)

    if plot:
        slist=nsectors_120.keys().values[2:]
        sectors = [int(k[1:]) for k in slist]
        _plot_mag_hist(target_tic,flag_df_120,f"{proposal}_120_brightness.png")
        _plot_visibility(sectors, nsectors_120, target_tic, f"{proposal}_120_visibility.png")
        _plot_sectors_visible(nsectors_120, cycle, f"{proposal}_120_nsectors.png")

        if(np.any(t20s)):
            _plot_mag_hist(target_tic[t20s],flag_df_020,f"{proposal}_020_brightness.png")     
            _plot_visibility(sectors, nsectors_020,target_tic[t20s], f"{proposal}_020_visibility.png")
            _plot_sectors_visible(nsectors_020, cycle, f"{proposal}_020_nsectors.png")

    if write and create_120_file:
        flag_df_120.to_csv(f"{proposal}_120_flags.csv")
        if(np.any(t20s) and create_020_file):
            flag_df_020.to_csv(f"{proposal}_020_flags.csv")
    
    return flag_df_120, flag_df_020


def _tr_files_exist(proposal):
    files_exist = (
        os.path.isfile(f"{proposal}_120_brightness.png")
        & os.path.isfile(f"{proposal}__120_visibility.png")
        & os.path.isfile(f"{proposal}__120_nsectors.png")
        & os.path.isfile(f"{proposal}_120_flags.csv")
    ) or (
        os.path.isfile(f"{proposal}_020_brightness.png")
        & os.path.isfile(f"{proposal}__020_visibility.png")
        & os.path.isfile(f"{proposal}__020_nsectors.png")
        & os.path.isfile(f"{proposal}_020_flags.csv")
    )

    return files_exist

def _plots_exist(proposal, cadence):
    plots_exist = (os.path.isfile(f"{proposal}_{cadence}_brightness.png") 
       & os.path.isfile(f"{proposal}_{cadence}_visibility.png") 
       & os.path.isfile(f"{proposal}_{cadence}_nsectors.png")
       )
    return plots_exist

def _write_quarto_column(f,proposal,
                         cadence, 
                         data_location, 
                         plot_location):
    if _plots_exist(proposal, cadence):
        
        f.write(f'### Summary of Requested {int(cadence)}s Targets:\n')
        f.write(f'![Target Magnitudes]({plot_location}{proposal}_{cadence}_brightness.png)\n')
        f.write(f'![Target Vsibility]({plot_location}{proposal}_{cadence}_visibility.png)\n')
        f.write(f'![Target Number Sectors Visible]({plot_location}{proposal}_{cadence}_nsectors.png)\n')
        f.write('\n')
        

        if(cadence == '120'):
            targets = _proposal_targets_from_csv("Cycle9_TargetList_mod.csv",proposal)
            ntot = len(targets)
        if(cadence == '020'):
            targets = _proposal_targets_from_csv("Cycle9_TargetList_mod.csv", proposal)
            tcut = targets.cadence_20sec == 'Y'
            ntot = len(targets[tcut])
    
        f.write(f'- **{ntot} Total {int(cadence)}s Targets Requested**\n')
        flags=pd.read_csv(f'{proposal}_{cadence}_flags.csv', index_col=0)
        if(flags.value.sum() > 0):
            if(flags.loc['extremely_bright'].value > 0):
                f.write(f'- There are **{flags.loc['extremely_bright'].value} target(s) that are extremely bright** and likely to be completely saturated\n')
            if(flags.loc['bright'].value > 0):
                f.write(f'- There are **{flags.loc['bright'].value} target(s) that are bright** and likely to be partially saturated\n')
            if(flags.loc['faint'].value > 0):
                f.write(f'- There are **{flags.loc['faint'].value} target(s) that are faint** and likely to have lower quality mission lightcurves\n')
            if(flags.loc['extremely_faint'].value > 0):
                f.write(f'- There are **{flags.loc['extremely_faint'].value} target(s) that are extremely faint** and likely to require non-mission, specialized photometry\n')
            if(flags.loc['not_visible'].value > 0):
                f.write(f'- There are **{flags.loc['not_visible'].value} target(s) that are likely to not be visible**.')
                f.write(f'Targets flagged as not_visible near the edge of ccds may or may not be observed.\n')
        else:

            f.write('- No flags noted\n')
            f.write('\n')            
    else:
        f.write(f'No {int(cadence)} Target List Found\n')

def _write_quarto_panel(cycle, panel_name, panel_proposals):
        # First set up directory structure
        # 
        os.makedirs(panel_name, exist_ok=True)
        shutil.copyfile('src/tessproposaltool/resources/index.qmd',f'{panel_name}/index.qmd')
        shutil.copyfile('src/tessproposaltool/resources/TESS-CDPP.png',f'{panel_name}/TESS-CDPP.png')

        # Make the Quarto book yaml
        with open(f'{panel_name}/_quarto.yml', "w") as f:
            f.write('\n')
            f.write(f'project:\n')
            f.write(f'  type: book\n')
            f.write(f'output-dir: "_{panel_name}"\n')
            f.write('book:\n')
            f.write(f'  title: "Panel {panel_name} Technical Reviews"\n')
            f.write(f'  output-file: "C{cycle}_{panel_name}_TechnicalReviews" \n')
            f.write(f'  chapters:\n')
            f.write(f'    - index.qmd\n')
            #f.write(f'    - intro.qmd\n')
            for proposal in panel_proposals:
                f.write(f'    - {proposal}.qmd\n')
            f.write('\n')
            f.write('format:\n')
            f.write('  pdf:\n')
            f.write('    documentclass: scrreprt\n')
            f.write('    toc: true\n')
            f.write('    toc-depth: 1\n')
            f.write('    mainfont: Helvetica\n')
            f.write('    geometry:\n')
            f.write('      - top=0.15in\n')
            f.write('      - bottom=0.15in\n')
            f.write('      - left=0.25in\n')
            f.write('      - right=0.25in\n')
        #Now write Each Panel Quarto .qmd file
        for proposal in panel_proposals:
            with open(f'{panel_name}/{proposal}.qmd', "w") as f:
                f.write('\n')
                f.write('---\n')
                #f.write(f'title: "{proposal} Technical Review"\n')
                f.write(f'format: pdf\n')
                f.write('---\n')
                f.write('\n')
                f.write(f'## Proposal: {proposal}\n')
                if(_plots_exist(proposal, '120') or _plots_exist(proposal,'020')):
                    f.write(f'\n')
                    #f.write(f'This proposal has requested high cadencence targets.')
                    #f.write(f'\n')
                else:
                    f.write(f'\n')
                    f.write(f'This proposal has not requested any 120-second or 20-second targets, therefore no target summaries are available. \n')
                    f.write(f'\n')
                
                f.write(f'\n')
                f.write(':::: {layout="[0.475, -0.05, 0.475]"}\n')
                f.write(f'\n')

                #left column
                f.write(f'\n')
                f.write(':::{#firstcol}\n')
                _write_quarto_column(f,proposal,'120','../','../')
                f.write(':::\n\n')

                # Right 
                #left column
                f.write(':::{#secondcol}\n')
                _write_quarto_column(f,proposal,'020','../','../')
                f.write(':::\n')
                f.write('::::')

def panel_review(
    panel_proposals,
    tlfile,
    proposal_plots=True,
    summary_plot=True,
    panel_name="",
    cycle=Review_Cycle,
    overwrite=False,
    quarto=True
):
    panel_summary = pd.DataFrame(
        {"proposal": panel_proposals, "flags": [""] * len(panel_proposals)}
    )

    for index, row in panel_summary.iterrows():
        targets = _proposal_targets_from_csv(tlfile, row["proposal"])
        if(targets is not None):
            tic_targets = targets.tic_number.values.astype(np.int64)
            total_targets = len(targets.tic_number.values)
            #if _tr_files_exist(row["proposal"]) and not overwrite:
            #    prop_flags = pd.read_csv(f"{row["proposal"]}_flags.csv", index_col=0)
            #else:
            prop_flags_120, prop_flags_020 = technical_review(
                tic_targets, proposal=row["proposal"], plot=proposal_plots, cycle=cycle, targets=targets
            )
            panel_summary.loc[index, flags] = prop_flags_120.T.loc["value", flags]
            panel_summary.loc[index, "Total Targets"] = total_targets
        
    if quarto:
        _write_quarto_panel(cycle, panel_name, panel_proposals)

    
    if summary_plot:
        summary_file = f"TESS_C{cycle}_{panel_name}_TR.md"
        with open(summary_file, "w") as f:
            f.write(f"# Panel {panel_name} Technical Reviews\n")
            f.write(f"\n")

            for proposal in panel_proposals:
                f.write(f"## {proposal}\n")
                mask = panel_summary["proposal"] == proposal
                total_flags = panel_summary.loc[mask, flags].iloc[0].sum()
                if total_flags > 0:
                    # Write Text Summary
                    for flag in flags:
                        flag_value = panel_summary.loc[mask, flag].values[0]
                        if flag_value > 0:
                            f.write(
                                _write_summary_flag(
                                    flag,
                                    flag_value,
                                    panel_summary.loc[mask, "Total Targets"]
                                    .values[0]
                                    .astype(int),
                                )
                            )
                    # Add Summary plots for thrown flags
                    for flag in flags:
                        flag_value = panel_summary.loc[mask, flag].values[0]
                        if flag_value > 0:
                            f.write(_write_flag_plot(proposal, flag))
                else:
                    f.write("  - No potential issues noted for this proposal\n")
                f.write(f"\n")

    return panel_summary


def _write_summary_flag(flag, value, total_targets):
    """write a written summary of the raised flag and"""
    match flag:
        case "extremely_bright":
            summary_string = f"   - {value}/{total_targets} targets flagged as being extremely bright with Tmag < {flag_thresholds["extremely_bright"]} \n"
        case "bright":
            summary_string = f"   - {value}/{total_targets} targets flagged as being bright with Tmag < {flag_thresholds["bright"]} \n"
        case "faint":
            summary_string = f"   - {value}/{total_targets} targets flagged as being faint with Tmag < {flag_thresholds["faint"]} \n"
        case "extremely_faint":
            summary_string = f"   - {value}/{total_targets} targets flagged as being extremely faint with Tmag < {flag_thresholds["extremely_faint"]} \n"
        case "not_visible":
            summary_string = f"   - {value}/{total_targets} targets flagged as being not visible this Cycle \n"
        case _:
            raise logger.error("Flag {flag} not found when writing summary")

    return summary_string


def _write_flag_plot(proposal, flag):
    # Plots should get created in technical_review
    # this then writes a line in md summary file that includes the image
    if flag in flags[0:4]:
        flag_plot_name = f"{proposal}_brightness.png"
    if flag in flags[4]:
        flag_plot_name = f"{proposal}_visibility.png"

    return f"![{flag}]({flag_plot_name} '{flag}')\n"
