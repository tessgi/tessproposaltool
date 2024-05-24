import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tesswcs import WCS, pointings
from astroquery.mast import Catalogs
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

from . import get_logger
logger = get_logger()

flags = ["extremely_bright",
         "bright",
         "faint",
         "extremely_faint",
         "not_visible",
         ]

flag_thresholds = {"extremely_bright": 4,
                   "bright": 7.5, 
                   "faint":16,
                   "extremely_faint":20,
                   "not_visible":0,
                   }
panel_configuration = {"Name": "Default Panel",

}
Review_Cycle = 6

def _proposal_targets_from_csv(file, proposal):
    target_list = pd.read_csv(file)
    prop_numbs = target_list["prnb"].unique()
    output_columns = ['tic_number','ra [ deg ]',
       'dec [ deg ]', 'pm_ra [ mas/yr ]', 'pm_dec [ mas/yr ]',
       'tess_mag [ mag ]', 'target_name', 'extended', 'special_handling']
    if proposal in prop_numbs:
        mask = target_list["prnb"] == proposal
        return target_list.loc[mask, output_columns]
    else:
        logger.error("Proposal Number Not Found in Target List File")

def check_visibility(targets, cycle=Review_Cycle, plot=True, proposal=0):
    target_coords = SkyCoord(targets['ra'].values * u.deg, 
                             targets['dec'].values * u.deg, 
                             pm_ra_cosdec = targets['pmRA'].values * u.mas / u.yr,
                             pm_dec = targets['pmDEC'].values * u.mas / u.yr, equinox='J2000')
    sectors_visible = pd.DataFrame({"tic":targets['ID'], "#sectors":[0] * len(targets['ID'])})
    sectors = pointings['Sector'][pointings['Cycle'] == cycle].data
    for sector in sectors:
        sectors_visible[f"s{sector}"] = 0
        ra, dec, roll = pointings[sector - 1][['RA', 'Dec', 'Roll']]
        for camera in np.arange(1, 5):
            for ccd in np.arange(1, 5):
                # predict the WCS
                wcs = WCS.predict(ra, dec, roll, camera=camera, ccd=ccd)
                # check if the target falls inside the CCD
                for tic, target in zip(targets['ID'].values, target_coords):
                    if wcs.footprint_contains(target):
                        sectors_visible.loc[sectors_visible["tic"] == tic,f"s{sector}"] = 1
    scol = [f"s{s}" for s in sectors]
    for item in sectors_visible["tic"]:
        sectors_visible.loc[sectors_visible["tic"] == item, "#sectors"] = np.sum(sectors_visible.loc[sectors_visible["tic"] == item, scol].values)
    
    if(plot):
        #Make a Visibility Plot
        fig = plt.figure(figsize=(3,1.5))
        ax=fig.add_subplot(111, projection="mollweide")
        ax.scatter(target_coords.ra.wrap_at(180 * u.deg).rad,
                      target_coords.dec.rad,
                      lw=0.5,
                      s=0.1,)
        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        #ax1=fig.add_subplot(122)
        #ax1.axis('off')
        #ax1.annotate("some Details Here",(0.6,0.75),xycoords='figure fraction')
        plt.tight_layout()
        plt.savefig(f'{proposal}_visibility.png')

    return sectors_visible
                        


def technical_review(ticinfo, plot=True, cycle=Review_Cycle, proposal=0):
    """Inclusions for a complete Technical Review:
        - Are Targets Observable?
        - Are they too bright?  
        - Are they too faint?

        Diagnostics:
        - what is their magnitude distribution?
        - what is their spatial distribution?
        - Have they been observed before? """
    target_tic = Catalogs.query_criteria(catalog="TIC", ID = ticinfo).to_pandas()
    target_flags = {}
    
    #Check Brightness Distribution
    target_flags["extremely_bright"] =  target_tic.Tmag.lt(flag_thresholds["extremely_bright"]).sum()
    target_flags["bright"] =  (target_tic.Tmag.lt(flag_thresholds["bright"]) & 
                               target_tic.Tmag.gt(flag_thresholds["extremely_bright"])).sum()
    
    target_flags["faint"] =  (target_tic.Tmag.gt(flag_thresholds["faint"]) & 
                              target_tic.Tmag.lt(flag_thresholds["extremely_faint"])).sum()
    target_flags["extremely_faint"] =  target_tic.Tmag.gt(flag_thresholds["extremely_faint"]).sum()

    #Check Target Visibility
    sectors_visible = check_visibility(target_tic, cycle=cycle, plot=plot, proposal=proposal)
    not_vis = np.sum(sectors_visible['#sectors'] == 0)
    target_flags["not_visible"] = not_vis * (not_vis > flag_thresholds["not_visible"])

    flag_df = pd.DataFrame({"value": target_flags.values()}, index=target_flags.keys())
    if(plot):
        """ These are just plot sketches, TBD Improved
        Current idea is they should be about the width of a printed page and the 
        height we would want in the summary, with the lot on the left and 
        any additional summary information on the right
        """
        
        #Make A Magnitude Histogram Plot
        rcParams['font.weight'] = 'bold'
        #hrange = np.arange(np.ceil(target_tic.Tmag.values.min()),
        #          np.ceil(target_tic.Tmag.values.max()))
        mag_hist = np.histogram(target_tic.Tmag.values)
        fig, ax = plt.subplots(1,2,figsize=(7,1.5))
        #gs=gridspec.GridSpec(1,2,figure=fig)
        #ax1 = plt.subplot(gs[0,0])
        ax[0].stairs(mag_hist[0], edges = mag_hist[1])
        ax[0].set_xlabel("TESS Magnitude")
        ax[0].set_ylabel("# of Targets")
        ax[1].axis('off')
        ax[1].annotate("some Details Here",(0.6,0.75),xycoords='figure fraction')
        plt.tight_layout()
        plt.savefig(f'{proposal}_brightness.png')
        
        ##Make a Visibility Plot
        #fig, ax = plt.subplots(1,2,figsize=(7,1.5), projection="mollweide")
        #ax[0].scatter()
        #ax[0].set_xlabel("TESS Magnitude")
        #ax[0].set_ylabel("# of Targets")
        #ax[1].axis('off')
        #ax[1].annotate("some Details Here",(0.6,0.75),xycoords='figure fraction')
        #plt.tight_layout()
        #plt.savefig(f'{proposal}_visibility.png')

    return flag_df

def panel_review(panel_proposals, tlfile, 
                 proposal_plots= True, 
                 summary_plot = True,
                 panel_name = "",
                 cycle=Review_Cycle):
    panel_summary = pd.DataFrame({"proposal":panel_proposals,
                                  "flags":[""] * len(panel_proposals) })

    for index, row in panel_summary.iterrows():
        targets = _proposal_targets_from_csv(tlfile, row["proposal"])
        tic_targets = targets.tic_number.values.astype(np.int64)
        total_targets = len(targets.tic_number.values)
        prop_flags = technical_review(tic_targets, plot = proposal_plots)
        panel_summary.loc[index, flags] = prop_flags.T.loc['value',flags]
        panel_summary.loc[index, "Total Targets"] = total_targets

    if(summary_plot):
        summary_file = f"TESS_Cycle{cycle}_Panel{panel_name}_TechnicalReviews.md"
        with open(summary_file, "w") as f:
            f.write(f"# Panel {panel_name} Technical Reviews\n")
            f.write(f"\n")

            for proposal in panel_proposals:
                f.write(f"## {proposal}\n")
                mask = panel_summary["proposal"] == proposal
                total_flags = panel_summary.loc[mask, flags].iloc[0].sum()
                if(total_flags > 0):
                    for flag in flags:
                        flag_value = panel_summary.loc[mask, flag].values[0]
                        if(flag_value > 0):
                            f.write(_write_summary_flag(flag, 
                                                        flag_value,
                                                        panel_summary.loc[mask,"Total Targets"].values[0].astype(int)))
                    # Add Summary plots for thrown flags
                    # TO-DO: still need to make these plots
                            f.write(_write_flag_plot(proposal, flag))
                else:
                    f.write("  - No potential issues noted for this proposal\n")       
                f.write(f"\n")

    return panel_summary

def _write_summary_flag(flag, value, total_targets):
    #This could be made more complex/descriptive
    return f"   - {value}/{total_targets} targets flagged as being {flag} \n"

def _write_flag_plot(proposal, flag):
    # Plots should get created in technical_review
    # this then writes a line in md summary file that includes the image
    if(flag in flags[0:4]):
        flag_plot_name = f'{proposal}_brightness.png'
    if(flag in flags[4]):
        flag_plot_name = f'{proposal}_visibility.png'

    return f"![{flag}]({flag_plot_name} '{flag}')\n"