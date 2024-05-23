import numpy as np
import pandas as pd
import matplotlib as plt
from tesswcs import WCS, pointings
from astroquery.mast import Catalogs
from astropy import units as u
from astropy.coordinates import SkyCoord


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

def check_visibility(targets, cycle=Review_Cycle):
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
    return sectors_visible
                        


def technical_review(ticinfo, plot=True, cycle=Review_Cycle):
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
    sectors_visible = check_visibility(target_tic, cycle=cycle)
    not_vis = np.sum(sectors_visible['#sectors'] == 0)
    target_flags["not_visible"] = not_vis * (not_vis > flag_thresholds["not_visible"])

    flag_df = pd.DataFrame({"flag":target_flags.keys() , "value": target_flags.values()})
    return flag_df

def panel_review(panel_proposals, tlfile, 
                 proposal_plots= True, summary_plot = True):
    panel_summary = pd.DataFrame({"proposal":panel_proposals,
                                  "flags":[""] * len(panel_proposals) })

    for index, row in panel_summary.iterrows():
        targets = _proposal_targets_from_csv(tlfile, row["proposal"])
        tic_targets = targets.tic_number.values.astype(np.int64)
        row["flags"] = technical_review(tic_targets, plot = proposal_plots)
    
    if(summary_plot):
        # do some summarizing from raised flags
        raise NotImplementedError