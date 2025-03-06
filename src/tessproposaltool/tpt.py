from __future__ import absolute_import

import argparse
import asyncio
import os
import tempfile


import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from astropy.io.votable import from_table
from astropy.table import Table
from astroquery.utils.tap.core import TapPlus

from lksearch.catalogsearch import query_id

import warnings


from . import _sync_call  # noqa
from . import get_logger

logger = get_logger()

TIC_COLUMNS = ["TIC", "RA", "DEC", "pmRA", "pmDEC", "Tmag"]
INPUT_COLUMNS = ["tic", "ra", "dec", "tmag"]
OUTPUT_COLUMNS = ["tic", "ra", "dec", "pmra", "pmde", "tmag"]
RENAME_OPTIONS = {
    "tic": ["tic", "tic_id", "id", "ticid", "tic_number", "tic_no", "#"],
    "ra": ["ra", "j2000ra", "raj2000"],
    "dec": ["dec", "j2000dec", "decj2000", "dej2000"],
    "pmra": ["pmra", "pmRA"],
    "pmde": ["pmde", "pmDE", "pmDEC", "pmdec"],
    "tmag": ["tmag", "tessmag", "mag", "vmag"],
}
TARGETLIST_OPTIONS = {
    "name": ["common_name", "common", "name"],
    "extended": ["extended", "extended_flag", "sg", "s'/'g", "s''g"],
    "20s_request": [
        "20s_request",
        "20s",
        "20s_cadence_flag",
        "20s_cadence",
        "20scadence",
    ],
    "swift_request": ["swift_request", "swift", "swiftrequest", "swift-request"],
    "nicer_request": ["nicer_request", "nicer", "nicerrequest", "nicer-request"],
    "remarks": ["remarks", "comments", "notes"],
}


# Assume this function is the asynchronous version of `launch_job`
async def async_launch_job(tap, query, upload_resource, upload_table_name):
    # Simulate asynchronous behavior (e.g., using aiohttp)
    return tap.launch_job(
        query=query,
        upload_resource=upload_resource,
        upload_table_name=upload_table_name,
        verbose=False,
    )


async def _xmatch_chunk(dataframe, task_number, radius=2, semaphore=None):
    tap = TapPlus(url="http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap")
    query = """
        SELECT tic.tic, tic.RAJ2000 as RA, tic.DEJ2000 as DEC, tic.pmRA, tic.pmDE as pmDEC, tic.Tmag, tic.dist, tic.teff, 
        tic.Disp as disposition, tic.m_TIC as duplicate_id, tic.plx, christina.index, christina.ra, christina.dec, christina.tmag,
        (DISTANCE(
          POINT('ICRS', tic.RAJ2000, tic.DEJ2000),
          POINT('ICRS', christina.ra, christina.dec)) * 3600/21) + 0.0000001 as sep,
        POWER(10, ABS(christina.tmag - tic.Tmag) * -0.4) as mweight
        FROM "IV/39/tic82" as tic, TAP_UPLOAD.table_test_{0} AS christina
        WHERE 1=CONTAINS(POINT('ICRS',tic.RAJ2000,tic.DEJ2000), CIRCLE('ICRS',christina.ra, christina.dec, {1} * 21/3600.)) 
        ORDER BY index
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, f"tic_table_{task_number}.xml")

        from_table(
            Table.from_pandas(dataframe[["ra", "dec", "tmag"]].fillna(0).reset_index())
        ).to_xml(filename)

        if semaphore:
            async with semaphore:
                j = await async_launch_job(
                    tap,
                    query.format(task_number, radius),
                    upload_resource=filename,
                    upload_table_name=f"table_test_{task_number}",
                )
        else:
            j = await async_launch_job(
                tap,
                query.format(task_number, radius),
                upload_resource=filename,
                upload_table_name=f"table_test_{task_number}",
            )

        r = j.get_results().to_pandas()
        mweight = np.min(
            [
                np.max([np.abs(np.log10(r.mweight)), 0.1 * np.ones(len(r))], axis=0),
                np.ones(len(r)) * 2,
            ],
            axis=0,
        )
        sepweight = np.min(
            [np.max([r.sep**2, np.ones(len(r)) * 0.01], axis=0), np.ones(len(r)) * 4],
            axis=0,
        )
        r["weight"] = (1 / sepweight) * 1 / mweight
        r = r[(r.disposition != "ARTIFACT") & (r.disposition != "DUPLICATE")]
        limit = 2000
        if len(r) > limit:
            r = r[r["index"] != r.iloc[-1]["index"]]

        r = (
            r.sort_values(["index", "mweight"], ascending=False)
            .drop_duplicates(subset=["index"], keep="first")
            .set_index("index")[TIC_COLUMNS]
        )
    return r


async def xmatch(dataframe, max_entries=50, radius=2, concurrency=5):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    results = []
    for idx in range(0, int(np.ceil(len(dataframe) / max_entries))):
        chunk_df = dataframe[idx * max_entries : (idx + 1) * max_entries]
        task_number = idx % concurrency
        task = asyncio.create_task(
            _xmatch_chunk(chunk_df, task_number, radius=radius, semaphore=semaphore)
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return dataframe.join(pd.concat(results))


def _parse_dataframe(input):
    logger.debug("Parsing dataframe columns")
    if isinstance(input, str):
        # Assume is a filename
        if input.endswith("csv"):
            input = pd.read_csv(input, comment="#")
        elif input.endswith("xlsx"):
            input = pd.read_excel(input)
        else:
            raise ValueError("Can not parse file extension.")
    if isinstance(input, pd.DataFrame):
        cols = np.asarray(input.columns)
        rename_dict = {}
        renamed_cols = []
        for key, item in RENAME_OPTIONS.items():
            if np.any([c.lower().strip() in item for c in cols]):
                rename_dict[[c for c in cols if c.lower().strip() in item][0]] = key
                renamed_cols.append(key)
            else:
                logger.debug(f"`{key}` not in input dataframe.")
        df = input.copy()
        df[list(set(list(RENAME_OPTIONS.keys())) - set(renamed_cols))] = np.nan
        return df.rename(columns=rename_dict)[list(RENAME_OPTIONS.keys())].astype(float)
    else:
        raise ValueError("Can not parse input.")


def fill_tics(
    dataframe, concurrency=5, parse_input=True, include_questionable_crossmatch=True
):
    logger.debug("Crossmatching nan TICs")
    if parse_input:
        df = _parse_dataframe(dataframe)[INPUT_COLUMNS]
    else:
        df = dataframe

    tic_df = df[df["tic"].isna()]
    if len(tic_df) == 0:
        logger.debug("All TICs specified.")
        return df
    logger.debug(f"{len(df)} missing TICs.")
    logger.start_spinner("Fixing missing TICs...")

    tic_df = _sync_call(
        xmatch, dataframe=tic_df, max_entries=40, radius=2, concurrency=concurrency
    )

    def iterate(max_entries=40, radius=2):
        nonlocal tic_df
        nfilled = (~tic_df["TIC"].isna()).sum()
        if nfilled < len(tic_df):
            newfilled = nfilled + 1
            while newfilled > nfilled:
                if not (tic_df.TIC.isna().any()):
                    return
                nfilled = (~tic_df["TIC"].isna()).sum()
                ndf = _sync_call(
                    xmatch,
                    dataframe=tic_df[tic_df.TIC.isna()][["ra", "dec", "tmag"]],
                    max_entries=max_entries,
                    radius=radius,
                    concurrency=concurrency,
                )
                if len(ndf) != 0:
                    tic_df = pd.concat([tic_df.dropna(subset=["TIC"]), ndf])
                newfilled = (~tic_df["TIC"].isna()).sum()

    # TODO include_questionable_crossmatch should be processed somewhere in here
    # Radius = 2 pixel
    iterate(50, 2)
    # Radius = 4 pixel
    iterate(5, 4)
    # Radius = 4 pixel
    iterate(1, 4)

    # This was throwing a setting with copy warning for me,
    # but this is using loc as per best practice and seems to be working
    # so catching & ignoring for now
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
        df.loc[tic_df.index, OUTPUT_COLUMNS] = np.asarray(tic_df[TIC_COLUMNS])
    logger.stop_spinner()
    return df[OUTPUT_COLUMNS]


def to_csv(target_list, filename="target_list.csv"):
    """Takes a dataframe of target information from tpt and writes an ark-compatible target-list in csv format.

    Required Columns Returned:
        -   TIC ID (if available)
        -   Right Ascension (decimal degrees)
        -   Declination (decimal degrees)
        -   Proper motion in Right Ascension (mas/yr)
        -   Proper motion in Declination (mas/yr)
        -   TESS mag

    Optional Columns Returned with default:
        -   Common name of target
        -   Extended flag (Is this S/G?  do we want to auto set this?)
        -   Special handling flag
        -   20-second cadence flag
        -   Swift time request (ksec)
        -   Nicer time request (ksec)
        -   Remarks

    Parameters
    ----------
    target_list : `~pandas.DataFrame`
        DataFrame of information from tpt
    filename : str
        name of the csv file to write to
    """

    """
    How do we want to handle optional columns?
    Two options?
        1. No optional inputs taken, a "default" target list is returned
            - we only ever allow up 3 columns of input - ra, dec, tmag
            - Assume default value for all Optional is 0/None - e.g. most basic 20s proposal.
            - ALL optional columns will be output, with a default value.    
            - User recieves this target list then customizes to their use case
            - We display some warning text that describes the default values and when/how a user should modify.
            - This is dangerous because it may be no loneger ark compatible if a user reads/writes the file
            - this is the "easiest" user option for the default 120s target proposal

        1.5 - Check Boxes set values across all columns
            
        2. We parse the input text file for potential optional columns.  
            - We have some sort of dictionary for a bunch of potnential column names  
            - We need documentation for what these names might be.  
                - alternatively we can enforce a strict column order and no names
                - people will inevitably screw this up so we'll need type checking here instead maybe?
            - If an optional column isn't present we throw a warning 
                - in that warning we saying describe the column that is missing and relevant column name(s) we accept
            - If a user supplies a column that isn't recognized we throw an error
                - we proibably require that a user have column name/labels in their csb
            - we potentially(?) omit any optional columns in the output table that are not supplied
            - this is more rubust to ark submissions but probably more annoying to the user 
                - it may require being somewhat "fiddly" for input csvs
            - the "default" proposal will have the MOST warnings
            - this is more annoying to code 
                -  conditional logic, name matching
                -  we'll have to track and assign more columns from the input df to the output df along the matched rows
        
        ** use lksearch to make giving a list of tics a valid input option**
        ** can include separation, contrast under remarks **
        ** tuning inputs for good crossmatch or everything? ** 
        ** how to send warnings to Rebekah? **
    """
    required_columns = [
        "tic",
        "ra",
        "dec",
        "pmra",
        "pmde",
        "tmag",
    ]
    required_keys = [col in target_list.keys() for col in required_columns]
    if any(~required_keys):
        logger.error(
            f"Missing Required Columns for a valid target list in Output DataFrame: {required_columns[~required_keys]}"
        )

    target_list.to_csv(filename, index=False)


def create_target_list(
    input_df,
    write_file=False,
    filename="target_list.csv",
    include_questionable_crossmatch=True,
):
    input_df = _parse_dataframe(input_df)
    # TODO some thing recursion parse mixed list
    new_dfs = []
    if input_df.tic.isnull().any():
        ticless = input_df.loc[input_df.tic.isnull()]
        new_dfs.append(
            fill_tics(
                ticless,
                parse_input=False,
                include_questionable_crossmatch=include_questionable_crossmatch,
            )
        )
    if input_df.tic.notnull().any():
        ticfull = input_df.loc[input_df.tic.notnull()]
        if (ticfull.ra.notnull().any()) or (ticfull.dec.notnull().any()):
            logger.warning(
                "TIC and ra/dec supplied: ignoring RA, DEC and using TIC to construct target list"
            )
        # TODO this should work for "TIC XXX" "XXX" XXX
        tic_df = query_id(ticfull.tic.astype(int).to_list(), output_catalog="tic")
        tic_df.index = ticfull.index
        tic_df = _parse_dataframe(tic_df)
        new_dfs.append(tic_df[OUTPUT_COLUMNS])

    # Merge crossmatched df and tic-queried df together
    if len(new_dfs) == 0:
        logger.exception("Cannot Retrieve ANY TIC values for supplied dataframe")
    else:
        new_df = pd.concat(new_dfs).sort_index()

    # Merge extra columns from the input file onto the columns from the TIC
    # parse_input is strippid columns so the below isn't adding in the remaining
    # evaluate
    # new_df = new_df.join(input_df[input_df.columns.difference(new_df.columns)])

    for key, item in TARGETLIST_OPTIONS.items():
        cols = np.asarray(input_df.columns)
        rename_dict = {}
        renamed_cols = []
        if np.any([c.lower().strip() in item for c in cols]):
            rename_dict[[c for c in cols if c.lower().strip() in item][0]] = key
            renamed_cols.append(key)
        else:
            logger.debug(f"`{key}` not in input dataframe.")  # Use a default key value

    if write_file:
        to_csv(new_df, filename=filename)
    return new_df


def tpt(args=None):
    """
    exposes tessproposaltool to the command line
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Creates Proposal Target Lists")
        parser.add_argument(
            "input",
            help="Input file.",
        )
        parser.add_argument(
            "-o",
            "--output",
            default="tess-targets.csv",
            help="Output filename. Defaults to tess-targets.csv",
        )

        args = parser.parse_args(args)
        args = vars(args)

    # Process
    create_target_list(args["input"], filename=args["output"])
