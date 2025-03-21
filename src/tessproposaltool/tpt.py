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
# columns to return from the fill_tics crossmatch
CROSSMATCH_COLUMNS = [
    "TIC",
    "RA",
    "DEC",
    "pmRA",
    "pmDEC",
    "Tmag",
    "sep",
    "weight",
    "mweight",
]
# Columns from the TIC Catalog we want
TIC_COLUMNS = ["TIC", "RA", "DEC", "pmRA", "pmDEC", "Tmag"]
# Columns we accept as input
INPUT_COLUMNS = ["tic", "ra", "dec", "tmag"]
# Required Output Columns
OUTPUT_COLUMNS = ["tic", "ra", "dec", "pmra", "pmde", "tmag"]
# Rename options for the required columns
RENAME_OPTIONS = {
    "tic": ["tic", "tic_id", "id", "ticid", "tic_number", "tic_no", "#"],
    "ra": ["ra", "j2000ra", "raj2000"],
    "dec": ["dec", "j2000dec", "decj2000", "dej2000"],
    "pmra": ["pmra", "pmRA"],
    "pmde": ["pmde", "pmDE", "pmDEC", "pmdec"],
    "tmag": ["tmag", "tessmag", "mag", "vmag"],
}
# Rename options for the optional columns
TARGETLIST_OPTIONS = {
    "name": ["common_name", "common", "name"],
    "extended": ["extended", "extended_flag", "sg", "s'/'g", "s''g"],
    "special_handling": ["special_handling", "special", "handling"],
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
# default values for optional columns if they are not specified
TARGETLIST_DEFAULTS = {
    "name": pd.NA,
    "extended": "N",
    "special_handling": "N",
    "20s_request": "N",
    "swift_request": pd.NA,
    "nicer_request": pd.NA,
    "remarks": pd.NA,
}

# preferred type for the required columns
# they do not need to be this on intput, but must be castable to this type
TARGETLIST_REQUIRED_TYPES = {
    "tic": int,
    "ra": float,
    "dec": float,
    "pmra": float,
    "pmde": float,
    "tmag": float,
}

# preferred type for the optional columns
# they do not need to be this on intput, but must be castable to this type
TARGETLIST_OPTIONAL_TYPES = {
    "name": str,
    "extended": str,
    "special_handling": str,
    "20s_request": str,
    "swift_request": float,
    "nicer_request": float,
    "remarks": str,
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
            .set_index("index")
        )
        r = r[CROSSMATCH_COLUMNS]
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


def _file2df(input):
    # Assume is a filename
    if input.endswith("csv"):
        input = pd.read_csv(input, comment="#")
    elif input.endswith("xlsx"):
        input = pd.read_excel(input)
    else:
        raise ValueError("Can not parse file extension.")
    return input


def _parse_dataframe(input):
    logger.debug("Parsing dataframe columns")
    if isinstance(input, str):
        input = _file2df(input)

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
        df = df.rename(columns=rename_dict)[list(RENAME_OPTIONS.keys())].astype(float)
    else:
        raise ValueError("Can not parse input.")

    validate_file = all(df.tic.notnull() | (df.ra.notnull() & df.dec.notnull()))
    if not (validate_file):
        raise ValueError(
            "Can't find a tic or ra/dec for all rows in the input.  Do you have a header?"
        )
    if len(df) == 0:
        logger.warning("Length of parsed file is zero - no data found")
        raise UserWarning
    return df


def _add_xmatch_column(df, tic_df):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
        df.loc[tic_df.index, OUTPUT_COLUMNS] = np.asarray(tic_df[TIC_COLUMNS])
        for row in tic_df.iterrows():
            sep_str = f"separation: {row[1]['sep']}"
            weight_str = f"weight: {row[1]['weight']}"
            mweight_str = f"mweight:  {row[1]['mweight']} | "
            crossmatch_string = (
                "Crossmatch Parameters: " + sep_str + weight_str + mweight_str
            )

            if row[1]["sep"] >= 3:
                logger.warning(
                    f"WARNING: TIC Crossmatch Uncertain for TIC {row[1]['TIC']} - see remarks column"
                )
                crossmatch_string = (
                    "WARNING: TIC Crossmatch Uncertain | " + crossmatch_string
                )
            df.loc[row[0], "xmatch"] = crossmatch_string
    return df


def fill_tics(dataframe, concurrency=5, parse_input=True):
    """
    Fills tic values using a positional crossmatch with a combined magnitude, positional weighted algorithm

    crossmatch parameters:
    separation: pixel fraction
    mweight:POWER(10, ABS(input.tmag - tic.Tmag) * -0.4) as mweight
    weight: (1 / sepweight) * 1 / mweight

    Parameters
    ----------
    dataframe : DataFrame
        input DataFrame of ra, dec, (tmag) to crossmatch and fill with TIC ids
    concurrency : int, optional
        parrallization of async TIC queries, by default 5
    parse_input : bool, optional
       parse an input file or data-frame to make it conform to tpt standards, by default True

    Returns
    -------
    df : Data Frame
        DataFrame of crossmatched TIC RESULTS to return
    """
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

    # Radius = 2 pixel
    iterate(50, 2)
    # Radius = 4 pixel
    iterate(5, 4)
    # Radius = 4 pixel
    iterate(1, 4)
    # This was throwing a setting with copy warning for me,
    # but this is using loc as per best practice and seems to be working
    # so catching & ignoring for now

    # broke this out so we could test
    df = _add_xmatch_column(df, tic_df)

    logger.stop_spinner()
    return df


def _validate_df(targetlist_df):
    # Enforce Typing
    for key in TARGETLIST_REQUIRED_TYPES.keys():
        try:
            targetlist_df[key] = targetlist_df[key].astype(
                TARGETLIST_REQUIRED_TYPES[key]
            )
        except ValueError:
            logger.exception(
                f"Cannot force {key} to be {TARGETLIST_REQUIRED_TYPES[key]} in: {targetlist_df[key]}"
            )

    for key in TARGETLIST_OPTIONAL_TYPES.keys():
        ind = targetlist_df[key].notnull()
        if any(ind):
            try:
                targetlist_df.loc[ind, key] = targetlist_df.loc[ind, key].astype(
                    TARGETLIST_OPTIONAL_TYPES[key]
                )
            except ValueError:
                logger.exception(
                    f"Cannot force {key} to be {TARGETLIST_OPTIONAL_TYPES[key]} in: {targetlist_df[key]}"
                )
    return targetlist_df


def _sort_tic_df(ticfull, tic_df):
    # TODO lksearch.catalog.search.query_id returns items in a different order than input
    # PR should go in to fix this, in the mean time, sort this list.
    new_index = np.zeros(len(tic_df))
    # astroquery does not return files in the same order they are delivered
    # We'll match the files downloaded here and re-create the index
    for i in range(len(new_index)):
        loc = ticfull["tic"].astype(int) == tic_df.iloc[i]["TIC"].astype(int)
        ind = ticfull.index[loc]
        ind = np.atleast_1d(ind)
        if len(ind) > 1:
            # Numpy deprecation warning - indexing with a 1 element array
            # For safety we'll make sure everything is at least a 1 element array and then grab the first item
            # To make sure there's not multiple matching items we'll make sure the length is not >1
            logger.exception("Multiple files in table matched in the download manifest")
        new_index[i] = ind[0]
        tic_df.index = new_index
    tic_df.sort_index()

    return tic_df


def create_target_list(
    user_df,
    write_file=True,
    filename="target_list.csv",
    include_questionable_crossmatch=True,
):
    """Takes a dataframe of target information  and writes an returns an ark-compatible target-list
    with optional file-write in csv format.
    If a CSV is supplied, it should be pandas readable with a column header to provide column names,
        with the names as state below

    The input dataframe requires EITHER:
        - RA, DEC and optionally Tmag
        - TIC ID
    If a TIC is supplied, the supplied information that is in the TIC catalog will be over-written

    OPTIONAL Columns can be supplied,  are:
        - name : The common name of the object (string or blank, Default blank)
        - extended: Is this an exended object? (Y/N,  Default N)
        - special_handling : Does this object require special handling? (Y/N Default N)
        - 20s_request : IS
        - swift_request : IF THIS IS FOR A TESS-SWIFT JOINT PROPOSAL, THE Swift Exposure time in ks.
            - (blank for non-joint proposals, int or float for joint proposals, Default blank)
        - nicer_request : IF THIS IS FOR A TESS-NICER JOINT PROPOSAL, THE NICER Exposure time in ks.
            - (blank for non-joint proposals, int or float for joint proposals, Default blank)
        - remarks : user supplied notes (string or blank, Default blank.)

    We parse the input text file for required and optional columns assuming they will be labeled as above.
    We attempt a minimal standard disambiguation of the input file/DataFrame, see RENAME_OPTIONS and TARGETLIST_OPTIONS
    If an ra, dec (and optionally but recommended Tmag) is supplied, we attempt a crossmatch with the TIC.
        - a diagnostic will be added to the remarks column
        - tic/ra/dec/pmra/pmdec/tmag will be filled from the TIC with the crossmatch
    If a tic is supplied, ra/dec/pmra/pmdec/tmag from the input will be ignored
        - the values for these columns in the output will be populated from the TIC
    Warnings will be raised for optional columns that are not supplied.
    For these Columns a sensible default will be used, see TARGETLIST_OPTIONAL_TYPES
    A dataframe containing ALL columns will be returned.  This can be optionally written to an ARK-compatible csv file.

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
    user_df : DataFrame, string
        dataframe or file path of input
    write_file : bool, optional
        Whether or not to save the output targetlist as a file, by default True
    filename : str, optional
       name of the file to save, by default "target_list.csv"
    include_questionable_crossmatch : bool, optional
        _description_, by default True

    Returns
    -------
    DataFrame
        Dataframe containing the output TargetList
    """

    # original _parse_dataframe downselects user colums.  This is usefull for filling tics but problematic for target lists
    # we'll differentiate here and not modify parse to preserve current functionality.  WE could modify # TODO

    if isinstance(user_df, str):
        user_df = _file2df(user_df)

    input_df = _parse_dataframe(user_df)

    new_dfs = []
    missing_tics = input_df.tic.isnull().any()
    if missing_tics:
        ticless = input_df.loc[input_df.tic.isnull()]
        new_tics = fill_tics(
            ticless,
            parse_input=False,
        )
        new_dfs.append(new_tics[OUTPUT_COLUMNS])

    if input_df.tic.notnull().any():
        ticfull = input_df.loc[input_df.tic.notnull()]
        if (ticfull.ra.notnull().any()) or (ticfull.dec.notnull().any()):
            logger.warning(
                "TIC and ra/dec supplied: ignoring RA, DEC and using TIC to construct target list"
            )

        tic_df = query_id(ticfull.tic.astype(int).to_list(), output_catalog="tic")
        tic_df = _sort_tic_df(ticfull, tic_df)
        tic_df = _parse_dataframe(tic_df)
        new_dfs.append(tic_df[OUTPUT_COLUMNS])

    # Merge crossmatched df and tic-queried df together
    if len(new_dfs) == 0:
        logger.exception("Cannot Retrieve ANY TIC values for supplied dataframe")
    else:
        targetlist_df = pd.concat(new_dfs).sort_index()

    # Merge extra columns from the input file onto the columns from the TIC
    # parse_input is strippid columns so the below isn't adding in the remaining
    # evaluate
    # new_df = new_df.join(input_df[input_df.columns.difference(new_df.columns)])

    for key, item in TARGETLIST_OPTIONS.items():
        cols = np.asarray(user_df.columns)
        col_match = [c.lower().strip() in item for c in cols]
        if np.any(col_match):
            targetlist_df[key] = user_df[user_df.columns[col_match]]
            # These keys should be "Y/N" so lets validate
            if key in ["extended", "special_handling", "20s_request"]:
                targetlist_df[key] = targetlist_df[key].str.strip().str.upper()

        else:
            default = TARGETLIST_DEFAULTS[key]
            if default is pd.NA:
                default = "null"
            logger.warning(
                f"`{key}` not found in input dataframe - assuming: {default}"
            )  # Use a default key value
            targetlist_df[key] = TARGETLIST_DEFAULTS[key]
        # add crossmatch parameters from fill_tics to the final remarks section
        if (key == "remarks") and (missing_tics):
            for row in new_tics.iterrows():
                crossmatch_string = f"{row[1]['xmatch']}"
                if pd.isnull(targetlist_df.loc[row[0], key]):
                    targetlist_df.loc[row[0], key] = crossmatch_string
                else:
                    targetlist_df.loc[row[0], key] = (
                        targetlist_df.loc[row[0], key] + crossmatch_string
                    )

    # flake8 complained
    targetlist_df = _validate_df(targetlist_df)

    if write_file:
        targetlist_df.to_csv(filename, index=False)

    return targetlist_df


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
