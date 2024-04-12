from __future__ import absolute_import

import argparse
import asyncio
import os
import tempfile

import numpy as np
import pandas as pd
from astropy.io.votable import from_table
from astropy.table import Table
from astroquery.utils.tap.core import TapPlus

from . import _sync_call  # noqa
from . import get_logger

logger = get_logger()


RENAME_OPTIONS = {
    "tic": ["tic", "tic_id", "id", "ticid", "tic_number", "tic_no", "#"],
    "ra": ["ra", "j2000ra", "raj2000"],
    "dec": ["dec", "j2000dec", "decj2000"],
    "tmag": ["tmag", "tessmag", "mag", "vmag"],
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
            .set_index("index")[["TIC", "RA", "DEC", "Tmag"]]
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
                logger.warn(f"`{key}` not in dataframe.")
        df = input.copy()
        df[list(set(list(RENAME_OPTIONS.keys())) - set(renamed_cols))] = np.nan
        return df.rename(columns=rename_dict)[list(RENAME_OPTIONS.keys())].astype(float)
    else:
        raise ValueError("Can not parse input.")


def fill_tics(dataframe, concurrency=5):
    logger.debug("Crossmatching nan TICs")
    df = _parse_dataframe(dataframe)[["tic", "ra", "dec", "tmag"]]
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
    df.loc[tic_df.index, ["tic", "ra", "dec", "tmag"]] = np.asarray(
        tic_df[["TIC", "RA", "DEC", "Tmag"]]
    )
    logger.stop_spinner()
    return df


def tpt(args=None):
    """
    exposes toco to the command line
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
    new_df = fill_tics(args["input"])
    new_df.to_csv(args["output"], index=False)
