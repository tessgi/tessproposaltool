import numpy as np
import pandas as pd

from tessproposaltool import PACKAGEDIR, fill_tics, get_logger
from tessproposaltool.tpt import _parse_dataframe

logger = get_logger()
logger.setLevel("DEBUG")

testdir = "/".join([*PACKAGEDIR.split("/")[:-2], "tests", "data"]) + "/"


def test_parse():
    df = pd.read_csv(testdir + "test1.csv")
    new_df = _parse_dataframe(df)
    assert np.in1d(new_df.columns, ["tic", "ra", "dec", "tmag"]).all()
    assert new_df.loc[0, "ra"] == 120


def test_fill():
    df = pd.read_csv(testdir + "test1.csv")
    new_df = fill_tics(df)
    assert new_df.loc[0, "tic"] == 252969141.0
    df.loc[0, "tmag"] = 18.0
    new_df = fill_tics(df)
    assert new_df.loc[0, "tic"] == 802056325.0


def test_fill_larger():
    df = pd.read_csv(testdir + "test2.csv")
    new_df = _parse_dataframe(df)
    assert np.in1d(new_df.columns, ["tic", "ra", "dec", "tmag"]).all()
    new_df = fill_tics(df)
    assert np.isfinite(new_df.tic).all()
