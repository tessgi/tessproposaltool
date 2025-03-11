import numpy as np
import pandas as pd

from tessproposaltool import PACKAGEDIR, fill_tics, get_logger, create_target_list
from tessproposaltool.tpt import _parse_dataframe

logger = get_logger()
logger.setLevel("DEBUG")

testdir = "/".join([*PACKAGEDIR.split("/")[:-2], "tests", "data"]) + "/"


def test_parse():
    df = pd.read_csv(testdir + "test1.csv")
    new_df = _parse_dataframe(df)
    # we do not require or expect a pmra/pmdec to be supplied
    assert np.in1d(
        new_df[["tic", "ra", "dec", "tmag"]].columns, ["tic", "ra", "dec", "tmag"]
    ).all()
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
    # we do not require or expect a pmra/pmdec to be supplied
    assert np.in1d(
        new_df[["tic", "ra", "dec", "tmag"]].columns, ["tic", "ra", "dec", "tmag"]
    ).all()
    new_df = fill_tics(df)
    assert np.isfinite(new_df.tic).all()


def test_inputs():
    # Test a file with:
    #   - ONLY RA, DEC
    #   - ONLY TIC
    #   - MIX TIC, Ra, Dec
    # Check to make sure:
    # we have the right number of responses
    # the responses are not null
    # they are returned in the sanme order as the input
    files = ["test3.csv", "test4.csv", "test5.csv"]

    def test_file(file):
        true_ticlist = [245701221, 5121803, 149605432, 462915110, 459811015, 466105108]
        test_out = create_target_list(testdir + file)
        # Make sure our output is the right length
        length = len(test_out) == 6
        # make sure we have matched a tic for every object
        complete = not test_out.tic.isnull().any()
        # Make sure our tics match and that the output order is the input order
        correct_order = all(
            [out == true for out, true in zip(test_out.tic, true_ticlist)]
        )
        return length, complete, correct_order

    for file in files:
        length, complete, correct_order = test_file(file)
        logger.debug(
            f"File: {file} Length:{length} Complete: {complete} Correct Order: {correct_order}"
        )
        assert length
        assert complete
        assert correct_order


def test_optional_columns():
    testfile = testdir + "test6.csv"

    optional_columns = [
        "name",
        "extended",
        "special_handling",
        "20s_request",
        "swift_request",
        "nicer_request",
        "remarks",
    ]

    test_df = create_target_list(testfile)
    for col in optional_columns:
        assert col in test_df.columns
    i_y = [0, 2]
    i_n = [1, 3, 4, 5]
    for i in i_y:
        assert test_df.iloc[i]["20s_request"] == "Y"
    for i in i_n:
        assert test_df.iloc[i]["20s_request"] == "N"
    assert all(test_df["extended"] == "N")
    assert all(test_df["special_handling"] == "N")

    assert test_df.name.isnull().all()
    assert test_df.swift_request.isnull().all()
    assert test_df.nicer_request.isnull().all()
