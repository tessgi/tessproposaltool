import numpy as np
import pandas as pd
import pytest

from tessproposaltool import PACKAGEDIR, create_target_list, fill_tics, get_logger
from tessproposaltool.tpt import _add_xmatch_column, _parse_dataframe

logger = get_logger()
logger.setLevel("DEBUG")

testdir = "/".join([*PACKAGEDIR.split("/")[:-2], "tests", "data"]) + "/"


def test_parse():
    df = pd.read_csv(testdir + "test_single_radec.csv")
    new_df = _parse_dataframe(df)
    # we do not require or expect a pmra/pmdec to be supplied
    assert np.in1d(
        new_df[["tic", "ra", "dec", "tmag"]].columns, ["tic", "ra", "dec", "tmag"]
    ).all()
    assert new_df.loc[0, "ra"] == 120


def test_fill():
    df = pd.read_csv(testdir + "test_single_radec.csv")
    new_df = fill_tics(df)
    assert new_df.loc[0, "tic"] == 252969141.0
    df.loc[0, "tmag"] = 18.0
    new_df = fill_tics(df)
    assert new_df.loc[0, "tic"] == 802056325.0


def test_fill_larger():
    df = pd.read_csv(testdir + "test_radec_long.csv")
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
    files = ["test_radec_only.csv", "test_tic_only.csv", "test_mix_radectic.csv"]

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
    testfile = testdir + "test_optional_column.csv"

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


def test_input_edge():
    with pytest.raises(ValueError):
        _parse_dataframe(testdir + "test_no_header.csv")
    with pytest.raises(UserWarning):
        _parse_dataframe(testdir + "test_empty.csv")
    nonsense = create_target_list(testdir + "test_nonsense_col.csv")
    assert "blah" not in nonsense.columns


def test_LargeSep_Warning():
    import io
    from contextlib import redirect_stdout

    from tessproposaltool import OUTPUT_COLUMNS, TIC_COLUMNS

    df = pd.read_csv(testdir + "test_mix_radectic.csv")
    new_df = create_target_list(df)
    tic_df = new_df[OUTPUT_COLUMNS]

    tic_df = tic_df.rename(columns=dict(zip(OUTPUT_COLUMNS, TIC_COLUMNS)))
    tic_df["sep"] = [3, 3, 0.5, 0.5, 0.5, 0.5]
    tic_df["weight"] = [0.1] * 6
    tic_df["mweight"] = [0.1] * 6

    f = io.StringIO()
    with redirect_stdout(f):
        test_df = _add_xmatch_column(tic_df)
    s = f.getvalue()

    assert "WARNING" in test_df.loc[1, "xmatch"]
    assert str(tic_df.loc[1, "TIC"]) in s
    assert "WARNING" not in test_df.loc[3, "xmatch"]
    assert "Crossmatch Parameters: " in new_df.loc[1, "remarks"]
