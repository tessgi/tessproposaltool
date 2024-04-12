<a href="https://github.com/tessgi/tessproposaltool/actions/workflows/tests.yml"><img src="https://github.com/tessgi/tessproposaltool/workflows/pytest/badge.svg" alt="Test status"/></a> [!
[![PyPI version](https://badge.fury.io/py/tessproposaltool.svg)](https://badge.fury.io/py/tessproposaltool)

# TESS Proposal Tool

Under development tool to help create TESS proposals and target lists.

## Use Case 1: Filling in missing TICs in a target list

If you have a list of RAs, Decs, and optionally TESS magnitudes and you would like to crossmatch them against TIC, you can do this with the `tessproposaltool`. You can do this from the command line using:

```shell
tpt radec_file.csv -o output.csv
```

Which will result in a file `output.csv` that contains the corrected RA, Dec, TESS magnitude and TIC numbers.

Your file should have a structure that contains RAs, Decs, and optionally TESS magnitudes, e.g.:

```shell
ra,dec,tmag
40.2986,56.7305,9.39
110.093,-22.2673,13.51
116.243,-30.0918,9.5
152.633,-59.3549,10.05
163.437,-58.4871,9.91
165.091,-60.7688,12.974
```

To do this inside Python you can use

```python
from tessproposaltool import fill_tics

# read in your radecs, or convert them into a pandas dataframe
RA, Dec, Tmag = ..., ..., ...
df = pd.DataFrame(np.asarray([RA, Dec, Tmag]).T, columns=['ra', 'dec', 'tmag'])
new_df = fill_tics(df)
```
