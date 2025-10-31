# coreVIEWERcsvReader

- A Utility to convert coreVIEWER csv files to Pandas DataFrames.
- pip: https://pypi.org/project/coreviewercsvreader/

## Features

- Read Core-Viewer CSV files and return a Pandas DataFrame

## Requirements

- Python 3.9 to 3.13 tested 
- Poetry for dependency and environment management

## example usage
```python
import coreviewercsvreader
if __name__ == "__main__":
   dataFrame = coreviewercsvreader.read_coresensing_csv("data/1-01107_2025-10-24_16-14-56.csv")
   print(dataFrame)
```


## License

See the LICENSE file included in the repository.

