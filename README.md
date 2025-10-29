# coreVIEWERcsvReader

A small Python utility to read and prepare Core-Viewer CSV files for validation, filtering, and further processing. The project provides a simple CLI entry point and an importable library module.

## Features

- Read Core-Viewer CSV files and return a Pandas DataFrame

## Requirements

- Python 3.13 (the project was developed with Python 3.13)
- Optional: Poetry for dependency and environment management

## example usage
```python
import coreviewercsvreader
if __name__ == "__main__":
   dataFrame = coreviewercsvreader.read_coresensing_csv("data/1-01107_2025-10-24_16-14-56.csv")
   print(dataFrame)
```

## License

See the LICENSE file included in the repository.

