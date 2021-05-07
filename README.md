# cytoxnet ![ci](https://github.com/Zymergen-DIRECT-2021/cytoxnet/actions/workflows/ci.yml/badge.svg?branch=main) [![codecov](https://codecov.io/gh/Zymergen-DIRECT-2021/cytoxnet/branch/main/graph/badge.svg?token=IVAS3MWX5B)](https://codecov.io/gh/Zymergen-DIRECT-2021/cytoxnet)

Toolbox for the machine learning prediction of microbe cytotoxicity.

## installation
Create and activate conda environment, then run the package setup from the repository root:
```
conda env create --file environment.yml
conda activate cytoxnet
pip install tensorflow~=2.4
pip install .
```
The package can then be imported as `import cytoxnet`
