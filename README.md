# SRLST
[![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org/)
(https://github.com/lanbiolab/SRLST/edit/master/README.md)
## Spatially Representation Learning of Spatial Transcriptomics With SRLST

## (1)Installation
This project is built on [Python3](https://www.python.org/).
   - Prerequisites: \
       [Python3.*](https://www.python.org/) (version>=3.10)
   - Direct dependencies: \
       [numpy](http://www.numpy.org/) \
       [scikit-learn](https://scikit-learn.org/stable/) \
       [PyTorch](https://pytorch.org/) (version >=1.2.0, <=1.11.0) \
       [torch_scatter](https://pytorch-geometric.com/whl/) \
       [torch_sparse](https://pytorch-geometric.com/whl/) \
       [scipy](https://scipy.org/) \
       [pandas](https://pandas.pydata.org/) \
       [matplotlib](https://pypi.org/project/matplotlib/)\
       [numpy](https://pypi.org/project/numpy/)\
       [python-igraph](https://pypi.org/project/python-igraph/)\
       [krb5](https://web.mit.edu/kerberos/)\
       [louvain](https://pypi.org/project/louvain/)\
       [torch-geometric](https://pypi.org/project/torch-geometric/)\
       [leidenalg](https://pypi.org/project/leidenalg/)\
       [scikit-misc](https://pypi.org/project/scikit-misc/)\
       [rpy2](https://pypi.org/project/rpy2/)\
       [opencv-python](https://pypi.org/project/opencv-python/)
## (2)Create an environment
We highly recommend using a virtual environment for the installation of SRLST and its dependencies. A virtual environment can be created and (de)activated as follows using [conda](https://conda.io/docs/):
```bash
# create
conda env create --name srlst -f /path/to/SRLST/environment.yml

# activate
conda activate srlst

# deactivate
conda deactivate
```
## (3) Install pytorch, torch_scatter and torch_sparse
After the environment being created and activated,[PyTorch](https://pytorch.org/) can be automatically installed during the installation. However, if the version of [PyTorch](https://pytorch.org/) installed is not appropriate for your OS, an appropriate version should be re-installed in the same environment as the [instructions](https://pytorch.org/get-started/locally/):
```
#install torch using pip 
bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

download the '.whl' file of torch_scatter and torch_sparse from [pytorch-geometric](https://pytorch-geometric.com/whl/).
#install torch_scatter and torch_sparse using pip 
bash
pip install torch_scatter-2.1.2+pt27cpu-cp311-cp311-win_amd64.whl
pip install torch_sparse-0.6.18+pt27cpu-cp311-cp311-win_amd64.whl
```
## (4)Install R
[R](https://cran.r-project.org/) is required to be installed, and installation address is also required to be recorded.

## (5)Example data

   * _[data.zip](https://zenodo.org/records/15624210)_: including DLPFC,fish,Human_Breast_Cancer,Liver_Cell_Human.After decompression, place the "data" directory under the main directory of the code

## (6)Train models
A new model can be trained as follows:
```bash
python DLPFC_main.py /R/R-4.5.0
python fish.py /R/R-4.5.0
python Human_Breast_Cancer.py /R/R-4.5.0
python Liver_Cell_Human.py /R/R-4.5.0
```
