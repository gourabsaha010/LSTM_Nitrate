### This code contains deep learning code used to model hydrologic and earth systems including anthropogenic activities to stream nitrate variability estimation and projection.

# LSTM_Nitrate modeling

This is the hydroDL model, which is an LSTM model used for daily stream nitrate (NO3) concentration prediction modeling. The model can be run using a main script named 'StreamTemp_integ.py.' The hydroDL model is available at 'https://github.com/mhpi/hydroDL/tree/release.' README file in the hydroDL package has details on installtion, python package requirements, and quick start guide (https://github.com/mhpi/hydroDL/blob/release/README.md).

# GPU requirement

The LSTM_Nitrate model requires graphics processing unit (GPU) to run the model. 

# Work Instruction to simulate LSTM_nitrate

1. Under ‘LSTM_Nitrate’ folder, ‘LSTM_temp2’ folder has the hydroDL model (LSTM model).
   
2. Script ‘StreamTemp-Integ.py’ should be used to simulate the model. forcing_list, attr_list, Batch_list, Hidden_list, RHO_list, Randomseed, Action, EPOCH, target variable, and duration of simulation need to be selected and changed based on input data for model training.

3. The entire LSTM model is in ‘hydroDL’ folder.

4. ‘scratch’ folder has a subfolder ‘SNTemp’, which contains another folder called ‘Forcing’. ‘Forcing’ folder has two subfolders named ‘attr_new’ and ‘Forcing_new’.  ‘attr_new’ should contains attribute data files and ‘Forcing_new’ folder should have forcing/time series data files. While simulating the model a .json file will generate automatically named ‘Statistics_basinnorm.json’ under ‘SNTemp’ folder. ‘Statistics_basinnorm.json’ contains the normalized values of input training data of each forcing for each site.
   
5. The simulated output files will be saved under ‘TempDemo/FirstRun’ folder.

# Packages required in the virtual environment (to simulate the LSTM model appropriately)

# Name                    Version          Build  Channel
argon2-cffi               20.1.0           py37h2bbff1b_1

arrow-cpp                 3.0.0            py37h0d1d0e5_0

async_generator           1.10             py37h28b3542_0

attrs                     20.3.0             pyhd3eb1b0_0
backcall                  0.2.0              pyhd3eb1b0_0
basemap                   1.2.2                    pypi_0    pypi
blas                      1.0                         mkl
bleach                    3.3.0              pyhd3eb1b0_0
boost-cpp                 1.73.0              h2bbff1b_11
brotli                    1.0.9                ha925a31_2
bzip2                     1.0.8                he774522_0
c-ares                    1.17.1               h2bbff1b_0
ca-certificates           2021.4.13            haa95532_1
certifi                   2020.12.5        py37haa95532_0
cffi                      1.14.5           py37hcd4344a_0
colorama                  0.4.4              pyhd3eb1b0_0
cudatoolkit               10.0.130                      0
cycler                    0.10.0                   py37_0
decorator                 5.0.6              pyhd3eb1b0_0
defusedxml                0.7.1              pyhd3eb1b0_0
double-conversion         3.1.5                ha925a31_1
entrypoints               0.3                      py37_0
et_xmlfile                1.0.1                   py_1001
freetype                  2.10.4               hd328e21_0
gflags                    2.2.2                ha925a31_0
glog                      0.4.0                h33f27b4_0
grpc-cpp                  1.26.0               h351948d_0
icc_rt                    2019.0.0             h0cc432a_1
icu                       58.2                 ha925a31_3
importlib-metadata        3.10.0           py37haa95532_0
importlib_metadata        3.10.0               hd3eb1b0_0
intel-openmp              2020.2                      254
ipykernel                 5.3.4            py37h5ca1d4c_0
ipython                   7.22.0           py37hd4e2768_0
ipython_genutils          0.2.0              pyhd3eb1b0_1
jdcal                     1.4.1                      py_0
jedi                      0.17.0                   py37_0
jinja2                    2.11.3             pyhd3eb1b0_0
jpeg                      9b                   hb83a4c4_2
jsonschema                3.2.0                      py_2
jupyter_client            6.1.12             pyhd3eb1b0_0
jupyter_core              4.7.1            py37haa95532_0
jupyterlab_pygments       0.1.2                      py_0
kiwisolver                1.3.1            py37hd77b12b_0
libboost                  1.73.0              h6c2663c_11
libpng                    1.6.37               h2a8f88b_0
libprotobuf               3.11.2               h7bd577a_0
libsodium                 1.0.18               h62dcd97_0
libtiff                   4.1.0                h56a325e_1
lz4-c                     1.9.3                h2bbff1b_0
m2w64-gcc-libgfortran     5.3.0                         6
m2w64-gcc-libs            5.3.0                         7
m2w64-gcc-libs-core       5.3.0                         7
m2w64-gmp                 6.1.0                         2
m2w64-libwinpthread-git   5.0.0.4634.697f757               2
markupsafe                1.1.1            py37hfa6e2cd_1
matplotlib                3.2.2                         0


matplotlib-base           3.2.2            py37h64f37c6_0
mistune                   0.8.4           py37hfa6e2cd_1001
mkl                       2020.2                      256
mkl-service               2.3.0            py37h196d8e1_0
mkl_fft                   1.3.0            py37h46781fe_0
mkl_random                1.1.1            py37h47e9c7a_0
msys2-conda-epoch         20160418                      1
nbclient                  0.5.3              pyhd3eb1b0_0
nbconvert                 6.0.7                    py37_0
nbformat                  5.1.3              pyhd3eb1b0_0
nest-asyncio              1.5.1              pyhd3eb1b0_0
ninja                     1.10.2           py37h6d14046_0
notebook                  6.3.0            py37haa95532_0
numpy                     1.19.2           py37hadc3359_0
numpy-base                1.19.2           py37ha3acd2a_0
olefile                   0.46                     py37_0
openpyxl                  3.0.7              pyhd3eb1b0_0
openssl                   1.1.1k               h2bbff1b_0
packaging                 20.9               pyhd3eb1b0_0
pandas                    1.2.3            py37hf11a4ad_0
pandoc                    2.12                 haa95532_0
pandocfilters             1.4.3            py37haa95532_1
parso                     0.8.2              pyhd3eb1b0_0
patsy                     0.5.1                      py_0    conda-forge
pickleshare               0.7.5           pyhd3eb1b0_1003
pillow                    8.2.0            py37h4fa10fc_0
pip                       21.0.1           py37haa95532_0
prometheus_client         0.10.1             pyhd3eb1b0_0
prompt-toolkit            3.0.17             pyh06a4308_0
pyarrow                   3.0.0                    pypi_0    pypi
pycparser                 2.20                       py_2
pygments                  2.8.1              pyhd3eb1b0_0
pyparsing                 2.4.7              pyhd3eb1b0_0
pyproj                    3.0.1                    pypi_0    pypi
pyqt                      5.9.2            py37h6538335_2
pyrsistent                0.17.3           py37he774522_0
pyshp                     2.1.3                    pypi_0    pypi
python                    3.7.10               h6244533_0
python-dateutil           2.8.1              pyhd3eb1b0_0
python_abi                3.7                     1_cp37m    conda-forge
pytorch                   1.2.0           py3.7_cuda100_cudnn7_1    pytorch
pytz                      2021.1             pyhd3eb1b0_0
pywin32                   227              py37he774522_1
pywinpty                  0.5.7                    py37_0
pyzmq                     20.0.0           py37hd77b12b_1
qt                        5.9.7            vc14h73c81de_0
re2                       2020.11.01           hd77b12b_1
scipy                     1.6.2            py37h14eb087_0
send2trash                1.5.0              pyhd3eb1b0_1
setuptools                52.0.0           py37haa95532_0
sip                       4.19.8           py37h6538335_0
six                       1.15.0           py37haa95532_0
snappy                    1.1.8                h33f27b4_0
sqlite                    3.35.4               h2bbff1b_0
statsmodels               0.12.2           py37hda49f71_0    conda-forge
terminado                 0.9.4            py37haa95532_0
testpath                  0.4.4              pyhd3eb1b0_0
tk                        8.6.10               he774522_0
torchvision               0.4.0                py37_cu100    pytorch
tornado                   6.1              py37h2bbff1b_0
traitlets                 5.0.5              pyhd3eb1b0_0
typing_extensions         3.7.4.3            pyha847dfd_0
uriparser                 0.9.3                h33f27b4_1
utf8proc                  2.6.1                h2bbff1b_0
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wcwidth                   0.2.5                      py_0
webencodings              0.5.1                    py37_1
wheel                     0.36.2             pyhd3eb1b0_0
wincertstore              0.2                      py37_0
winpty                    0.4.3                         4
xlrd                      2.0.1              pyhd3eb1b0_0
xz                        5.2.5                h62dcd97_0
zeromq                    4.3.3                ha925a31_3
zipp                      3.4.1              pyhd3eb1b0_0
zlib                      1.2.11               h62dcd97_4
zstd                      1.4.5                h04227a9_0

# Citation for LSTM_Nitrate

Saha, G.K., Rahmani, F., Shen, C., Li, L., Raj, C., 2023. A Deep Learning-based Novel Approach to Generate Continuous Daily Stream Nitrate Concentration for Nitrate Data-sparse Watersheds. Science of The Total Environment 162930. https://doi.org/10.1016/j.scitotenv.2023.162930

# Citations for hydroDL

Feng, DP., Lawson, K., and CP. Shen, Mitigating prediction error of deep learning streamflow models in large data-sparse regions with ensemble modeling and soft data, Geophysical Research Letters (2021), https://doi.org/10.1029/2021GL092999

Feng, DP, K. Fang and CP. Shen, Enhancing streamflow forecast and extracting insights using continental-scale long-short term memory networks with data integration, Water Resources Research (2020), https://doi.org/10.1029/2019WR026793


The file is under construction ...

The content will be available soon. 
