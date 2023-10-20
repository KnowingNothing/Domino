# Domino

## Prerequisite
1. Python >= 3.8
2. CMake >= 3.12

## Download
```sh
git clone git@github.com:KnowingNothing/Domino.git
cd Domino
git submodule update --init --recursive
```

## Build & Test C++ library
Build up dominoc
```shell
$ mkdir build && cd build
$ cmake ..
$ make
$ ctest
```
## Install python dependencies
```sh
python3 -m pip install -r requirements.txt
```

## Setup python environment

Setup python environment for domino, dominoc, mculib (`testing/mculib`), tileflow (`testing/tileflow`).
```sh
# recommended
cd python
python3 setup.py develop
```
or
```sh
cd python
python3 setup.py install --user
```
or
```sh
source set-env.sh
```
## Other dependencies (deprecated)

```
accelergy
timeloop
```
refer to https://timeloop.csail.mit.edu/timeloop/installation

we prepare download script in `download_utils`

```
maestro
```

we prepare download script in `download_utils`

## Run dataflow comparison experiment

The experiment folder is `testing/tileflow/test/experiments`, please check the `readme` file under the folder to reproduce the experiment.
