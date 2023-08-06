# Domino

## Prerequisite
1. Python >= 3.8
2. CMake >= 3.12

## Download
```sh
git clone git@github.com:KnowingNothing/Domino.git
git submodule update --init --recursive
```

## Build & Test

```shell
$ mkdir build && cd build
$ cmake ..
$ make
$ ctest
```

## Setup environment
The environment for Python3
```sh
source set-env.sh
```
or
```sh
cd python
python setup.py develop --no-deps
```
or
```sh
cd python
python setup.py install --user
```

## Python part dependencies

```
pip install -r requirements.txt
```

others (deprecated):

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
