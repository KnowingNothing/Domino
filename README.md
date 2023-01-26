# Domino

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
source setup-env.sh
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
tflite
onnx
networkx
pandas
```

others:

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
