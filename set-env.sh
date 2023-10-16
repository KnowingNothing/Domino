export PYTHONPATH=$(pwd)/python:$(pwd)/build:$PYTHONPATH

pushd .
cd testing/mculib
source set-env.sh
popd 

pushd .
cd testing/tileflow
source set-env.sh
popd
