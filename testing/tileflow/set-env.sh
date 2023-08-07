X=$(which tileflow)
if [ $? != 0 ]; then 
	echo "No tileflow in path.";
	return 1;
fi
export TILEFLOW_BIN_PATH=$X
export TILEFLOW_PYTHONPATH=$(pwd)/python
export PYTHONPATH=$TILEFLOW_PYTHONPATH:$PYTHONPATH
