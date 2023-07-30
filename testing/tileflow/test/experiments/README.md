## Experiment Files

This directory contains the test files for TileFlow Python Interface.
The different files are used for different dataflows.

### Setup
0. Make sure you are using Python >= 3.8 (the code is only tested using Python 3.8; other versions may cause unexpected issues.)
1. Build Domino (follow the instructions of Domino's README) and install all the dependencies.
2. In the root directory of Domino, setup Domino's environment. Note that it is important to setup the environment in the root directory.
```sh
source set-env.sh
``` 
3. Change directory to TileFlow Python Interface and setup the environment. Note that it is important to setup the environment in `tileflow`` directory.
```sh
cd testing/tileflow
source set-env.sh
```
4. Build TileFlow [C++ repo](https://github.com/pku-liang/TileFlow). Make sure the TileFlow root directory is under your home directory so that we can find `~/TileFlow/build/bin`.

### Self-attention dataflows
- layerwise dataflow: `no_fuse_self_attention.py`
    *Usage:* 
    ```sh
    python no_fuse_self_attention.py --trials 1000 --define_tiling_space --logfile no_fuse_self_attention.log
    ```

- flat dataflow: `flat_dataflow.py`
    *Usage:* 
    ```sh
    python flat_dataflow.py --trials 1000 --define_tiling_space --logfile flat_self_attention.log --dataflow rgran
    ```

- chimera dataflow: `chimera_self_attention.py`
    *Usage:* 
    ```sh
    python chimera_self_attention.py --trials 1000 --define_tiling_space --logfile chimera_self_attention.log
    ```

- tileflow dataflow: `tileflow_self_attention.py`
    *Usage:* 
    ```sh
    python tileflow_self_attention.py --trials 1000 --define_tiling_space --logfile tileflow_self_attention.log
    ```

### Conv-chain dataflows
- layerwise dataflow: `no_fuse_conv_chain.py`
    *Usage:* 
    ```sh
    python no_fuse_conv_chain.py --trials 1000 --define_tiling_space --logfile no_fuse_conv_chain.log --layout nhwc
    ```

- fused-layer dataflow: `fused_layer_dataflow.py`
    *Usage:* 
    ```sh
    python fused_layer_dataflow.py --trials 1000 --define_tiling_space --logfile fused_layer_conv_chain.log --layout nhwc
    ```

- isos dataflow: `isos_dataflow.py`
    *Usage:* 
    ```sh
    python isos_conv_chain.py --trials 1000 --define_tiling_space --logfile isos_conv_chain.log --layout nhwc
    ```

- tileflow dataflow: `tileflow_convpy`
    *Usage:* 
    ```sh
    python tileflow_conv.py --trials 1000 --define_tiling_space --logfile tileflow_conv_chain.log --layout nhwc
    ```

### Notes
- `--define_tiling_space` uses the GA search for tiling factors. If this is not specified, it will use MCTS for tiling factors.

- `from_scratch_self_attention.py` and `from_scratch_conv_chain.py` use GA search for fusing structures.