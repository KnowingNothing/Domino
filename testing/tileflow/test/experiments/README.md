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

3. Build TileFlow [C++ repo](https://github.com/pku-liang/TileFlow). Make sure `tileflow`'s binary is under your system path.

4. Change directory to TileFlow Python Interface and setup the environment. Note that it is important to setup the environment in `tileflow`` directory.
```sh
cd testing/tileflow
source set-env.sh
```

### Reproducing Experiments
- Experiment on self_attention block/CNN Network
    - Description: this folder contains the source code to run the experiments described in Figure 9 and Figure 10, which compares different dataflow designs on different hardware platforms using for the self-attention and CNN workloads. Every dataflow used in experiment is tuned by TileFlow's searching algorithm.
    - Running:
        ```sh
        python script.py --all
        ```
    - Sample outputs are included in `sample_output`, corresponding to experiments in Fig.9 and Fig.10.
    - This experiment takes about 3 hours on a CPU with 112 cores. 

#### Self-attention dataflows
- layerwise dataflow: `no_fuse_self_attention.py`
    *Usage:* 
    ```sh
    python no_fuse_self_attention.py --trials 10 --define_tiling_space
    ```

- flat dataflow: `flat_dataflow.py`
    *Usage:* 
    ```sh
    python flat_dataflow.py --trials 10 --define_tiling_space --dataflow rgran
    ```

- chimera dataflow: `chimera_self_attention.py`
    *Usage:* 
    ```sh
    python chimera_self_attention.py --trials 10 --define_tiling_space
    ```

- tileflow dataflow: `tileflow_self_attention.py`
    *Usage:* 
    ```sh
    python tileflow_self_attention.py --trials 10 --define_tiling_space
    ```

#### Conv-chain dataflows
- layerwise dataflow: `no_fuse_conv_chain.py`
    *Usage:* 
    ```sh
    python no_fuse_conv_chain.py --trials 10 --define_tiling_space --layout nhwc
    ```

- fused-layer dataflow: `fused_layer_dataflow.py`
    *Usage:* 
    ```sh
    python fused_layer_dataflow.py --trials 10 --define_tiling_space --layout nhwc
    ```

- isos dataflow: `isos_dataflow.py`
    *Usage:* 
    ```sh
    python isos_conv_chain.py --trials 10 --define_tiling_space --layout nhwc
    ```

- tileflow dataflow: `tileflow_conv.py`
    *Usage:* 
    ```sh
    python tileflow_conv.py --trials 10 --define_tiling_space --layout nhwc
    ```

### Notes
- `--define_tiling_space` uses the GA search for tiling factors. If this is not specified, it will use MCTS for tiling factors.

- `from_scratch_self_attention.py` and `from_scratch_conv_chain.py` use GA search for fusing structures.