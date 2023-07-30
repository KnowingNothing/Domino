python conv_dataflow_l1_bw_profile.py --dataflow no_fuse --number 1 --inference --logfile data-no-fuse-conv-chain-nchw-random-tune-latency-2023-4-24/trace-new.log --layout nchw --inference --inference_hw 0 --second_kernel_size 1 --l1_bw 1 --define_tiling_space --metric L1::SlowDown --begin 0



python conv_dataflow_l1_bw_profile.py --dataflow fused_layer --number 1 --inference --logfile data-fused-layer-nchw-random-tune-latency-2023-4-24/trace-new.log --layout nchw --inference --inference_hw 0 --second_kernel_size 1 --l1_bw 1 --define_tiling_space --metric L1::SlowDown --begin 0

python conv_dataflow_l1_bw_profile.py --dataflow isos --number 1 --inference --logfile data-isos-nchw-random-tune-latency-2023-4-24/trace-new.log --layout nchw --inference --inference_hw 0 --second_kernel_size 1 --l1_bw 1 --define_tiling_space --metric L1::SlowDown --begin 0

python conv_dataflow_l1_bw_profile.py --dataflow tileflow --number 1 --inference --logfile data-tileflow-nchw-random-tune-latency-2023-4-24/trace-new.log --layout nchw --inference --inference_hw 0 --second_kernel_size 1 --l1_bw 1 --define_tiling_space --metric L1::SlowDown --begin 0



python conv_dataflow_l2_bw_profile.py --dataflow tileflow --number 1 --inference --logfile data-tileflow-nchw-random-tune-latency-2023-4-24/trace-new.log --layout nchw --inference --inference_hw 0 --second_kernel_size 1 --l1_bw 450 --l2_bw 1 --define_tiling_space --metric L2::SlowDown --begin 0


python flat_dataflow.py --inference --define_tiling_space --number 9 --logfile data-hgran-random-tune-latency-2023-4-23/trace.log --inference_hw 0 --dataflow hgran --begin 0
python flat_dataflow.py --inference --define_tiling_space --number 9 --logfile data-rgran-random-tune-latency-2023-4-23-2/trace.log --inference_hw 0 --dataflow rgran --begin 0


python metric_analysis.py --file data-rgran-random-tune-latency-2023-4-23-2/trace.log --hw_id 0 --metric Energy
python metric_analysis.py --file data-tileflow-self-attention-random-tune-latency-constraint-2023-4-23-2/trace.log --hw_id 0 --metric Cycle
python metric_analysis.py --file data-tileflow-self-attention-random-tune-latency-2023-4-23/trace.log --hw_id 0 --metric Cycle
python metric_analysis.py --file data-tileflow-conv-chain-random-tune-constraint-latency-2023-4-23//trace.log --hw_id 0 --metric Cycle