import argparse
import multiprocessing
import os
import sys
import pandas as pd

__METRICS__ = ["Cycle", "Energy", "MEM::L1", "MEM::L2", "MEM::L0"]

_run_all = False

def kernel(wkl, args, params):
    name, script, dataflow = args
    number = 9 if wkl == 'self_attention' else 5
    logfile = f'logs/{wkl}/{name}.log'
    cmd = f'python {script} --number {number}  --trials 1000 --define_tiling_space > {logfile}'
    if dataflow is not None:
        cmd += f' --dataflow {dataflow}'
    cmd += ' ' + params
    
    if _run_all or not os.path.exists(logfile):
        err = os.system(cmd)
        if err: return 
    
    for hw_id in [0,1]:
        for metric in __METRICS__:
            output_file = f'outs/{wkl}/{name}-{hw_id}-{metric}.csv'
            cmd = f'python metric_analysis.py --wkl {wkl} --file {logfile} --hw_id {hw_id} --metric {metric} > {output_file}'
            err = os.system(cmd)
            if err: return 

def run(wkl, dataflows, shapes, params):
    os.system(f'mkdir -p logs/{wkl}')
    os.system(f'mkdir -p outs/{wkl}')
    os.system(f'mkdir -p pics/{wkl}')
    
    if _run_all or not os.path.exists(f'csvs/{wkl}.csv'):
        procs = []
        
        for args in dataflows:
            name, _, _ = args
            if _run_all or not os.path.exists(f'logs/{wkl}/{name}.log'):
                proc = multiprocessing.Process(target = kernel, args = (wkl, args, params))
                proc.start()
                procs.append(proc)
                
        for proc in procs: proc.join()
        
        values = []
        keys = []
        for name, _, _ in dataflows:
            for hw_id in [0,1]:
                for metric in __METRICS__:
                    if not os.path.exists(f'outs/{wkl}/{name}-{hw_id}-{metric}.csv'):
                        print (f'[ERROR] running {name} {hw_id} {metric}', file=sys.stderr)
                        continue
                    values.append(pd.read_csv(f'outs/{wkl}/{name}-{hw_id}-{metric}.csv'))
                    keys.append((name, hw_id, metric))
        print (keys)
        df = pd.concat(values, keys = keys, names = ['Dataflow', 'Architecture', 'metric'], sort=False)
        if wkl == 'self_attention':
            df['Shape'] = df.apply(lambda row: shapes[(row['num_heads'],row['seq_len'],row['hidden'])], axis = 1)
        else: 
            df['Shape'] = df.apply(lambda row: shapes[(row['in_channel'],row['height'],row['width'],row['out_channel_1'],row['out_channel_2'])], axis = 1)
        df.to_csv(f'csvs/{wkl}.csv')
    
    df = pd.read_csv(f'csvs/{wkl}.csv')
    
    for hw_id, hw_name in enumerate(['Edge', 'Cloud']):
        for metric in __METRICS__:
            ddff = df[df['Architecture'] == hw_id]
            ddff = ddff[ddff['metric'] == metric][['Dataflow', metric, 'Shape']]
            ddff = ddff.set_index(['Shape', 'Dataflow'])[metric].unstack()
            # print (ddff)
            ax = ddff.plot.bar(rot = 45)
            ax.set_title(f'{wkl}-{hw_name}-{metric}')
            ax.get_figure().savefig(f'pics/{wkl}/' + hw_name + f'-{metric}' + '.png', bbox_inches = 'tight')


_params = {
    'self_attention': {
        'dataflows': [
        ('Naive', 'no_fuse_self_attention.py', None),
        ('UniPipe', 'flat_dataflow.py', 'bgran'),
        ('FLAT-HGran', 'flat_dataflow.py', 'hgran'),
        ('FLAT-RGran', 'flat_dataflow.py', 'rgran'),
        ('Chimera', 'chimera_self_attention.py', None),
        ('TileFlow', 'tileflow_self_attention.py', None)
    ],  
        'shapes': {
        # (num_heads, seq_len, hidden)
        (8, 512, 512): 'Bert-Small',
        (12, 512, 768): 'Bert-Base',
        (16, 512, 1024): 'Bert-Large',
        (12, 256, 768): 'ViT-Base/14',
        (16, 256, 1024): 'ViT-Large/14',
        (16, 256, 1280): 'ViT-Huge/14',
        (12, 196, 768): 'ViT-Base/16',
        (16, 196, 1024): 'ViT-Large/16',
        (16, 196, 1280): 'ViT-Huge/16'
        },
        'params': ''
    },
    
    'conv1x1': {
        'dataflows': [
            ('Naive', 'no_fuse_conv_chain.py', None),
            ('Fused-Layer', 'fused_layer_dataflow.py', None),
            ('ISOS', 'isos_dataflow.py', None),
            ('TileFlow', 'tileflow_conv.py', None)
        ],
        'shapes': {
            # (in_channel, height, width, out_channel_1, out_channel_2)
            (64, 112, 112, 192, 128): 'Yolo',
            (32, 147, 147, 64, 80): 'Inception-V3',
            (64, 56, 56, 128, 64): 'Darknet-19-0',
            (128, 28, 28, 256, 128): 'Darknet-19-1',
            (16, 227, 227, 64, 16): 'Squeezenet-V1.1'
        },
        'params': '--layout nhwc',
    },
    
    'conv3x3': {
        'dataflows': [
            ('Naive', 'no_fuse_conv_chain.py', None),
            ('Fused-Layer', 'fused_layer_dataflow.py', None),
            ('ISOS', 'isos_dataflow.py', None),
            ('TileFlow', 'tileflow_conv.py', None)
        ],
        'shapes': {
            # (in_channel, height, width, out_channel_1, out_channel_2)
            (64, 112, 112, 192, 128): 'Yolo',
            (32, 147, 147, 64, 80): 'Inception-V3',
            (64, 56, 56, 128, 64): 'Darknet-19-0',
            (128, 28, 28, 256, 128): 'Darknet-19-1',
            (16, 227, 227, 64, 16): 'Squeezenet-V1.1'
        },
        'params': '--layout nhwc --second_kernel_size 3',
    }
}

def main(workload):
    os.system('mkdir -p logs')
    os.system('mkdir -p outs')
    os.system('mkdir -p pics')
    os.system('mkdir -p csvs')
    run(workload, _params[workload]['dataflows'], _params[workload]['shapes'], _params[workload]['params'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wkl', type = str, choices = ['self_attention', 'conv1x1', 'conv3x3'], 
                        nargs='+', default = ['self_attention', 'conv1x1', 'conv3x3'], 
                        )  
    parser.add_argument('--all', action = 'store_true')
    args = parser.parse_args()
    _run_all = args.all
    if os.environ.get('TILEFLOW_BIN_PATH') is None:
        print ('[ERROR]: TILEFLOW_BIN_PATH not set. Did you run set-env.sh in ../..?', file = sys.stderr)
        raise RuntimeError
    procs = []
    for wkl in args.wkl:
        proc = multiprocessing.Process(target = main, args = (wkl,))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
