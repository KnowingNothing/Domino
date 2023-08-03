import argparse
import multiprocessing
import os
import sys
import pandas as pd

__METRICS__ = ["Cycle", "Energy", "MEM::L1", "MEM::L2", "MEM::L0"]

def kernel(wkl, args):
    name, script, dataflow, layout = args
    number = 5 if wkl == 'conv' else 9
    logfile = f'logs/{wkl}/{name}.log'
    cmd = f'python {script} --number {number}  --trials 1000 --define_tiling_space > {logfile}'
    if dataflow is not None:
        cmd += f' --dataflow {dataflow}'
    if layout is not None:
        cmd += f' --layout {layout}'
    print (cmd)
    
    if not os.path.exists(logfile):
        err = os.system(cmd)
        if err: return 
    
    for hw_id in [0,1]:
        for metric in __METRICS__:
            output_file = f'outs/{wkl}/{name}-{hw_id}-{metric}.csv'
            cmd = f'python metric_analysis.py --wkl {wkl} --file {logfile} --hw_id {hw_id} --metric {metric} > {output_file}'
            print(cmd)
            err = os.system(cmd)
            if err: return 

def run(wkl, dataflows, shapes):
    os.system(f'mkdir -p logs/{wkl}')
    os.system(f'mkdir -p outs/{wkl}')
    os.system(f'mkdir -p pics/{wkl}')
    
    if not os.path.exists(f'csvs/{wkl}.csv'):
        procs = []
        
        for args in dataflows:
            name, _, _, _ = args
            # if not os.path.exists(f'logs/{name}.log'):
            proc = multiprocessing.Process(target = kernel, args = (wkl, args,))
            proc.start()
            procs.append(proc)
                
        for proc in procs: proc.join()
        
        values = []
        keys = []
        for name, _, _, _ in dataflows:
            for hw_id in [0,1]:
                for metric in __METRICS__:
                    if not os.path.exists(f'outs/{wkl}/{name}-{hw_id}-{metric}.csv'):
                        print (f'[ERROR] running {name} {hw_id} {metric}', file=sys.stderr)
                        continue
                    values.append(pd.read_csv(f'outs/{wkl}/{name}-{hw_id}-{metric}.csv'))
                    keys.append((name, hw_id, metric))
        df = pd.concat(values, keys = keys, names = ['Dataflow', 'Architecture', 'metric'], sort=False)
        if wkl == 'self_attention':
            df['Shape'] = df.apply(lambda row: shapes[(row['num_heads'],row['seq_len'],row['hidden'])], axis = 1)
        else: 
            df['Shape'] = df.apply(lambda row: shapes[(row['in_channel'],row['height'],row['width'],row['out_channel_1'],row['out_channel_2'])], axis = 1)
        df.to_csv(f'{wkl}.csv')
    
    df = pd.read_csv(f'csvs/{wkl}.csv')
    
    for hw_id, hw_name in enumerate(['Edge', 'Cloud']):
        for metric in __METRICS__:
            ddff = df[df['Architecture'] == hw_id]
            ddff = ddff[ddff['metric'] == metric][['Dataflow', metric, 'Shape']]
            ddff = ddff.set_index(['Shape', 'Dataflow'])[metric].unstack()
            print (ddff)
            ax = ddff.plot.bar(rot = 45)
            ax.set_title(f'{wkl}-{hw_name}-{metric}')
            ddff.to_csv(f'{wkl}-{hw_name}-{metric}.csv')
            ax.get_figure().savefig(f'pics/{wkl}/' + hw_name + f'-{metric}' + '.png', bbox_inches = 'tight')


params = {
    'self_attention': {
        'dataflows': [
        ('Naive', 'no_fuse_self_attention.py', None, None),
        ('FLAT-HGran', 'flat_dataflow.py', 'hgran', None),
        ('FLAT-RGran', 'flat_dataflow.py', 'rgran', None),
        ('Chimera', 'chimera_self_attention.py', None, None),
        ('TileFlow', 'tileflow_self_attention.py', None, None)
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
        }
    },
    'conv': {
        'dataflows': [
            ('Naive', 'no_fuse_conv_chain.py', None, 'nhwc'),
            ('Fused-Layer', 'fused_layer_dataflow.py', None, 'nhwc'),
            ('ISOS', 'isos_dataflow.py', None, 'nhwc'),
            ('TileFlow', 'tileflow_conv.py', None, 'nhwc')
        ],
        'shapes': {
            # (in_channel, height, width, out_channel_1, out_channel_2)
            (64, 112, 112, 192, 128): 'Yolo',
            (32, 147, 147, 64, 80): 'Inception-V3',
            (64, 56, 56, 128, 64): 'Darknet-19-0',
            (128, 28, 28, 256, 128): 'Darknet-19-1',
            (16, 227, 227, 64, 16): 'Squeezenet-V1.1'
        }
    }
}

def main(workload):
    os.system('mkdir -p logs')
    os.system('mkdir -p outs')
    os.system('mkdir -p pics')
    os.system('mkdir -p csvs')
    run(workload, params[workload]['dataflows'], params[workload]['shapes'])

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--wkl', type = str, choices = ['self_attention', 'conv'], default = 'self_attention')  
    # args = parser.parse_args()
    main('self_attention')
    main('conv')