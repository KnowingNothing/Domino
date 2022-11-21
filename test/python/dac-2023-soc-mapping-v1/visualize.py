import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd

#visualize proportions
def visualize_pe():
    os.system("mkdir -p pics_pe")
    for model in ['mixed3']:# ['vision1', 'vision2', 'mixed1', 'mixed2', 'mixed3']:
        for SoC in ['LargeSoC']:# ['SmallSoC', 'LargeSoC']:
            for bw in ['highBW']:# ['lowBW', 'highBW']:
                data = {}
                for alg in ['COMB', 'H2H', 'MAGMA']:
                    with open(f'pe_curve/{alg}_{model}_{SoC}_{bw}.pkl', 'rb') as f:
                        tmp = pkl.load(f)
                        for k in list(tmp.keys()):
                            if 'GemmTPU' in k:
                                tmp[k[4:]] = tmp.pop(k)
                        data[alg] = tmp 
                index = pd.MultiIndex.from_tuples([(j,i,k) for i in data for j in data[i] for k in data[i][j]])
                df = pd.DataFrame([data[i][j][k] for j,i,k in index], index = index).fillna(0).unstack()[0]
                df = df['occupancy'].unstack()
                print(df)
                df.to_csv(f'{model}_{SoC}_{bw}.csv')
                # print(df)
                # df['compute_proportion'] = df['compute_amount'] / df['amount']
                ax = df.plot(kind='bar', rot = 45)
                plt.savefig(f'pics_pe/{model}_{SoC}_{bw}.pdf', bbox_inches='tight')

visualize_pe()
