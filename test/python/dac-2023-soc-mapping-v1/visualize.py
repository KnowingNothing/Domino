import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd

#visualize proportions
def visualize_pe():
    os.system("mkdir -p pics_pe")
    for model in ['vision1', 'vision2', 'mixed1', 'mixed2', 'mixed3']:
        for SoC in ['SmallSoC', 'LargeSoC']:
            for bw in ['lowBW', 'highBW']:
                data = {}
                for alg in ['COMB', 'H2H', 'MAGMA']:
                    with open(f'pe_curve/{alg}_{model}_{SoC}_{bw}.pkl', 'rb') as f:
                        data[alg] = pkl.load(f)
                index = pd.MultiIndex.from_tuples([(j,i,k) for i in data for j in data[i] for k in data[i][j]])
                df = pd.DataFrame([data[i][j][k] for j,i,k in index], index = index).fillna(0).unstack()[0]
                # df['compute_proportion'] = df['compute_amount'] / df['amount']
                ax = df['occupancy'].unstack().plot(kind='bar')
                plt.savefig(f'pics_pe/{model}_{SoC}_{bw}.pdf', bbox_inches='tight')

visualize_pe()
