import os
import json
import sys
sys.path.append("../")
from utils.setup_funcs import PROJECT_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robustbench.utils import clean_accuracy
from robustbench.utils import load_model
from datasets import *


root_path = os.path.join(PROJECT_ROOT,"saved_data/metrics/gen_last_bn_eval_diverse")

corruption_list = ["clean","frost", "gaussian_noise", "glass_blur", "contrast", "pixelate"]
seeds = 10
num_vals = seeds

results_dict = {corr : np.zeros(num_vals) for corr in corruption_list}

for corr_i, corr in enumerate(corruption_list):
    # get data for specific corruption
    if corr == 'clean':
        test_model = load_model(model_name='Standard',dataset='cifar10',threat_model='corruptions').eval().cuda()
        x_clean, y_clean = x_clean,y_clean = load_cifar10(1000)
        for seed_i in range(1,seeds+1):
            results_dict[corr][seed_i-1] = clean_accuracy(test_model.to('cuda'),x_clean,y_clean,256,'cuda')
    else:
        paths = os.listdir(os.path.join(root_path,corr))
        for seed_i,path in zip(range(1,seeds+1),paths):
            p = os.path.join(root_path,corr,path)
            with open(p) as f:
                # Load the JSON data
                metrics = json.load(f)
                results_dict[corr][seed_i-1] = metrics['test_acc']

cs = ['gray','r','orange','green', 'blue', 'brown']
fig,ax = plt.subplots(1,1,figsize=(6,5))
test_accs = pd.read_csv(os.path.join(PROJECT_ROOT,f"saved_data/cond_diff_results_embd/acc"),index_col=False)
# print(test_accs)
for corr_i, corr in enumerate(corruption_list):
    ax.axhline(results_dict[corr].mean(),label=corr,c=cs[corr_i],linestyle='--')
    try:
        test_acc = test_accs[corr].values
        # test_acc = [float(s.split('[')[1].split(']')[0]) for s in test_acc]
        plt.plot(test_acc,c=cs[corr_i])
    except:
        continue

ax.set_xlabel('Training Iteration (x500)',fontsize=14)
ax.set_ylabel('avg. test accuracy',fontsize=14)
ax.grid()
ax.legend(fontsize=12)
ax.set_ylim((0,1))
ax.set_xlim((0,10))
plt.tick_params(axis='both', which='major', labelsize=14)

plt.savefig(f'cond_diff_embd2.pdf')