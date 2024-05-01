import os
import json
import sys
sys.path.append("../")
from utils.setup_funcs import PROJECT_ROOT

import matplotlib.pyplot as plt
import numpy as np


root_path = os.path.join(PROJECT_ROOT,"saved_data/metrics/gen_last_bn_eval_diverse")

corruption_list = ["frost", "gaussian_noise", "glass_blur", "contrast", "pixelate"]
seeds = 10
num_vals = seeds

results_dict = {corr : np.zeros(num_vals) for corr in corruption_list}

for corr_i, corr in enumerate(corruption_list):
    paths = os.listdir(os.path.join(root_path,corr))
    for seed_i,path in zip(range(1,seeds+1),paths):
        p = os.path.join(root_path,corr,path)
        with open(p) as f:
            # Load the JSON data
            metrics = json.load(f)
            results_dict[corr][seed_i-1] = metrics['test_acc']

cs = ['r','orange','green', 'blue', 'brown']
fig,ax = plt.subplots(1,1,figsize=(6,5))
for corr_i, corr in enumerate(corruption_list):
    ax.axhline(results_dict[corr].mean(),label=corr,c=cs[corr_i],linestyle='--')
    try:
        test_acc = np.load(os.path.join(PROJECT_ROOT,f"saved_data/diff_results/{corr}_torch.Size([8, 160])","acc"+corr+".npy"))
        plt.plot(test_acc,c=cs[corr_i])
    except:
        continue

ax.set_xlabel('Training Iteration (x500)',fontsize=14)
ax.set_ylabel('avg. test accuracy',fontsize=14)
ax.grid()
ax.legend(fontsize=12)
ax.set_ylim((0,0.75))
# ax.set_xlim((0,13))
plt.tick_params(axis='both', which='major', labelsize=14)

shape = "8_160"
plt.savefig(f'diff_{shape}.pdf')