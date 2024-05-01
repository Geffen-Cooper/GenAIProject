import os
import json
import sys
sys.path.append("../")
from utils.setup_funcs import PROJECT_ROOT

import matplotlib.pyplot as plt
import numpy as np


root_path = os.path.join(PROJECT_ROOT,"saved_data/metrics")

# finetune_config_list = ["none","fc","first_conv","last_bn","last_two_bn","last_three_bn","fc_eval","last_three_bn_eval","last_bn_eval","last_two_bn_eval","first_conv_eval","gen_last_bn_eval_diverse"]
finetune_config_list = ["none","fc_eval","last_three_bn_eval","last_bn_eval","last_two_bn_eval","first_conv_eval"]
corruption_list = ["frost", "gaussian_noise", "glass_blur", "contrast", "pixelate"]
seeds = 10
num_vals = seeds*len(corruption_list)

results_dict = {config :
                    {'test_accs' : np.zeros(num_vals), 'train_times' : np.zeros(num_vals), 'num_params' : 0}
                for config in finetune_config_list}

for config_i, config in enumerate(finetune_config_list):
    for corr_i, corr in enumerate(corruption_list):
        paths = os.listdir(os.path.join(root_path,config,corr))
        for seed_i,path in zip(range(1,seeds+1),paths):
            p = os.path.join(root_path,config,corr,path)
            with open(p) as f:
                # Load the JSON data
                metrics = json.load(f)
                results_dict[config]['test_accs'][corr_i*seeds+seed_i-1] = metrics['test_acc']
                results_dict[config]['train_times'][corr_i*seeds+seed_i-1] = metrics['train_time']/4 # we run 4 in parallel
                results_dict[config]['num_params'] = metrics['num_params']

fig,ax = plt.subplots(1,1,figsize=(6,5))
for config_i, config in enumerate(finetune_config_list):
    ax.errorbar(results_dict[config]['num_params'],results_dict[config]['test_accs'].mean(),yerr=results_dict[config]['test_accs'].std(),capsize=4)
    ax.scatter(results_dict[config]['num_params'],results_dict[config]['test_accs'].mean(), s=results_dict[config]['train_times'].mean()*4, alpha=0.5,label=config)
    print(results_dict[config]['train_times'].mean())
ax.set_xlabel('# of parameters',fontsize=14)
ax.set_ylabel('avg. test accuracy',fontsize=14)
ax.set_yticks(np.linspace(0,1,11),np.round(np.linspace(0,1,11),2),fontsize=14)
ax.set_xticks(np.linspace(0,7000,8),np.round(np.linspace(0,7000,8),0).astype(int),fontsize=14)
ax.grid()
ax.legend(fontsize=14)
ax.set_ylim((0,1))

plt.savefig('metrics.pdf')