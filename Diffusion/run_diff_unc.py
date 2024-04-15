import torch
import torch.nn as nn

from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from param_dataset import *
import sys
sys.path.append("../")
sys.path.append("../finetuning")
from finetuning.datasets import *
from finetuning.train import validate
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model


corrs = ["frost", "gaussian_noise", "glass_blur", "contrast", "pixelate"]

for corr in corrs:
    model = Unet1D(
        dim = 16,
        dim_mults = (1, 2, 2, 2),
        channels = 32
    )
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000)
    # exit()

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 40,
        timesteps = 1000,
        objective = 'pred_noise',
        auto_normalize=False
    )

    root_path = os.path.expanduser(f"~/Projects/GenAIProject/saved_data/checkpoints/gen_last_bn_eval_diverse/{corr}")
    dataset = ParamDataset(root_path)

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 64,
        train_lr = 4e-3,
        train_num_steps = 5000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        corr=corr,
        results_folder=f'{corr}_results_40/'
    )
    trainer.train()

# # after a lot of training

# sampled_seq = diffusion.sample(batch_size = 4)
# sampled_seq.shape # (4, 32, 128)
