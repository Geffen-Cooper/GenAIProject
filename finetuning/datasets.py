from robustbench.data import load_cifar10c
from robustbench.data import load_cifar10
from torch.utils.data import TensorDataset

import numpy as np
import os


CORRUPTIONS = ["gaussian_noise","shot_noise","impulse_noise", 
               "defocus_blur","glass_blur","motion_blur","zoom_blur",
               "snow","frost", "fog",
               "brightness", "contrast","elastic_transform","pixelate","jpeg_compression"]
imgs_per_corr = 10000


def get_cifar10c_data(corruptions,train_size):
    # load relevant corruption data
    x_corr,y_corr = load_cifar10c(imgs_per_corr*len(corruptions),5,corruptions=corruptions,data_dir=os.path.expanduser('~/Projects/data/'))

    labels = {}
    num_classes = int(max(y_corr)) + 1
    for i in range(num_classes):
        labels[i] = (y_corr==i).nonzero().view(-1)
    num_ex = train_size // num_classes
    tr_idxs = []
    val_idxs = []
    test_idxs = []
    for i in range(len(labels.keys())):
        np.random.shuffle(labels[i])
        tr_idxs.append(labels[i][:num_ex])
        val_idxs.append(labels[i][num_ex:num_ex+int(0.2*num_ex)])
        test_idxs.append(labels[i][num_ex+int(0.2*num_ex):])
    tr_idxs = np.concatenate(tr_idxs)
    val_idxs = np.concatenate(val_idxs)
    test_idxs = np.concatenate(test_idxs)
    
    tr_dataset = TensorDataset(x_corr[tr_idxs], y_corr[tr_idxs])
    val_dataset = TensorDataset(x_corr[val_idxs], y_corr[val_idxs])
    te_dataset = TensorDataset(x_corr[test_idxs], y_corr[test_idxs])

    return tr_dataset, val_dataset, te_dataset