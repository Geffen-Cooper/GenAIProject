import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import os

import sys
sys.path.append("../")
sys.path.append("../finetuning")
from finetuning.datasets import *
from finetuning.train import validate
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model
from pathlib import Path

class ParamDataset(Dataset):
    def __init__(self, root_dir, conditional=False, corr_list=None,embd_cond=False):
        self.root_dir = root_dir
        self.conditional = conditional
        self.embd_cond = embd_cond

        # get the filenames and labels
        self.files = []
        self.labels = []
        for root, _, filenames in os.walk(root_dir):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                self.files.append(full_path)
                self.labels.append(os.path.basename(os.path.dirname(full_path)))

        unique_labels = set(self.labels)
        self.label_to_int = {label_str : i for i,label_str in enumerate(unique_labels)}
        self.int_to_label = {val : key for key,val in self.label_to_int.items()}
        
        if corr_list is not None:
            self.label_to_int = {corr : i for i,corr in enumerate(corr_list)}
            self.int_to_label = {val : key for key,val in self.label_to_int.items()}

        if embd_cond == True:
            root_dir = os.path.join(root_dir,'../../gen_embds')
            self.dom_labels = []
            self.embds = {corr: None for corr in corr_list}
            for root, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    dom_name = os.path.basename(os.path.dirname(full_path))
                    self.embds[dom_name] = torch.load(full_path).cpu()
                    self.dom_labels.append(dom_name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # load params
        state_dict = torch.load(self.files[index])['model_state_dict']
        params = torch.cat([param.flatten() for param in state_dict.values()])
        params = params.reshape((8,160))
        
        # load label
        label = self.labels[index]

        if self.conditional == True:
            return params,self.label_to_int[label]
        elif self.embd_cond == True:
            random_index = torch.randint(0, self.embds[label].size(0), (1,)).item()
            # print(f'------error: {random_index},{label}')
            # print(params, self.embds[label][random_index])
            # exit()
            return params, self.embds[label][random_index]
        else:
            return params
    
# put back into statedict
def put_back_params(model,params):
    params = params.view(-1)
    with torch.no_grad():
        weights = params[:len(params)//2]
        biases = params[len(params)//2:]
        model.bn1.weight[:] = weights[:]
        model.bn1.bias[:] = biases[:]
        return model
    

# Define a forward hook function to extract activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = F.avg_pool2d(input[0].detach(), 8).view(-1,640)
    return hook

# generate embeddings for fine-tuned models
def gen_embeds(path_to_saved_data,corruption,og_cifar_model):
    # get the corresponding cifar data
    if corruption == 'clean':
        x_clean,y_clean = load_cifar10(1000)
        train_ds = TensorDataset(x_clean, y_clean)
    else:
        train_ds, val_ds, test_ds = get_cifar10c_data([corruption],1000)
    dl = DataLoader(train_ds,1000)

    with torch.no_grad():
        # add a hook to the model to get the activations right before bn
        og_cifar_model.bn1.register_forward_hook(get_activation('bn1'))

        # then do a forward pass to collect these embeddings
        x,y = next(iter(dl))
        out = og_cifar_model(x.to('cuda'))
        input_activations = activations['bn1']

    # save the embeddings under the proper folder
    print(input_activations.shape)

    Path(os.path.join(path_to_saved_data,corruption)).mkdir(parents=True, exist_ok=True)
    torch.save(input_activations,os.path.join(path_to_saved_data,corruption,'embeds.pt'))

if __name__ == '__main__':
    test_model = load_model(model_name='Standard',dataset='cifar10',threat_model='corruptions').eval().to('cuda')
    root_path = os.path.expanduser(f"~/Projects/GenAIProject/saved_data/gen_embds")
    for corr in ["clean","frost", "gaussian_noise", "glass_blur", "contrast", "pixelate"]:
        gen_embeds(root_path,corr,test_model.eval())
    