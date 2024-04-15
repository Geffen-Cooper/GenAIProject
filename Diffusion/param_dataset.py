import torch
from torch.utils.data import Dataset,DataLoader
import os


class ParamDataset(Dataset):
    def __init__(self, root_dir, conditional=False, corr_list=None):
        self.root_dir = root_dir
        self.conditional = conditional

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