from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from torch.utils.data import Dataset
import numpy as np

class GCNDataset(Dataset):  
    def __init__(self, args, ft_dict_list, label, adj_dict_list=None,weight=None): 
        super(GCNDataset, self).__init__()
        self.ft_dict_list = ft_dict_list  
        self.adj_dict_list=adj_dict_list
        self.label = label 
        self.args=args
        self.weight=weight
  
    def __len__(self):  
        return len(self.label)  
  
    def __getitem__(self, idx):  
        #返回单个样本和对应的标签 
        if self.weight is None:
            return self.ft_dict_list[idx],self.label[idx]
        else:
            tf=self.ft_dict_list[idx]['tf'].reshape(-1)
            gene=self.ft_dict_list[idx]['gene'].reshape(-1)
            y_true=torch.cat((tf,gene),0)
            weight=self.weight[idx]
            return self.ft_dict_list[idx], self.adj_dict_list[idx],y_true,weight
