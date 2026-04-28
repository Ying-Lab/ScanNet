from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from torch.utils.data import Dataset
import numpy as np

# class GCNDataset(Dataset):  
#     def __init__(self, args, ft_dict_list, label, adj_dict_list=None,weight=None): 
#         super(GCNDataset, self).__init__()
#         self.ft_dict_list = ft_dict_list  
#         self.adj_dict_list=adj_dict_list
#         self.label = label 
#         self.args=args
#         self.weight=weight
  
#     def __len__(self):  
#         return len(self.label)  
  
#     def __getitem__(self, idx):  
#         #返回单个样本和对应的标签 
#         if self.weight is None:
#             return self.ft_dict_list[idx],self.label[idx]
#         else:
#             tf=self.ft_dict_list[idx]['tf'].reshape(-1)
#             gene=self.ft_dict_list[idx]['gene'].reshape(-1)
#             y_true=torch.cat((tf,gene),0)
#             weight=self.weight[idx]
#             return self.ft_dict_list[idx], self.adj_dict_list[idx],y_true,weight

class GCNDataset(Dataset):  
    def __init__(self, args, tf_value,gene_value, label, adj_dict_list=None,weight=None): 
        super(GCNDataset, self).__init__()
        self.tf_value = tf_value
        self.gene_value = gene_value
        self.adj_dict_list=adj_dict_list
        self.label = label 
        self.args=args
        self.weight=weight
  
    def __len__(self):  
        return len(self.label)  
  
    def __getitem__(self, idx):  
        #返回单个样本和对应的标签 
        tf_vec=self.tf_value[idx].toarray().reshape(-1, 1)
        gene_vec=self.gene_value[idx].toarray().reshape(-1, 1)
        ft_dict = {
            'tf': torch.tensor(tf_vec, dtype=torch.float32),
            'gene': torch.tensor(gene_vec, dtype=torch.float32)
        }
        return ft_dict,self.label[idx]

