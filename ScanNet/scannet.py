import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.manifold import TSNE
from iHGCNlayer import iHGCNLayer



class ScanNet(nn.Module):
    
    def __init__(self, net_schema,  layer_shape, all_nodes,tf_nodes, type_fusion='att', type_att_size=64):
        super(ScanNet, self).__init__()
        
        self.hgc1 = iHGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size)
        self.hgc2 = iHGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size)

        self.LN1=nn.LayerNorm(layer_shape[1])
        self.LN2=nn.LayerNorm(layer_shape[2])
        self.poolsize = 8
        self.flattenF = layer_shape[2]*(all_nodes//self.poolsize)   
        self.graphembed=nn.Linear(self.flattenF,layer_shape[3])
        self.reconstruct=nn.Linear(layer_shape[3],all_nodes)
        self.nn_fc1=nn.Linear(all_nodes,256)
        self.nn_fc2=nn.Linear(256,layer_shape[3])
        self.embe2class=nn.Linear(2*layer_shape[3],layer_shape[-1])



    def forward(self, ft_dict, adj_dict):
        # Regulation-level Encoder with iHGCN
        h1 = self.hgc1(ft_dict, adj_dict)
        x_dict = {}
        for k in h1:
            x_dict[k] = self.LN1(h1[k])
        x_dict = self.non_linear(x_dict)  
        x_dict = self.dropout_ft(x_dict, 0.2)  

        h2 = self.hgc2(x_dict, adj_dict)
        x_dict = {}
        for k in h2:
            x_dict[k] = self.LN2(h2[k])
        x_dict = self.non_linear(x_dict)


        x_cat=torch.cat((x_dict['tf'],x_dict['gene']),dim=1)
        x = self.graph_max_pool(x_cat, self.poolsize)
        x = x.view(-1, self.flattenF) #B x V/p*F
        x = self.graphembed(x)
        x = F.relu(x)
        x_hidden_gae = x
        x_decode_gae = self.reconstruct(x_hidden_gae)

        # Expression-lvel Encoder(NN)
        x_nn=torch.cat((ft_dict['tf'],ft_dict['gene']),dim=1)
        x_nn=x_nn.squeeze(2)  # B x V
        x_nn = self.nn_fc1(x_nn) 
        x_nn = F.relu(x_nn)
        x_nn = self.nn_fc2(x_nn)
        x_nn = F.relu(x_nn)

        cell_embd = torch.cat((x_hidden_gae, x_nn),1) 
        logits = self.embe2class(cell_embd)

        return logits,x_decode_gae,cell_embd


    def non_linear(self, x_dict):
        y_dict = {}
        for k in x_dict:
            y_dict[k] = F.elu(x_dict[k])
        return y_dict


    def dropout_ft(self, x_dict, dropout):
        y_dict = {}
        for k in x_dict:
            y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
        return y_dict
     

    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p   maxpool的是gene
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x
    
    def loss(self, gnn_y1, y_target1,y2, y_target2,args,cell_embd,l2_regularization=5e-4):
     
        loss1 = nn.MSELoss()(gnn_y1, y_target1) 
        loss2 = nn.CrossEntropyLoss()(y2,y_target2)           
        loss = 1 * loss1 + 1 * loss2

        l2_loss = 0.0
        for param in self.parameters():
            data = param* param
            l2_loss += data.sum()


        loss =loss + 0.2* l2_regularization* l2_loss 

        return loss

