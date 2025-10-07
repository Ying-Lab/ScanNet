import torch
import torch.nn as nn
import torch.nn.functional as F



class iHGCNLayer(nn.Module):

	def __init__(self, net_schema, in_layer_shape, out_layer_shape, type_fusion, type_att_size):
		super(iHGCNLayer, self).__init__()
		
		self.net_schema = net_schema
		self.in_layer_shape = in_layer_shape
		self.out_layer_shape = out_layer_shape

		self.hete_agg = nn.ModuleDict()
		for k in net_schema.keys():
			self.hete_agg[k] = iHGCNByType(k, net_schema[k], in_layer_shape, out_layer_shape, type_fusion, type_att_size)

	def forward(self, x_dict, adj_dict):
		ret_x_dict = {}
		for k in self.hete_agg.keys():
			ret_x_dict[k] = self.hete_agg[k](x_dict, adj_dict)
		return ret_x_dict



class iHGCNByType(nn.Module):
	
	def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, type_fusion, type_att_size):
		super(iHGCNByType, self).__init__()
		
		self.nb_list = nb_list
		self.curr_k = curr_k
		self.type_fusion = type_fusion
		
		self.W_rel = nn.ParameterDict()
		for k in nb_list:
			self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
			nn.init.xavier_uniform_(self.W_rel[k].data)
		
		self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
		nn.init.xavier_uniform_(self.w_self.data)

		self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
		nn.init.xavier_uniform_(self.bias.data)


		self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
		nn.init.xavier_uniform_(self.w_query.data)
		self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
		nn.init.xavier_uniform_(self.w_keys.data)
		self.w_att = nn.Parameter(torch.FloatTensor(2*type_att_size, 1))
		nn.init.xavier_uniform_(self.w_att.data)


	def forward(self, x_dict, adj_dict):
		
		self_ft = torch.matmul(x_dict[self.curr_k], self.w_self)
		
		nb_ft_list = [self_ft]
		threshold=0.2
		for k in self.nb_list:
			x_nb = torch.matmul(x_dict[k], self.W_rel[k])  ## 投影在common空间
			A=adj_dict[self.curr_k]
			x1=torch.matmul(A,x_nb)
			nb_ft_list.append(x1)

		att_query = torch.matmul(self_ft, self.w_query).repeat(1,len(nb_ft_list), 1)
		att_keys = torch.matmul(torch.cat(nb_ft_list, 1), self.w_keys)
		att_input = torch.cat([att_keys, att_query], 2)
		att_input = F.dropout(att_input, 0.5, training=self.training)
		e = F.elu(torch.matmul(att_input, self.w_att)) 
		attention = F.softmax(e.view(x_dict[self.curr_k].shape[0],len(nb_ft_list), -1).transpose(1,2), dim=2)
		agg_nb_ft = torch.cat([nb_ft.unsqueeze(2) for nb_ft in nb_ft_list], 2).mul(attention.unsqueeze(-1)).sum(2)   ## TF,TG融合
			
		output = agg_nb_ft + self.bias

		return output