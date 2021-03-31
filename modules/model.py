# coding: utf-8

import torch
import math, collections
# import sparsemax
from . import sparsemax as sparsemax_module


class SelfAttentionEncoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_individuals, num_heads=8, num_layers=1, dropout=0.0, bottleneck_layers=list(), sparsemax=False):
		super(SelfAttentionEncoder, self).__init__()
		self.to_hidden = MLP(input_size, hidden_size, hidden_size)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_individuals = num_individuals
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.bottleneck_layers = bottleneck_layers
		self.sparsemax = sparsemax
		self.dropout = torch.nn.Dropout(dropout)
		self.self_attention = torch.nn.Sequential(
									collections.OrderedDict([
									('layer{}'.format(l),
									SelfAttentionLayer(
										hidden_size,
										num_individuals,
										num_heads=1 if l in bottleneck_layers else num_heads,
										dropout=dropout,
										sparsemax=sparsemax if l in bottleneck_layers else False
										)
									)
									for l in range(num_layers)
									]))

	def forward(self, batched_input):
		hidden = self.to_hidden(batched_input.view(-1,self.input_size)).view(batched_input.size()[:-1]+(-1,))

		hidden = self.dropout(hidden)

		mask = torch.eye(batched_input.size(1), device=batched_input.device).bool()[None,None,:,:]

		input_as_dict = {
			'values':torch.zeros_like(hidden),
			'original_input':hidden,
			'mask':mask,
			'weights':[]
		}
		out_as_dict = self.self_attention(input_as_dict)
		return out_as_dict['values'], out_as_dict['weights']



	def pack_init_args(self):
		args = {
			'input_size':self.input_size,
			'hidden_size':self.hidden_size,
			'num_individuals':self.num_individuals,
			'num_heads':self.num_heads,
			'num_layers':self.num_layers,
			'bottleneck_layers':self.bottleneck_layers,
			'dropout':self.dropout.p,
			'sparsemax':self.sparsemax
		}
		return args


class SelfAttentionLayer(torch.nn.Module):
	def __init__(self, hidden_size, num_individuals, num_heads=8, dropout=0.0, sparsemax=False):
		super(SelfAttentionLayer, self).__init__()
		hidden_size_per_head = hidden_size // num_heads
		self.to_query = LinearSplit(hidden_size, hidden_size_per_head, num_heads)
		self.to_key = LinearSplit(hidden_size, hidden_size_per_head, num_heads)
		self.to_value = LinearSplit(hidden_size, hidden_size_per_head, num_heads)

		self.attention = DotProductAttention(sparsemax=sparsemax)
		self.linear_combine_heads = torch.nn.Linear(hidden_size_per_head*num_heads, hidden_size)
		self.top_feedfoward = MLP(hidden_size, hidden_size, hidden_size, nonlinearity='GELU')
		self.dropout = torch.nn.Dropout(dropout)
		self.layer_norm = torch.nn.LayerNorm(hidden_size)

		self.register_parameter(
			'individual_query',
			torch.nn.Parameter(
				torch.randn(1,num_individuals,hidden_size),
				requires_grad=True
			)
			)
		self.register_parameter(
			'individual_memory',
			torch.nn.Parameter(
				torch.randn(1,num_individuals,hidden_size),
				requires_grad=True
			)
			)
		self.register_parameter(
			'individual_post_attention',
			torch.nn.Parameter(
				torch.randn(1,num_individuals,hidden_size),
				requires_grad=True
			)
			)

	def forward(self, input_as_dict):
		input_to_query = input_as_dict['values']
		input_to_memory = input_as_dict['original_input']
		mask = input_as_dict['mask']
		batch_size, num_individuals, hidden_size = input_to_memory.size()

		input_to_query = input_to_query + self.individual_query
		input_to_memory = input_to_memory + self.individual_memory

		input_to_query = input_to_query.view(-1,hidden_size)
		input_to_memory = input_to_memory.view(-1,hidden_size)
		query = torch.stack(self.to_query(input_to_query), dim=-2) # Only receive the positional embedding.
		key = torch.stack(self.to_key(input_to_memory), dim=-2)
		value = torch.stack(self.to_value(input_to_memory), dim=-2)
		query, key, value = [x.view((-1,num_individuals)+x.size()[1:]).transpose(1,2).contiguous()
							for x in [query, key, value]]
		attention, weight = self.attention(query, key, value, mask)
		attention = attention.transpose(1,2).contiguous().view(-1, hidden_size)
		attention = self.linear_combine_heads(attention)
		attention = self.dropout(attention)
		attention = self.layer_norm(
						(self.individual_post_attention
						+
						attention.view(-1,num_individuals,hidden_size)
						).view(-1, hidden_size)
						)

		out = self.top_feedfoward(attention)
		out = self.dropout(out)
		out = self.layer_norm(attention + out)
		input_as_dict['values'] = out.view(batch_size, num_individuals, hidden_size)
		input_as_dict['weights'].append(weight)
		return input_as_dict



class LinearSplit(torch.nn.Module):
	def __init__(self, input_size, output_size, num_splits):
		super(LinearSplit, self).__init__()
		self.linears = torch.nn.ModuleList([
								torch.nn.Linear(input_size, output_size)
								for ix in range(num_splits)
							])
		
	def forward(self, x):
		return [l(x) for l in self.linears]

class DotProductAttention(torch.nn.Module):
	def __init__(self, sparsemax=False):
		super(DotProductAttention, self).__init__()
		if sparsemax:
			self.softmax = sparsemax_module.Sparsemax(dim=-1)
		else:
			self.softmax = torch.nn.Softmax(dim=-1)

	def forward(self, query, key, value, mask=None):
		"""
		query: batch_size (x num_heads) x length_1 x hidden_size
		key, value: batch_size (x num_heads) x length_2 x hidden_size
		"""
		weight = query.matmul(key.transpose(-2,-1))
		weight = weight / math.sqrt(key.size(-1)) # batch_size (x num_heads) x length_1 x length_2
		if not mask is None:
			weight = weight.masked_fill(
				mask,
				torch.finfo().min
				)
		weight = self.softmax(weight) # No attention dropout as # of individuals is supposed to be small.
		out = weight.matmul(value)
		return out, weight




class MLP(torch.jit.ScriptModule):
# class MLP(torch.nn.Module):
	"""
	Multi-Layer Perceptron.
	"""
	def __init__(self, input_size, hidden_size, output_size, nonlinearity='GELU'):
		super(MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.nonlinearity = nonlinearity
		if nonlinearity=='GELU':
			nonlinearity = GELU()
		else:
			nonlinearity = getattr(torch.nn, nonlinearity)()
		self.whole_network = torch.nn.Sequential(
			torch.nn.Linear(input_size, hidden_size),
			nonlinearity,
			torch.nn.Linear(hidden_size, output_size)
			)

	@torch.jit.script_method
	def forward(self, batched_input):
		return self.whole_network(batched_input)

	def pack_init_args(self):
		init_args = {
			'input_size':self.input_size,
			'hidden_size':self.hidden_size,
			'output_size':self.output_size,
			'nonlinearity':self.nonlinearity
		}
		return init_args

class RandomMask(torch.nn.Module):
	def __init__(self, mask_value='learn', input_size=None, random_rate=0.2):
		super(RandomMask, self).__init__()
		self.input_size = input_size
		self.random_rate = random_rate
		if mask_value == 'learn':
			assert not input_size is None, "input_size must be specified when mask_value='learn'"
			self.mask_learning = True
			self.register_parameter('mask_value', torch.nn.Parameter(torch.rand(input_size).view(1,1,-1), requires_grad=True))
		else:
			self.mask_learning = False
			self.mask_value = mask_value

	def forward(self, batched_input):
		mask = torch.zeros_like(batched_input[0,:,0])
		mask[0] = 1.0
		mask = torch.stack(
				[mask[torch.randperm(mask.size(0))]
				for batch_ix in range(batched_input.size(0))],
				dim=0)
		# Step 0: Fill zeros in the mased entries.
		masked_input = batched_input*(1-mask[:,:,None])
		# Step 1: Sample from Uniform to determine if the mask type is Gaussian random.
		random_value = torch.randn_like(batched_input[:,0,0]).view(-1,1,1)
		samples = torch.rand_like(random_value)
		masked_input = masked_input + (samples<self.random_rate).float() * mask[:,:,None] * random_value
		# Step 2: Fill the other with self.mask_value
		masked_input = masked_input + (self.random_rate<=samples).float() * mask[:,:,None] * self.mask_value
		return masked_input,mask

	def pack_init_args(self):
		args = {
			'mask_value':self.mask_value,
			'input_size':self.input_size,
			'random_rate':self.random_rate,
			}
		if self.mask_learning:
			args['mask_value'] = 'learn'
		return args

class GELU(torch.jit.ScriptModule):
	"""
	Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
	Copied from BERT-pytorch:
	https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py
	"""
	__constants__ = ['sqrt_2_over_pi']
	def __init__(self):
		super(GELU, self).__init__()
		self.sqrt_2_over_pi = math.sqrt(2 / math.pi)

	@torch.jit.script_method
	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))))