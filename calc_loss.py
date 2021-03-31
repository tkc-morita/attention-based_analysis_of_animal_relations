# encoding: utf-8

import torch
import numpy as np
import pandas as pd
from modules import data_utils
import learning
import os, argparse, itertools


class Tester(learning.Learner):
	def __init__(self, model_config_path, discrete_pred, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		for param in self.parameters():
			param.requires_grad = False
		[m.eval() for m in self.modules]
		if discrete_pred:
			self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
		else:
			self.loss_func = torch.nn.MSELoss(reduction='none')
		self.loss_func.eval()


	def test(self, data, target, to_numpy = True):
		with torch.no_grad():
			data = data.to(self.device)
			target = target.to(self.device)
			att,_ = self.attention(data)
			pred = self.last_linear(att.view(-1,att.size(-1))).view(att.size()[:-1]+(-1,))
			if isinstance(self.loss_func, torch.nn.MSELoss):
				loss = self.loss_func(pred, batched_target).sum(-1)
			else:
				loss = self.loss_func(
							pred.view(-1,pred.size(-1)),
							target.view(-1)
							).view(pred.size()[:-1])
		if to_numpy:
			loss = loss.data.cpu().numpy()
		return loss


	def test_dataset(self, dataset, to_numpy = True, batch_size=1, num_workers=1, ix2individual=None):
		dataloader = data_utils.get_data_loader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers
			)
		results = []
		for data, target, ix_in_list in dataloader:
			loss = self.test(data, target, to_numpy=to_numpy)
			results += [(data_ix,indiv_ix,loss_per_indiv)
				for data_ix, loss_per_data in zip(ix_in_list, loss)
				for indiv_ix, loss_per_indiv in enumerate(loss_per_data)]
		df_tested = pd.DataFrame(results,
						columns=['data_ix','individual_ix','loss']
						)
		if not ix2individual is None:
			df_tested.loc[:,'individual_ix'] = df_tested.individual_ix.map(ix2individual)
		return df_tested

def get_parameters():
	par_parser = argparse.ArgumentParser()

	par_parser.add_argument('model_path', type=str, help='Path to the configuration file of a trained model.')
	par_parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	par_parser.add_argument('annotation_file', type=str, help='Path to the annotation csv file.')
	par_parser.add_argument('save_path', type=str, default=None, help='Path to the directory where results are saved.')
	par_parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	par_parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size.')
	par_parser.add_argument('--xmin', type=float, default=0.0, help='Minimum on the x axis.')
	par_parser.add_argument('--xmax', type=float, default=5.0, help='Maximum on the x axis.')
	par_parser.add_argument('--xstep', type=float, default=0.5, help='Step size between grids on the x axis.')
	par_parser.add_argument('--ymin', type=float, default=0.0, help='Minimum on the y axis.')
	par_parser.add_argument('--ymax', type=float, default=4.0, help='Maximum on the y axis.')
	par_parser.add_argument('--ystep', type=float, default=0.5, help='Step size between grids on the y axis.')
	par_parser.add_argument('--zmin', type=float, default=0.0, help='Minimum on the z axis.')
	par_parser.add_argument('--zmax', type=float, default=2.5, help='Maximum on the z axis.')
	par_parser.add_argument('--zstep', type=float, default=0.5, help='Step size between grids on the z axis.')
	par_parser.add_argument('--num_workers', type=int, default=1, help='# of workers for dataloading.')
	par_parser.add_argument('--discrete_pred', action='store_true', help='If selected, the prediction is discretized based on the grid space.')

	return par_parser.parse_args()

if __name__ == '__main__':
	args = get_parameters()

	if not os.path.isdir(os.path.dirname(args.save_path)):
		os.makedirs(os.path.dirname(args.save_path))

	tester = Tester(args.model_path, args.discrete_pred, device=args.device)

	normalizer = data_utils.build_normalizer(
							args.xmin, args.xmax,
							args.ymin, args.ymax,
							args.zmin, args.zmax
							)
	if isinstance(tester.loss_func, torch.nn.CrossEntropyLoss):
		assign2block, num_blocks = data_utils.build_blocks(
										args.xmin, args.xmax, args.xstep,
										args.ymin, args.ymax, args.ystep,
										args.zmin, args.zmax, args.zstep
									)
		transform = data_utils.Transform(normalizer,assign2block)
	else:
		transform = data_utils.Transform(normalizer,normalizer)

	dataset = data_utils.Dataset(args.input_root, args.annotation_file, transform=transform)

	ix2individual = {ix:indiv for ix,indiv in enumerate(sorted(dataset.get_individuals()))}


	df_tested = tester.test_dataset(dataset, batch_size=args.batch_size, ix2individual=ix2individual, num_workers=args.num_workers)
	df_tested.loc[:,'data_ix'] = df_tested.data_ix.astype(int)
	df_tested = df_tested.sort_values(['data_ix','individual_ix'])
	df_time = dataset.df_annotation.loc[:,['data_path','time_ix']]
	df_tested = df_tested.merge(df_time, how='left', left_on='data_ix', right_index=True)
	df_tested.to_csv(os.path.join(args.save_path), index=False)
