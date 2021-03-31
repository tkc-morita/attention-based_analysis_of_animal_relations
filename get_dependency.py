# encoding: utf-8

import torch
import numpy as np
import pandas as pd
from modules import data_utils
import learning
import os, argparse, itertools


class Tester(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		for param in self.parameters():
			param.requires_grad = False
		[m.eval() for m in self.modules]


	def test(self, data, to_numpy = True):
		with torch.no_grad():
			data = data.to(self.device)
			_,deps = self.attention(data)
		if to_numpy:
			deps = [d.data.cpu().numpy() for d in deps]
		return deps


	def test_dataset(self, dataset, save_path, to_numpy = True, batch_size=1, num_workers=1, ix2individual=None):
		dataloader = data_utils.get_data_loader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers
			)
		# results = []
		rename_existing_file(save_path)
		for data, target, ix_in_list in dataloader:
			deps = self.test(data, to_numpy=to_numpy)
			results = [(data_ix,layer_ix,head_ix,depender,dependee,d)
				for layer_ix, deps_per_layer in enumerate(deps)
				for data_ix, deps_per_data in zip(ix_in_list, deps_per_layer)
				for head_ix,deps_per_head in enumerate(deps_per_data)
				for depender, deps_per_depender in enumerate(deps_per_head)
				for dependee, d in enumerate(deps_per_depender)]
			df_dep = pd.DataFrame(results,
							columns=['data_ix','layer_ix','head_ix','depender','dependee','weight']
							)
			if not ix2individual is None:
				df_dep.loc[:,'dependee'] = df_dep.dependee.map(ix2individual)
				df_dep.loc[:,'depender'] = df_dep.depender.map(ix2individual)

			df_dep.loc[:,'data_ix'] = df_dep.data_ix.astype(int)
			df_dep = df_dep.sort_values(['data_ix','layer_ix','head_ix','depender','dependee'])
			df_time = dataset.df_annotation.loc[
							dataset.df_annotation.index.isin(df_dep.data_ix),
							['data_path','time_ix']
							]
			df_dep = df_dep.merge(df_time, how='left', left_on='data_ix', right_index=True)

			if os.path.isfile(save_path):
				df_dep.to_csv(save_path, index=False, mode='a', header=False)
			else:
				df_dep.to_csv(save_path, index=False)
		# return df_dep
	
def rename_existing_file(filepath):
	if os.path.isfile(filepath):
		new_path = filepath+'.prev'
		rename_existing_file(new_path)
		os.rename(filepath, new_path)

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

	return par_parser.parse_args()

if __name__ == '__main__':
	args = get_parameters()

	if not os.path.isdir(os.path.dirname(args.save_path)):
		os.makedirs(os.path.dirname(args.save_path))

	tester = Tester(args.model_path, device=args.device)

	normalizer = data_utils.build_normalizer(
							args.xmin, args.xmax,
							args.ymin, args.ymax,
							args.zmin, args.zmax
							)
	transform = data_utils.Transform(normalizer,normalizer)

	dataset = data_utils.Dataset(args.input_root, args.annotation_file, transform=transform)

	ix2individual = {ix:indiv for ix,indiv in enumerate(sorted(dataset.get_individuals()))}


	tester.test_dataset(dataset, args.save_path, batch_size=args.batch_size, ix2individual=ix2individual, num_workers=args.num_workers)
