# coding: utf-8

import torch
import torch.utils.data
import pandas as pd
import numpy as np
import os.path
import warnings
warnings.simplefilter("error")


class Dataset(torch.utils.data.Dataset):
	def __init__(self, input_root, annotation_file, transform = None):
		self.df_annotation = pd.read_csv(annotation_file)
		self.input_root = input_root
		self.transform = transform

	def __len__(self):
		"""Return # of data strings."""
		return self.df_annotation.shape[0]

	def __getitem__(self, ix):
		"""Return """
		data_path = self.df_annotation.loc[ix, 'data_path']
		df_data = pd.read_csv(os.path.join(self.input_root,data_path))
		time_ix = self.df_annotation.loc[ix, 'time_ix']
		df_data = df_data[df_data.time_ix==time_ix].sort_values(['individual','axis'])

		input_data = df_data.loc[:,'value'].values.astype(np.float32).reshape(-1,3)

		input_data = torch.from_numpy(input_data)

		if self.transform:
			input_data, output_data = self.transform(input_data, input_data)
		return input_data, output_data, ix

	def get_individuals(self):
		data_path = self.df_annotation.loc[0, 'data_path']
		df_data = pd.read_csv(os.path.join(self.input_root,data_path))
		return df_data.individual.unique()


class Transform(object):
	def __init__(self, in_trans, out_trans):
		self.in_trans = in_trans
		self.out_trans = out_trans

	def __call__(self, input_data, output_data):
		in_transformed = self.in_trans(input_data)
		out_transformed = self.out_trans(output_data)
		return in_transformed, out_transformed

def build_blocks(xmin,xmax,xstep,ymin,ymax,ystep,zmin,zmax,zstep):
	x_boundaries = torch.arange(xmin,xmax,xstep)[1:]
	y_boundaries = torch.arange(ymin,ymax,ystep)[1:]
	z_boundaries = torch.arange(zmin,zmax,zstep)[1:]
	num_divisions_x = x_boundaries.size(0)+1
	num_divisions_y = y_boundaries.size(0)+1
	num_divisions_z = z_boundaries.size(0)+1
	block_ixs = torch.arange(
					num_divisions_x*num_divisions_y*num_divisions_z
					).view(num_divisions_x,num_divisions_y,num_divisions_z)
	def assign2block(coordinates):
		"""
		coordinates: * x 3 tensor
		"""
		broadcaster4coordinates = coordinates.size()[:-1] + (1,)
		broadcaster4boundaries = ((1,) * (coordinates.dim() - 1)) + (-1,)
		count_over_x = (coordinates[...,0].view(broadcaster4coordinates)>=x_boundaries.view(broadcaster4boundaries)).sum(-1)
		count_over_y = (coordinates[...,1].view(broadcaster4coordinates)>=y_boundaries.view(broadcaster4boundaries)).sum(-1)
		count_over_z = (coordinates[...,2].view(broadcaster4coordinates)>=z_boundaries.view(broadcaster4boundaries)).sum(-1)
		return block_ixs[count_over_x,count_over_y,count_over_z]
	return assign2block, block_ixs.view(-1).size(0)


def build_normalizer(xmin,xmax,ymin,ymax,zmin,zmax):
	def normalizer(coordinates):
		"""
		coordinates: N x 3 tensor
		"""
		coordinates[...,0] -= (xmax + xmin) * 0.5 # Move the center to 0
		coordinates[...,0] /= (xmax - xmin) * 0.5 # Rescale so that the range becomes [-1.0, 1.0].

		coordinates[...,1] -= (ymax + ymin) * 0.5
		coordinates[...,1] /= (ymax - ymin) * 0.5

		coordinates[...,2] -= (zmax + zmin) * 0.5
		coordinates[...,2] /= (zmax - zmin) * 0.5
		return coordinates
	return normalizer


class IterationBasedBatchSampler(torch.utils.data.BatchSampler):
	"""
	Wraps a BatchSampler, resampling from it until
	a specified number of iterations are sampled.
	Partially Copied from maskedrcnn-benchmark.
	https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
	"""

	def __init__(self, batch_sampler, num_iterations, start_iter=0):
		self.batch_sampler = batch_sampler
		self.num_iterations = num_iterations
		self.start_iter = start_iter
		if hasattr(self.batch_sampler.sampler, 'set_start_ix'):
			start_ix = (self.start_iter % len(self.batch_sampler)) * self.batch_sampler.batch_size
			self.batch_sampler.sampler.set_start_ix(start_ix)

	def __iter__(self):
		iteration = self.start_iter
		epoch = iteration // len(self.batch_sampler)
		while iteration <= self.num_iterations:
			if hasattr(self.batch_sampler.sampler, 'set_epoch'):
				self.batch_sampler.sampler.set_epoch(epoch)
			for batch in self.batch_sampler:
				iteration += 1
				if iteration > self.num_iterations:
					break
				yield batch
			epoch += 1

	def __len__(self):
		return self.num_iterations

class RandomSampler(torch.utils.data.RandomSampler):
	"""
	Custom random sampler for iteration-based learning.
	"""
	def __init__(self, *args, seed=111, **kwargs):
		super(RandomSampler, self).__init__(*args, **kwargs)
		self.epoch = 0
		self.start_ix = 0
		self.seed = seed

	def set_epoch(self, epoch):
		self.epoch = epoch

	def set_start_ix(self, start_ix):
		self.start_ix = start_ix

	def __iter__(self):
		g = torch.Generator()
		g.manual_seed(self.epoch+self.seed)
		start_ix = self.start_ix
		self.start_ix = 0
		n = len(self.data_source)
		if self.replacement:
			return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=g).tolist()[start_ix:])
		return iter(torch.randperm(n, generator=g).tolist()[start_ix:])


def get_data_loader(dataset, batch_size=1, shuffle=False, num_iterations=None, start_iter=0, num_workers=1, random_seed=111):
	if shuffle:
		sampler = RandomSampler(dataset, replacement=False, seed=random_seed)
	else:
		sampler = torch.utils.data.SequentialSampler(dataset)
	batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
	if not num_iterations is None:
		batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations, start_iter=start_iter)

	data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler)
	return data_loader