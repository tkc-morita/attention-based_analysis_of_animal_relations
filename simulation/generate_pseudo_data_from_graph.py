# coding:utf-8

import numpy as np
import pandas as pd
import os, argparse, json



def generate_pseudo_data(
		graph_info, data_size, save_path, radius,
		distance_type='proximal', random_state=None, **bounds
		):
	if distance_type=='proximal':
		sample_func = generate_proximal_data
	elif distance_type=='sphere':
		sample_func = generate_sphere_data
	dfs = []
	node2samples = {}
	for referrer in graph_info['ordered_nodes']:
		parents = graph_info['node2parents'][referrer]
		if parents:
			referee_samples = node2samples[random_state.choice(parents)]
			referrer_samples = sample_func(referee_samples,radius,random_state=random_state,**bounds)
		else:
			referrer_samples = generate_uniform_data(data_size, random_state=random_state, **bounds)
		node2samples[referrer] = referrer_samples
		df = to_df(referrer_samples, referrer)
		dfs.append(df)
	df = pd.concat(dfs, axis=0, ignore_index=True)
	df.to_csv(save_path, index=False)

def to_df(data,name):
	df = pd.DataFrame(data, columns=['x','y','z'])
	df['time_ix'] = df.index
	df['individual'] = name
	df = df.melt(id_vars=['time_ix','individual'], var_name='axis', value_name='value')
	return df

def generate_uniform_data(
	data_size,
	xmin=0.0,
	xmax=1.0,
	ymin=0.0,
	ymax=1.0,
	zmin=0.0,
	zmax=1.0,
	random_state=None
	):
	if random_state is None:
		random_state = np.random.RandomState()
	data = random_state.rand(data_size,3)
	data[:,0] *= xmax - xmin
	data[:,0] += xmin
	data[:,1] *= ymax - ymin
	data[:,1] += ymin
	data[:,2] *= zmax - zmin
	data[:,2] += zmin
	return data

def generate_proximal_data(
	reference,
	max_distance,
	random_state=None,
	**bounds
	):
	data = np.zeros_like(reference)
	data_size = data.shape[0]
	indices = np.arange(data_size)
	while True:
		proposal = generate_uniform_data(data_size,random_state=random_state,**bounds)
		newly_accepted = np.linalg.norm(reference[indices,:]-proposal, axis=-1)<max_distance
		data[indices[newly_accepted],:] = proposal[newly_accepted,:]
		if newly_accepted.all():
			break
		else:
			indices = indices[~newly_accepted]
			data_size = indices.size
	return data

# def generate_distance_data(
# 	reference,
# 	min_distance,
# 	random_state=None,
# 	**bounds
# 	):
# 	data = np.zeros_like(reference)
# 	data_size = data.shape[0]
# 	indices = np.arange(data_size)
# 	while True:
# 		proposal = generate_uniform_data(data_size,random_state=random_state,**bounds)
# 		newly_accepted = np.linalg.norm(reference[indices,:]-proposal, axis=-1)>=min_distance
# 		# print(indices.size, np.linalg.norm(reference[indices,:]-proposal, axis=-1).min())
# 		data[indices[newly_accepted],:] = proposal[newly_accepted,:]
# 		if newly_accepted.all():
# 			break
# 		else:
# 			indices = indices[~newly_accepted]
# 			data_size = indices.size
# 	return data

def generate_sphere_data(
	reference,
	distance,
	random_state=None,
	xmin=0.0,
	xmax=1.0,
	ymin=0.0,
	ymax=1.0,
	zmin=0.0,
	zmax=1.0,
	):
	data = np.zeros_like(reference)
	data_size = data.shape[0]
	indices = np.arange(data_size)
	mins = np.array([xmin,ymin,zmin])[None,:]
	maxs = np.array([xmax,ymax,zmax])[None,:]
	if random_state is None:
		random_state = np.random.RandomState()
	while True:
		proposal = _sample_from_sphere(reference[indices,:],distance,random_state)
		newly_accepted = (mins<=proposal).all(1) & (proposal<=maxs).all(1)
		data[indices[newly_accepted],:] = proposal[newly_accepted,:]
		if newly_accepted.all():
			break
		else:
			indices = indices[~newly_accepted]
			data_size = indices.size
	return data

def _sample_from_sphere(reference, distance, random_state):
	data_size = reference.shape[0]
	elevs,azims = random_state.rand(2,data_size)
	elevs = elevs*np.pi
	azims = azims*np.pi*2.0
	z = distance * np.cos(elevs)
	sin_elevs = np.sin(elevs)
	x = distance * sin_elevs * np.cos(azims)
	y = distance * sin_elevs * np.sin(azims)
	out = reference + np.stack([x,y,z], axis=-1)
	return out

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('graph_path', type=str, help='Path to the json file containing grpah info.')
	parser.add_argument('save_dir', type=str, help='Path to the directory where results are saved.')
	parser.add_argument('data_size', type=int, help='Data size.')
	parser.add_argument('--seed', type=int, default=111, help='Random seed.')
	parser.add_argument('--xmin', type=float, default=0.0, help='Minimum on the x axis.')
	parser.add_argument('--xmax', type=float, default=5.0, help='Maximum on the x axis.')
	parser.add_argument('--ymin', type=float, default=0.0, help='Minimum on the y axis.')
	parser.add_argument('--ymax', type=float, default=4.0, help='Maximum on the y axis.')
	parser.add_argument('--zmin', type=float, default=0.0, help='Minimum on the z axis.')
	parser.add_argument('--zmax', type=float, default=2.5, help='Maximum on the z axis.')
	parser.add_argument('--radius', type=float, default=1.5, help='(Maximum) distance b/w agents.')
	parser.add_argument('--distance_type', type=str, default='proximal', choices=['proximal','sphere'], help='Type of inter-individual distance. "proximal" or "sphere".')
	args = parser.parse_args()

	max_sub_data_size = 5000
	file_ix = 0
	random_state = np.random.RandomState(args.seed)
	dfs = []
	with open(args.graph_path, 'r') as f:
		graph_info = json.load(f)
	if not os.path.isdir(os.path.join(args.save_dir, 'data')):
		os.makedirs(os.path.join(args.save_dir, 'data'))
	with open(os.path.join(args.save_dir, 'info.json'), 'w') as f:
		json.dump(vars(args), f)
	while args.data_size>0:
		sub_data_size = min(max_sub_data_size, args.data_size)
		filename = '{:03d}.csv'.format(file_ix)
		generate_pseudo_data(
			graph_info,
			sub_data_size,
			os.path.join(args.save_dir, 'data', filename),
			args.radius,
			distance_type=args.distance_type,
			xmin=args.xmin,
			xmax=args.xmax,
			ymin=args.ymin,
			ymax=args.ymax,
			zmin=args.zmin,
			zmax=args.zmax,
			random_state=random_state
		)
		df = pd.DataFrame(np.arange(sub_data_size)[:,None], columns=['time_ix'])
		df['data_path'] = filename
		args.data_size -= sub_data_size
		dfs.append(df)
		file_ix += 1
	df = pd.concat(dfs, axis=0)
	df.to_csv(os.path.join(args.save_dir, 'director.csv'), index=False)
	info = {}