# coding: utf-8

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from load_graph import load_graph,dict2graph
from plot_graph import plot_graph
import os,argparse,glob,json

def sample_graphs(num_nodes, num_graphs, graph_dir, seed=111):
	if os.path.isdir(graph_dir):
		graphs = [load_graph(path)
					for path in glob.glob(os.path.join(graph_dir,'*.json'))]
	else:
		os.makedirs(graph_dir)
		graphs = []
	excluded_graph_dir = os.path.join(graph_dir, 'exclude')
	if os.path.isdir(excluded_graph_dir):
		excluded_graphs = [load_graph(path)
								for path in glob.glob(os.path.join(excluded_graph_dir,'*.json'))]
	else:
		excluded_graphs = []
	num_excluded_graphs = len(excluded_graphs)
	graphs += excluded_graphs

	prev_studied_dir = os.path.join(graph_dir, 'previously_studied')
	if os.path.isdir(prev_studied_dir):
		prev_studied_graphs = {os.path.basename(os.path.splitext(path)[0]):load_graph(path)
								for path in glob.glob(os.path.join(prev_studied_dir,'*.json'))}
	else:
		prev_studied_graphs = dict()

	random_state = np.random.RandomState(seed)
	duplications = []
	for graph_ix in range(num_graphs):
		while True:
			ordered_nodes,node2parents,g = generate_directed_acyclic_graph(num_nodes,random_state=random_state)
			overlap = False
			for g_prev in graphs:
				if nx.algorithms.isomorphism.is_isomorphic(g,g_prev):
					overlap = True
					break
			if not overlap:
				break
		graphs.append(g)

		isomorphic_name = None
		for name_prev,prev_g in prev_studied_graphs.items():
			if nx.algorithms.isomorphism.is_isomorphic(g,prev_g):
				isomorphic_name = name_prev
				break

		basename_wo_ext = '{:02d}'.format(len(graphs)-num_excluded_graphs-1)
		if not isomorphic_name is None:
			duplications.append((basename_wo_ext,isomorphic_name))
		save_path_wo_ext = os.path.join(graph_dir,basename_wo_ext)
		with open(save_path_wo_ext + '.json', 'w') as f:
			json.dump({'ordered_nodes':ordered_nodes,'node2parents':node2parents}, f)
		plot_graph(g, save_path_wo_ext + '.png')

	if duplications:
		df = pd.DataFrame(duplications,columns=['sampled_graph','previous_graph'])
		save_path = os.path.join(graph_dir, 'duplications.csv')
		if os.path.isfile(save_path):
			df.to_csv(save_path, index=False, mode='a',header=False)
		else:
			df.to_csv(save_path, index=False)


def generate_directed_acyclic_graph(num_nodes, random_state=None):
	if random_state is None:
		random_state = np.random.RandomState()
	while True:
		node2parents = dict()
		ordered_nodes = []
		for node_ix in range(num_nodes):
			parents = [node for node in node2parents.keys() if random_state.rand()>0.5]
			node_name = 'node_{}'.format(node_ix)
			node2parents[node_name] = parents
			ordered_nodes.append(node_name)
		g = dict2graph(node2parents)
		if (not nx.algorithms.isolate.number_of_isolates(g)) and (max(map(len,node2parents.values()))>1): # Reject if any node is isolated (no incoming nor outgoing neighbors).
			break
	return ordered_nodes,node2parents,g



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('save_dir', type=str, help='Path to the directory where sampled graphs are saved.')
	parser.add_argument('num_graphs', type=int, help='# of graphs to sample.')
	parser.add_argument('--num_nodes', type=int, default=5, help='# of nodes in the sampled graphs.')
	parser.add_argument('--seed', type=int, default=111, help='Random seed.')
	args = parser.parse_args()
		
	sample_graphs(args.num_nodes, args.num_graphs, args.save_dir, seed=args.seed)