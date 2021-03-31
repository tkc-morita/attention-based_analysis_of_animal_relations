# coding: utf-8

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from load_graph import load_graph,dict2graph
from plot_graph import plot_graph
import os,argparse,glob,json,itertools

def sample_graphs(
		num_nodes,
		# num_graphs,
		graph_dir,
		# seed=111
		):
	if not os.path.isdir(graph_dir):
		os.makedirs(graph_dir)
	excluded_graph_dir = os.path.join(graph_dir, 'exclude')
	if os.path.isdir(excluded_graph_dir):
		excluded_graphs = [load_graph(path)
								for path in glob.glob(os.path.join(excluded_graph_dir,'*.json'))]
	else:
		excluded_graphs = []
	
	prev_studied_dir = os.path.join(graph_dir, 'previously_studied')
	if os.path.isdir(prev_studied_dir):
		prev_studied_graphs = {os.path.basename(os.path.splitext(path)[0]):load_graph(path)
								for path in glob.glob(os.path.join(prev_studied_dir,'*.json'))}
	else:
		prev_studied_graphs = dict()

	duplications = []
	for graph_ix,(g,info) in enumerate(all_graphs(num_nodes,excluded_graphs=excluded_graphs)):
		isomorphic_name = None
		for name_prev,prev_g in prev_studied_graphs.items():
			if nx.algorithms.isomorphism.is_isomorphic(g,prev_g):
				isomorphic_name = name_prev
				break

		basename_wo_ext = '{:02d}'.format(graph_ix)
		if not isomorphic_name is None:
			duplications.append((basename_wo_ext,isomorphic_name))
		save_path_wo_ext = os.path.join(graph_dir,basename_wo_ext)
		with open(save_path_wo_ext + '.json', 'w') as f:
			json.dump(info, f)
		plot_graph(g, save_path_wo_ext + '.png')

	if duplications:
		df = pd.DataFrame(duplications,columns=['sampled_graph','previous_graph'])
		save_path = os.path.join(graph_dir, 'duplications.csv')
		if os.path.isfile(save_path):
			df.to_csv(save_path, index=False, mode='a',header=False)
		else:
			df.to_csv(save_path, index=False)

def all_graphs(num_nodes,excluded_graphs=list()):
	graphs = []
	previous_graphs = excluded_graphs
	node2possible_parents = [[[]] + [['node_{}'.format(parent)] for parent in range(node_ix)] for node_ix in range(num_nodes)]
	ordered_nodes = ['node_{}'.format(node_ix) for node_ix in range(num_nodes)]
	for node2parents in itertools.product(*node2possible_parents):
		node2parents = {'node_{}'.format(node_ix):parents for node_ix,parents in enumerate(node2parents)}
		g = dict2graph(node2parents)
		if (not nx.algorithms.isolate.number_of_isolates(g)) and (not check_isomorphicity(g, previous_graphs)): # Reject if any node is isolated (no incoming nor outgoing neighbors).
			graphs.append((g,{'ordered_nodes':ordered_nodes,'node2parents':node2parents}))
			previous_graphs.append(g)
	return graphs

def check_isomorphicity(graph,previous_graphs):
	for prev_g in previous_graphs:
		if nx.algorithms.isomorphism.is_isomorphic(graph,prev_g):
			return True
	return False



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('save_dir', type=str, help='Path to the directory where sampled graphs are saved.')
	# parser.add_argument('num_graphs', type=int, help='# of graphs to sample.')
	parser.add_argument('--num_nodes', type=int, default=5, help='# of nodes in the sampled graphs.')
	# parser.add_argument('--seed', type=int, default=111, help='Random seed.')
	args = parser.parse_args()
		
	sample_graphs(args.num_nodes,
					# args.num_graphs,
					args.save_dir,
					# seed=args.seed
					)