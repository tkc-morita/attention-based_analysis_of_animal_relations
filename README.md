# Attention-Based Analysis of Animal Inter-Individual Relations

This repository provides Python (PyTorch) programs used in [Morita et al. (2021) "Non-Parametric Analysis of Inter-Individual Relations Using an Attention-Based Neural Network"](https://doi.org/10.1101/2020.03.25.994764).

## Training

```
python learning.py ...
```

The following options were used in the paper.

```
python learning.py /path/to/data_dir /path/to/annotation.csv -b 512 --num_workers 8 -j any_name_you_like -d cuda -S /path/to/directory/where/results/are/saved/ --attention_hidden_size 512 --num_attention_layers 2 --num_attention_heads 4 --bottleneck_layers 1 --learning_rate 0.005 -i 30000 --warmup_iters 3000 --saving_interval 600 --discrete_pred
```

## Get dependency weights after training.

```
python get_dependency.py ...
```

## Get simulation data

Sample random graphs.

```
python generate_graphs_single_parent.py ...
python generate_graphs_multiple_parents.py ...
```

Generate pseudo location data from graphs.

```
python generate_pseudo_data_from_graph.py ...
```
