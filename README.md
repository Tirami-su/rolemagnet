# RoleMagnet

RoleMagnet is a method of  role clustering for flow networks.

### Introduction

- **rolemagnet/**    Source code folder

- **experiment_*.ipynb**    Experiments mentioned in the paper. You can learn how to use Role Magnet by reading them.

### Prerequisites

- Python 3
- Python libraries: numpy  scipy  sklearn  networkx

### Usage

Copy source code folder rolemagnet/  to your project.

Make a graph G by networkx, a list b representing balance of flow (or anything else you want to use as weight of nodes), then run:

```python
import rolemagnet as rm
vec,role,label=rm.role_magnet(G, balance=b)
```

`vec` is representations of nodes , `label` is a list consisting of role labels of nodes, `vec` and `label` are both arranged in same order as `G.nodes()`. `role` is a dict whose key is a role name and value is a list containing center of cluster and nodes in the cluster. 

`balance` is optional. If node weight is insignificant in your network,  just run:

```python
vec,role,label=rm.role_magnet(G)
```

Other optional parameters:

- **shape**	  The shape of competitive layer in SOM. If not provided, Role Magnet will compute according to the distribution of data.
- **sample**    Evenly spaced sampling points. Default is `np.linspace(0,100,25)`.

### Acknowledgements

We would like to thank the authors of [struc2vec](https://github.com/leoribeiro/struc2vec), [RolX](https://github.com/Lab41/Circulo/blob/master/circulo/algorithms/rolx.py) and [GraphWave](https://github.com/snap-stanford/graphwave)  for the open access of the implementation of their method.

### License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/Tirami-su/rolemagnet/blob/master/LICENSE.md) file for details.

