import numpy as np
import os
import sys
from tqdm import tqdm
import torch
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader
import atom3d.util.graph as gr

_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)
_DEFAULT_V_DIM = (64, 16)
_DEFAULT_E_DIM = (32, 1)
    
class GNNTransformLBA(object):
    def __init__(self, pocket_only=True):
        self.pocket_only = pocket_only
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        if self.pocket_only:
            item = prot_graph_transform(item, atom_keys=['atoms_pocket'], label_key='scores')
        else:
            item = prot_graph_transform(item, atom_keys=['atoms_protein', 'atoms_pocket'], label_key='scores')
        # transform ligand into PTG graph
        item = mol_graph_transform(item, 'atoms_ligand', 'scores', use_bonds=True, onehot_edges=False)
        node_feats, edges, edge_feats, node_pos = gr.combine_graphs(item['atoms_pocket'], item['atoms_ligand'], edges_between=True)
        combined_graph = Data(node_feats, edges, edge_feats, y=item['scores']['neglog_aff'], pos=node_pos)
        return combined_graph
    
def load_data(data_dir, transform = GNNTransformLBA()):
    train_data = LMDBDataset(os.path.join(data_dir, "train"), transform = transform)
    valid_data = LMDBDataset(os.path.join(data_dir, "val"), transform =transform )
    test_data = LMDBDataset(os.path.join(data_dir, 'test'), transform = transform)
    return train_data, valid_data, test_data

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


if __name__== "__main__":
    data_dir = "/media/Z/minhnh46/khangnn3/gvp/atom3d-data/lba/split-by-sequence-identity-30/data"
    train_data, valid_data, test_data = load_data(data_dir)
    print(train_data[0])
