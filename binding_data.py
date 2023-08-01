import os 
import torch
from torch_geometric.data import Data
import selfies as sf
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset
import pickle
from torch_geometric.data import Data, Batch
import numpy as np

class PairDataset(Dataset):
    def __init__(self, root, df):
        super().__init__()
        self.df = df 
        self.root = root 
        self.ligand_graphs = None 
        with open(os.path.join(root, "ligand_to_graph.pkl"), "rb") as f:
            self.ligand_graphs = pickle.load(f)
        with open(os.path.join(root, "ligand_to_ecfp.pkl"), "rb") as f:
            self.ligand_mps= pickle.load(f)
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        items = self.df.iloc[idx]
        protein_name = items["protein"]
        ligand = items["ligand"]
        label = items["label"]
        
        protein_file = os.path.join(self.root, f"res_graph/{protein_name}.pdb.pt")
        protein_graph = torch.load(protein_file, map_location="cpu")
        protein_graph.y = label 

        ligand_graph = self.ligand_graphs[ligand]
        ligand_graph = Data(x = torch.from_numpy(ligand_graph[0].astype(np.float64)).float(), edge_index = torch.from_numpy(ligand_graph[1].astype(np.int64)).t(), 
                            edge_attr = torch.from_numpy(ligand_graph[2].astype(np.float64)).float())

        ligand_mp = torch.from_numpy(self.ligand_mps[ligand])
        
        return protein_graph, ligand_graph, ligand_mp
    

class CollaterLBA(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def collate(self, data_list):
        if len(data_list) != self.batch_size:
            bs = len(data_list)
        else:
            bs = self.batch_size
        batch_1 = Batch.from_data_list([d[0] for d in data_list])
        batch_2 = Batch.from_data_list([d[1] for d in data_list])
        mp = torch.stack([d[2] for d in data_list])
        return batch_1, batch_2, mp
    
    def adjust_graph_indices(self, graph, bs):
        total_n = 0
        for i in range(bs-1):
            n_nodes = graph.num_nodes[i].item()
            total_n += n_nodes
            #graph.ca_idx[i+1] += total_n
        return graph

    def __call__(self, batch):
        return self.collate(batch)


class TargetDataset(Dataset):
    def __init__(self, root, csv_file, symbol_to_idx, 
                idx_to_symbol, max_len, protein_set, transform = None, test = False,
                ):
        super().__init__()
        self.root = root
        self.csv_file = csv_file
        self.symbol_to_idx = symbol_to_idx
        self.idx_to_symbol = idx_to_symbol 
        self.max_len = max_len
        self.transform = transform 
        self.test = test
        self.df = pd.read_csv(self.csv_file)
        self.df = self.df[self.df.protein.isin(protein_set)]
 
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        items = self.df.iloc[idx]
        protein_name = items['protein']
        ligand = items['ligand']
        label = items["label"]
        name = protein_name + "_" + ligand
        s = sf.encoder(ligand)
        encoding = [self.symbol_to_idx[symbol] for symbol in sf.split_selfies(s)]
        if len(encoding) < self.max_len:
            ligand_tensor = torch.tensor(encoding + [self.symbol_to_idx['[nop]'] for i in range(self.max_len - len(encoding))])
        else:
            ligand_tensor = torch.tensor(encoding)
        protein_file = os.path.join(self.root, f"res_graph/{protein_name}.pdb.pt")
        protein_graph = torch.load(protein_file, map_location="cpu")
        protein_graph.y = label 
        
        if self.test:
            return ligand_tensor, protein_graph, int(label), ligand, protein_name
        return ligand_tensor, protein_graph, int(label)
    
class VAECollate():
    def __init__(self, test = False):
        self.test = test

    def __call__(self, data_list):
        ligand_tensor = torch.stack([d[0] for d in data_list], dim = 0)
        protein_graph = Batch.from_data_list([d[1] for d in data_list])
        return ligand_tensor, protein_graph