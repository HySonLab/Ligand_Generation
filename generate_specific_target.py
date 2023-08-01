import os
import argparse
import random
import warnings
from collections import defaultdict
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch_geometric.loader import DataLoader
import copy
from protein_utils import featurize_as_graph
from conditional_vae import ThreeD_Conditional_VAE
import json


warnings.filterwarnings('ignore')

def get_prop(prop, x):
    return torch.tensor(props[prop](x), device=device).unsqueeze(1).float()


parser = argparse.ArgumentParser()
parser.add_argument("--target_idx", type = int, default=0)
parser.add_argument("--root", type = str, default = "./data/")
parser.add_argument("--autodock_executable", type= str, default = "AutoDock-GPU/bin/autodock_gpu_128wi")
parser.add_argument("--protein_name", type = str, default = "1err")
parser.add_argument("--device", type = int, default = 0)
parser.add_argument("--num_epoch", type = int, default =200)
parser.add_argument("--chkpt_path", type = str, default = "./chkpt/")
parser.add_argument("--num_mols", type = int, default = 1000)
args = parser.parse_args()


protein_file = f"/data/{args.protein_name}/{args.protein_name}.maps.fld"
props = {'logp': one_hots_to_logp, 
         'penalized_logp': one_hots_to_penalized_logp, 
         'qed': one_hots_to_qed, 
         'sa': one_hots_to_sa, 
         'binding_affinity': lambda x: one_hots_to_affinity(x, args.autodock_executable, protein_file),
         'cycles': one_hots_to_cycles}


num_mols = args.num_mols

protein_pdb = f"/data/{args.protein_name}/{args.protein_name}.pdb"
protein_df = fo.bp_to_df(fo.read_pdb(protein_pdb))
protein_graph = featurize_as_graph(protein_df, device = "cuda")

vae = ThreeD_Conditional_VAE(max_len=72, vocab_len=108, 
          latent_dim=1024, embedding_dim=128, condition_dim = 128, checkpoint_path = None, freeze = False)
vae.load_state_dict(torch.load('/checkpoints/conditional_vae.pt', map_location = "cpu"))
vae = vae.to(device)
vae.eval()

batch_mol = 500 if num_mols >= 1000 else 100
cnt = 0
smiles = []

kds = []
qeds = []
sas = []
dockings = []
while cnt < num_mols:
    print("Generate Batch: ", cnt)
    cond = [copy.deepcopy(protein_graph) for idx in range(batch_mol)]
    cond = Batch.from_data_list(cond).to(device)    
     
    with torch.no_grad():
        x_prot = vae.protein_model((cond.node_s, cond.node_v), 
                                        cond.edge_index, (cond.edge_s, cond.edge_v), cond.seq, cond.batch)
        out = vae.prior_network(x_prot).view(-1, 2, 1024)
        prior_mu, prior_log_var = out[:, 0, :], out[:, 1, :]
        prior_std = torch.exp(prior_log_var * 0.5)
    z = torch.normal(mean = prior_mu, std = prior_std)
    x = vae.decode(z, None).cpu()
    smiles  += [one_hot_to_smiles(hot) for hot in x]
    qed = get_prop("qed", x).detach().cpu().numpy().flatten().tolist()
    sa = get_prop("sa", x).detach().cpu().numpy().flatten().tolist()
    
    affinity = get_prop("binding_affinity", x).detach().cpu().numpy().flatten().tolist()
    kd = list(map(lambda x: delta_g_to_kd(x), affinity))
    kds += kd
    dockings += affinity
    qeds += qed
    sas += sa
    cnt += batch_mol

print("Number of generated smiles: ", len(smiles))
print("Mean docking score:  ", sum(dockings) / len(dockings))
for idx in np.argpartition(kds, 6)[:6]:
    print(qeds[idx], kds[idx], sas[idx])