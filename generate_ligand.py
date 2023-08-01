import os
import argparse
import random
import warnings
import torch
from torch_geometric.data import Batch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import json

from binding_data import TargetDataset, VAECollate
from utils.transform import BaseTransform
from models.conditional_vae import ThreeD_Conditional_VAE
from utils.chem_evaluator import diversity, kl_divergence, uniqueness, validity, novelty, fcd_distance_torch
from utils.mol_utils import *


warnings.filterwarnings('ignore')

def get_prop(prop, x):
    return torch.tensor(props[prop](x), device=device).unsqueeze(1).float()




parser = argparse.ArgumentParser()
parser.add_argument("--target_idx", type = int, default=0)
parser.add_argument("--root", type = str, default = "./data/")
parser.add_argument("--autodock_executable", type= str, default = "AutoDock-GPU/bin/autodock_gpu_128wi")
parser.add_argument("--device", type = int, default = 0)
parser.add_argument("--num_epoch", type = int, default =200)
parser.add_argument("--chkpt_path", type = str, default = "./chkpt/")
parser.add_argument("--num_mols", type = int, default = 1000)
args = parser.parse_args()

props = {'logp': one_hots_to_logp, 
         'penalized_logp': one_hots_to_penalized_logp, 
         'qed': one_hots_to_qed, 
         'sa': one_hots_to_sa, 
         'binding_affinity': lambda x: one_hots_to_affinity(x, args.autodock_executable, args.protein_file),
         'cycles': one_hots_to_cycles}

csv_file = os.path.join(args.root, "filter.csv")


df = pd.read_csv(csv_file)

train_smiles = df['ligand'].to_list()
    
with open("symbol_to_idx.json", "r") as f:
    symbol_to_idx = json.load(f)
with open("idx_to_symbol.json", "r") as f:
    idx_to_symbol = json.load(f)

protein = os.listdir(os.path.join(args.root, "res_graph"))
protein = list(map(lambda x: x.replace(".pdb.pt", ""), protein))
protein = sorted(protein)
num_train = int(len(protein) * 0.9)
train_protein = protein[:num_train]
test_protein = protein[num_train:]

idx = args.target_idx
protein_name = test_protein[idx]

test_protein = [protein_name]
ref_smiles = df[df['protein'] == protein_name].ligand.to_list()


test_dataset = TargetDataset(
    csv_file=csv_file, root=args.root, symbol_to_idx=symbol_to_idx,
    idx_to_symbol=idx_to_symbol, max_len=72, protein_set=test_protein, test = True
)

vae = ThreeD_Conditional_VAE(max_len=72, vocab_len=108, 
          latent_dim=1024, embedding_dim=128, condition_dim = 128, checkpoint_path = None, freeze = False).cuda()
vae.load_state_dict(torch.load('checkpoints/conditional_vae.pt'))
vae.eval()


batch_mol = 1000 if args.num_mols >= 1000 else args.num_mols
cnt = 0
smiles = []

while cnt < args.num_mols:
    print("Generate Batch: ", cnt)
    cond = [test_dataset[0][1] for idx in range(batch_mol)]
    cond = Batch.from_data_list(cond).cuda() 
    with torch.no_grad():
        x_prot = vae.protein_model((cond.node_s, cond.node_v), 
                                        cond.edge_index, (cond.edge_s, cond.edge_v), cond.seq, cond.batch)
        out = vae.prior_network(x_prot).view(-1, 2, 1024)
        prior_mu, prior_log_var = out[:, 0, :], out[:, 1, :]
        prior_std = torch.exp(prior_log_var * 0.5)
    z = torch.normal(mean = prior_mu, std = prior_std).cuda()
    x = vae.decode(z, x_prot).cpu()
    smiles  += [one_hot_to_smiles(hot) for hot in x]
    cnt += batch_mol

print("Number of generated smiles: ", len(smiles))

diversity_score = diversity(smiles)
uniqueness_score = uniqueness(smiles)
novelty_score = novelty(smiles, train_smiles)
validity_score = validity(smiles)
fcd_distance = fcd_distance_torch(smiles, ref_smiles)

print("Diversity: ", diversity_score)
print("Uniqueness: ", uniqueness_score)
print("Validity: ", validity_score)
print("Novelty: ", novelty_score)
print("FCD Distance: ", fcd_distance)


gen_mols = [Chem.MolFromSmiles(smi) for smi in smiles]

qed = get_prop("qed", x).detach().cpu().numpy().flatten()
sa = get_prop("sa", x).detach().cpu().numpy().flatten().tolist()
logp = get_prop("penalized_logp", x).detach().cpu().numpy().flatten().tolist()


qed_s = sorted(qed.tolist(), reverse = True)
sa = sorted(sa)
logp = sorted(logp, reverse = True)

print("Top 6 QED: ", qed_s[:6])
print("Top 6 SA: ", sa[:6])
print("Top 6 logP: ", logp[:6])