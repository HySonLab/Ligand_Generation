import os
import argparse
import pytorch_lightning as pl
import torch
from tqdm import tqdm
import torch_geometric
from torch.utils.data import DataLoader
import numpy as np
from binding_data import TargetDataset, VAECollate
from models.conditional_vae import ThreeD_Conditional_VAE
import json

import warnings
warnings.filterwarnings('ignore')

torch_geometric.seed_everything(1)

parser = argparse.ArgumentParser()
parser.add_argument("--root", type = str, default = "./data/")
parser.add_argument("--data", type = str)
parser.add_argument("--fold_idx", type = int, default = 0)
parser.add_argument("--feature_type", type = str, default = "default")
parser.add_argument("--device", type = int, default = 0)
parser.add_argument("--num_epoch", type = int, default =200)
parser.add_argument("--chkpt_path", type = str, default = "./checkpoints/")
args = parser.parse_args()


csv_file = os.path.join(args.root, "filter.csv")

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

train_dataset = TargetDataset(
    csv_file=csv_file, root=args.root, symbol_to_idx=symbol_to_idx,
    idx_to_symbol=idx_to_symbol, max_len=72, protein_set=train_protein
)
test_dataset = TargetDataset(
    csv_file=csv_file, root=args.root, symbol_to_idx=symbol_to_idx,
    idx_to_symbol=idx_to_symbol, max_len=72, protein_set=test_protein
)
print("Number of train protein: ", len(train_protein))
print("Number of test protein: ", len(test_protein))
print("Number of training sample: ", len(train_dataset))
print("Number of test samples: ", len(test_dataset))

loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                    num_workers=1, collate_fn=VAECollate(False))


# vae = ThreeD_Conditional_VAE(max_len=72, vocab_len=108, latent_dim=1024, embedding_dim=128, condition_dim=128, checkpoint_path="vae.pt", freeze=True)
# trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=30, logger=pl.loggers.CSVLogger('logs'),
#                      enable_checkpointing=False)

# print('Training..')
# trainer.fit(vae, loader)
print('Saving..')
torch.save(vae.state_dict(), f"{args.chkpt_path}/conditional_vae.pt")
