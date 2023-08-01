import os 
import argparse
import atom3d.util.formats as fo 
from tqdm import tqdm
import selfies as sf
import pandas as pd
import torch
from utils.protein_utils import featurize_as_graph
from utils.transform import BaseTransform


parser = argparse.ArgumentParser()
parser.add_argument("--root", type = str, default = "./data/")
parser.add_argument("--data", type = str, default = "kiba")
parser.add_argument("--device", type = str, default = "cuda:0")

args = parser.parse_args()
transform = BaseTransform(device = args.device)

if args.data == "davis":
    root = "/data/davis/pdb_file/prot_3d_for_Davis"
else:
    root = "/data/kiba/prot_3d_for_KIBA"

folders = os.listdir(root)
for folder in tqdm(folders): 
    path = os.path.join(root, folder)
    protein_df = fo.bp_to_df(fo.read_pdb(path))
    protein_graph = featurize_as_graph(protein_df)
    torch.save(protein_graph, os.path.join(root, "res_graph", f"{folder}.pt"))