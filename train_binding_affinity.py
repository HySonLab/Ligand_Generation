import os
import argparse
from tqdm import tqdm
import torch
import pytorch_lightning as pl
import json
import warnings
import torch_geometric
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd

from binding_data import PairDataset, CollaterLBA
from utils.metrics import get_metrics_reg
from models.protein_model import BindingAffinityPredictor


torch_geometric.seed_everything(12345)

warnings.filterwarnings("ignore")

def draw_r2(predictions, ground_truths, figname): 
    
    # Calculate the R2 score
    r2 = r2_score(ground_truths, predictions)

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, ground_truths, s=50, alpha=0.7)

    # Plot a diagonal line for reference (y = x)
    plt.plot([min(predictions), max(predictions)], [min(predictions), max(predictions)], 'k--', lw=2)

    # Set labels and title
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truths")
    plt.title(f"R2 Score: {r2:.3f}")

    # Display the plot
    plt.savefig(figname, transparency=True, bbox_inches = 'tight')

parser = argparse.ArgumentParser()
parser.add_argument("--root", type = str, default = "./data/")
parser.add_argument("--data", type = str, default = "kiba")
parser.add_argument("--fold_idx", type = int, default = 0)
parser.add_argument("--device", type = int, default = 0)
parser.add_argument("--num_epoch", type = int, default =200)
parser.add_argument("--chkpt_path", type = str, default = "./chkpt/")
args = parser.parse_args()

# root = os.path.join(args.root, args.data)
# csv_file = os.path.join(root, "full.csv")

root = f"/cm/shared/khangnn4/data/lba/data/{args.data}"
args.root = root
csv_file = f"/cm/shared/khangnn4/data/lba/data/{args.data}/full.csv"



df = pd.read_csv(csv_file)


test_fold = json.load(open(f"{args.root}/folds/test_fold_setting1.txt"))
folds = json.load(open(f"{args.root}/folds/train_fold_setting1.txt"))

val_fold = folds[args.fold_idx]
df_train = df[~ df.index.isin(test_fold)]
df_val = df_train[df_train.index.isin(val_fold)]
df_train = df_train[~ df_train.index.isin(val_fold)]
df_test = df[df.index.isin(test_fold)]

train_dataset = PairDataset(root = root, df = df_train)
val_dataset = PairDataset(root = root, df = df_val)
test_dataset = PairDataset(root = root, df = df_test) 


print("Number of train samples: ", len(train_dataset))
print("Number of validation sample: ", len(val_dataset))
print("Number of test samples: ", len(test_dataset))
print("Fold: ", args.fold_idx)

train_loader = DataLoader(train_dataset, batch_size=128,
                            shuffle=True, num_workers=6, collate_fn=CollaterLBA(128))
val_loader = DataLoader(val_dataset, batch_size=128,
                        shuffle=False, num_workers=6, collate_fn= CollaterLBA(128))
test_loader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, num_workers=6, collate_fn= CollaterLBA(128))

model = BindingAffinityPredictor()
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val loss",
    mode="min",
    dirpath=args.chkpt_path,
    filename=f"checkpoint_{args.data}_{args.fold_idx}",
    every_n_epochs=1
)

trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=200, logger=pl.loggers.CSVLogger('logs'),
                        enable_checkpointing=True, callbacks=[checkpoint_callback])

trainer.fit(model, train_loader, val_loader)
print(checkpoint_callback.best_model_path)
checkpoint = torch.load(checkpoint_callback.best_model_path)
model.load_state_dict(checkpoint['state_dict'])
results = trainer.test(model, dataloaders=test_loader)
predictions = torch.cat(model.predictions, dim=0).view(-1).numpy()
targets = torch.cat(model.targets, dim=0).view(-1).numpy()

draw_r2(predictions, targets, f"plot/{args.data}_{args.fold_idx}.pdf")