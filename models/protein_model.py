import torch 
import torch.nn as nn 
import torch.nn.functional as F
from gvp.models import ThreeD_Protein_Model
import pytorch_lightning as pl
from linear_attention_transformer import LinearAttentionTransformerLM, LinformerSettings
from torch_geometric.nn.models import PNA, GAT
from torch_geometric.nn import global_mean_pool, global_add_pool

class Ligand_Graph_Model(nn.Module):
    def __init__(self, hidden_dim = 256):
        super().__init__()
        self.gnn = GAT(in_channels=23, hidden_channels=hidden_dim, dropout=0.1, norm="layer",
                    num_layers = 4, out_channels=hidden_dim, v2 = True, edge_dim = 6, jk = "last")
        
    def forward(self, batch):
        x = self.gnn(batch.x, batch.edge_index, edge_attr = batch.edge_attr)
        return global_mean_pool(x, batch.batch)


class BindingAffinityPredictor(pl.LightningModule):
    def __init__(self, hidden_dim = 128, model_type = "protein_model"):
        super().__init__()

        if model_type == "transfomer_gvp":
            self.protein_model = TransformerGVP(node_in_dim = (6,3), node_h_dim = (128, 32), edge_in_dim = (32, 1), edge_h_dim=(32, 1), 
                                      seq_in = True, num_layers = 3, drop_rate=0.1, attention_type="performer")
        else:
            self.protein_model = ThreeD_Protein_Model(node_in_dim = (6,3), node_h_dim = (128, 32), edge_in_dim = (32, 1), edge_h_dim=(32, 1), 
                                      seq_in = True, num_layers = 3, drop_rate=0.1, attention_type="performer")
        
        self.ligand_model = Ligand_Graph_Model(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.ReLU()

        )

        self.ligand_mp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
        )
        self.predictions = []
        self.targets = []

    def forward(self, batch):
        protein_batch, ligand_batch, ligand_mp = batch
        x_prot = self.protein_model((protein_batch.node_s, protein_batch.node_v), 
                                    protein_batch.edge_index, (protein_batch.edge_s, protein_batch.edge_v), protein_batch.seq,protein_batch.batch)
        x_ligand =self.ligand_model(ligand_batch)
        x_mp = self.ligand_mp(ligand_mp)
        x = torch.cat([x_prot, x_ligand, x_mp], dim = -1)
        return self.fc(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience=20, verbose = True)
        return {'optimizer': optimizer, 'scheduler' : scheduler}
    
    def loss_function(self, pred, target):
        mse = F.mse_loss(pred, target)
        return mse
    
    def training_step(self, train_batch, batch_idx):
        out = self(train_batch)
        loss = self.loss_function(out.view(-1), train_batch[0].y.float().view(-1))
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        out = self(val_batch)
        loss = self.loss_function(out.view(-1), val_batch[0].y.float().view(-1))
        self.log("val loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss
    
    def reset_predictions(self):
        self.predictions = []
    
    @torch.no_grad()
    def test_step(self, test_batch, batch_idx):
        out = self(test_batch) 
        y_true = out.cpu()
        y_pred = test_batch[0].y.cpu()

        self.predictions.append(y_true)
        self.targets.append(y_pred)
        results = get_metrics_reg(y_true, y_pred, with_rm2=True, with_ci = True)
        self.log("MSE", results["mse"], prog_bar=True, logger=True)
        self.log("RM2", results["rm2"])
        self.log("CI", results["ci"])
        return results