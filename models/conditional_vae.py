import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.distributions as td
from gvp.models import ThreeD_Protein_Model

def gaussian_analytical_kl(mu1, logsigma1, mu2, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)

class VAE(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(max_len * embedding_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
        
    def encode(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.encoder(x).view(-1, 2, self.latent_dim)
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, x):
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target, mu, log_var, batch_size, p):
        nll = F.nll_loss(pred, target)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * pred.shape[1])
        return (1 - p) * nll + p * kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(train_batch)
        p = 0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch.flatten(), mu, log_var, len(train_batch), p)
        self.log('train_loss', loss)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(val_batch)
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch.flatten(), mu, log_var, len(val_batch), 0.5)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss
    

class ThreeD_Conditional_VAE(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim, 
                condition_dim, checkpoint_path = None, freeze = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.condition_dim = condition_dim

        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(embedding_dim * self.max_len, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, latent_dim * 2))

        self.protein_model = ThreeD_Protein_Model(node_in_dim = (6,3), node_h_dim = (128, 32), edge_in_dim = (32, 1), edge_h_dim=(32, 1), 
                                      seq_in = True, num_layers = 3, drop_rate=0.1)
    
        self.prior_network = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze

        if self.checkpoint_path is not None:
            self.load_checkpoint()
        if self.freeze:
            self.freeze_params()
    
    def freeze_params(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_checkpoint(self):   
        params = torch.load(self.checkpoint_path)
        state_dict = self.decoder.state_dict()
        for name, p in params.items():
            if "encoder" in name:
                name = name[8:]
                self.encoder.state_dict()[name].copy_(p)
            if "decoder" in name:
                name = name[8:]
                self.decoder.state_dict()[name].copy_(p)

    def encode(self, batch):
        input_ids = batch[0]
        cond = batch[1]
        x = self.embedding(input_ids).view(len(input_ids), -1)
        x = self.encoder(x).view(-1, 2, self.latent_dim)
        x_prot = self.protein_model((cond.node_s, cond.node_v), 
                                    cond.edge_index, (cond.edge_s, cond.edge_v), cond.seq, cond.batch)
        prior = self.prior_network(x_prot).view(-1, 2, self.latent_dim)
        prior_mu, prior_log_var = prior[:, 0, :], prior[:, 1, :]
        mu, log_var = x[:, 0, :], x[:, 1, :]
        log_std = torch.exp(0.5 * log_var)
        prior_std = torch.exp(0.5 * prior_log_var)
        eps = torch.randn_like(prior_std)
        z = mu + eps * log_std
        return z, mu, log_var, prior_mu, prior_log_var

    def decode(self, z, cond):
        x = z
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, batch):
        z,  mu, log_var, prior_mu, prior_log_var = self.encode(batch)
        cond = batch[1]
        return self.decode(z, cond), z, mu, log_var, prior_mu, prior_log_var
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target, mu, log_var, prior_mu, prior_log_var, batch_size, p):
        nll = F.nll_loss(pred, target)
        #nll  = 0
        kld = gaussian_analytical_kl(mu, log_var, prior_mu, prior_log_var).sum() / (batch_size * pred.shape[1])
        return (1-p)*nll + p*kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var, prior_mu, prior_log_var = self(train_batch)
        p = 0.1
        input_ids = train_batch[0].view(-1)
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), input_ids,mu, log_var, prior_mu, prior_log_var, len(train_batch), p)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var, prior_mu, prior_log_var = self(val_batch)
        input_ids = val_batch[0].view(-1)
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), input_ids,mu, log_var, prior_mu, prior_log_var, len(val_batch), 0.5)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss