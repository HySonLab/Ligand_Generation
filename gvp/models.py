import numpy as np
import torch
import torch.nn as nn
from . import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from performer_pytorch import Performer, PerformerLM
import torch_geometric
from torch_geometric.utils import to_dense_batch
from linear_attention_transformer import LinearAttentionTransformerLM, LinformerSettings
from performer_pytorch import PerformerLM

class TransformerGVP(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1, attention_type = "performer"):
        
        super().__init__()
        
        if seq_in:
            self.W_s = nn.Embedding(20, 64)
            node_in_dim = (node_in_dim[0], node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.W_in = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (node_h_dim[0], 0), vector_gate=True)
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
            
        self.attention_type = attention_type
       
        if attention_type == "performer":
            self.transformer = Performer(
                            dim = ns,
                            depth = 2,
                            heads = 4,
                            dim_head = ns // 4, 
                            causal = False
                        )
        else:
            layer = nn.TransformerEncoderLayer(ns, 4, ns * 2, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, 2)

        self.final_readout = nn.Sequential(
            nn.Linear(ns + ns, 128), nn.ReLU(), nn.Linear(128, 128)
        )
        self.seq_transformer = LinearAttentionTransformerLM(
                        num_tokens = 20,
                        dim = 128,
                        heads = 8,
                        depth = 2,
                        max_seq_len = 640,
                        return_embeddings=True,
                        linformer_settings = LinformerSettings(256))
        
    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if seq is not None:
            #seq = self.W_s(seq)
            seq, mask = to_dense_batch(seq, batch, max_num_nodes=640)
            seq_emb = self.seq_transformer(seq)
            seq_rep = torch.sum(seq_emb, dim = 1)
        
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        h_t = self.W_in(h_V)
        h_t, mask = to_dense_batch(h_t, batch)
        h_t = self.transformer(h_t)
        h_t = h_t[mask]

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        node_rep = torch.cat([h_t, out], dim = -1)
        node_rep = self.final_readout(node_rep)
        
        geo_rep =  scatter_mean(node_rep, batch, dim = 0)
        return torch.cat([geo_rep, seq_rep], dim = -1)

class ThreeD_Protein_Model(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.5, attention_type = "performer"):
        
        super().__init__()
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0], node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0), vector_gate=True))
            
        self.attention_type = attention_type
        if attention_type == "performer":
            self.transformer = Performer(
                            dim = ns,
                            depth = 2,
                            heads = 4,
                            dim_head = ns // 4, 
                            causal = False
                        )
        else:
            layer = nn.TransformerEncoderLayer(ns, 4, ns * 2, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, 2)

        self.seq_transformer = LinearAttentionTransformerLM(
                        num_tokens = 20,
                        dim = 128,
                        heads = 4,
                        depth = 4,
                        dim_head = 128 // 4,
                        max_seq_len = 640,
                        return_embeddings=True,
                        linformer_settings = LinformerSettings(256), 
                        ff_dropout=drop_rate, 
                        attn_dropout=drop_rate,
                        attn_layer_dropout=drop_rate)
        
        self.skip_connection = nn.Sequential(nn.Linear(ns * 2, ns), nn.ReLU(), nn.Linear(ns, ns))

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if seq is not None:
            seq, mask = to_dense_batch(seq, batch, max_num_nodes=640)
            seq_emb = self.seq_transformer(seq)
            seq_rep = torch.mean(seq_emb, dim = 1)
        
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
       
        x, mask = to_dense_batch(out, batch)
        x_o = self.transformer(x)
        x = torch.cat([x, x_o], dim = -1)
        x = self.skip_connection(x)
        geo_rep = x.mean(dim = 1)
        if seq is not None:
            z = torch.cat([geo_rep, seq_rep], dim = -1)
            return z
        return geo_rep