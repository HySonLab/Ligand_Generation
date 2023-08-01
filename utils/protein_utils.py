import json
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster

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

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}

@torch.no_grad()
def featurize_as_graph(protein, num_rbf = 16, device = "cuda"):
    name = protein['name']
    N_coords = protein[protein.name == 'N'][['x', 'y', 'z']].to_numpy()
    CA_coords =protein[protein.name == 'CA'][['x', 'y', 'z']].to_numpy()
    C_coords = protein[protein.name == 'C'][['x', 'y', 'z']].to_numpy()
    O_coords = protein[protein.name == 'O'][['x', 'y', 'z']].to_numpy()
    max_ = min([CA_coords.shape[0], C_coords.shape[0], N_coords.shape[0], O_coords.shape[0]])
    N_coords = N_coords[:max_, :]
    CA_coords = CA_coords[:max_, :]
    C_coords = C_coords[:max_, :]
    O_coords = O_coords[:max_, :]
    coords = np.stack((N_coords, CA_coords,C_coords, O_coords), axis = 1)
    #print(coords.shape)
    with torch.no_grad():
        coords = torch.from_numpy(coords).float().to(device)
        seq = torch.as_tensor([_amino_acids(a) for a in protein[protein.name == 'CA']['resname'][:max_]],
                                dtype=torch.long).to(device)
        #print(coords.shape)
        mask = torch.isfinite(coords.sum(dim=(1,2)))
        coords[~mask] = np.inf
        
        X_ca = coords[:, 1]
        #print(X_ca.shape)
        #X_ca = coords
        edge_index = torch_cluster.radius_graph(X_ca, r = 5.0)
        #edge_index = torch_cluster.knn_graph(X_ca, k=30)
        
        pos_embeddings = positional_embeddings(edge_index)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf, device = device)
        
        dihedrals = get_dihedrals(coords)                     
        orientations = get_orientations(X_ca)
        sidechains = get_sidechains(coords)
        
        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                (node_s, node_v, edge_s, edge_v))
        
    data = torch_geometric.data.Data(x=X_ca, seq = seq,
                                        node_s=node_s, node_v=node_v,
                                        edge_s=edge_s, edge_v=edge_v,
                                        edge_index=edge_index) #mask=mask)
    return data
                            
def get_dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def positional_embeddings(edge_index, 
                            num_embeddings=16,
                            period_range=[2, 1000], device ="cuda"):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device = device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def get_sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec 