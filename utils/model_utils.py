import dgl
import torch
import numpy as np
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return one_of_k_encoding(atom.GetSymbol(),
                             ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'B', 'Unknown']) \
           + one_of_k_encoding(atom.GetDegree(), list(range(7))) \
           + one_of_k_encoding(atom.GetImplicitValence(), list(range(7))) \
           + one_of_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) \
           + [atom.GetIsAromatic()]


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()], dtype=np.float)


def create_graph(smiles, device):

    mol = Chem.MolFromSmiles(smiles)

    # Extract atom features (node features)
    atoms = mol.GetAtoms()
    atom_tensor = np.zeros((len(atoms), 32))
    for atoms_ix, atom in enumerate(atoms):
        atom_tensor[atoms_ix, :] = atom_features(atom)
    atom_tensor = torch.from_numpy(atom_tensor).double()

    # Extract bond features (edge features)
    bonds = mol.GetBonds()
    bond_tensor = []
    for bond in bonds:
        begin_idx = bond.GetBeginAtom().GetIdx()
        end_idx = bond.GetEndAtom().GetIdx()
        features = np.array(bond_features(bond))
        bond_tensor.append([begin_idx, end_idx, features])
        bond_tensor.append([end_idx, begin_idx, features])

    bond_tensor.sort()

    bond_idx1 = [t[0] for t in bond_tensor]
    bond_idx2 = [t[1] for t in bond_tensor]
    bond_tensor = [t[2] for t in bond_tensor]
    bond_tensor = torch.from_numpy(np.array(bond_tensor)).double()

    G = dgl.DGLGraph().to(device)

    # Add N nodes
    G.add_nodes(len(atom_tensor))
    # Add edges
    G.add_edges(bond_idx1, bond_idx2)

    # Add node features
    G.ndata['x'] = atom_tensor.to(device)
    # Add edge features
    G.edata['y'] = bond_tensor.to(device)

    return G
