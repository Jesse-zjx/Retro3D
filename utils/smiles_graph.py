from rdkit import Chem
import numpy as np
import re

from .smiles_utils import smi_tokenizer


BONDTYPES = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
BONDTOI = {bond: i for i, bond in enumerate(BONDTYPES)}


class SmilesGraph:
    def __init__(self, smi, existing=None):
        if existing is not None:
            assert len(existing) == 3
            self.adjacency_matrix, self.bond_type_dict, self.bond_attributes = existing
        else:
            self.smi = smi
            self.V = len(smi_tokenizer(smi))
            self.adjacency_matrix, self.bond_type_dict, self.bond_attributes = self.construct_graph_struct(smi)
        self.adjacency_matrix_attr = np.zeros((len(self.adjacency_matrix), len(self.adjacency_matrix), 7), dtype=int)
        for i in range(len(self.adjacency_matrix)):
            for cand_j in self.adjacency_matrix[i]:
                self.adjacency_matrix_attr[i][cand_j] = self.bond_attributes[(i, cand_j)]

    def construct_graph_struct(self, smi, verbose=False):
        # V * V
        adjacency_matrix = [[] for _ in range(self.V)]
        bond_types = {}
        bond_attributes = {}
        mol = Chem.MolFromSmiles(smi)
        atom_order = [atom.GetIdx() for atom in mol.GetAtoms()]
        atom_smarts = [atom.GetSmarts() for atom in mol.GetAtoms()]

        neighbor_smiles_list = []
        neighbor_bonds_list, neighbor_bond_attr_list = [], []
        for atom in mol.GetAtoms():
            sig_neighbor_bonds = []
            sig_neighbor_bonds_attr = []
            sig_atom_smart = atom_smarts[:]
            sig_atom_smart[atom.GetIdx()] = '[{}:1]'.format(atom.GetSymbol())
            for i, neighbor_atom in enumerate(atom.GetNeighbors()):
                sig_atom_smart[neighbor_atom.GetIdx()] = '[{}:{}]'.format(neighbor_atom.GetSymbol(), 900 + i)
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_atom.GetIdx())
                sig_neighbor_bonds.append(str(bond.GetBondType()))
                sig_neighbor_bonds_attr.append(self.get_bond_feature(bond))
            neighbor_featured_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_order, canonical=False,
                                                             atomSymbols=sig_atom_smart)
            neighbor_smiles_list.append(neighbor_featured_smi)
            neighbor_bonds_list.append(sig_neighbor_bonds)
            neighbor_bond_attr_list.append(sig_neighbor_bonds_attr)
        for cur_idx, ne_smi in enumerate(neighbor_smiles_list):
            neighbor_featured_smi_tokens = smi_tokenizer(ne_smi)
            neighbor_bonds = neighbor_bonds_list[cur_idx]
            neighbor_bonds_attr = neighbor_bond_attr_list[cur_idx]
            pivot, cand_js, order = -1, [], []
            for j in range(len(neighbor_featured_smi_tokens)):
                if re.match('\[.*:1]', neighbor_featured_smi_tokens[j]):
                    pivot = j
                if re.match('\[.*:90[0-9]]', neighbor_featured_smi_tokens[j]):
                    cand_js.append(j)
                    order.append(int(re.match('\[.*:(90[0-9])]', neighbor_featured_smi_tokens[j]).group(1)) - 900)
            if pivot > -1:
                assert len(neighbor_bonds) == len(cand_js)
                neighbor_bonds = list(np.array(neighbor_bonds)[order])
                neighbor_bonds_attr = list(np.array(neighbor_bonds_attr)[order])
                if verbose:
                    print(ne_smi)
                    print(pivot, cand_js, neighbor_bonds, '\n')
                adjacency_matrix[pivot] = cand_js
                for cur_j in cand_js:
                    bond_types[(pivot, cur_j)] = BONDTOI[neighbor_bonds.pop(0)]
                    bond_attributes[(pivot, cur_j)] = neighbor_bonds_attr.pop(0)
        return adjacency_matrix, bond_types, bond_attributes

    def one_hot_vector(self, value, lst):
        if value not in lst:
            value = lst[-1]
        return map(lambda x: x == value, lst)

    def get_bond_feature(self, bond):
        attr = []
        attr += self.one_hot_vector(
            bond.GetBondTypeAsDouble(),
            [1.0, 1.5, 2.0, 3.0]
        )
        attr.append(bond.GetIsAromatic())
        attr.append(bond.GetIsConjugated())
        attr.append(bond.IsInRing())

        return np.array(attr)
