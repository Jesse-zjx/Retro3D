from rdkit import Chem
import numpy as np
import re

from .smiles_utils import smi_tokenizer


BONDTYPES = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
BONDTOI = {bond: i for i, bond in enumerate(BONDTYPES)}


class SmilesThreeD:
    def __init__(self, rooted_smi, existing=None):
        if existing is not None:
            assert len(existing) == 3
            self.adjacency_matrix, self.bond_type_dict, self.bond_attributes = existing
        else:
            self.rooted_smi = rooted_smi
            self.V = len(smi_tokenizer(rooted_smi))
            self.adjacency_matrix, self.bond_type_dict, self.bond_attributes = self.construct_graph_struct(smi)
        self.adjacency_matrix_attr = np.zeros((len(self.adjacency_matrix), len(self.adjacency_matrix), 7), dtype=int)
        for i in range(len(self.adjacency_matrix)):
            for cand_j in self.adjacency_matrix[i]:
                self.adjacency_matrix_attr[i][cand_j] = self.bond_attributes[(i, cand_j)]

    def construct_dist_matrix(self, smi):
        """从分子结构图构建基于smiles token的图"""
        # V * V
        adjacency_matrix = [[] for _ in range(self.V)]
        bond_types = {}
        bond_attributes = {}
        mol = Chem.MolFromSmiles(smi)
        atom_order = [atom.GetIdx() for atom in mol.GetAtoms()]
        atom_smarts = [atom.GetSmarts() for atom in mol.GetAtoms()]

        # 标记了领居节点的smiles表达式
        neighbor_smiles_list = []
        # 存储edge(bond) 的类型，以及7个特征信息
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
            # 标记了邻居节点的smiles表达式
            neighbor_featured_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_order, canonical=False,
                                                             atomSymbols=sig_atom_smart)
            neighbor_smiles_list.append(neighbor_featured_smi)
            neighbor_bonds_list.append(sig_neighbor_bonds)
            neighbor_bond_attr_list.append(sig_neighbor_bonds_attr)
        # 构建邻接矩阵
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

    def get_atoms_coordinate(rooted_smi, smi=None, atoms_coord=None):
        if atoms_coord is not None:
            atoms_coord = atoms_coord[1:]   # 去掉pad的[0.0, 0.0, 0.0]
            smi_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", smi)))
            rooted_smi_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", rooted_smi)))
            positions = [atoms_coord[smi_map_numbers.index(i)] for i in rooted_smi_map_numbers]
        else:
            mol = Chem.MolFromSmiles(rooted_smi)
            if mol.GetNumAtoms() < 2:
                return None
            mol = Chem.AddHs(mol)
            ignore_flag1 = 0
            while Chem.AllChem.EmbedMolecule(mol, randomSeed=10) == -1:
                ignore_flag1 = ignore_flag1 + 1
                if ignore_flag1 >= 20:
                    return None
            Chem.AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            positions = mol.GetConformer().GetPositions().tolist()
        return positions