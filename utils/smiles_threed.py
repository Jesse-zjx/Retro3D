from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import re

from .smiles_utils import smi_tokenizer

# ATOMTYPES = ['C', 'H', 'O', 'N', 'S', 'Li', 'Mg', 'F', 'K', 'B', 'Cl', \
# 'I', 'Se', 'Si', 'Sn', 'P', 'Br', 'Zn', 'Cu', 'Pt', 'Fe', 'Pd', 'Pb']
# ATOMTOI = {atom: i for i, atom in enumerate(ATOMTYPES)}


class SmilesThreeD:
    def __init__(self, rooted_smi, before=None, existing=None):
        if existing is not None:
            assert len(existing) == 3
            self.atoms_coord, self.atom_token, self.atom_index = existing
        else:
            self.atoms_coord = self.get_atoms_coordinate(rooted_smi, before)
            self.atom_token, self.atom_index = self.get_atoms_dict(rooted_smi)

        V = len(smi_tokenizer(rooted_smi))
        self.dist_matrix = np.zeros((V, V), dtype=float)
        if self.atoms_coord is not None:
            atoms_coord = np.array(self.atoms_coord)
            for i, index_i in enumerate(self.atom_index):
                for j, index_j in enumerate(self.atom_index):
                    self.dist_matrix[index_i, index_j] = \
                        np.linalg.norm(atoms_coord[i]-atoms_coord[j])

        
    def get_atoms_dict(self, smi):
        atom_token = []
        atom_index = []
        for i, atom in enumerate(smi_tokenizer(smi)):
            if any(c.isalpha() for c in atom):
                atom_token.append(atom)
                atom_index.append(i)
        return atom_token, atom_index

    def get_atoms_coordinate(self, rooted_smi, before=None):
        if before is not None:
            old_smi, threed_contents = before
            atoms_coord = threed_contents[0]
            smi_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", old_smi)))
            rooted_smi_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", rooted_smi)))
            positions = [atoms_coord[smi_map_numbers.index(i)] for i in rooted_smi_map_numbers]
        else:
            mol = Chem.MolFromSmiles(rooted_smi)
            if mol.GetNumAtoms() < 2:
                return None
            mol = Chem.AddHs(mol)
            ignore_flag1 = 0
            while AllChem.EmbedMolecule(mol, randomSeed=10) == -1:
                ignore_flag1 = ignore_flag1 + 1
                if ignore_flag1 >= 20:
                    return None
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            positions = mol.GetConformer().GetPositions().tolist()
        return positions

if __name__ == '__main__':
    # smi = '[O:1]=[N+:2]([O-:3])[c:4]1[cH:5][c:6]([CH:7]=[O:8])[cH:9][cH:10][c:11]1[F:12]'
    # print(smi_tokenizer(smi))
    # smi_3d = SmilesThreeD(smi)
    # print(smi_3d.atom_token)
    # print(smi_3d.atom_index)
    # print(smi_3d.atoms_coord)

    # rooted_smi = get_rooted_smiles_with_am(smi)
    # print(smi_tokenizer(rooted_smi))
    # smi_3d = SmilesThreeD(rooted_smi, (smi, [[0,0,0]]+smi_3d.atoms_coord))
    # print(smi_3d.atom_token)
    # print(smi_3d.atom_index)
    # print(smi_3d.atoms_coord)
    pass