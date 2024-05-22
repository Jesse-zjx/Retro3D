import re
from rdkit import Chem
import numpy as np


# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# console = logging.StreamHandler()
# console.setLevel(logging.WARNING)
# logger.addHandler(console)
# fileter = logging.FileHandler('log.txt')
# fileter.setLevel(logging.INFO)
# logger.addHandler(fileter)

def canonical_smiles_with_am(smi):
    """Canonicalize a SMILES with atom mapping"""
    atomIdx2am, pivot2atomIdx = {}, {}
    mol = Chem.MolFromSmiles(smi)
    atom_ordering = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atomIdx2am[atom.GetIdx()] = atom.GetProp('molAtomMapNumber')
            atom.ClearProp('molAtomMapNumber')
        else:
            atomIdx2am[atom.GetIdx()] = '0'
        atom_ordering.append(atom.GetIdx())
    unmapped_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_ordering, canonical=False)
    mol = Chem.MolFromSmiles(unmapped_smi)
    cano_atom_ordering = list(Chem.CanonicalRankAtoms(mol))
    for i, j in enumerate(cano_atom_ordering):
        pivot2atomIdx[j + 1] = i
        mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', j + 1)
    new_tokens = []
    for token in smi_tokenizer(Chem.MolToSmiles(mol)):
        if re.match('.*:([0-9]+)]', token):
            pivot = re.match('.*(:[0-9]+])', token).group(1)
            token = token.replace(pivot, ':{}]'.format(atomIdx2am[pivot2atomIdx[int(pivot[1:-1])]]))
        new_tokens.append(token)
    canonical_smi = ''.join(new_tokens)
    # canonical reactants order
    if '.' in canonical_smi:
        canonical_smi_list = canonical_smi.split('.')
        canonical_smi_list = sorted(canonical_smi_list, key=lambda x: (len(x), x))
        canonical_smi = '.'.join(canonical_smi_list)
    return canonical_smi


def clear_map_smiles(smi, canonical=False, randomize=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, doRandom=randomize, canonical=canonical)
    else:
        return smi


def get_cooked_smi(atommap_smi, randomize=False):
    """
    get cooked[canonical/random] smiles (with, without) atom-map
    """
    if '.' in atommap_smi:
        atommap_smi_list = atommap_smi.split('.')
        cooked_smi_am_list = []
        for smi in atommap_smi_list:
            cooked_smi_am = get_cooked_smi(smi)
            cooked_smi_am_list.append(cooked_smi_am)
        # repermute reacts by specific probability(np.random.rand()) if randomize
        cooked_smi_am_list = sorted(cooked_smi_am_list, key=lambda x: len(x), reverse=(randomize and np.random.rand() > 0.5))
        cooked_smi_am = '.'.join(cooked_smi_am_list)
    else:
        atommap_mol = Chem.MolFromSmiles(atommap_smi)
        cooked_smi = clear_map_smiles(atommap_smi, canonical=True, randomize=randomize)
        cooked_mol = Chem.MolFromSmiles(cooked_smi)
        cooked2atommapIdx = atommap_mol.GetSubstructMatch(cooked_mol)
        id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]
        try:
            cooked_atom_map = [id2atommap[cooked2atommapIdx[i]] for i in range(len(cooked_mol.GetAtoms()))]
            for i, atom_map in enumerate(cooked_atom_map):
                # if atom_map != 0:
                cooked_mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', atom_map)
            cooked_smi_am = Chem.MolToSmiles(cooked_mol, isomericSmiles=True, canonical=False)
        except:
            # logger.info(atommap_smi)
            cooked_smi_am = atommap_smi
    return cooked_smi_am


def get_rooted_reacts_acord_to_prod(reacts, prod_am):
    reacts = reacts.split('.')
    cand_order = []
    cands = []
    prod_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", prod_am)))
    reacts_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reacts]

    for i, react_map_num in enumerate(reacts_map_numbers):
        for j, prod_atom_map_num in enumerate(prod_map_numbers):
            if prod_atom_map_num in react_map_num:
                # 找到reacts中的root
                rea_mol = Chem.MolFromSmiles(reacts[i])
                for atom in rea_mol.GetAtoms():
                    if atom.GetAtomMapNum() == prod_atom_map_num:
                        root_id = atom.GetIdx()
                        break
                rea_smi_am = Chem.MolToSmiles(rea_mol, isomericSmiles=True, rootedAtAtom=int(root_id), canonical=True)
                cands.append(rea_smi_am)
                cand_order.append(j)
                break
    sorted_reactants = sorted(list(zip(cands, cand_order)), key=lambda x: x[1])
    cands = [item[0] for item in sorted_reactants]
    reacts_am = '.'.join(cands)
    return reacts_am


def smi_tokenizer(smi):
    """Tokenize a SMILES sequence or reaction"""
    pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print('ERROR:', smi, ''.join(tokens))
    assert smi == ''.join(tokens)
    return tokens


def get_context_alignment(prod, reacts):
    prod_tokens = smi_tokenizer(prod)
    reacts_tokens = smi_tokenizer(reacts)
    prod_token2idx = {}

    for i, token in enumerate(prod_tokens):
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            am = int(re.match('.*:([0-9]+)]', token).group(1))
            prod_token2idx[am] = i
        else:
            prod_token2idx[token] = prod_token2idx.get(token, []) + [i]
    # 反应物中的位置对应产物中的位置
    context_alignment = []
    for i, token in enumerate(reacts_tokens):
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            am = int(re.match('.*:([0-9]+)]', token).group(1))
            pivot = prod_token2idx.get(am, -1)
            if pivot != -1:
                if (i, pivot) not in context_alignment:
                    context_alignment.append((i, pivot))
            # 向前向后遍历
            i_cursor = i + 1
            pivot_cursor = pivot + 1
            while i_cursor < len(reacts_tokens) and pivot_cursor < len(prod_tokens) and (
                    i_cursor, pivot_cursor) not in context_alignment:
                if reacts_tokens[i_cursor] == prod_tokens[pivot_cursor]:
                    context_alignment.append((i_cursor, pivot_cursor))
                    i_cursor += 1
                    pivot_cursor += 1
                else:
                    break

            i_cursor = i - 1
            pivot_cursor = pivot - 1
            while i_cursor > -1 and pivot_cursor > -1 and (i_cursor, pivot_cursor) not in context_alignment:
                if reacts_tokens[i_cursor] == prod_tokens[pivot_cursor]:
                    context_alignment.append((i_cursor, pivot_cursor))
                    i_cursor -= 1
                    pivot_cursor -= 1
                else:
                    break
    return context_alignment
