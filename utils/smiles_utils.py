import re
from rdkit import Chem
import numpy as np

from rdchiral.template_extractor import mols_from_smiles_list, replace_deuterated, get_changed_atoms


def get_rooted_smiles_with_am(smi, randomChoose=False):
    mol = Chem.MolFromSmiles(smi)
    if randomChoose:
        random_root = np.random.choice([atom.GetIdx() for atom in mol.GetAtoms()])
    else:
        random_root = -1
    random_root = int(random_root)
    rooted_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True,
                                  rootedAtAtom=random_root)
    return rooted_smi


def get_rooted_reacts_acord_to_prod(prod_am, reacts):
    mol = Chem.MolFromSmiles(prod_am)
    reacts = reacts.split('.')
    cand_order = []
    cands = []
    reacts_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reacts]

    for i, react_map_num in enumerate(reacts_map_numbers):
        for j, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomMapNum() in react_map_num:
                rea_mol = Chem.MolFromSmiles(reacts[i])
                # 找到reacts中的root
                rea_am2id = {}
                for a in rea_mol.GetAtoms():
                    if a.HasProp('molAtomMapNumber'):
                        rea_am2id[int(a.GetProp('molAtomMapNumber'))] = a.GetIdx()
                root_id = rea_am2id[atom.GetAtomMapNum()]
                rea_smi_am = Chem.MolToSmiles(rea_mol, isomericSmiles=True, canonical=True, rootedAtAtom=root_id)
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


def remove_am_without_canonical(smi_am):
    """Get the canonical SMILES by token modification (smiles arranged by CanonicalRankAtoms)
    :param smi_am: SMILES from `canonical_smiles_with_am`
    :return:
    """

    def check_special_token(token):
        pattern = "(Mg|Zn|Si|Sn|Se|se|Ge|K|Ti|Pd|Mo|Ce|Ta|As|te|Pb|Ru|Ag|W|Pt|Co|Ca|Xe|11CH3|Rh|Tl|V|131I|Re|13c|siH|La|pH|Y|Zr|Bi|125I|Sb|Te|Ni|Fe|Mn|Cr|Al|Na|Li|Cu|nH[0-9]?|NH[1-9]?\+|\+|-|@|PH[1-9]?)"
        regex = re.compile(pattern)
        return regex.findall(token)

    new_tokens = []
    for token in smi_tokenizer(smi_am):
        # Has atommapping:
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            # print
            token = token.replace(re.match('.*(:[0-9]+)]', token).group(1), '')
            explicitHs = re.match('.*(H[1-9]?).*', token)
            onlyH = re.match('\[[1-9]?H', token)
            if explicitHs and not check_special_token(token) and not onlyH:
                token = token.replace(explicitHs.group(1), '')[1:-1]
            elif not check_special_token(token) and not onlyH:
                token = token[1:-1]
            else:
                token = token
        new_tokens.append(token)

    canonical_smi = ''.join(new_tokens)
    return canonical_smi


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


def get_nonreactive_mask(cano_prod_am, prod, reacts, radius=0):
    reactants = mols_from_smiles_list(replace_deuterated(reacts).split('.'))
    products = mols_from_smiles_list(replace_deuterated(prod).split('.'))
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
    # 找到radius范围内的只需要循环radius遍即可
    for _ in range(radius):
        mol = Chem.MolFromSmiles(cano_prod_am)
        changed_neighbor = []
        for atom in mol.GetAtoms():
            if atom.GetSmarts().split(':')[1][:-1] in changed_atom_tags:
                for neighbor in atom.GetNeighbors():
                    changed_neighbor.append(neighbor.GetSmarts().split(':')[1][: -1])
        changed_atom_tags = list(set(changed_neighbor + changed_atom_tags))

    nonreactive_mask = []
    for i, token in enumerate(smi_tokenizer(cano_prod_am)):
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            tag = re.match('.*:([0-9]+)]', token).group(1)
            if tag in changed_atom_tags:
                nonreactive_mask.append(False)
                continue
        nonreactive_mask.append(True)

    # 如果没有找到反应中心
    if sum(nonreactive_mask) == len(nonreactive_mask):
        nonreactive_mask = [False] * len(nonreactive_mask)
    return nonreactive_mask


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
