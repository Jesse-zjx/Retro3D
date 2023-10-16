import re
from rdkit import Chem
import numpy as np

from rdchiral.template_extractor import mols_from_smiles_list, replace_deuterated, get_changed_atoms


def randomize_smiles_with_am(smi):
    """Randomize a SMILES with atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    random_root = np.random.choice([(atom.GetIdx()) for atom in mol.GetAtoms()])
    return Chem.MolToSmiles(mol, rootedAtAtom=int(random_root))


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


def clear_map_rooted_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=int(root), canonical=canonical)
    else:
        return smi


def get_rooted_prod(atommap_smi, random=False):
    atommap_mol = Chem.MolFromSmiles(atommap_smi)
    root = -1
    if random:
        root = np.random.choice([(atom.GetIdx()) for atom in atommap_mol.GetAtoms()])
    rooted_smi = clear_map_rooted_smiles(atommap_smi, root=root)
    rooted_mol = Chem.MolFromSmiles(rooted_smi)
    root2atommapIdx = atommap_mol.GetSubstructMatch(rooted_mol)
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]
    rooted_atom_map = [id2atommap[root2atommapIdx[i]] for i in range(len(rooted_mol.GetAtoms()))]
    
    for i, atom_map in enumerate(rooted_atom_map):
        if atom_map != 0:
            rooted_mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', atom_map)
    rooted_smi_am = Chem.MolToSmiles(rooted_mol, canonical=False, doRandom=False)
    return rooted_smi_am


def get_rooted_reacts_acord_to_prod(prod_am, reacts):
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


def re_atommap(prod, reacts):
    '''
    USPTO_MIT数据集atom_map重排
    '''
    prod = sorted(prod.split(' ')[0].split('.'), key=lambda x: len(x), reverse=True)[0]
    pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", prod)))
    reacts = '.'.join([react for react in reacts.split(".") if len(set(map(int, re.findall(r"(?<=:)\d+", react))) & set(pro_atom_map_numbers)) > 0])
    rea_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", reacts)))

    atom_map_comm = list(set(rea_atom_map_numbers) & (set(pro_atom_map_numbers)))
    atom_map_dict = {}

    reacts_mol = Chem.MolFromSmiles(reacts)
    num = 1
    for atom in reacts_mol.GetAtoms():
        map_number = atom.GetIntProp('molAtomMapNumber')
        if map_number in atom_map_comm:
            atom_map_dict[map_number] = num
            atom.SetIntProp('molAtomMapNumber', num)
            num += 1
        else:
            atom.ClearProp('molAtomMapNumber')
    
    prod_mol = Chem.MolFromSmiles(prod)
    for atom in prod_mol.GetAtoms():
        map_number = atom.GetIntProp('molAtomMapNumber')
        atom.SetIntProp('molAtomMapNumber', atom_map_dict[map_number])
    
    return Chem.MolToSmiles(prod_mol), Chem.MolToSmiles(reacts_mol)


if __name__ == '__main__':
    pass
    # prod = '[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1'
    
    # reacts = 'CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[CH3:8][C:9](=[O:10])[c:11]1[cH:12][cH:13][c:14]2[nH:15][cH:16][cH:17][c:18]2[cH:19]1'
    
    # print()

    # prod = '[c:1]1([CH:8]=[O:9])[cH:2][cH:3][c:4]([Br:5])[n:6][cH:7]1'
    # rooted_prod = get_rooted_prod(prod, True)
    # print(rooted_prod)

    # print()

    # reacts = 'Br[c:1]1[cH:2][cH:3][c:4]([Br:5])[n:6][cH:7]1.CN(C)[CH:8]=[O:9]'
    # rooted_reacts = get_rooted_reacts_acord_to_prod(rooted_prod, reacts)
    # print(rooted_reacts)