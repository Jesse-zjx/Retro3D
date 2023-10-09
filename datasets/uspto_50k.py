import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
from rdkit import Chem
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset

from utils.smiles_graph import SmilesGraph
from utils.smiles_threed import SmilesThreeD
from utils.smiles_utils import get_context_alignment, get_nonreactive_mask, \
                                smi_tokenizer, remove_am_without_canonical
from utils.smiles_utils import canonical_smiles_with_am, randomize_smiles_with_am, \
                                get_rooted_prod, get_rooted_reacts_acord_to_prod


class USPTO_50K(Dataset):
    def __init__(self, config, mode, rank=-1):
        self.root = config.DATASET.ROOT
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.augment = config.DATASET.AUGMENT
        self.known_class = config.DATASET.KNOWN_CLASS
        self.shared_vocab = config.DATASET.SHARED_VOCAB
        if rank < 1:
            print('Building {} data from: {}'.format(mode, self.root))
        self.vocab_file = ''
        if self.shared_vocab:
            self.vocab_file += 'vocab_share.pk'
        else:
            self.vocab_file += 'vocab.pk'
        
        # Build and load vocabulary
        if self.vocab_file in os.listdir(self.root):
            with open(os.path.join(self.root, self.vocab_file), 'rb') as f:
                self.src_i2t, self.tgt_i2t = pickle.load(f)
            self.src_t2i = {self.src_i2t[i]: i for i in range(len(self.src_i2t))}
            self.tgt_t2i = {self.tgt_i2t[i]: i for i in range(len(self.tgt_i2t))}
        else:
            if rank < 1:
                print('Building vocab...')
            train_data = pd.read_csv(os.path.join(self.root, 'raw_train.csv'))
            val_data = pd.read_csv(os.path.join(self.root, 'raw_val.csv'))
            raw_data = pd.concat([val_data, train_data])
            raw_data.reset_index(inplace=True, drop=True)
            self.build_vocab_from_raw_data(raw_data)

        self.data = pd.read_csv(os.path.join(self.root, 'raw_{}.csv'.format(mode)))
        if config.DATASET.SAMPLE:
            self.data = self.data.sample(n=100, random_state=0)
            self.data.reset_index(inplace=True, drop=True)

        # Build and load processed data into lmdb
        self.processed_data = []
        if 'cooked_{}.lmdb'.format(self.mode) not in os.listdir(self.root):
            self.build_processed_data(self.data)
        self.env = lmdb.open(os.path.join(self.root, 'cooked_{}.lmdb'.format(self.mode)),
                             max_readers=1, readonly=True, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))
            for key in self.product_keys:
                self.processed_data.append(pickle.loads(txn.get(key)))

    def build_vocab_from_raw_data(self, raw_data):
        reactions = raw_data['reactants>reagents>production'].to_list()
        prods, reacts = [], []
        for i in tqdm(range(len(reactions))):
            rxn = reactions[i]
            r, p = rxn.split('>>')
            if not r or not p:
                continue
            src, tgt = self.parse_smi(p, r, '<UNK>', build_vocab=True)
            if Chem.MolFromSmiles(src) is None or Chem.MolFromSmiles(tgt) is None:
                continue
            prods.append(src)
            reacts.append(tgt)

        if self.shared_vocab:  # Shared src and tgt vocab
            i2t = set()
            for i in range(len(prods)):
                i2t.update(smi_tokenizer(prods[i]))
                i2t.update(smi_tokenizer(reacts[i]))
            i2t.update(['<RX_{}>'.format(i) for i in range(1, 11)])
            i2t.add('<UNK>')
            i2t = ['<unk>', '<pad>', '<sos>', '<eos>'] + sorted(list(i2t))
            self.src_i2t, self.tgt_i2t = i2t, i2t
        else:  # Non-shared src and tgt vocab
            src_i2t, tgt_i2t = set(), set()
            for i in range(len(prods)):
                src_i2t.update(smi_tokenizer(prods[i]))
                tgt_i2t.update(smi_tokenizer(reacts[i]))
            src_i2t.update(['<RX_{}>'.format(i) for i in range(1, 11)])
            src_i2t.add('<UNK>')
            self.src_i2t = ['<unk>', '<pad>'] + sorted(list(src_i2t))
            self.tgt_i2t = ['<unk>', '<pad>', '<sos>', '<eos>'] + sorted(list(tgt_i2t))
        with open(os.path.join(self.root, self.vocab_file), 'wb') as f:
            pickle.dump([self.src_i2t, self.tgt_i2t], f)
        self.src_t2i = {self.src_i2t[i]: i for i in range(len(self.src_i2t))}
        self.tgt_t2i = {self.tgt_i2t[i]: i for i in range(len(self.tgt_i2t))}
        return

    def build_processed_data(self, raw_data):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['reactants>reagents>production'].to_list()
        multi_data = []
        for i in range(len(reactions)):
            r, p = reactions[i].split('>>')
            rt = '<RX_{}>'.format(raw_data['class'][i]) if self.known_class else '<UNK>'
            multi_data.append({"reacts":r, "prod":p, "class":rt})
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = list(tqdm(pool.imap(func=self.parse_smi_wrapper, iterable=multi_data)))
        pool.close()
        pool.join()

        env = lmdb.open(os.path.join(self.root, 'cooked_{}.lmdb'.format(self.mode)),
                        map_size=1099511627776)
        with env.begin(write=True) as txn:
            for i, result in enumerate(tqdm(results)):
                if result is not None:
                    p_key = '{} {}'.format(i, remove_am_without_canonical(result['rooted_product']))
                    try:
                        txn.put(p_key.encode(), pickle.dumps(result))
                    except Exception as e:
                        continue
        return

    def parse_smi_wrapper(self, react_dict):
        prod, reacts, react_class = react_dict['prod'], react_dict['reacts'], react_dict['class'] 
        if not prod or not reacts:
            return None
        return self.parse_smi(prod, reacts, react_class, build_vocab=False, randomize=False)

    def parse_smi(self, prod, reacts, react_class, build_vocab=False, randomize=False):
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # rooted_prod_am = get_rooted_prod(prod)
        # rooted_reacts_am = get_rooted_reacts_acord_to_prod(rooted_prod_am, reacts)
        rooted_prod_am = canonical_smiles_with_am(prod)
        rooted_reacts_am = canonical_smiles_with_am(reacts)
        rooted_prod = remove_am_without_canonical(rooted_prod_am)
        rooted_reacts = remove_am_without_canonical(rooted_reacts_am)

        if build_vocab:
            return rooted_prod, rooted_reacts
        
        if Chem.MolFromSmiles(rooted_prod) is None or Chem.MolFromSmiles(rooted_reacts) is None:
            return None
        
        # Get the smiles 3d
        before = None
        if randomize:
            # rooted_prod_am = get_rooted_prod(prod, randomize)
            # rooted_reacts_am = get_rooted_reacts_acord_to_prod(rooted_prod_am, reacts)
            # rooted_prod = remove_am_without_canonical(rooted_prod_am)
            # rooted_reacts = remove_am_without_canonical(rooted_reacts_am)
            rooted_prod_am = randomize_smiles_with_am(rooted_prod_am)
            rooted_prod = remove_am_without_canonical(rooted_prod_am)
            if np.random.rand() > 0.5:
                rooted_reacts_am = '.'.join(rooted_reacts_am.split('.')[::-1])
                rooted_reacts = remove_am_without_canonical(rooted_reacts_am)

            before = (prod, self.processed['threed_contents'])
        smiles_threed = SmilesThreeD(rooted_prod_am, before=before) 

        if smiles_threed.atoms_coord is None:
            return None

        # Get the smiles graph
        smiles_graph = SmilesGraph(rooted_prod)
        # Get the nonreactive mask
        nonreactive_mask = get_nonreactive_mask(rooted_prod_am, prod, reacts, radius=1)
        # Get the context alignment based on atom-mapping
        context_alignment = get_context_alignment(rooted_prod_am, rooted_reacts_am)

        context_attn = torch.zeros((len(smi_tokenizer(rooted_reacts_am))+1, len(smi_tokenizer(rooted_prod_am))+1)).long()
        for i, j in context_alignment:
            context_attn[i][j+1] = 1

        # Prepare model inputs
        src_token = [react_class] + smi_tokenizer(rooted_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(rooted_reacts) + ['<eos>']
        src_token = [self.src_t2i.get(st, self.src_t2i['<unk>']) for st in src_token]
        tgt_token = [self.tgt_t2i.get(tt, self.tgt_t2i['<unk>']) for tt in tgt_token]

        smiles_threed.atoms_token = [self.src_t2i.get(at, self.src_t2i['<unk>']) for at in smiles_threed.atoms_token]

        nonreactive_mask = [True] + nonreactive_mask
        graph_contents = smiles_graph.adjacency_matrix, smiles_graph.bond_type_dict, smiles_graph.bond_attributes
        threed_contents = smiles_threed.atoms_coord, smiles_threed.atoms_token, smiles_threed.atoms_index



        result = {
            'src': src_token,
            'tgt': tgt_token,
            'context_align': context_attn,
            'nonreact_mask': nonreactive_mask,
            'graph_contents': graph_contents,
            'threed_contents':threed_contents,
            'rooted_product': rooted_prod_am,
            'rooted_reactants': rooted_reacts_am,
            'reaction_class': react_class
        }
        return result

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_i2t[t] for t in tokens]
            else:
                return [self.src_i2t[t] for t in tokens if t != 1]
        else:
            if raw:
                return [self.tgt_i2t[t] for t in tokens]
            else:
                return [self.tgt_i2t[t] for t in tokens if t not in [1, 2, 3]]

    def __len__(self):
        return len(self.product_keys)

    def __getitem__(self, idx):
        self.processed = self.processed_data[idx]
        p = np.random.rand()
        if self.mode == 'train' and self.augment and p > 0.3:
            prod = self.processed['rooted_product']
            react = self.processed['rooted_reactants']
            rt = self.processed['reaction_class']
            new_processed = self.parse_smi(prod, react, rt, randomize=True)
            if new_processed is not None:
                self.processed = new_processed
        src, tgt, context_alignment, nonreact_mask, graph_contents, threed_contents = \
            self.processed['src'], self.processed['tgt'],  self.processed['context_align'], \
            self.processed['nonreact_mask'], self.processed['graph_contents'], self.processed['threed_contents']
        src_graph = SmilesGraph(self.processed['rooted_product'], existing=graph_contents)
        src_threed = SmilesThreeD(self.processed['rooted_product'], existing=threed_contents)
        return src, tgt, context_alignment, nonreact_mask, src_graph, src_threed


if __name__ == '__main__':
    pass

    # prod = '[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1'
    # Br[c:1]1[cH:2][cH:3][c:4]([Br:5])[n:6][cH:7]1.CN(C)[CH:8]=[O:9]
    # reacts = 'CC(C)(C)OC(=O)O[C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7].[CH3:8][C:9](=[O:10])[c:11]1[cH:12][cH:13][c:14]2[nH:15][cH:16][cH:17][c:18]2[cH:19]1'

    # rooted_prod_am, atoms_coordinate = get_rooted_smiles_with_am(prod)
    # print(rooted_prod_am)
    # print(atoms_coordinate)

    # print()
    # rooted_prod_am_r, atoms_coordinate_r = get_rooted_smiles_with_am(rooted_prod_am, randomChoose=True, atoms_coord=atoms_coordinate)
    # print(rooted_prod_am_r)
    # print(atoms_coordinate_r)

    # rooted_prod = remove_am_without_canonical(rooted_prod_am)
    # rooted_reacts = remove_am_without_canonical(rooted_reacts_am)

    # smiles_graph = SmilesGraph(rooted_prod)
    # nonreactive_mask = get_nonreactive_mask(rooted_prod_am, prod, reacts, radius=1)
    # context_alignment = get_context_alignment(rooted_prod_am, rooted_reacts_am)
    # atoms_coordinate = get_atoms_coordinate(rooted_prod_am)

    # rooted_prod_am = get_rooted_smiles_with_am(prod)
    # rooted_reacts_am = get_rooted_reacts_acord_to_prod(rooted_prod_am, reacts)
    # print(rooted_prod_am)
    # print(rooted_reacts_am)
    # atoms_coordinate = get_atoms_coordinate(rooted_prod_am)


    # context_attn = torch.zeros(
    #     (len(smi_tokenizer(rooted_reacts_am)) + 1, len(smi_tokenizer(rooted_prod_am)) + 1)).long()
    # for i, j in context_alignment:
    #     context_attn[i][j + 1] = 1

    # src_token = smi_tokenizer(rooted_prod)
    # tgt_token = ['<BOS>'] + smi_tokenizer(rooted_reacts) + ['<EOS>']
    # src_token = ['<UNK>'] + src_token

    # nonreactive_mask = [True] + nonreactive_mask
    # atoms_coordinate = [[0,0,0]] + atoms_coordinate
    # graph_contents = smiles_graph.adjacency_matrix, smiles_graph.bond_type_dict, smiles_graph.bond_attributes

    # src_token = [src_t2i.get(st, src_t2i['<unk>']) for st in src_token]
    # tgt_token = [tgt_t2i.get(tt, tgt_t2i['<unk>']) for tt in tgt_token]



    # env = lmdb.open('data/USPTO_50K/cooked_val.lmdb',
    #                 max_readers=1, readonly=True,
    #                 lock=False, readahead=False, meminit=False)

    # processed_data = []
    # with env.begin(write=False) as txn:
    #     product_keys = list(txn.cursor().iternext(values=False))
    #     for key in product_keys:
    #         processed_data.append(pickle.loads(txn.get(key)))

    # print(processed_data[999]['rooted_product'])
    # print(processed_data[999]['rooted_reactants'])
    # print(processed_data[999]['reaction_class'])