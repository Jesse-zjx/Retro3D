import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
from rdkit import Chem
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset

from utils.smiles_graph import SmilesGraph
from utils.smiles_threed import SmilesThreeD
from utils.smiles_utils import get_context_alignment, smi_tokenizer, clear_map_smiles
from utils.smiles_utils import get_cooked_smi, get_rooted_reacts_acord_to_prod


class USPTO_50K(Dataset):
    def __init__(self, config, mode, rank=-1):
        self.root = config.DATASET.ROOT
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.augment = config.DATASET.AUGMENT
        self.r_smiles = config.DATASET.RSMI
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
                             max_readers=126, readonly=True, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))
            for key in self.product_keys:
                self.processed_data.append(pickle.loads(txn.get(key)))
        if not self.known_class:
            for i in range(len(self.processed_data)):
                self.processed_data[i]['src'][0] = self.src_t2i['<UNK>']
                self.processed_data[i]['reaction_class'] = '<UNK>'

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
            rt = '<RX_{}>'.format(raw_data['class'][i])
            multi_data.append({"reacts":r, "prod":p, "class":rt})
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = list(tqdm(pool.imap(func=self.parse_smi_wrapper, iterable=multi_data)))
        pool.close()
        pool.join()

        env = lmdb.open(os.path.join(self.root, 'cooked_{}.lmdb'.format(self.mode)),
                        map_size=1099511627776)
        with env.begin(write=True) as txn:
            for i, result in enumerate(results):
                if result is not None:
                    p_key = '{} {}'.format(i, result['rooted_product'])
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
        parse_prod_am = get_cooked_smi(prod, randomize)
        if randomize and self.r_smiles:
            parse_reacts_am = get_rooted_reacts_acord_to_prod(reacts, parse_prod_am)
        else:
            parse_reacts_am = get_cooked_smi(reacts, randomize)

        parse_prod = clear_map_smiles(parse_prod_am)
        parse_reacts = clear_map_smiles(parse_reacts_am)


        if build_vocab:
            return parse_prod, parse_reacts
        
        if Chem.MolFromSmiles(parse_prod) is None or Chem.MolFromSmiles(parse_reacts) is None:
            return None
        
        # Get the smiles 3d
        before = None   # avoid re-compute 3d coordinate (time)
        if randomize:
            before = (prod, self.processed['threed_contents'])
        smiles_threed = SmilesThreeD(parse_prod_am, before=before) 

        if smiles_threed.atoms_coord is None:
            return None

        # Get the smiles graph
        smiles_graph = SmilesGraph(parse_prod)
        # Get the context alignment based on atom-mapping
        context_alignment = get_context_alignment(parse_prod_am, parse_reacts_am)

        context_attn = torch.zeros((len(smi_tokenizer(parse_reacts_am))+1, len(smi_tokenizer(parse_prod_am))+1)).long()
        for i, j in context_alignment:
            context_attn[i][j+1] = 1

        # Prepare model inputs
        src_token = [react_class] + smi_tokenizer(parse_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(parse_reacts) + ['<eos>']
        src_token = [self.src_t2i.get(st, self.src_t2i['<unk>']) for st in src_token]
        tgt_token = [self.tgt_t2i.get(tt, self.tgt_t2i['<unk>']) for tt in tgt_token]

        smiles_threed.atoms_token = [self.src_t2i.get(at, self.src_t2i['<unk>']) for at in smiles_threed.atoms_token]

        graph_contents = smiles_graph.adjacency_matrix, smiles_graph.bond_type_dict, smiles_graph.bond_attributes
        threed_contents = smiles_threed.atoms_coord, smiles_threed.atoms_token, smiles_threed.atoms_index

        result = {
            'src': src_token,
            'tgt': tgt_token,
            'context_align': context_attn,
            'graph_contents': graph_contents,
            'threed_contents':threed_contents,
            'rooted_product': parse_prod_am,
            'rooted_reactants': parse_reacts_am,
            'reaction_class': react_class
        }
        return result

    def reconstruct_smi(self, indexs):
        illgel_words = ['<pad>', '<sos>', '<eos>', '<UNK>'] + ['<RX_{}>'.format(i) for i in range(1, 11)]
        illgel_index = [self.tgt_t2i[word] for word in illgel_words]
        return [self.tgt_i2t[i] for i in indexs if i not in illgel_index]

    def __len__(self):
        return len(self.product_keys)

    def __getitem__(self, idx):
        self.processed = self.processed_data[idx]
        p = np.random.rand()
        if self.r_smiles or (self.mode == 'train' and self.augment and p > 0.3):
            prod = self.processed['rooted_product']
            react = self.processed['rooted_reactants']
            rt = self.processed['reaction_class']
            new_processed = self.parse_smi(prod, react, rt, randomize=True)
            if new_processed is not None:
                self.processed = new_processed
        src, tgt, context_alignment, graph_contents, threed_contents = \
            self.processed['src'], self.processed['tgt'],  self.processed['context_align'], \
            self.processed['graph_contents'], self.processed['threed_contents']
        src_graph = SmilesGraph(self.processed['rooted_product'], existing=graph_contents)
        src_threed = SmilesThreeD(self.processed['rooted_product'], existing=threed_contents)
        return src, tgt, context_alignment, src_graph, src_threed
