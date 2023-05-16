import torch
from torch.utils.data import dataloader, RandomSampler, SequentialSampler
import datasets


class Data():
    def __init__(self, config, train=True, val=True, test=True, rank=-1):
        self.config = config
        self.rank = rank
        self.action = {
            'train': train,
            'val': val,
            'test': test
        }
        self.loaders = {}
        for mode in self.action.keys():
            if self.action[mode]:
                self.loaders[mode] = self.get_data_loader(mode)
            else:
                self.loaders[mode] = None

    def get_data_loader(self, mode='train'):
        dataset = eval('datasets.' + self.config.DATASET.NAME)(config=self.config, mode=mode)
        self.src_t2i, self.tgt_t2i = dataset.src_t2i.copy(), dataset.tgt_t2i.copy()

        n_gpus = 1
        if mode == 'train':
            if self.rank != -1:
                batch_size = self.config.TRAIN.BATCH_SIZE_PER_GPU
                sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
            else:
                batch_size = self.config.TRAIN.BATCH_SIZE_PER_GPU * n_gpus
                sampler = RandomSampler(dataset, replacement=False)
            drop_last = True
        else:
            if self.rank != -1:
                batch_size = self.config.TEST.BATCH_SIZE_PER_GPU
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            else:
                batch_size = self.config.TEST.BATCH_SIZE_PER_GPU * n_gpus
                sampler = SequentialSampler(dataset)
            drop_last = False

        return dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        sampler=sampler, num_workers=self.config.TRAIN.NUM_WORKERS,
                        pin_memory=True, drop_last=drop_last, collate_fn=self.generate_batch)

    def get_loaders(self):
        return self.loaders

    def generate_batch(self, data):
        src, tgt, align, nonreact_mask, atoms_coord, src_graph = zip(*data)
        bsz = len(data)
        max_src_len = max([len(item) for item in src])
        max_tgt_len = max([len(item) for item in tgt])

        
        new_src = torch.full((max_src_len, bsz), self.src_t2i['<pad>'], dtype=torch.long)
        new_tgt = torch.full((max_tgt_len, bsz), self.tgt_t2i['<pad>'], dtype=torch.long)
        new_alignment = torch.zeros((bsz, max_tgt_len-1, max_src_len), dtype=torch.float)
        new_nonreactive_mask = torch.ones((max_src_len, bsz), dtype=torch.bool)
        new_bond_matrix = torch.zeros((bsz, max_src_len, max_src_len, 7), dtype=torch.long)
        new_dist_matrix = torch.zeros((bsz, max_src_len, max_src_len, 3), dtype=torch.long)

        for i in range(bsz):
            new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
            new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
            new_alignment[i, :align[i].shape[0], :align[i].shape[1]] = align[i].float()
            new_nonreactive_mask[:, i][:len(nonreact_mask[i])] = torch.BoolTensor(nonreact_mask[i])
            full_adj_matrix = torch.from_numpy(src_graph[i].adjacency_matrix_attr)
            # 加入了反应类型要偏移1
            new_bond_matrix[i, 1:full_adj_matrix.shape[0] + 1, 1:full_adj_matrix.shape[1] + 1] = full_adj_matrix
            
        return new_src, new_tgt, new_alignment, new_nonreactive_mask, (new_bond_matrix, src_graph)
        
        # src_batch, tgt_batch, pos_batch, z_batch, index_batch = [], [], [], [], []
        # atoms_batch = []
        # index = 0
        # for batch in data_batch:  # 开始对一个batch重的每一个样本进行处理
        #     src_item, tgt_item = batch[:2]
        #     src_batch.append(src_item)  # 编码器输入序列不需要加起止符
        #     # 在每个idx序列的首位加上 起始token 和 结束token
        #     tgt = torch.cat([torch.tensor([self.config.DATASET.TRG_BOS_IDX]), tgt_item, torch.tensor([self.config.DATASET.TRG_EOS_IDX])], dim=0)
        #     tgt_batch.append(tgt)

        #     if self.config.DATASET.ADD_3D:
        #         pos_item, z_item, atoms_item = batch[-3:]
        #         for pos, z, atoms in zip(pos_item, z_item, atoms_item):
        #             pos_batch.append(pos)
        #             z_batch.append(z)
        #             atoms_batch.append(atoms)
        #             index_batch.append(index)
        #     index += 1
        # # 以最长的序列为标准进行填充
        # src_batch = pad_sequence(src_batch, padding_value=self.src_t2i['<pad>'])  # [src_len,batch_size]
        # tgt_batch = pad_sequence(tgt_batch, padding_value=self.tgt_t2i['<pad>'])
        # if self.config.DATASET.ADD_3D:
        #     pos_batch = torch.stack(pos_batch)
        #     z_batch = torch.stack(z_batch)
        #     atoms_batch = torch.stack(atoms_batch)
        #     index_batch = torch.as_tensor(index_batch)
        #     return src_batch, tgt_batch, pos_batch, z_batch, index_batch, atoms_batch
        # return src_batch, tgt_batch
