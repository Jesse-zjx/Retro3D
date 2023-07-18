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
        dataset = eval('datasets.' + self.config.DATASET.NAME)(config=self.config, mode=mode, rank=self.rank)
        self.src_t2i, self.tgt_t2i = dataset.src_t2i.copy(), dataset.tgt_t2i.copy()

        n_gpus = torch.cuda.device_count()
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
        src, tgt, align, nonreact_mask, src_graph, src_threed = zip(*data)
        bsz = len(data)
        max_src_len = max([len(item) for item in src])
        max_tgt_len = max([len(item) for item in tgt])

        
        new_src = torch.full((max_src_len, bsz), self.src_t2i['<pad>'], dtype=torch.long)
        new_tgt = torch.full((max_tgt_len, bsz), self.tgt_t2i['<pad>'], dtype=torch.long)
        new_alignment = torch.zeros((bsz, max_tgt_len-1, max_src_len), dtype=torch.float)
        new_nonreactive_mask = torch.ones((max_src_len, bsz), dtype=torch.bool)
        new_bond_matrix = torch.zeros((bsz, max_src_len, max_src_len, 7), dtype=torch.long)
        new_dist_matrix = torch.zeros((bsz, max_src_len, max_src_len), dtype=torch.float)

        for i in range(bsz):
            new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
            new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
            new_alignment[i, :align[i].shape[0], :align[i].shape[1]] = align[i].float()
            new_nonreactive_mask[:, i][:len(nonreact_mask[i])] = torch.BoolTensor(nonreact_mask[i])
            full_adj_matrix = torch.from_numpy(src_graph[i].adjacency_matrix_attr)
            # 加入了反应类型要偏移1
            new_bond_matrix[i, 1:full_adj_matrix.shape[0] + 1, 1:full_adj_matrix.shape[1] + 1] = full_adj_matrix

            full_dist_matrix = torch.from_numpy(src_threed[i].dist_matrix)
            new_dist_matrix[i, 1:full_dist_matrix.shape[0] + 1, 1:full_dist_matrix.shape[1] + 1] = full_dist_matrix
            
        return new_src, new_tgt, new_alignment, new_nonreactive_mask, (new_bond_matrix, src_graph), (new_dist_matrix, src_threed)
        