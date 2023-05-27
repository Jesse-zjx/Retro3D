import os
import argparse
import timeit
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import models

from utils import update_config, config, get_output_dir, Data, \
                 set_seed, set_logger, Trainer, ScheduledOptim


def train():
    parser = argparse.ArgumentParser(description='Train stage')
    parser.add_argument('--config', help='config file path', required=True, type=str)
    args = parser.parse_args()
    update_config(config, args)
    rank = int(os.environ["LOCAL_RANK"])

    if rank != -1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl')

    output_dir, tensorboard_log_dir = get_output_dir(config, rank)
    logger = set_logger(output_dir, rank)
    # tensor board
    writer_dict = {
        'writer': SummaryWriter(tensorboard_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
        'test_global_steps': 0
    }

    set_seed(config, rank)

    # Dataloader
    data = Data(config, train=True, val=True, test=True, rank=rank)
    loaders = data.get_loaders()
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    # model
    model = eval('models.' + config.MODEL.NAME)(
        n_src_vocab=len(data.src_t2i),
        n_trg_vocab=len(data.tgt_t2i),
        src_pad_idx=data.src_t2i['<pad>'],
        tgt_pad_idx=data.tgt_t2i['<pad>'],
        d_model=config.MODEL.D_MODEL,
        d_inner=config.MODEL.D_INNER,
        n_enc_layers=config.MODEL.N_LAYERS,
        n_dec_layers=config.MODEL.N_LAYERS,
        n_head=config.MODEL.N_HEAD,
        dropout=config.MODEL.DROPOUT,
        shared_embed=config.MODEL.SHARED_EMBED,
        shared_encoder=config.MODEL.SHARED_ENCODER
    )
    
    if rank != -1:
        model = DDP(model.cuda(),
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model, device_ids=config.GPUS).cuda()

    optimizer = ScheduledOptim(
        optimizer = torch.optim.Adam(model.parameters(),
                                    betas=(config.TRAIN.ADAM_BETA1, config.TRAIN.ADAM_BETA2),
                                    eps=config.TRAIN.ADAM_EPSILON, weight_decay=config.TRAIN.WEIGHT_DECAY),
        lr_mul=config.TRAIN.LEARNING_RATE, d_model=config.MODEL.D_MODEL,
        n_warmup_steps=config.TRAIN.WARMUP_STEPS)

    start_time = timeit.default_timer()
    trainer = Trainer(model=model, config=config, writer_dict=writer_dict, logger=logger, rank=rank)
    for epoch in range(0, trainer.end_epoch):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        trainer.train(train_loader, optimizer)
        if ((epoch + 1) % config.TEST.EVAL_STEPS) == 0:
            trainer.val(val_loader)
            trainer.test(test_loader)
            early_stop = trainer.save_models(output_dir)
            if early_stop:
                break

    writer_dict['writer'].close()
    end_time = timeit.default_timer()
    logger.info('Elapse time: %d hour %d minute !' % (
        int((end_time - start_time) / 3600), int((end_time - start_time) % 3600 / 60)))
    logger.info('Done!')

if __name__ == '__main__':
    train()
