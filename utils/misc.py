import torch
from collections import OrderedDict
import time
import os
from contextlib import redirect_stdout
import pathlib
import logging
import numpy as np
import random
import torch.backends.cudnn as cudnn

import models

def get_saved_info(path):
    check_point = torch.load(path, map_location=torch.device('cpu'))
    return check_point['settings'], check_point['model']


def get_pretrained_model(config, model_dict, data):
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

    new_state_dict = OrderedDict()
    for k, v in model_dict.items():
        if k[:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def get_output_dir(config, rank=-1):
    root_output_dir = pathlib.Path(config.OUTPUT_DIR)
    if not root_output_dir.exists():
        root_output_dir.mkdir()

    dataset = config.DATASET.NAME
    if config.DATASET.RSMI:
        dataset += '_rsmi' 
    if config.DATASET.KNOWN_CLASS:
        dataset += '_known'
    model = config.MODEL.NAME + '_dim' + str(config.MODEL.D_MODEL) + '_wd' + str(config.TRAIN.WEIGHT_DECAY)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    saved_model_output_dir = root_output_dir / dataset / model / time_str / 'saved_model'
    tensorboard_log_dir = root_output_dir / dataset / model / time_str / 'tensorboard'
    saved_model_output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    with open(root_output_dir / dataset / model / time_str / 'config.yaml', 'w') as f:
        with redirect_stdout(f):
            print(config.dump())

    if rank < 1:
        print('model:{}, d_model:{}, wd:{}, dataset:{}, class:{}, rsmi:{}'.format( \
            config.MODEL.NAME, config.MODEL.D_MODEL, config.TRAIN.WEIGHT_DECAY, \
            config.DATASET.NAME, config.DATASET.KNOWN_CLASS, config.DATASET.RSMI))
    return str(saved_model_output_dir), str(tensorboard_log_dir)


def set_logger(output_dir=None, rank=-1):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank < 1 else logging.WARNING)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank < 1 else logging.WARNING)
    logger.addHandler(console)

    if output_dir is not None:
        fileter = logging.FileHandler(output_dir[:-11] + 'log.txt')
        fileter.setLevel(logging.INFO if rank < 1 else logging.WARNING)
        logger.addHandler(fileter)

    return logger


def set_seed(config, rank=-1):
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
