import logging
import os
import random
import shutil
import sys
from datetime import datetime

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import data
import models
from models.LogitScaleNetwork import LogitScaleNetwork
from models.view_module import ViewWeightsModule, ViewFusionModule
from trainers.mlp import MLP, MLP_ME
from trainers.stc_m3d_trainer import STC_M3D_Trainer
from utils.logger import setup_logging
from utils.misc import load_config, dump_config
from utils.scheduler import cosine_lr, const_lr
from param import parse_args


def main(cli_args, extras):
    # tcp_port = "12338"
    local_rank = int(os.environ["LOCAL_RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend='nccl',
    )
    rank = dist.get_rank()

    config = load_config(cli_args.config, cli_args=vars(cli_args), extra_args=extras)
    if config.autoresume:
        config.trial_name = config.get('trial_name') + "@autoresume"
    else:
        config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.ngpu = num_gpus

    # fix the seed
    if config.fix_seed:
        seed = config.seed
        torch.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)

    if rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)

    config.device = 'cuda:{0}'.format(rank)

    if rank == 0:
        config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.log')
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
        logging.info("Using {} GPU(s).".format(config.ngpu))

    if config.train:
        torch.cuda.set_device(rank)
        model = models.make(config).cuda(rank)
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info("Network:{}, Number of parameters: {}".format(config.model.name, total_params))
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if config.model.name.startswith('Mink'):
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)  # minkowski only
            logging.info("Using MinkowskiSyncBatchNorm")
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logging.info("Using SyncBatchNorm")

        logit_scale = LogitScaleNetwork(config.training.logit_scale_init).cuda(rank)
        logit_scale = DDP(logit_scale, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        mlp_type = config.training.get('mlp_type')
        if mlp_type == 'mlp':
            pc_adapter = MLP(in_features=config.training.pc_in_dim,
                                hidden_features=config.training.pc_hidden_dim,
                                out_features=config.training.pc_out_dim,
                                drop=config.training.pc_dropout,
                                activate=config.training.activate).cuda(rank)
        elif mlp_type == 'mlp_me':
            pc_adapter = MLP_ME(in_features=config.training.pc_in_dim,
                                hidden_features=config.training.pc_hidden_dim,
                                out_features=config.training.pc_out_dim,
                                drop=config.training.pc_dropout).cuda(rank)
        pc_adapter = DDP(pc_adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        view_weights_module = ViewWeightsModule(config.clip_embed_dim).cuda(rank)
        view_weights_module = DDP(view_weights_module, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        view_fusion_module = ViewFusionModule(config.clip_embed_dim).cuda(rank)
        view_fusion_module = DDP(view_fusion_module, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        train_loader = data.make(config, 'train', rank, num_gpus)

        objaverse_lvis_loader = data.make_objaverse_lvis(config)
        scanobjectnn_loader = data.make_scanobjectnntest(config)

        if rank == 0:
            if train_loader is not None:
                logging.info("Train iterations: {}".format(len(train_loader)))

        params = list(model.parameters()) + list(pc_adapter.parameters()) + list(view_weights_module.parameters()) + list(view_fusion_module.parameters()) + list(logit_scale.parameters())
        logging.info("Trainable parameters: {}".format(sum(p.numel() for p in params)))

        warmup_epoch = config.training.warmup_epoch
        warmup = warmup_epoch * len(train_loader)
        epoch = config.training.max_epoch
        total_steps = epoch * len(train_loader)
        lr = config.training.lr * num_gpus
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=(config.training.beta1, config.training.beta2),
            eps=config.training.eps,
        )

        if config.training.scheduler == "cosine":
            scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
        elif config.training.scheduler == "const":
            scheduler = const_lr(optimizer, lr, warmup, total_steps)
        else:
            lr_decay_step = config.training.lr_decay * len(train_loader)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step,
                                                        gamma=config.training.lr_decay_rate)
        if config.trainer == "STC_M3D_trainer":
            trainer = STC_M3D_Trainer(rank, config, model, logit_scale,
                                                      pc_adapter, view_weights_module, view_fusion_module, optimizer, scheduler, \
                                                      train_loader, objaverse_lvis_loader,
                                                      scanobjectnn_loader)
        else:
            raise NotImplementedError("Trainer {} not implemented".format(config.trainer))
        if config.resume is not None:
            trainer.load_from_checkpoint(config.resume)
        elif config.autoresume:
            if os.path.exists(os.path.join(config.ckpt_dir, '{}.pt'.format('latest'))):
                trainer.load_from_checkpoint(os.path.join(config.ckpt_dir, '{}.pt'.format('latest')))

        trainer.train()


if __name__ == '__main__':
    cli_args, extras = parse_args(sys.argv[1:])
    main(cli_args, extras)
