import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from numpy import *
from tqdm import tqdm

from trainers.trainer_utils import merge_results_dist

from tjco.loss import TriadicLoss, MMD
from tjco.similarity import TriSimilarity

class STC_M3D_Trainer(object):
    def __init__(self, rank, config, model, logit_scale, pc_adapter, view_weights_module, view_fusion_module,optimizer,scheduler, train_loader, objaverse_lvis_loader, scanobjectnn_loader):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.pc_adapter = pc_adapter
        self.view_weights_module = view_weights_module
        self.view_weights = None
        self.view_fusion_module = view_fusion_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 1
        self.step = 0
        self.alpha = config.training.alpha
        self.beta = config.training.beta
        self.best_lvis_acc = 0
        self.lvis_best_epoch = 0
        self.best_scanobjectnn_acc = 0
        self.scanobjectnn_best_epoch = 0
        self.config.ngpu = dist.get_world_size()

        self.triadic_loss = TriadicLoss(negative_sampling=config.negative_sampling)
        self.tri_similarity = TriSimilarity()
        
    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.pc_adapter.load_state_dict(checkpoint['pc_adapter'])
        self.logit_scale.load_state_dict(checkpoint['logit_scale'])
        self.view_weights_module.load_state_dict(checkpoint['view_weights_module'])
        self.view_fusion_module.load_state_dict(checkpoint['view_fusion_module'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch'] + 1
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
    

    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def train_one_epoch(self):
        self.model.train()
        self.pc_adapter.train()
        self.view_weights_module.train()
        self.view_fusion_module.eval()

        acc_list = []
        total_loss_list, triadic_loss_list, fusion_loss_list, pair_loss_list = [], [], [], []

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0.0
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(data['xyz'], data['features'], \
                                       device=self.config.device, \
                                       quantization_size=self.config.model.voxel_size)
            else:
                pred_feat = self.model(data['xyz_dense'], data['features_dense'])
            logit_scale = self.logit_scale(None)
            pc_feat = F.normalize(self.pc_adapter(pred_feat), dim=1) 
            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)

            if self.config.training.loss_type == "triple":
                if self.config.dataset.use_fusion:
                    img_feat = img_feat.view(-1, self.config.dataset.num_imgs, self.config.clip_embed_dim)
                    view_weights = self.view_weights_module(img_feat)
                    fusion_img_feat = self.view_fusion_module(img_feat, view_weights)
                    max_weight_idx = torch.argmax(view_weights, dim=1)
                    single_img_feat = img_feat[torch.arange(img_feat.size(0)), max_weight_idx, :]
                    single_img_feat = F.normalize(single_img_feat, dim=1)

                    single_text_feat = text_feat[:, 0:self.config.clip_embed_dim]
                    single_text_feat = F.normalize(single_text_feat, dim=1)

                    view_fusion_mmd_loss = 0.0
                    view_weights_mean = view_weights.mean(dim=0)
                    for i in range(self.config.dataset.num_imgs):
                        view_feat = img_feat[:, i, :]
                        view_mmd = MMD(fusion_img_feat, view_feat, kernel="rbf")
                        view_fusion_mmd_loss += view_weights_mean[i] * view_mmd
                    
                    triadic_loss, acc = self.triadic_loss([pc_feat, single_img_feat, single_text_feat], logit_scale)
                    loss_pi, _ = self.contras_loss(pc_feat, single_img_feat)
                    loss_pt, _ = self.contras_loss(pc_feat, single_text_feat)
                    loss_it, _ = self.contras_loss(single_img_feat, single_text_feat)
                    pair_loss = loss_pi + loss_pt + loss_it

                    loss += self.alpha * (triadic_loss + pair_loss) + self.beta * view_fusion_mmd_loss

                    acc_list.append(acc.item())

                    triadic_loss_list.append(triadic_loss.item())
                    pair_loss_list.append(pair_loss.item())
                    fusion_loss_list.append(view_fusion_mmd_loss.item())
                    total_loss_list.append(loss.item())
                else:
                    for i in range (self.config.dataset.num_imgs):
                        single_img_feat = F.normalize(img_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim], dim=1)
                        single_img_feat = single_img_feat.to(self.config.device)

                        single_text_feat = F.normalize(text_feat[:, 0:self.config.clip_embed_dim], dim=1)
                        single_text_feat = single_text_feat.to(self.config.device)

                        triadic_loss, acc = self.triadic_loss([pc_feat, single_img_feat, single_text_feat], logit_scale)

                        loss_pi, _ = self.contras_loss(pc_feat, single_img_feat)
                        loss_pt, _ = self.contras_loss(pc_feat, single_text_feat)
                        loss_it, _ = self.contras_loss(single_img_feat, single_text_feat)
                        pair_loss = loss_pi + loss_pt + loss_it

                        loss += self.alpha * (triadic_loss + pair_loss) 

                        acc_list.append(acc.item())

                        triadic_loss_list.append(triadic_loss.item())   
                        pair_loss_list.append(pair_loss.item())
                        
                    total_loss_list.append(loss.item()/self.config.dataset.num_imgs)
            loss.backward()
            self.optimizer.step()
            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()

        if self.rank == 0:
            logging.info(f'Train: acc: {np.mean(acc_list)}')
            logging.info(f'Train: total_loss: {np.mean(total_loss_list)} triadic_loss: {np.mean(triadic_loss_list)} pair_loss: {np.mean(pair_loss_list)} fusion_loss: {np.mean(fusion_loss_list)}' )


    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "pc_adapter": self.pc_adapter.state_dict(),
            "view_weights_module": self.view_weights_module.state_dict(),
            "view_fusion_module": self.view_fusion_module.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.scheduler == "default" else None,
            "epoch": self.epoch,
            "step": self.step,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))
        

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch + 1):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))

            self.train_one_epoch()
            if epoch >= self.config.training.test_epoch:
                self.test_objaverse_lvis()
                self.test_scanobjectnn()                                   
            if self.rank == 0:
                self.save_model('latest')

    def test_objaverse_lvis(self):
        self.model.eval()
        self.pc_adapter.eval()
        self.view_weights_module.eval()
        self.view_fusion_module.eval()

        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
        
        per_cat_correct = torch.zeros(1156).cuda()
        per_cat_count = torch.zeros(1156).cuda()

        labels_all = []
        logits_all = []
        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                
                pred_feat_pc = F.normalize(self.pc_adapter(pred_feat), dim=1).to(self.config.device)
                clip_text_feat = F.normalize(clip_text_feat, dim=1).to(self.config.device)
                img_feat = torch.vstack(data['img_feat']).to(self.config.device)
             
                if self.config.dataset.use_fusion:
                    img_feat = img_feat.view(-1, self.config.dataset.num_imgs, self.config.clip_embed_dim)
                    
                    view_weights = self.view_weights_module(img_feat)
                    max_weight_idx = torch.argmax(view_weights, dim=1)

                    img_feat = img_feat[torch.arange(img_feat.size(0)), max_weight_idx, :]          
                img_feat = F.normalize(img_feat, dim=1)
 
                logits_tri = self.tri_similarity(clip_text_feat, [pred_feat_pc, img_feat])
                 
                labels = data['category'].to(self.config.device)
                logits_all.append(logits_tri.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(
            os.path.join(self.config.ckpt_dir, "objaverse_dir"),
            logits_all, labels_all)

        if self.rank == 0:
            topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))
            for i in torch.unique(labels_all):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()

            if overall_acc > self.best_lvis_acc:
                self.best_lvis_acc = overall_acc
                self.lvis_best_epoch = self.epoch
                self.save_model('best_lvis')


            logging.info(
                f'Test ObjaverseLVIS: overall acc: {overall_acc}')
            logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                                topk_acc[1].item(),
                                                                                                topk_acc[2].item()))

    def test_scanobjectnn(self):
        self.model.eval()
        self.pc_adapter.eval()
        self.view_weights_module.eval()
        self.view_fusion_module.eval()

        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)    
    
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                pred_feat_pc = F.normalize(self.pc_adapter(pred_feat), dim=1).to(self.config.device)
                clip_text_feat = F.normalize(clip_text_feat, dim=1).to(self.config.device)
                
                img_feat = torch.vstack(data['img_feat']).to(self.config.device)
                if self.config.dataset.use_fusion:
                    img_feat = img_feat.view(-1, self.config.dataset.num_imgs, self.config.clip_embed_dim)
                    
                    view_weights = self.view_weights_module(img_feat)
                    max_weight_idx = torch.argmax(view_weights, dim=1)

                    img_feat = img_feat[torch.arange(img_feat.size(0)), max_weight_idx, :]
                img_feat = F.normalize(img_feat, dim=1)
                
                logits_tri = self.tri_similarity(clip_text_feat, [pred_feat_pc, img_feat])

                labels = data['category'].to(self.config.device)
                logits_all.append(logits_tri.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(
            os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
            logits_all, labels_all)

        if self.rank == 0:
            topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))
            for i in range(15):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()

            if overall_acc > self.best_scanobjectnn_acc:
                self.best_scanobjectnn_acc = overall_acc
                self.scanobjectnn_best_epoch = self.epoch
                self.save_model('best_scanobjectnn')

            logging.info(f'Test ScanObjectNN: overall acc: {overall_acc}')
            logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                               topk_acc[1].item(),
                                                                                               topk_acc[2].item()))
    

