#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:41:48 2020
Training code for GCN-CNN autoencoder with triplet network for metric learning.
Dataloader loads triplets of graph data, images.. 
Network is trained with reconstruction loss and triplet ranking loss.

@author: dipu
"""


import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import init_paths

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

from dataloaders.dataloader_triplet import *
from dataloaders.dataloader_triplet import RICO_TripletDataset
from dataloaders.dataloader_test import RICO_ComponentDataset
import models

import opts_dml
import shutil
import os
import os.path as osp
import errno
import time 
import numpy as np
from scipy.spatial.distance import cdist

from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc
from evaluate import getBoundingBoxes_from_info

import wandb

#%%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth'))

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise 
  
def set_lr2(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor
    print('\n', optimizer, '\n')            
        
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)        


def weighted_mse_loss(input, target, weight):
    mse = (input - target) ** 2      # B * C * W * H
    mse = torch.mean(mse, dim =(2,3))  #  B*C
    return torch.mean(weight*mse)
             
def compute_MSE_weights(images):
    weight = torch.sum(images,dim=(2,3))
    sumw = torch.sum(weight, dim=1, keepdim=True)
    weight = weight/sumw
    weight = 1/weight 
    weight[weight==float('inf')] = 0
    weight = weight / weight.sum(1, keepdim=True)
    weight = 10*weight
    weight[weight==0] = 1
    return weight

def weighted_mse_loss_2(input, target, weights):
    mse = (input - target) ** 2      # B * C * W * H
    mse = torch.mean(mse, dim =(0,2,3))  #  C
    return torch.mean(weights*mse) 


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.d = nn.PairwiseDistance(p=2)
        print ('Triplet loss intilized with margin {}'.format(self.margin))
    
    def forward(self, anchor, positive, negative, size_average=True):
        distance = self.d(anchor, positive) - self.d(anchor, negative) + self.margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss

#%%
def main(opt):
    print(opt)
    
    # data_transform = transforms.Compose([   # Only used if decoder is trained using 3-Channel RBG (not 25Channel Images)
    #         transforms.Resize([254,126]),  # transforms.Resize([254,126])
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    
    save_dir = 'trained_model/model_dec_{}_dim{}'.format(opt.decoder_model, opt.dim)
    mkdir_if_missing(save_dir)
    print ('\nOutput dir: ', save_dir)
    
    loader = RICO_TripletDataset(opt, transform=None) 
    loader_test = RICO_ComponentDataset(opt, transform=None)
    model: torch.nn.Module = models.create(opt.decoder_model, opt)
    model = model.cuda()

    # Should use square images for rplan
    expected_output_size = (239, 239)

    assert model.expected_output_size == expected_output_size, "Expected output size is not correct"

 
    if opt.pretrained:
        pt_model = opt.pt_model
        print('\n Loading from the Pretrained Network from... ')
        print(pt_model)       
        resume = load_checkpoint(pt_model)
        model.load_state_dict(resume['state_dict'])  
        model = model.cuda()
    
    wandb.init(project="gcn-cnn", name=save_dir.split('/')[-1])
    wandb.config.update(opt)
    
    if opt.loss =='mse':
        criterion = nn.MSELoss()    
    elif opt.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([30.0]).cuda())
    elif opt.loss == 'huber':
        criterion = nn.SmoothL1Loss()
    elif opt.loss == 'l1':
        criterion = nn.L1Loss()   
    
    optimizer = torch.optim.Adam(model.parameters(), opt.learning_rate)
    
    margin = opt.margin
    lambda_mul = opt.lambda_mul
    dml_loss = TripletLoss(margin=margin)
    dml_loss = dml_loss.cuda()
    
   
    # Bounding boxes are only used for evaluation
    # boundingBoxes = getBoundingBoxes_from_info()
    
    epoch_done = True
    iteration = 0
    epoch = 0

    total_epoch_time = 0
    
    model.train()
    torch.set_grad_enabled(True)
    time_s = time.time()
    while True:
        if epoch_done:

            epoch += 1
            
            if epoch in [5,12,17]: 
                decay_factor = opt.learning_rate_decay_rate
                set_lr2(optimizer, decay_factor)
                print('\n', optimizer, '\n') 
                
            losses = AverageMeter()
            losses_recon = AverageMeter()
            losses_dml = AverageMeter()

            epoch_done = False
            iteration = 0

            total_epoch_time = 0
        
        # Load a batch of data from train split
        data =  loader.get_batch('train')

        images_a, images_p, images_n = resize_25channel_images(data, decoder_model=opt.decoder_model, output_size=expected_output_size)

        #2. Forward model and compute loss
        emb_a, out_a, emb_p, out_p, emb_n, out_n = forward_train(model, optimizer, data)

        recon_loss = compute_recon_loss(criterion, out_a, out_p, out_n, images_a, images_p, images_n)

        ml_loss = dml_loss(emb_a, emb_p, emb_n)
        
        loss = ml_loss + lambda_mul*recon_loss
        
        # 3. Update model
        loss.backward()
        clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        torch.cuda.empty_cache()
        losses.update(loss.detach().item())
        losses_recon.update(recon_loss.detach().item())
        losses_dml.update(ml_loss.detach().item())
        
#        torch.cuda.synchronize()

        # Update the iteration and epoch
        iteration += 1
        
        if epoch == 0 and iteration == 1:
            print("Training Started ")

        # if iteration%10 == 0:
        #     print(f"{iteration=}")

        if data['bounds']['wrapped']:
            epoch_done = True 

        log_interval = 10

        if iteration % log_interval == 0 or epoch_done:


            elsp_time = (time.time() - time_s)

            total_epoch_time += elsp_time

            wandb.log({
                "loss": losses.val,
                "loss_recon": losses_recon.val,
                "loss_dml": losses_dml.val,

                "loss_avg": losses.avg,
                "loss_recon_avg": losses_recon.avg,
                "loss_dml_avg": losses_dml.avg,

                "time_per_sample": elsp_time / (log_interval * opt.batch_size),
                "time_per_sample_epoch_avg": total_epoch_time / ((iteration + 1) * opt.batch_size),
            })


            print( 'Epoch [%02d] [%05d / %05d] Average_Loss: %.5f    Recon Loss: %.4f  DML Loss: %.4f'%(epoch, iteration*opt.batch_size, len(loader), losses.avg, losses_recon.avg, losses_dml.avg ))
            with open(save_dir+'/log.txt', 'a') as f:
                f.write('Epoch [%02d] [%05d / %05d  ] Average_Loss: %.5f  Recon Loss: %.4f  DML Loss: %.4f\n'%(epoch, iteration*opt.batch_size, len(loader), losses.avg, losses_recon.avg, losses_dml.avg ))
                f.write('Completed {} images in {}'.format(iteration*opt.batch_size, elsp_time))
                
            
            print('Completed {} images in {}'.format(iteration*opt.batch_size, total_epoch_time))
            time_s = time.time()
        
        if epoch_done:
            log_validation_loss(model, loader_test, criterion, expected_output_size, epoch)

                
        if (epoch+1) % 5 == 0  and epoch_done:
            state_dict = model.state_dict()  
            
            save_checkpoint({
            'state_dict': state_dict,
            'epoch': (epoch+1)}, is_best=False, fpath=osp.join(save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))

            # perform_tests_dml takes a lot of time
            # perform_tests_dml(model, loader_test, boundingBoxes,  save_dir, epoch)
            
            # set model to training mode again.
            model.train()
            torch.set_grad_enabled(True)
               
        if epoch > opt.max_epochs:
            break

def resize_25channel_images(data, decoder_model, output_size):

    images_a = data['images_a'].cuda()
    images_p = data['images_p'].cuda()
    images_n = data['images_n'].cuda()

    if decoder_model == 'strided': 
        images_a = F.interpolate(images_a, size=output_size)
        images_p = F.interpolate(images_p, size=output_size)
        images_n = F.interpolate(images_n, size=output_size)
    else:
        raise NotImplementedError("Expected to use strided decoder")
    
    return images_a, images_p, images_n

def forward_train(model, optimizer, data):
    # torch.cuda.synchronize()   # Waits for all kernels in all streams on a CUDA device to complete. 
    optimizer.zero_grad()

    return model_forward(model, data)



def model_forward(model, data):
    sg_data_a = {key: torch.from_numpy(data['sg_data_a'][key]).cuda() for key in data['sg_data_a']}
    sg_data_p = {key: torch.from_numpy(data['sg_data_p'][key]).cuda() for key in data['sg_data_p']}
    sg_data_n = {key: torch.from_numpy(data['sg_data_n'][key]).cuda() for key in data['sg_data_n']}
        
    emb_a, out_a = model(sg_data_a)
    emb_p, out_p = model(sg_data_p)
    emb_n, out_n = model(sg_data_n)
        
    emb_a = F.normalize(emb_a)
    emb_p = F.normalize(emb_p)
    emb_n = F.normalize(emb_n)

    return emb_a,out_a,emb_p,out_p,emb_n,out_n


def compute_recon_loss(criterion, out_a, out_p, out_n, images_a, images_p, images_n):

    loss_a = criterion(out_a, images_a)
    loss_p = criterion(out_p, images_p)
    loss_n = criterion(out_n, images_n)
        
    recon_loss = torch.mean(torch.stack([loss_a , loss_p , loss_n]))

    return recon_loss


def log_validation_loss(model, loader, criterion, expected_output_size, epoch):
    torch.set_grad_enabled(False)
    model.eval()

    # Setup meter
    losses_recon = AverageMeter()


    while True:
        data =  loader.get_batch('val')

        try:
            images = data['images'].cuda()
        except Exception as e:
            print(f"{data.keys()=}")
            raise e
        
        images = F.interpolate(images, size=expected_output_size)

        sg_data = {key: torch.from_numpy(data['sg_data'][key]).cuda() for key in data['sg_data']}


        #2. Forward model and compute loss

        x_enc, x_dec = model(sg_data)
        x_enc = F.normalize(x_enc)

        recon_loss_x = criterion(x_dec, images)

        losses_recon.update(recon_loss_x.item())

        if data['bounds']['wrapped']:
            break

    wandb.log({
        "val/loss_recon_avg": losses_recon.avg,
        "epoch": epoch,
    })

    print(f"Validation recon loss: {losses_recon.avg}")

    torch.set_grad_enabled(True)
    model.train()

def perform_tests_dml(model, loader_test, boundingBoxes,  save_dir, ep):
    model.eval()  
    save_file = save_dir + '/result.txt' 
    q_feat, q_fnames = extract_features(model, loader_test, split='val')
    g_feat, g_fnames = extract_features(model, loader_test, split='gallery')
    
    onlyGallery = True
    if not(onlyGallery):
        t_feat, t_fnames = extract_features(model, loader_test, split='train')
        g_feat = np.vstack((g_feat,t_feat))
        g_fnames = g_fnames + t_fnames 

    # Use train images as gallery
    # g_feat, g_fnames = extract_features(model, loader_test, split='train')

    q_feat = np.concatenate(q_feat)
    g_feat = np.concatenate(g_feat)
    
    distances = cdist(q_feat, g_feat, metric= 'euclidean')
    sort_inds = np.argsort(distances)

    overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1])
    overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1])
    

    print('\n\nep:%s'%(ep))
    print(save_dir)
    print('GAlleryOnly Flag:', onlyGallery)
    
    print('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]))        
    print('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]))
          
    with open(save_file, 'a') as f:
        f.write('\n\ep: {}\n'.format(ep))
        f.write('Model name: {}\n'.format(save_dir))
        f.write('GAlleryOnly Flag: {}\n'.format(onlyGallery))     
       
        f.write('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')
        f.write('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
        
 
def extract_features(model, loader, split):
    """Extract features from model.
    
    Args:
        model
        loader
        split: 'train', 'val', or 'gallery'
    """
    epoch_done = False 
    feat = []
    fnames = [] 
    c=0 
    
    torch.set_grad_enabled(False)
    while epoch_done == False:
        c+=1
        data = loader.get_batch(split)
        sg_data = {key: torch.from_numpy(data['sg_data'][key]).cuda() for key in data['sg_data']}
        x_enc, x_dec = model(sg_data)
        x_enc = F.normalize(x_enc)
        outputs = x_enc.detach().cpu().numpy()
        feat.append(outputs)
        fnames += [x['id'] for x in data['infos']]
    
        if data['bounds']['wrapped']:
            #print('Extracted features from {} images from {} split'.format(c, split))
            epoch_done = True
    
    print('Extracted features from {} images from {} split'.format(len(fnames), split))
    return feat, fnames


opt = opts_dml.parse_opt()
for arg in vars(opt):
    print(arg, getattr(opt,arg))

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
if __name__ == '__main__':
    main(opt)
    

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='elementwise_mean', pos_weight=None):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        return loss.mean()
    else:
        return loss.sum() 
    
    
    