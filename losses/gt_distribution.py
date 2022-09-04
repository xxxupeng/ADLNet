from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

# groud truth Laplace distribution
def LaplaceDisp2Prob(Gt,maxdisp=192,m=3,n=3,mode=False,dense_disp=None):
    N,H,W = Gt.shape
    if mode:
        if dense_disp is None:
            dense_disp = Gt
                                
        Gtpad = F.pad(dense_disp,(int(n/2-0.5),int(n/2+0.5),int(m/2-0.5),int(m/2+0.5)),mode='replicate')
        x = torch.zeros(N,m*n,H,W,device=Gt.device)
        for i in range(m):
            for j in range(n):
                x[:,n*i+j,:,:] = Gtpad[:,i:-(m-i),j:-(n-j)]
                
        x_var, _ = torch.var_mean(x,dim=1,keepdim=True)
        
        b = x_var.sqrt()
        b = 1 + torch.pow(b/25,0.5)

    else:
        b = 0.8
            
    Gt = torch.unsqueeze(Gt,1)
    disp = torch.arange(maxdisp,device=Gt.device)
    disp = disp.reshape(1,maxdisp,1,1).repeat(N,1,H,W)
    cost = -torch.abs(disp-Gt) / b

    return F.softmax(cost,dim=1)

# groud truth Gaussian distribution
def GaussianDisp2Prob(Gt,maxdisp=192):
    N,H,W = Gt.shape
    sigma2 = 2
    
    Gt = torch.unsqueeze(Gt,1)
    disp = torch.arange(maxdisp).reshape(1,maxdisp,1,1).repeat(N,1,H,W).cuda()
    cost = -torch.pow((disp-Gt),2) / (2*sigma2)
    return F.softmax(cost,dim=1)

def loss_sm_fixed(x,disp,mask,maxdisp=192):
    num = mask.sum()
    x = torch.log(x + 1e-30)
    mask = torch.unsqueeze(mask,1).repeat(1,maxdisp,1,1)
    Gt = LaplaceDisp2Prob(disp,maxdisp,mode=False).detach_()
#     Gt = GaussianDisp2Prob(disp,maxdisp).detach_()
    loss =  - (Gt[mask]*x[mask]).sum() / num 
    return loss

def loss_sm(x,disp,mask,maxdisp=192,dense_disp=None):
    num = mask.sum()
    x = torch.log(x + 1e-30)
    mask = torch.unsqueeze(mask,1).repeat(1,maxdisp,1,1)
    Gt = LaplaceDisp2Prob(disp,maxdisp,dense_disp=dense_disp).detach_()
    loss = - (Gt[mask]*x[mask]).sum() / num 

    return loss

def loss_mm_fixed(x,disp,mask,maxdisp,dense_disp=None,m=3,n=3):
    N,H,W = disp.shape
    
    num = mask.sum()
    x = torch.log(x + 1e-30)
    mask = torch.unsqueeze(mask,1).repeat(1,maxdisp,1,1)
    
    Gt1 = LaplaceDisp2Prob(disp,maxdisp,mode=False)
    
    Gtpad = F.pad(disp,(int(n/2-0.5),int(n/2+0.5),int(m/2-0.5),int(m/2+0.5)),mode='replicate')
    Gt_list = torch.zeros(N,m*n,H,W,device=x.device)
    for i in range(m):
        for j in range(n):
            Gt_list[:,n*i+j,:,:] = Gtpad[:,i:-(m-i),j:-(n-j)]
    mean = torch.mean(Gt_list,dim=1)
    valid = (torch.abs(mean-disp) > 5).unsqueeze(1)
    index = (Gt_list < (disp.unsqueeze(1)-10)) + (Gt_list > (disp.unsqueeze(1)+10))
    Gt_list = Gt_list * index

    mean = torch.mean(Gt_list,dim=1) * (m*n) / torch.sum(index,dim=1).clamp(min=1)
    Gt2 = LaplaceDisp2Prob(mean,maxdisp,mode=False)
    
    w = torch.sum(index,dim=1,keepdim=True) * 0.025   # 0.5-0.0625   0.8-0.025

    Gt = ((1-w*valid)*Gt1 + (w*valid)*Gt2).detach_()
    loss =  - (Gt[mask]*x[mask]).sum() / num 

    return loss

def loss_mm_fixed_KITTI(x,disp,mask,maxdisp,dense_disp=None,m=5,n=5):
    N,H,W = disp.shape
    
    num = mask.sum()
    x = torch.log(x + 1e-30)
    mask = torch.unsqueeze(mask,1).repeat(1,maxdisp,1,1)
    
    Gt1 = LaplaceDisp2Prob(disp,maxdisp,mode=False)
    
    Gtpad = F.pad(disp,(int(n/2-0.5),int(n/2+0.5),int(m/2-0.5),int(m/2+0.5)),mode='replicate')
    Gt_list = torch.zeros(N,m*n,H,W,device=x.device)
    for i in range(m):
        for j in range(n):
            Gt_list[:,n*i+j,:,:] = Gtpad[:,i:-(m-i),j:-(n-j)]
    invalid_num = torch.sum((Gt_list == 0),dim=1,keepdim=True)
    mean = torch.mean(Gt_list,dim=1,keepdim=True) * m*n / (m*n - invalid_num).clamp(min=1)
    valid = (torch.abs(mean-disp.unsqueeze(1)) > 5) #N1HW

    index = ((Gt_list < (disp.unsqueeze(1)-10)) + (Gt_list > (disp.unsqueeze(1)+10))) * (Gt_list > 0)
    Gt_list = Gt_list * index

    mean = torch.mean(Gt_list,dim=1) * (m*n) / torch.sum(index,dim=1).clamp(min=1)
    Gt2 = LaplaceDisp2Prob(mean,maxdisp,mode=False)
    
    w = torch.sum(index,dim=1,keepdim=True) * 0.2 / (m*n - 1 - invalid_num).clamp(min=1)   # alpha = 0.8

    Gt = ((1-w*valid)*Gt1 + (w*valid)*Gt2).detach_()
    loss =  - (Gt[mask]*x[mask]).sum() / num 

    return loss