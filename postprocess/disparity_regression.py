from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


# softargmax
class mean_disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(mean_disparityregression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        disp = torch.arange(self.maxdisp,dtype=x.dtype,device=x.device).reshape(1,self.maxdisp,1,1)
        out = torch.sum(x*disp,1, keepdim=True)
        return out



# unimodal by monotony range
class unimodal_disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(unimodal_disparityregression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        disp = torch.arange(self.maxdisp,dtype=x.dtype,device=x.device).reshape(1,self.maxdisp,1,1)
        index = torch.argmax(x,1,keepdim=True)
        mask = disp.repeat(x.size(0),1,x.size(2),x.size(3))
        mask2 = torch.arange(self.maxdisp+1,dtype=x.dtype,device=x.device).reshape([1,self.maxdisp+1,1,1]).repeat(x.size(0),1,x.size(2),x.size(3))

        x_diff_r = torch.diff(x,dim=1,prepend=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device),\
                            append=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device))
        x_diff_l = torch.diff(x,dim=1,prepend=torch.ones(x.size(0),1,x.size(2),x.size(3),dtype=x.dtype,device=x.device))
        
        
        index_r = torch.gt(x_diff_r * torch.gt(mask2,index),0).int()
        index_r = torch.argmax(index_r,1,keepdim=True)-1
        
        index_l = torch.lt(x_diff_l * torch.le(mask,index),0).int()
        index_l = (self.maxdisp-1) - torch.argmax(torch.flip(index_l,[1]),1,keepdim=True)
        
        mask = torch.ge(mask,index_l) * torch.le(mask,index_r)
        x = x * mask.data
        x = x / torch.sum(x,1,keepdim=True)
        
        out = torch.sum(x*disp,1, keepdim=True)
        return out

# modal by hard range
class hard_unimodal_disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(hard_unimodal_disparityregression, self).__init__()
        self.maxdisp = maxdisp
        

    def forward(self, x):
        disp = torch.arange(self.maxdisp,dtype=x.dtype,device=x.device).reshape(1,self.maxdisp,1,1)
        index = torch.argmax(x,1,keepdim=True).repeat(1,self.maxdisp,1,1)
        mask = disp.repeat(x.size(0),1,x.size(2),x.size(3))
        mask = torch.ge(mask,index-4) * torch.le(mask,index+4)
        x = x * mask.data
        x = x / torch.sum(x,1,keepdim=True)
        out = torch.sum(x*disp,1, keepdim=True)
        return out