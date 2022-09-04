import torch.nn.functional as F
from losses.gt_distribution import *


__loss__ = {
    "SL1" : F.smooth_l1_loss,
    "SM" : loss_sm,
    "SMF" : loss_sm_fixed,
    "MMF" : loss_mm_fixed,
    "MMFK" : loss_mm_fixed_KITTI,

}