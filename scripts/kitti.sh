#!/usr/bin/env bash
set -x
CUDA_VISIBLE_DEVICES="9" python -W ignore submission.py \
    --dataset kitti \
    --datapath /ssd/ssd_central3/xp/KITTI_15 \
    --trainlist ./datalists/kitti15_train.txt \
    --testlist ./datalists/kitti15_sub.txt \
    --epochs 1 --lr 0.001  \
    --batch_size 4 --test_batch_size 1 \
    --maxdisp 192 \
    --model PSMNet \
    --postprocess monotony \
    --loss_func MMFK \
    --savemodeldir /ssd/ssd_central3/xp/Check_Point/MMGD/KITTI/   \
    --model_name  "MMF" \
    --loadmodel "/ssd/ssd_central3/xp/Check_Point/MMGD/KITTI/KITTI15_MMF_monotony_laplace_b=0.8_alpha=0.8_plus_mixed_modal_sparse_AMGD_train_999.tar"


# /data0/xp/KITTI_2015/data_scene_flow
# KITTI15_MMF_monotony_laplace_b=0.8_alpha=0.8_plus_mixed_modal_sparse_AMGD
# KITTI15_SMF_monotony_laplace_b=0.8_AMGD
# SceneFlow_MMF_monotony_laplace_b=0.8_alpha=0.8_plus_mixed_modal_train_49.tar
# KITTI/KITTI15_SMF_monotony_laplace_b=0.8_AMGD_train_19.tar
# KITTI/KITTI15_MMF_monotony_laplace_b=0.8_alpha=0.8_plus_mixed_modal_sparse_AMGD_train_499.tar