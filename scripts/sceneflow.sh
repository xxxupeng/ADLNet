#!/usr/bin/env bash
set -x
CUDA_VISIBLE_DEVICES='8,9' python -W ignore train_DDP.py \
    --dataset sceneflow \
    --datapath /ssd/ssd_central3/xp/Scene_Flow \
    --trainlist ./datalists/sceneflow_train.txt \
    --testlist ./datalists/sceneflow_test.txt \
    --epochs 10 --lr 0.001  \
    --batch_size 2 --test_batch_size 2 \
    --maxdisp 192 \
    --model PSMNet \
    --postprocess monotony \
    --loss_func MMF \
    --savemodeldir /ssd/ssd_central3/xp/Check_Point/MMGD/   \
    --model_name  'test' \
    --loadmodel "/ssd/ssd_central3/xp/Check_Point/MMGD/SceneFlow_MMF_monotony_laplace_b=0.8_alpha=0.8_plus_mixed_modal_train_49.tar"


# /ssd/ssd_central3/xp/Scene_Flow