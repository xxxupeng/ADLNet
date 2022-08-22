#!/usr/bin/env bash
set -x
CUDA_VISIBLE_DEVICES=9 python -W ignore main_mm.py \
    --dataset sceneflow_mm \
    --datapath /ssd/ssd_central3/xp/Scene_Flow \
    --trainlist ./datalists/sceneflow_train_mm.txt \
    --testlist ./datalists/sceneflow_test.txt \
    --epochs 10 --lr 0.001  \
    --batch_size 2 --test_batch_size 2 \
    --maxdisp 192 \
    --model PSMNet \
    --postprocess monotony \
    --loss_func MM \
    --savemodeldir /ssd/ssd_central3/xp/Check_Point/MMGD/ \
    --model_name "SceneFlow_MM_monotony_1+pow(150,0.2)_ALL" \
    --loadmodel "/ssd/ssd_central3/xp/Check_Point/MMGD/MM_monotony_1+pow(150,0.2)_ALL_train_9.tar"
    