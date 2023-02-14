#!/bin/sh
# 1 - BASELINE - mosaic experiment
EXP_NAME=2Task-NutAssembly-Debug
TASK_str=nut_assembly
EPOCH=20
BSIZE=45
python repo/mosaic/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME}   \
      bsize=${BSIZE} vsize=${BSIZE} actions.n_mixtures=2 actions.out_dim=64 attn.attn_ff=128  simclr.mul_intm=0  \
      simclr.compressor_dim=128 simclr.hidden_dim=256 epochs=${EPOCH}