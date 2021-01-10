#!/bin/bash
python test.py --dataset_name='SHA' --mode='crop' --nThreads=1 --gpu_ids='0' --batch_size=1 --net_name='csrpersp_crop' --test_model_name='model_path'
