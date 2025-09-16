import argparse
import os
import os.path as osp
import torch
torch.cuda.device_count()
import numpy as np
from main import main
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose gpu id to train on the server

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('--city', type=str, default='Wuhu', help='path to config file')
    
    parser.add_argument('--model', type=str, default='softgroup')
    parser.add_argument('--config', type=str, default='', help='path to config file')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--resume', default='epoch_108.pth', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    args = parser.parse_args()
    args.work_dir = './experiment/test_model_'+args.city
    if(args.model =='softgroup'):
        args.config='./configs/softgroup/softgroup_urbanbis.yaml'
    elif(args.model =='softgroup++'):
        args.config='./configs/softgroup++/softgroup++_urbanbis_building_pre.yaml'
    else:
        if('softgroup' in args.model):
            args.config='./configs/softgroup++/softgroup++_urbanbis_building.yaml'
    return args


if __name__ == '__main__':
    args = get_args()
    main(args,is_test=True)

