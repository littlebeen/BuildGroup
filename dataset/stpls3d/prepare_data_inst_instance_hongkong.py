# https://github.com/meidachen/STPLS3D/blob/main/HAIS/data/prepare_data_inst_instance_stpls3d.py
import glob
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from prepare_data_inst_instance_urbanbis import is_exist


def split_pointcloud(cloud, size=50.0, stride=50, split='train'):
    all_cloud = []
    limit_max = np.amax(cloud[:, 0:3], axis=0)
    blocks = []
    if 'val' in split:
        size=250
        stride=125
    n=0
    if (limit_max[0] > size or limit_max[1] > size) and ( split!='test'):
        width = int(np.ceil((limit_max[0] - size) / stride)) + 1
        depth = int(np.ceil((limit_max[1] - size) / stride)) + 1
        if(width==0):
            width=1
        if(depth==0):
            depth=1
        cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]

        for (x, y) in cells:
            xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
            ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
            cond = xcond & ycond
            block = cloud[cond, :]
            instance_ids=np.unique(block[:,-1])

            #blocks.append(block)

            if not is_exist(all_cloud, instance_ids):
                all_cloud.append(instance_ids)
                mask = np.isin(cloud[:,-1], instance_ids)  # 检查 cloud 中是否有 array 的值
                block = cloud[mask]
                blocks.append(block)
            else:
                n+=1
    else:
        blocks.append(cloud[:, :])
    return blocks

def dataAug(points, semantic_keep):
    angle = random.randint(1, 359)
    angle_radians = math.radians(angle)
    rotation_matrix = np.array(
        [[math.cos(angle_radians), -math.sin(angle_radians), 0], [math.sin(angle_radians), math.cos(angle_radians), 0],
         [0, 0, 1]])
    points[:, :3] = points[:, :3].dot(rotation_matrix)
    points_kept = points[np.in1d(points[:, -2], semantic_keep)]
    return points_kept

def preparePthFiles(files_dir, split, output_folder, aug_times=0):
    ### save the coordinates so that we can merge the data to a single scene after segmentation for visualization
    out_json_path = os.path.join(output_folder, 'coord_shift.json')
    coord_shift = {}
    ### used to increase z range if it is smaller than this, over come the issue where spconv may crash for voxlization.
    z_threshold = 6

    # Map relevant classes to {1,...,14}, and ignored classes to -100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([0, 1, 2, 3, 4, 5, 6]):
        remapper[x] = i
    # Map instance to -100 based on selected semantic (change a semantic to -100 if you want to ignore it for instance)
    remapper_disable_instance_by_semantic = np.ones(150) * (-100)
    for i, x in enumerate([-100, 1, -100, -100, -100, -100, -100]):
        remapper_disable_instance_by_semantic[x] = i

    ### only augment data for these classes
    semantic_keep = [0, 1, 2, 3, 4, 5, 6]

    files = glob.glob(os.path.join(files_dir, "*.txt"))
    counter = 0
    for file in files:

        for aug_time in range(aug_times + 1):
            points = pd.read_csv(file, header=None, delimiter=' ').values
            indices = np.where(points[:, -1] >=0)[0]
            points = points[indices]  
            if aug_time != 0:
                points = dataAug(points, semantic_keep)
            name = os.path.basename(file).strip('.txt') + '_%d' % aug_time
            
            points = np.unique(points, axis=0)
            points = points.astype(float)
            if('1718' in name):
                points[:, 2]+=8329
            if('11SW8B(e833n816,e834n816)' in name):
                points[:, 2]+=361
            if('11SW18B(e833n813,e834n814)' in name):
                points[:, 2]+=14996
            points[:, :3] = points[:, :3] / 100
            points[:, :3] = np.around(points[:, :3], decimals=2)
            zeros_columns = np.zeros((points.shape[0], 3))
            points = np.concatenate((points[:,0:3], zeros_columns,points[:,3:5]), axis=1)
            points[:,6] = 1
            if split != 'test':
                coord_shift['globalShift'] = list(points[:, :3].min(0))
            min_z = points[:, :3].min(0)[2]
            points[:, :3] = points[:, :3] - points[:, :3].min(0)

            blocks = split_pointcloud(points, size=250, stride=125, split=split)
            for blockNum, block in enumerate(blocks):
                if (len(block) > 10000):
                    out_file_path = os.path.join(output_folder,  name + str(blockNum) + '_inst_nostuff.pth')
                    if (block[:, 2].max(0) - block[:, 2].min(0) < z_threshold):
                        block = np.append(
                            block, [[
                                block[:, 0].mean(0), block[:, 1].mean(0), block[:, 2].max(0) +
                                (z_threshold -
                                 (block[:, 2].max(0) - block[:, 2].min(0))), block[:, 3].mean(0),
                                block[:, 4].mean(0), block[:, 5].mean(0), -100, -100
                            ]],
                            axis=0)
                        print("range z is smaller than threshold ")
                        print(name + str(blockNum) + '_inst_nostuff')
                    if split != 'test':
                        out_file_name = name + str(blockNum) + '_inst_nostuff'
                        coord_shift[out_file_name] = list(block[:, :3].mean(0))
                    coords = np.ascontiguousarray(block[:, :3] - block[:, :3].mean(0))
                    coords_ground = np.ascontiguousarray(block[:, :3])
                    coords_ground[:,2] +=min_z
                    # coords = block[:, :3]
                    colors = np.ascontiguousarray(block[:, 3:6]) / 127.5 - 1

                    coords = np.float32(coords)
                    colors = np.float32(colors)
                    coords_ground = np.float32(coords_ground)
                    if split != 'test':
                        sem_labels = np.ascontiguousarray(block[:, -2]) #光取语义信息
                        sem_labels = sem_labels.astype(np.int32)
                        sem_labels = remapper[np.array(sem_labels)]  

                        instance_labels = np.ascontiguousarray(block[:, -1])
                        instance_labels = instance_labels.astype(np.float32)

                        disable_instance_by_semantic_labels = np.ascontiguousarray(block[:, -2])
                        disable_instance_by_semantic_labels = disable_instance_by_semantic_labels.astype(np.int32)
                        disable_instance_by_semantic_labels = remapper_disable_instance_by_semantic[
                            np.array(disable_instance_by_semantic_labels)]
                        instance_labels = np.where(disable_instance_by_semantic_labels == -100, -100, instance_labels)

                        # map instance from 0.
                        # [1:] because there are -100
                        if (np.min(instance_labels)<=-1):
                            unique_instances = (np.unique(instance_labels))[1:].astype(np.int32)
                        else:
                            unique_instances = (np.unique(instance_labels))[0:].astype(np.int32)
                        remapper_instance = np.ones(50000) * (-100)
                        for i, j in enumerate(unique_instances):
                            remapper_instance[j] = i

                        instance_labels = remapper_instance[instance_labels.astype(np.int32)]

                        unique_semantics = (np.unique(sem_labels)).astype(np.int32)

                        if (split == 'train' or split == 'val') and (
                                len(unique_instances) < 1):
                            print("unique insance: %d" % len(unique_instances))
                            print("unique semantic: %d" % len(unique_semantics))
                            print()
                            counter += 1
                        else:
                            torch.save((coords, colors, sem_labels, instance_labels,coords_ground), out_file_path)
                            print(len(coords))
                    else:
                        torch.save((coords, colors), out_file_path)
                    
    print("Total skipped file :%d" % counter)
    #json.dump(coord_shift, open(out_json_path, 'w'))


if __name__ == '__main__':
    city = 'Hongkong'
    data_folder = '/root/autodl-tmp/Urbanbis/reconstructed/'+city
    save_folder = '/root/autodl-tmp/Urbanbis/reconstructed/'+city+'_softgroup_building'
    filesOri = sorted(glob.glob(data_folder + '/*.txt'))

    split = 'train'
    train_files_dir = os.path.join(data_folder, split)
    train_out_dir = os.path.join(save_folder, split)
    os.makedirs(train_out_dir, exist_ok=True)
    preparePthFiles(train_files_dir, split, train_out_dir,aug_times=1)

    split = 'test_val'
    val_files_dir = os.path.join(data_folder, split)
    val_out_dir = os.path.join(save_folder, 'test_val_250m')
    os.makedirs(val_out_dir, exist_ok=True)
    preparePthFiles(val_files_dir, split, val_out_dir)

    split = 'val'
    val_files_dir = os.path.join(data_folder, split)
    val_out_dir = os.path.join(save_folder, 'val_250m')
    os.makedirs(val_out_dir, exist_ok=True)
    preparePthFiles(val_files_dir, split, val_out_dir)
