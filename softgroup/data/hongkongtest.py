from .custom import CustomDataset
import torch
import numpy as np
from softgroup.ops import voxelization_idx
from torchvision import transforms
from PIL import Image
import os.path as osp
import math
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
import cv2
import os
class HongkongDataset(CustomDataset):

    CLASSES = (
                'building',)

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # ignore instance of class 0 and reorder class id
        instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def save_image(self,tensor):
        tensor = tensor.byte()
        to_pil = transforms.ToPILImage()
 
        # 应用转换
        pil_image = to_pil(tensor)
        
        # 保存 PIL 图像为 PNG 文件
        pil_image.save('./output_image.png')

    def get2D(self,cloud):
        allcloud2 = cloud[:,1:4].clone()
        allcloud2 = torch.floor(allcloud2.type(torch.float))
        maxxy,_ = torch.max(allcloud2,dim=0)
        weight, height=math.floor(maxxy[0].item())+1, math.floor(maxxy[1].item())+1
        allcloud2[:,2] = allcloud2[:,2]/(maxxy[2].item()+1)
        allcloud2 = torch.cat([cloud[:,0].unsqueeze(-1),allcloud2],dim=1)
        p2i = allcloud2[:,0:3]
        # p2i = torch.cat([cloud[:,0].unsqueeze(-1),p2i.unsqueeze(-1)],dim=1)
        image = torch.zeros(4,weight,height)
        for i in range(4):
            indices = np.where(allcloud2[:, 0] == i)[0]
            points = allcloud2[indices]
            cloud2 = points[:,1:4].clone()

            unique_array = np.unique(cloud2, axis=0)
            for point in unique_array:
                image[i, math.floor(point[0]),math.floor(point[1])]= 127.5+127.5*point[2]
                #image[i, math.floor(point[0]),math.floor(point[1])]= 255


        tensor_with_new_dim = image.unsqueeze(1)  # 形状变为 (1, 4, w*h)

        expanded_tensor = tensor_with_new_dim.repeat(1, 3, 1, 1)  
        #expanded_tensor = expanded_tensor.permute(0, 2, 3, 1)
        #self.save_image(expanded_tensor[0])
        return expanded_tensor,p2i.type(torch.long)

    def transform_train(self, xyz, rgb, semantic_label, instance_label, aug_prob=1.0):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        # xyz_middle = xyz / self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)
        max_tries = 5
        while (max_tries > 0):
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label,valid_idxs

    def transform_test(self, xyz, rgb, semantic_label, instance_label):
        xyz_middle = self.dataAugment(xyz, False, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def cut(self,ground_xyz):
        size=250
        stride=230.
        cloud = ground_xyz
        blocks = []
        limit_max = np.amax(cloud[:, 0:3], axis=0)
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
            if(len(cloud[cond])>50):
                blocks.append(cond)
        return blocks

    def random(self, data):
        new_data=[]
        xyz = data[0]
        size=2
        length= len(xyz)//size
        indices = np.random.choice(len(xyz), size=length, replace=False)
        for i in range(len(data)):
            new_data.append(data[i][indices])
        return new_data


    def get_cond_item(self,data,cond):
        xyz, rgb, semantic_label, instance_label = data
        xyz = xyz[cond]
        rgb = rgb[cond]
        semantic_label = semantic_label[cond]
        instance_label= instance_label[cond]
        return xyz, rgb, semantic_label, instance_label
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        
        #dst = './experiment/test_model'
        # find=False
        # for item2 in os.listdir(dst):
        #     if 'volume' not in item2:
        #             for item in self.filenames:
        #                 scan_id = osp.basename(item).replace(self.suffix, '')
        #                 if scan_id in item2:
        #                     find=True
        #                     break
        #             if find==False:
        #                 print(item2)

        scan_id = osp.basename(filename).replace(self.suffix, '')
        data = self.load(filename)
        #data= self.random(data)
        ground_xyz = data[-1]
        data=data[0:-1]
        blocks = self.cut(ground_xyz)
        ground_xyz = torch.from_numpy(ground_xyz)
        all_data=[]
        for block in blocks:
            new_data = self.get_cond_item(data,block)
            xyz_pre = new_data[0].copy()
            xyz_pre = xyz_pre - xyz_pre.min(0)
            new_data = self.transform_train(*new_data) if self.training else self.transform_test(*new_data)
            if new_data is None:
                return None
            xyz, xyz_middle, rgb, semantic_label, instance_label = new_data  

            xyz[:,2]=0
            xyz_middle[:,2]=0

            # info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
            # inst_num, inst_pointnum, inst_cls, pt_offset_label = info
            coord = torch.from_numpy(xyz).long()

            coord_float = torch.from_numpy(xyz_middle)
            xyz_pre = torch.from_numpy(xyz_pre).long()
            block = torch.from_numpy(block)
            feat = torch.from_numpy(rgb).float()
            if self.training:
                feat += torch.randn(feat.size(1)) * 0.1
            all_data.append((scan_id, coord, coord_float, feat,xyz_pre,ground_xyz,block))

        
        return all_data


    def collate_fn(self, batch):
        new_datas=[]
        for data in batch:
            for i in range(len(data)):
                scan_ids = []
                coords = []
                coords_float = []
                feats = []
               
                xyz_pres =[]
                ground_xyzs=[]
                conds=[]
                total_inst_num = 0
                batch_id = 0
                if data[i] is None:
                    continue
                (scan_id, coord, coord_float, feat,xyz_pre,ground_xyz,cond) = data[i]
                
                # total_inst_num += inst_num
                scan_ids.append(scan_id)
                coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
                xyz_pres.append(torch.cat([xyz_pre.new_full((xyz_pre.size(0), 1), batch_id), xyz_pre], 1))
                ground_xyzs.append(ground_xyz)
                conds.append(cond)
                coords_float.append(coord_float)
                feats.append(feat)
                
                # instance_pointnum.extend(inst_pointnum)
                # instance_cls.extend(inst_cls)

                batch_id += 1
                assert batch_id > 0, 'empty batch'
                if batch_id < len(batch):
                    self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

                # merge all the scenes in the batch
                coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
                xyz_pres = torch.cat(xyz_pres, 0)
                ground_xyzs = torch.cat(ground_xyzs, 0)
                conds = torch.cat(conds, 0)
                batch_idxs = coords[:, 0].int()
                coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
                feats = torch.cat(feats, 0)  # float (N, C)
               
                #instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
                #instance_cls = torch.tensor(np.array(instance_cls), dtype=torch.long)  # long (total_nInst)
                

                spatial_shape = np.clip(
                    coords.max(0)[0][1:].numpy() + 5, self.voxel_cfg.spatial_shape[0]+5, None)
                
                voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
                image,p2i = self.get2D(xyz_pres)
                conds = conds.numpy()
                new_datas.append({
                    'scan_ids': scan_ids,
                    'coords': coords,
                    'batch_idxs': batch_idxs,
                    'voxel_coords': voxel_coords,
                    'p2v_map': p2v_map,
                    'v2p_map': v2p_map,
                    'coords_float': coords_float,
                    'feats': feats,
                    
                    
                   
                    'spatial_shape': spatial_shape,
                    'batch_size': batch_id,
                    'image': image,
                    'p2i': p2i,
                    'xyz_pres': xyz_pres,
                    'ground_xyzs':ground_xyzs,
                    'conds':conds
                })
        return new_datas
        
def save_image(tensor):
    cv2.imwrite('output.png', tensor)

def save_point(points,name):
    with open(name+'.txt', 'w') as f:
        for row in points:
        # 将每一行转换为字符串并写入文件，行与行之间用换行符分隔
            f.write(' '.join(map(str, row.tolist())) + '\n')


 