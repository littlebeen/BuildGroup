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
class UrbanbisDataset(CustomDataset):

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
        max_leng= max(math.floor(maxxy[0].item())+1, math.floor(maxxy[1].item())+1)
        weight, height=max_leng,max_leng
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
        return xyz, xyz_middle, rgb, semantic_label, instance_label,None

    
    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        data = self.load(filename)
        # if(not self.training):
        #     name = filename.split('/')[-1].replace('_inst_nostuff.pth','').split('_')[0]
        #     f = filename.replace(filename.split('/')[-1],'')
        #     f = f.replace('test_val_250m','ground')+name+'.npy'
        #     ground = np.load(f)
        #     ground_xyz=data[-1]
        #     for i in range(len(ground_xyz)):
        #         ground_xyz[i,2] = ground_xyz[i,2]- ground[math.floor(ground_xyz[i,0]), math.floor(ground_xyz[i,1])]
        #     mask = ground_xyz[:, 2] < 0
        #     ground_xyz[mask, 2] = 0  
        #     data=data[0:-1]
        # ground_xyz = data[-1]
        
        #data=data[0:-1]
        ground_xyz = data[0]


        xyz_pre = data[0].copy()
        xyz_pre = xyz_pre - xyz_pre.min(0)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        if data is None:
            return None
        xyz, xyz_middle, rgb, semantic_label, instance_label,valid_idxs = data  

        xyz[:,2]=0
        xyz_middle[:,2]=0
        xyz_pre =xyz_pre[valid_idxs] if self.training else xyz_pre
        #save_point(xyz_pre,'xyz_pre')
        edge = getEdge(xyz_pre,instance_label)

        info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
        inst_num, inst_pointnum, inst_cls, pt_offset_label = info
        coord = torch.from_numpy(xyz).long()

        coord_float = torch.from_numpy(xyz_middle)
        xyz_pre = torch.from_numpy(xyz_pre).long()
        ground_xyz = torch.from_numpy(ground_xyz)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(feat.size(1)) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        edge = torch.from_numpy(edge)
        #save_point(coord_float[edge.bool()],'xyz_pre')
        return (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
                inst_pointnum, inst_cls, pt_offset_label,xyz_pre,edge,ground_xyz)    


    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        instance_labels = []

        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long
        pt_offset_labels = []
        xyz_pres =[]
        edges=[]
        ground_xyzs=[]
        total_inst_num = 0
        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
             inst_pointnum, inst_cls, pt_offset_label,xyz_pre,edge,ground_xyz) = data
            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num
            scan_ids.append(scan_id)
            edges.append(edge)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            xyz_pres.append(torch.cat([xyz_pre.new_full((xyz_pre.size(0), 1), batch_id), xyz_pre], 1))
            ground_xyzs.append(ground_xyz)
            coords_float.append(coord_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)
            pt_offset_labels.append(pt_offset_label)
            batch_id += 1
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        xyz_pres = torch.cat(xyz_pres, 0)
        ground_xyzs = torch.cat(ground_xyzs, 0)
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        edges = torch.cat(edges, 0).long()  # long (N)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
        pt_offset_labels = torch.cat(pt_offset_labels).float()

        spatial_shape = np.clip(
            coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        image,p2i = self.get2D(xyz_pres)

        return {
            'scan_ids': scan_ids,
            'coords': coords,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
            'image': image,
            'p2i': p2i,
            'edges': edges,
            'xyz_pres': xyz_pres,
            'ground_xyzs':ground_xyzs
        }

def getEdge(input_cloud,instance_label):
    n = input_cloud.shape[0]
    
    # 初始化输出数组
    output_cloud = np.zeros((n), dtype=np.float32)
    ins_uni = np.unique(instance_label)
    for instance_id in ins_uni:
        end_idx = np.where(instance_label == instance_id)[0]
        instance_points = input_cloud[end_idx]

        boundery = detect_edges_uniform(instance_points[:,0:2])


        # 设置边缘点信息
        output_cloud[end_idx] = boundery
        
    return output_cloud


def detect_edges_uniform(cloud_point):
    # save_point(cloud_point,'ins')
    allcloud2 = cloud_point.copy()
    allcloud2 = np.floor(allcloud2).astype(np.int32)
    maxx= np.max(allcloud2[:,0])
    maxy= np.max(allcloud2[:,1])
    weight, height=math.floor(maxx.item())+1, math.floor(maxy.item())+1
    image = np.zeros((weight,height),dtype=np.uint8)

    unique_array = np.unique(allcloud2, axis=0)
    for point in unique_array:
        image[math.floor(point[0]),math.floor(point[1])]= np.uint8(255)
    
    image_3 = np.empty((weight, height,3), dtype=image.dtype)
    image_3[:] = image[:, :, np.newaxis] 

    edges = cv2.Canny(image, 100, 200)
    edges[edges > 0] = 1.5
    edges[edges < 1] = 1.
    #save_image(edges)
    edges_point =np.zeros(allcloud2.shape[0])
    slice_inds = np.arange(0, allcloud2.shape[0],dtype=np.int32)
    edges_point[slice_inds]= edges[allcloud2[slice_inds,0],allcloud2[slice_inds,1]]
    return edges_point

def save_image(tensor):
    cv2.imwrite('output.png', tensor)

def save_point(points,name):
    with open(name+'.txt', 'w') as f:
        for row in points:
        # 将每一行转换为字符串并写入文件，行与行之间用换行符分隔
            f.write(' '.join(map(str, row.tolist())) + '\n')