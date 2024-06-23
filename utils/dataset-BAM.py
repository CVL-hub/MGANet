import os
import os.path
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm
from utils.get_weak_anns import transform_anns

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', 'tif']



def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):    
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []     

        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label
                                              == c)
                        tmp_label[target_pix[0],target_pix[1]] = 1
                        if tmp_label.sum() >= 2 * 32 * 32:      
                            new_label_class.append(c)     
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1 
                    if tmp_label.sum() >= 2 * 32 * 32:      
                        new_label_class.append(c)            

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list

from scipy.ndimage.morphology import distance_transform_edt
def onehot_to_binary_edges(mask, radius, num_classes):
    if radius < 0:
        return mask

    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])
    # maskmap = [[0, 0, 0], [255, 255, 255]]

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = (edgemap > 0).astype(np.uint8)
    # masknodes = np.array(maskmap)
    # edgemap = np.uint8(masknodes[edgemap.astype(np.uint8)])
    return edgemap


def mask_to_onehot(mask, num_classes):
    _mask = [mask == i for i in range(num_classes + 1)]
    # _mask = [mask == (i+1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, data_set=None, use_split_coco=False, \
                        transform=None, mode='train', ann_type='mask', \
                        ft_transform=None, ft_aug_size=None, \
                        ms_transform=None):

        assert mode in ['train', 'val', 'demo', 'finetune']
        assert data_set in ['iSAID','LoveDA']
        if mode == 'finetune':
            assert ft_transform is not None
            assert ft_aug_size is not None
        if data_set == 'iSAID':
            self.num_classes = 15
        self.mode = mode #train
        self.split = split   # 0代表现在是第几个fold
        self.shot = shot # 1
        self.data_root = data_root # ./data/iSAID
        self.ann_type = ann_type # mask

        if data_set == 'iSAID':
            self.class_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 2:
                self.sub_list = list(range(1, 11))  # [1,2,3,4,5,6,7,8,9,10]
                self.sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
            elif split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 16))  # [1,2,3,4,5,11,12,13,14,15]
                self.sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
            elif split == 0:
                self.sub_list = list(range(6, 16))  # [6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(1, 6))  # [1,2,3,4,5]
                # to do
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))     

        if self.mode == 'train':
            self.img_path = os.path.join(data_root, 'train/images')
            self.ann_path = os.path.join(data_root, 'train/semantic_png')
        else:
            self.img_path = os.path.join(data_root, 'val/images')
            self.ann_path = os.path.join(data_root, 'val/semantic_png')      

        print('sub_list: ', self.sub_list) # [6-15]
        print('sub_val_list: ', self.sub_val_list)#[1-5]    
        mode = 'train' if self.mode=='train' else 'val' # train

        self.transform = transform

        self.ft_transform = ft_transform
        self.ft_aug_size = ft_aug_size
        self.ms_transform_list = ms_transform
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.x = 1
    

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.num_classes):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise # 字典列表形式

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('data/splits/isaid/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.mode == 'train':  # For training, read image-metadata of "the other" folds
            for fold_id in range(3):
                if fold_id == self.split:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.mode, fold_id)
        elif self.mode == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.mode, self.split)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.mode, len(img_metadata)))

        return img_metadata
      
    def __len__(self):
        return len(self.img_metadata)
    
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        # mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')))
        mask = cv2.imread(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png', cv2.IMREAD_GRAYSCALE)
        return mask
    
    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        path = os.path.join(self.img_path, img_name) + '.png'
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        return image

    def __getitem__(self, index):
        label_class = []
        #读取query set中的图片和掩码
        image_path, class_sample = self.img_metadata[index] # 从零开始的class_sample
        image = self.read_img(image_path)
        label_path, _ = self.img_metadata[index]
        label = self.read_mask(label_path)

        # image的shape是[H,W,C], label的shape是[H,W]
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        # 标签中有哪些类别          
        label_class = np.unique(label).tolist()
        
        # 去掉背景和边缘
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []
        # 如果是val模式，就把属于sub_val_list中的类别加入到new_label中
        # 如果是train模式，就把类别加入到sub_list中，即训练类别中
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class    
        assert len(label_class) > 0
        # 选择任意的一个类别
        class_chosen = class_sample + 1
        # 返回的是一个二维的tuple，target_pix[0]表示x位置
        # target_pix[1]表示y位置，对应的label[x][y] == class_chosen
        # 因此每个列表的长度与符合条件位置一样
        target_pix = np.where(label == class_chosen)
        # 下面一样
        ignore_pix = np.where(label == 255)
        # 将对应的位置设置为1，表示前景
        # 边缘位置设置为255
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        label[ignore_pix[0],ignore_pix[1]] = 255


        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = set()
        # 取得k张support和其对应的mask图片
        # support图片不能与query中的图片一样
        # 并且support中每张图片也不一样
        for k in range(self.shot):
            # 任选一张图片下标
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_image_path in support_idx_list):
                support_image_path = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
                support_label_path = support_image_path              
            support_idx_list.add(support_image_path)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)


        # 得到support_image,support_mask, 其中在mask模型下
        # support_label_list_ori_mask与support_mask一样
        support_image_list_ori = []
        support_label_list_ori = []
        support_label_list_ori_mask = []
        subcls_list = []
        # 用于训练的的类别当中，当前选择的类别是第几类
        # 得到其下标
        if self.mode == 'train':
            subcls_list.append(self.sub_list.index(class_chosen))
        else:
            subcls_list.append(self.sub_val_list.index(class_chosen))
                
        for k in range(self.shot):  
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = self.read_img(support_image_path)
            support_label = self.read_mask(support_label_path)

            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            # 一样的，都是mask
            support_label, support_label_mask = transform_anns(support_label, self.ann_type)   # mask/bbox
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            support_label_mask[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list_ori.append(support_image)
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot                    
        
        raw_image = image.copy()
        raw_label = label.copy()
        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]
        # 使用transform转化图片和mask,放入到上面的新建的列表当中
        if self.transform is not None:
            image, label = self.transform(image, label)   # transform the triple
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = \
                    self.transform(support_image_list_ori[k], support_label_list_ori[k])

        s_xs = support_image_list
        s_ys = support_label_list

        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)

        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        image_name = image_path.split('/')[-1]

    
        # Return 训练的时候只有一个类别;
        if self.mode == 'train':
            return image, label, s_x, s_y, subcls_list
        elif self.mode == 'val':
            return image, label, s_x, s_y, subcls_list, raw_label, class_chosen
        elif self.mode == 'demo':
            total_image_list = support_image_list_ori.copy()
            total_image_list.append(raw_image)            
            return image, label, s_x, s_y, subcls_list, total_image_list, support_label_list_ori, support_label_list_ori_mask, raw_label

if __name__ == '__main__':
    train_data=SemData(split=0,shot=1 ,data_root='/workspace/BAM/data/LoveDA/',mode='val',data_set='LoveDA')