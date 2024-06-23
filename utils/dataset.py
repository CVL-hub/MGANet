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
from utils.seed_init import place_seed_points
from utils.get_weak_anns import transform_anns

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', 'tif']



def mask_small_object(mask):
    """
    :param mask: input mask
    :return: weight map
    """
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, retval):
        mask[labels == i] = (1.0 - np.log(stats[i][4] / 65536))
    mask[labels == 0] = 1.0
    return mask


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
                        ms_transform=None, max_sp= 5):

        assert mode in ['train', 'val', 'demo', 'finetune']
        assert data_set in ['iSAID','LoveDA']
        if mode == 'finetune':
            assert ft_transform is not None
            assert ft_aug_size is not None

        # 15个类别，10个用来训练，5个用来测试
        if data_set == 'iSAID':
            self.num_classes = 15
        elif data_set == 'LoveDA':
            self.num_classes = 6
        self.max_sp = max_sp
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

        elif data_set == 'LoveDA':
            self.class_list = list(range(1, 7))  # [1,2,3,4,5,6]
            if self.split == 2:
                self.sub_list = list(range(1, 5))  # [1,2,3,4]
                self.sub_val_list = list(range(5, 7))  # [5,6]
            elif split == 1:
                self.sub_list = list(range(1, 3)) + list(range(5, 7))   # [1,2,5,6]
                self.sub_val_list = list(range(3, 5))  # [3,4]
            elif split == 0:
                self.sub_list = list(range(3, 7))  # [3,4,5,6]
                self.sub_val_list = list(range(1, 3))  # [1,2]s

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

        print('sub_list: ', self.sub_list) # [6-15]
        print('sub_val_list: ', self.sub_val_list)#[1-5]    

        # @@@ For convenience, we skip the step of building datasets and instead use the pre-generated lists @@@
        # if self.mode == 'train':
        #     data_list = '/workspace/BAM/lists/LoveDA/few_train.txt'
        #     self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list, True)
        #     assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        # elif self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
        #     data_list = '/workspace/BAM/lists/LoveDA/few_val.txt'
        #     self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list, False)
        #     assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)

        mode = 'train' if self.mode=='train' else 'val' # train

        fss_list_root = './lists/{}/fss_list/{}/'.format(data_set, mode) # ./lists/iSAID/fss_list/train/
        fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(split)
        fss_sub_class_file_list_path = fss_list_root + 'sub_class_file_list_{}.txt'.format(split)

        # Write FSS Data
        # with open(fss_data_list_path, 'w') as f:
        #     for item in self.data_list:
        #         img, label = item
        #         f.write(img + ' ')
        #         f.write(label + '\n')
        # with open(fss_sub_class_file_list_path, 'w') as f:
        #     f.write(str(self.sub_class_file_list))
        # exit()

        # Read FSS Data
        # f_str当中每一行的格式是image-mask
        with open(fss_data_list_path, 'r') as f:
            f_str = f.readlines()
        self.data_list = []
        for line in f_str:
            img, mask = line.split(' ')
            self.data_list.append((img, mask.strip()))

        with open(fss_sub_class_file_list_path, 'r') as f:
            f_str = f.read()
        # sub_class_file_list是一个dict
        # sub_class_file_list["key"]是一个列表
        # 每个列表包含对应的图片和其掩码的路径位置
        self.sub_class_file_list = eval(f_str)

        self.transform = transform

        self.ft_transform = ft_transform
        self.ft_aug_size = ft_aug_size
        self.ms_transform_list = ms_transform
      
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        #读取query set中的图片和掩码
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_ori = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_ori = label
        
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
        class_chosen = label_class[random.randint(1,len(label_class))-1]
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

        # for cls in range(1,self.num_classes+1):
        #     select_pix = np.where(label_b_tmp == cls)
        #     if cls in self.sub_list:
        #         label_b[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
        #     else:
        #         label_b[select_pix[0],select_pix[1]] = 0    
        
        # 得到列表
        file_class_chosen = self.sub_class_file_list[class_chosen]
        # 列表长度
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        # 取得k张support和其对应的mask图片
        # support图片不能与query中的图片一样
        # 并且support中每张图片也不一样
        for k in range(self.shot):
            # 任选一张图片下标
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]                
            support_idx_list.append(support_idx)
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
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image_ori = support_image
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            support_label_ori = support_label

            # mask = mask_to_onehot(support_label, self.num_classes)
            # mask = mask[1:, :, :]
            # support_edge = onehot_to_binary_edges(mask, 2, self.num_classes)
            # support_edge = support_edge * support_label
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

        scale_score = mask_small_object(np.array(label.clone(), dtype=np.uint8))
        scale_score = torch.from_numpy(scale_score)

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