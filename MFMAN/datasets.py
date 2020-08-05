# -*- coding: utf-8 -*-
# @Author  : CS
import os
import torch.utils.data
import numpy as np
from PIL import Image
import torch
import cv2
import sys
sys.path.append('..')

class my_dataset(torch.utils.data.Dataset):

    def __init__(self,class_name, is_train=True, data_dir=None,transform=None):
        """VOC格式数据集
        Args:
            data_dir: VOC格式数据集根目录，该目录下包含：
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
            split： train、test 或者 eval， 对应于 ImageSets/Main/train.txt,eval.txt
        """
        # 类别
        self.class_names = class_name
        self.data_dir = data_dir
        self.is_train = is_train
        if data_dir:
            self.data_dir = data_dir
        self.split = 'train'       # train     对应于ImageSets/Main/train.txt
        if not self.is_train:
            self.split = 'test'    # test      对应于ImageSets/Main/test.txt
        self.transform = transform
        self.txt_file = os.path.join(self.data_dir, "{}.txt".format(self.split))
        # 从train.txt 文件中读取图片 id 返回ids列表
        self.ids = self._read_image_ids(self.txt_file)
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_name = self.ids[index]

        #print(image_name)
        # 解析Annotations/id.xml 读取id图片对应的 boxes, labels, is_difficult 均为列表
        labels = self._get_labels(image_name)
        # 读取 JPEGImages/id.jpg 返回Image.Image
        image = self._read_image(image_name)
        
        #print(image.shape)
        if self.transform:
            # print(image.shape)          
            image= self.transform(image)
        return image, labels
    # 返回 id, boxes， labels， is_difficult
    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip()[:8])
        return ids

    # 解析xml，返回 boxes， labels， is_difficult   numpy.array格式
    def _get_labels(self, image_name):
        labels = []
        with open(self.txt_file) as f:
            for line in f:
                #print(line.rstrip()[:8])
                if line.rstrip()[:8] == image_name:
                    line=line.strip('\n')  
                    class_name = line[9:]
                    #if class_name in ['people','plane','streetlamp','train','tree','truck']:
                        #labels.append(6)
                    #else:
                    labels.append(self.class_dict[class_name])
        return np.array(labels, dtype=np.int64)
    # 读取图片数据，返回Image.Image
    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, self.split, "{}".format(image_id))
        image = Image.open(image_file).convert("RGB")
        """ f self.darkchannel_dehaze == True:      
            img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            im1 = cv2.GaussianBlur(img, ksize=(29, 29), sigmaX=0, sigmaY=0)    
            image = Image.fromarray(cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)) """
        #image = np.array(image)
        return image
