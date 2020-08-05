import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import numpy as np 
from Data.datasets import my_dataset
'''
Constructing training and testing dataset.
'''
IMAGE_SIZE=(224,224)
IMAGE_SIZE_2 = (224,224)
CROP_SIZE = 224

def my_baseline_dataset(FLAG):
    #classname = {'bird':1,'boat':2,'bridge':3,'building':4,'bus':5,'car':6,'people':7,'plane':7,'streetlamp':7,'train':7,'tree':7,'truck':7}
    classname = {'bird':1,'boat':2,'bridge':3,'building':4,'bus':5,'car':6,'people':7,'plane':8,'streetlamp':9,'train':10,'tree':11,'truck':12}
    root_path = FLAG.root_dir
    data_name = FLAG.source
    data_name_tgt = FLAG.target
    data_path = os.path.join(root_path,data_name)
    data_path_tgt = os.path.join(root_path,data_name_tgt)
    train_path = os.path.join(data_path)
    #test_path = os.path.join(root_path,FLAG.target,'test/')
    test_path = os.path.join(data_path)
    test_target_path = os.path.join(data_path_tgt)
    #Following imagenet parameters
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,0.224,0.225])
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE_2),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = my_dataset(class_name = classname,data_dir=train_path,is_train=True,transform=train_transform)
    #print(train_dataset[1])
    #test_dataset = my_dataset(class_name = classname,data_dir=test_path,is_train=False,transform=test_transform)
    test_dataset_tgt = my_dataset(class_name = classname,data_dir=test_target_path,is_train=False,transform=test_transform)

    classes=['bird','boat','bridge','building','bus','car','people','plane','streetlamp','train','tree','truck']
    #classname = {'bird':1,'boat':2,'bridge':3,'building':4,'bus':5,'car':6,'people':7,'plane':7,'streetlamp':7,'train':7,'tree':7,'truck':7}

    return train_dataset,test_dataset_tgt,classes


def my_test_dataset(FLAG):
    #classname = {'bird':1,'boat':2,'bridge':3,'building':4,'bus':5,'car':6,'people':7,'plane':7,'streetlamp':7,'train':7,'tree':7,'truck':7}
    classname = {'bird':1,'boat':2,'bridge':3,'building':4,'bus':5,'car':6,'people':7,'plane':8,'streetlamp':9,'train':10,'tree':11,'truck':12}
    darkchannel_dehaze = FLAG.darkchannel_dehaze
    root_path = FLAG.root_dir
    saliency_map_add = FLAG.saliency_map_add
    saliency_map_cat = FLAG.saliency_map_cat
    data_path = os.path.join(root_path)

    train_path = os.path.join(data_path)
    #test_path = os.path.join(root_path,FLAG.target,'test/')
    test_path = os.path.join(data_path)
    
    #Following imagenet parameters
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,0.224,0.225])

    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    #print(train_dataset[1])
    test_dataset = my_dataset(class_name = classname,data_dir=train_path,is_train=False,transform=test_transform,saliency_map_add=saliency_map_add,saliency_map_cat=saliency_map_cat,darkchannel_dehaze=darkchannel_dehaze)
    classes=['bird','boat','bridge','building','bus','car','people','plane','streetlamp','train','tree','truck']
    
    return test_dataset,classes



def my_cross_dataset(FLAG):
    #classname = {'bird':1,'boat':2,'bridge':3,'building':4,'bus':5,'car':6,'people':7,'plane':7,'streetlamp':7,'train':7,'tree':7,'truck':7}
    classname = {'bird':1,'boat':2,'bridge':3,'building':4,'bus':5,'car':6,'people':7,'plane':8,'streetlamp':9,'train':10,'tree':11,'truck':12}
    root_path = FLAG.root_dir
    source_name = FLAG.source
    target_name = FLAG.target
    source_path = os.path.join(root_path,source_name)
    target_path = os.path.join(root_path,target_name)

    source_train_path = os.path.join(source_path)
    target_train_path = os.path.join(target_path)
    target_test_path = os.path.join(target_path)
    
    #Following imagenet parameters
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,0.224,0.225])
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE_2),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    source_train_dataset = my_dataset(class_name = classname,data_dir=source_train_path,is_train=True,transform=train_transform)
    #source_test_dataset = my_dataset(class_name = classname,data_dir=source_train_path,is_train=False,transform=train_transform)
    target_train_dataset = my_dataset(class_name = classname,data_dir=target_train_path,is_train=True,transform=train_transform)
    target_test_dataset = my_dataset(class_name = classname,data_dir=target_test_path,is_train=False,transform=test_transform)
    classes=['bird','boat','bridge','building','bus','car','people','plane','streetlamp','train','tree','truck']
    #classes=['bird','boat','bridge','building','bus','car','other']

    return source_train_dataset,target_train_dataset,target_test_dataset,classes
