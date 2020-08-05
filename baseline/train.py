import torch
import torchvision
from options import parser
import models
import os
import math
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dataset import my_baseline_dataset,my_cross_dataset
from utils import log_to_csv,adjust_lr,adjust_lr_v2,mmd_linear,CORAL,weights_init,mmd_rbf_noaccelerate
from tensorboardX import SummaryWriter
from torch.utils.data import dataloader
import numpy as np
from focalloss import FocalLoss as focal_loss
from focalloss import CELoss as ce_loss
global FLAG 
FLAG = parser.parse_args()
if FLAG.mode == 'single':
    print("Root_dir: {}\n".format(FLAG.root_dir),
        "dataset name: {}\n".format(FLAG.source),
        "Net: {}\n".format(FLAG.arch),
        "Training Mode: {}\n".format(FLAG.mode),
        "Batchsize: {}\n".format(FLAG.batch_size),
        "Epoches: {}\n".format(FLAG.epoch),
        "Gpu group: {}\n".format(FLAG.gpus)
        )

else:
    print("Root_dir: {}\n".format(FLAG.root_dir),
        "source name: {}\n".format(FLAG.source),
        "target name: {}\n".format(FLAG.target),
        "Net: {}\n".format(FLAG.arch),
        "Training Mode: {}\n".format(FLAG.mode),
        "Adaptation Mode: {}\n".format(FLAG.adapt_mode),
        "Batchsize: {}\n".format(FLAG.batch_size),
        "Epoches: {}\n".format(FLAG.epoch),
        "Gpu group: {}\n".format(FLAG.gpus)
        )


        
#os.environ['CUDA_VISIBLE_DEVICES']=FLAG.gpus
if len(FLAG.gpus)>1:
    gpus_id = FLAG.gpus.split(',')
    DEVICE = torch.device('cuda:{}'.format(FLAG.gpus) if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cuda:{}'.format(FLAG.gpus) if torch.cuda.is_available() else 'cpu')

def baseline_test(model,test_loader,epoch):

    model.eval()
    num_class = FLAG.num_class
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))

    with torch.no_grad():
        
        for i,data in enumerate(test_loader,0):
            
            inputs,labels = data[0].to(DEVICE),data[1].to(DEVICE).squeeze()
            outputs = model(inputs)
            _,predicted_ind = torch.max(outputs.data,1)
            c = (predicted_ind==labels).squeeze()
            for v in range(len(labels)):
                label = labels[v]
                class_correct[label]+=c[v].item()
                class_total[label] += 1
    
    acc_test = float(sum(class_correct)/sum(class_total))
    print('Epoch: [{}/{}], accuracy on {} test dataset: {:.2f}%'.format(epoch+1, FLAG.epoch, FLAG.source,100*acc_test))
    return acc_test,class_correct,class_total
def cross_test1(model,test_loader,epoch):

    model.eval()
    num_class = FLAG.num_class
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))

    with torch.no_grad():
        
        for i,data in enumerate(test_loader,0):
            
            inputs,labels = data[0].to(DEVICE),data[1].to(DEVICE).squeeze()
            outputs = model(inputs)
            _,predicted_ind = torch.max(outputs.data,1)
            c = (predicted_ind==labels).squeeze()
            for v in range(len(labels)):
                label = labels[v]
                class_correct[label]+=c[v].item()
                class_total[label] += 1
    
    acc_test = float(sum(class_correct)/sum(class_total))
    print('Epoch: [{}/{}], accuracy on {} test dataset: {:.2f}%'.format(epoch, FLAG.epoch, FLAG.target,100*acc_test))
    return acc_test,class_correct,class_total
def cross_test(model,test_loader,epoch):

    model.eval()
    num_class = FLAG.num_class
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))

    with torch.no_grad():
        
        for i,data in enumerate(test_loader,0):
            
            inputs,labels = data[0].to(DEVICE),data[1].to(DEVICE).squeeze()
            outputs,_,_ = model(inputs,inputs)
            _,predicted_ind = torch.max(outputs.data,1)
            c = (predicted_ind==labels).squeeze()
            for v in range(len(labels)):
                label = labels[v]
                class_correct[label]+=c[v].item()
                class_total[label] += 1
    
    acc_test = float(sum(class_correct)/sum(class_total))
    print('Epoch: [{}/{}], accuracy on {} test dataset: {:.2f}%'.format(epoch, FLAG.epoch, FLAG.target,100*acc_test))
    return acc_test,class_correct,class_total

def rev_test(model,test_loader,epoch):

    model.eval()
    num_class = FLAG.num_class
    mode = FLAG.mode
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))
    len_test_loader = len(test_loader.dataset)
    test_loss =0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        
        for i,data in enumerate(test_loader,0):
            
            inputs,labels = data[0].to(DEVICE),data[1].to(DEVICE).squeeze()

            if mode == 'rev':
                outputs,_ = model(inputs)
            elif mode == 'rev_2':
                outputs,_ = model(inputs,alpha=0)
            test_loss += criterion(outputs,labels).item()
            _,predicted_ind = torch.max(outputs.data,1)
            c = (predicted_ind==labels).squeeze()
            for v in range(len(labels)):
                label = labels[v]
                class_correct[label]+=c[v].item()
                class_total[label] += 1
    
    acc_test = float(sum(class_correct)/sum(class_total))
    print('Epoch: [{}/{}], accuracy on {} test dataset: {:.2f}%, test_loss: {:.4f}'.format(epoch, 
        FLAG.epoch, FLAG.target,100*acc_test,test_loss/len_test_loader))
    return acc_test,class_correct,class_total,test_loss

def mcd_test(G,C1,C2,test_loader,epoch):

    G.eval()
    C1.eval()
    C2.eval()

    num_class = FLAG.num_class
    class_correct_1 = list(0. for i in range(num_class))
    class_correct_2 = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))
    len_test_loader = len(test_loader.dataset)
    test_loss_1 = 0
    test_loss_2 = 0
    correct = 0
    correct2 = 0
    size = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i,data in enumerate(test_loader,0):

            inputs,labels = data[0].to(DEVICE),data[1].to(DEVICE).squeeze()
            features = G(inputs)
            output1 = C1(features)
            output2 = C2(features)
            test_loss_1 += criterion(output1,labels).item()
            test_loss_2 += criterion(output2,labels).item()
            _,predicted_ind_1 = torch.max(output1.data,1)
            _,predicted_ind_2 = torch.max(output2.data,1)
            ac_1 = (predicted_ind_1==labels).squeeze()
            ac_2 = (predicted_ind_2==labels).squeeze()
            for v in range(len(labels)):
                label = labels[v]
                class_correct_1[label]+=ac_1[v].item()
                class_correct_2[label]+=ac_2[v].item()
                class_total[label] += 1
    
    acc_test_1 = float(sum(class_correct_1)/sum(class_total))
    acc_test_2 = float(sum(class_correct_2)/sum(class_total))
    print('Epoch: [{}/{}], accuracy on {} test dataset: {:.2f}% /and {:.2f}%, test_loss: {:.4f} /and {:.4f}'.format(epoch, 
        FLAG.epoch, FLAG.target,100*acc_test_1,100*acc_test_2,test_loss_1/len_test_loader,test_loss_2/len_test_loader))
    return acc_test_1,acc_test_2,class_correct_1,class_correct_2,class_total,test_loss_1/len_test_loader,test_loss_2/len_test_loader

'''
===========================
Baseline
===========================
'''
def baseline_train():
    #Basic parameters
    gpus = FLAG.gpus
    batch_size = FLAG.batch_size
    epoches = FLAG.epoch
    LOG_INTERVAL = 10
    TEST_INTERVAL = 2
    data_name = FLAG.source
    target_name =FLAG.target
    model_name = FLAG.arch
    l2_decay = 5e-4
    lr = FLAG.lr
    #Loading dataset
    if FLAG.isLT:
        train_dataset,test_dataset,classes = baseline_LT_dataset(FLAG)
    else:    
        train_dataset,test_dataset_tgt,classes = my_baseline_dataset(FLAG)
        #print(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,
                    shuffle=True,num_workers=8,drop_last=True)
    #test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,
     #               shuffle=False,num_workers=8)

    test_loader_tgt = torch.utils.data.DataLoader(dataset=test_dataset_tgt,batch_size=batch_size,
                    shuffle=False,num_workers=8)
    #Define model
    Nets = models.Models(FLAG)
    base_model = Nets.model()
    if len(gpus)>1:
        gpus = gpus.split(',')
        gpus = [int(v) for v in gpus]
        base_model = nn.DataParallel(base_model,device_ids=gpus)

    base_model.to(DEVICE)
    #print(base_model)
    #Define Optimizer
    paras = dict(base_model.named_parameters())
    paras_new = []
    if 'resnet' or 'resnest' in model_name:
        for k,v in paras.items():
            if 'fc' not in k:
                paras_new.append({'params':[v],'lr':1e-3})
            else:
                paras_new.append({'params':[v],'lr':1e-2})

    elif model_name == 'vgg' or model_name == 'alexnet':
        for k,v in paras.items():
            if 'classifier.6' not in k:
                paras_new.append({'params':[v],'lr':1e-3})
            else:
                paras_new.append({'params':[v],'lr':1e-2})

    #print(paras_new)
    optimizer = optim.SGD(paras_new,lr=lr,momentum=0.9,weight_decay=l2_decay)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[50,80],gamma=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.94)
    #print(optimizer.param_groups[-1]['lr'])
    #Define loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    #Training
    
    best_result = 0.0
    best_result1 = 0.0
    #Model store
    if FLAG.isLT:
        model_dir = os.path.join('./models/','baseline-'+data_name+'-'+target_name+'-'+model_name)
    else:
        model_dir = os.path.join('./models/','baseline-'+data_name+'-'+model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #Tensorboard configuration
    if FLAG.isLT:
        log_dir = os.path.join('./logs/','baseline-'+data_name+'-'+target_name+'-'+model_name)
    else:
        log_dir = os.path.join('./logs/','baseline-'+data_name+'-'+model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    for epoch in range(epoches):
        base_model.train()
        #scheduler.step(epoch)
        running_loss = 0.0
        
        for i,data in enumerate(train_loader,0):
            label = torch.squeeze(data[1].to(DEVICE))
            inputs,labels = data[0].to(DEVICE),label
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs,labels)
            #log training loss
            if i%5 ==0:
                n_iter = epoch*len(train_loader)+i
                writer.add_scalar('data/training loss',loss,n_iter)
                #print(optimizer.param_groups[0]['lr'])
            loss.backward()
            optimizer.step()

            #Print statistics
            running_loss += loss.item()
            if i%LOG_INTERVAL == 0: #Print every 30 mini-batches
                print('Epoch:[{}/{}],Batch:[{}/{}] loss: {:4f}'.format(epoch+1,epoches,i+1,len(train_loader),running_loss/30))
                running_loss = 0

        scheduler.step(epoch)

        if epoch%TEST_INTERVAL ==0:   #Every 2 epoches
            
            #acc_test,class_corr,class_total=baseline_test(base_model,test_loader,epoch)
            acc_test1,class_correct1,class_total1=baseline_test(base_model,test_loader_tgt,epoch)
            #log test acc
            writer.add_scalar('data/test accuracy',acc_test,epoch)
            #Store the best model
            #if acc_test>best_result:
                #log results for classes
                #log_path = model_path = os.path.join(model_dir,
                            #'{}-{}-epoch_{}-accval_{}.csv'.format(data_name,model_name,epoch,round(acc_test,3)))
                #log_to_csv(log_path,classes,class_corr,class_total)
                #best_result = acc_test
           # else:
              #  print('The results in this epoch cannot exceed the best results !')
            if acc_test1>best_result1:
                #log results for classes
                log_path1 = model_path = os.path.join(model_dir,
                            '{}-{}-epoch_{}-accvaltgt_{}.csv'.format(data_name,model_name,epoch,round(acc_test1,3)))
                log_to_csv(log_path1,classes,class_correct1,class_total1)
                best_result1 = acc_test1
            else:
                print('The results in this epoch cannot exceed the best results !')
    
    writer.close()

'''
===========================
Dann
===========================
'''
def Revgrad_train():

    #Basic parameters
    gpus = FLAG.gpus
    batch_size = FLAG.batch_size
    epoches = FLAG.epoch
    init_lr = FLAG.lr
    LOG_INTERVAL = 10
    TEST_INTERVAL = 2
    source_name = FLAG.source
    target_name = FLAG.target
    model_name = FLAG.arch
    adapt_mode = FLAG.adapt_mode
    momentum=0.9
    l2_decay= 5e-4

    #Loading dataset
    if FLAG.isLT:
        source_train,target_train,target_test,classes = cross_dataset_LT(FLAG)
    else:
        source_train,target_train,target_test,classes = my_cross_dataset(FLAG)
    source_train_loader = torch.utils.data.DataLoader(dataset=source_train,batch_size=batch_size,
                    shuffle=True,num_workers=8,drop_last=True)
    target_train_loader = torch.utils.data.DataLoader(dataset=target_train,batch_size=batch_size,
                    shuffle=True,num_workers=8,drop_last=True)
    target_test_loader = torch.utils.data.DataLoader(dataset=target_test,batch_size=batch_size,
                    shuffle=False,num_workers=8)
    #Define model
    cross_model = models.RevGrad_v3(FLAG)
    
    if len(gpus)>1:
        gpus = gpus.split(',')
        gpus = [int(v) for v in gpus]
        cross_model = nn.DataParallel(cross_model,device_ids=gpus)
        cross_model.to(DEVICE)
        #Define Optimizer
        optimizer_fea = optim.SGD([{'params':cross_model.module.sharedNet.parameters()},
                            {'params':cross_model.module.cls_fc.parameters(),'lr':init_lr}],
                            lr=init_lr/10,momentum=momentum,weight_decay=l2_decay)
        optimizer_critic = optim.SGD([{'params':cross_model.module.domain_fc.parameters(),'lr':init_lr}],
                            lr=init_lr,momentum=momentum,weight_decay=l2_decay)

    else:
        cross_model.to(DEVICE)
        #Define Optimizer
        optimizer_fea = optim.SGD([{'params':cross_model.sharedNet.parameters()},
                            {'params':cross_model.cls_fc.parameters(),'lr':init_lr}],
                            lr=init_lr/10,momentum=momentum,weight_decay=l2_decay)
        optimizer_critic = optim.SGD([{'params':cross_model.domain_fc.parameters(),'lr':init_lr}],
                            lr=init_lr,momentum=momentum,weight_decay=l2_decay)

    #loss function
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
    #criterion1 = focal_loss
    #Training
    
    best_result = 0.0
    #Model store
    model_dir = os.path.join('./cross_models/',adapt_mode+'-'+source_name+'2'+target_name+'-'+model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #Tensorboard configuration
    log_dir = os.path.join('./cross_logs/',adapt_mode+'-'+source_name+'2'+target_name+'-'+model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)

    for epoch in range(1,epoches+1):
        cross_model.train()
        len_source_loader= len(source_train_loader)
        len_target_loader = len(target_train_loader)
        iter_source = iter(source_train_loader)
        iter_target = iter(target_train_loader)
        dlabel_src = torch.ones(batch_size).long()
        dlabel_tgt = torch.zeros(batch_size).long()

        if len_target_loader <= len_source_loader:
            iter_num = len_target_loader
            which_dataset = True
        else:
            iter_num = len_source_loader
            which_dataset = False
        #Adaptive learning rate
        optimizer_fea,optimizer_critic = adjust_lr_v2(optimizer_fea,optimizer_critic,epoch,FLAG)
        writer.add_scalar('data/SharedNet lr',optimizer_fea.param_groups[0]['lr'],epoch)
        for i in range(1,iter_num+1):

            if which_dataset:
                target_data,target_label = next(iter_target)
                if i % len_target_loader == 0:
                    iter_source  = iter(source_train_loader)
                source_data,source_label = next(iter_source)
            else:
                source_data,source_label = next(iter_source)
                if i % len_source_loader == 0:
                    iter_target  = iter(target_train_loader)
                target_data,target_label = next(iter_target)
                
                
            input_source_data,input_source_label = source_data.to(DEVICE),source_label.to(DEVICE).squeeze()
            src_dlabel,tgt_dlabel = dlabel_src.to(DEVICE),dlabel_tgt.to(DEVICE)

            clabel_src,dlabel_pred_src = cross_model(input_source_data)
            label_loss = criterion(clabel_src , input_source_label)
            critic_loss_src = criterion1(dlabel_pred_src,src_dlabel)
            confusion_loss_src = 0.5*criterion1(dlabel_pred_src,src_dlabel)+criterion1(dlabel_pred_src,tgt_dlabel)

            
            
            
            input_target_data,input_target_label = target_data.to(DEVICE),target_label.to(DEVICE)

            clabel_tgt,dlabel_pred_tgt = cross_model(input_target_data)
            critic_loss_tgt = criterion1(dlabel_pred_tgt,tgt_dlabel)
            confusion_loss_tgt = 0.5*criterion1(dlabel_pred_tgt,src_dlabel)+criterion1(dlabel_pred_tgt,tgt_dlabel)

            confusion_loss_total = (confusion_loss_src+confusion_loss_tgt)*0.2
            fea_loss_total = confusion_loss_total+label_loss
            critic_loss_total = (critic_loss_src+critic_loss_tgt)*0.2
            
            optimizer_fea.zero_grad()
            fea_loss_total.backward(retain_graph=True)
            optimizer_fea.step()
            optimizer_fea.zero_grad()
            optimizer_critic.zero_grad()
            critic_loss_total.backward()
            optimizer_critic.step()

            
            if i%5 ==0:
                n_iter = (epoch-1)*len_target_loader+i
                #writer.add_scalar('data/fea_loss',fea_loss_total,n_iter)
                writer.add_scalar('data/train_loss',critic_loss_total,n_iter)

            #Print statistics
            if i%LOG_INTERVAL == 0: #Print every 30 mini-batches
                print('Epoch:[{}/{}],Batch:[{}/{}] \tconfusion_loss: {:.4f} \tsource_label_loss: {:.4f} \tdomain_loss: {:.4f}'
                        .format(epoch,epoches,i,len_target_loader,confusion_loss_total.item(),label_loss.item(),critic_loss_total.item()))


        if epoch%TEST_INTERVAL ==0:   #Every 2 epoches
            
            acc_test,class_corr,class_total,_=rev_test(cross_model,target_test_loader,epoch)
            #log test acc
            writer.add_scalar('data/test accuracy',acc_test,epoch)
            #Store the best model
            if acc_test>best_result:
                model_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}.pth'.format(source_name,target_name,model_name,epoch,round(acc_test,3)))
                torch.save(cross_model,model_path)
                #log results for classes
                log_path = model_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}.csv'.format(source_name,target_name,model_name,epoch,round(acc_test,3)))
                log_to_csv(log_path,classes,class_corr,class_total)
                best_result = acc_test
            else:
                print('The results in this epoch cannot exceed the best results !')

    writer.close()
    

'''
===========================
DDC and CORAL
===========================
'''

def cross_train():

    #Basic parameters
    gpus = FLAG.gpus
    batch_size = FLAG.batch_size
    epoches = FLAG.epoch
    init_lr = FLAG.lr
    LOG_INTERVAL = 10
    TEST_INTERVAL = 2
    source_name = FLAG.source
    target_name = FLAG.target
    model_name = FLAG.arch
    adapt_mode = FLAG.adapt_mode
    l2_decay = 5e-4

    #Loading dataset
    if FLAG.isLT:
        source_train,target_train,target_test,classes = cross_dataset_LT(FLAG)
    else:
        source_train,target_train,target_test,classes = my_cross_dataset(FLAG)
    source_train_loader = torch.utils.data.DataLoader(dataset=source_train,batch_size=batch_size,
                    shuffle=True,num_workers=8,drop_last=True)
    target_train_loader = torch.utils.data.DataLoader(dataset=target_train,batch_size=batch_size,
                    shuffle=True,num_workers=8,drop_last=True)
    target_test_loader = torch.utils.data.DataLoader(dataset=target_test,batch_size=batch_size,
                    shuffle=False,num_workers=8)
    #Define model
    if adapt_mode == 'ddc':
        cross_model = models.DDCNet(FLAG)
        
        #adapt_loss_function = mmd_linear
        adapt_loss_function = mmd_rbf_noaccelerate
        #print(model)

    elif adapt_mode == 'coral':
        cross_model = models.DeepCoral(FLAG)
        adapt_loss_function = CORAL

    elif adapt_mode == 'mmd':
        cross_model = models.DDCNet(FLAG)
        adapt_loss_function = mmd_linear

    else:
        print('The adaptive model name is wrong !')
    
    if len(gpus)>1:
        gpus = gpus.split(',')
        gpus = [int(v) for v in gpus]
        cross_model = nn.DataParallel(cross_model,device_ids=gpus)

    cross_model.to(DEVICE)
    #Define Optimizer
    if len(gpus)>1:
        optimizer = optim.SGD([{'params':cross_model.module.sharedNet.parameters()},
                            {'params':cross_model.module.cls_fc.parameters(),'lr':init_lr}],
                            lr=init_lr/10,momentum=0.9,weight_decay=l2_decay)

    else:
        optimizer = optim.SGD([{'params':cross_model.sharedNet.parameters()},
                            {'params':cross_model.cls_fc.parameters(),'lr':init_lr}],
                            lr=init_lr/10,momentum=0.9,weight_decay=l2_decay)
    #print(optimizer.param_groups)
    #loss function
    criterion = torch.nn.CrossEntropyLoss()
    #Training
    
    best_result = 0.0
    #Model store
    model_dir = os.path.join('./cross_models/',adapt_mode+'-'+source_name+'2'+target_name+'-'+model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #Tensorboard configuration
    log_dir = os.path.join('./cross_logs/',adapt_mode+'-'+source_name+'2'+target_name+'-'+model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)

    for epoch in range(1,epoches+1):
        cross_model.train()
        len_source_loader= len(source_train_loader)
        len_target_loader = len(target_train_loader)
        iter_source = iter(source_train_loader)
        iter_target = iter(target_train_loader)

        if len_target_loader <= len_source_loader:
            iter_num = len_target_loader
            which_dataset = True
        else:
            iter_num = len_source_loader
            which_dataset = False
        #Adaptive learning rate
        optimizer = adjust_lr(optimizer,epoch,FLAG)
        writer.add_scalar('data/SharedNet lr',optimizer.param_groups[0]['lr'],epoch)
        running_loss = 0.0
        for i in range(1,iter_num+1):

            if which_dataset:
                target_data,_ = next(iter_target)
                if i % len_target_loader == 0:
                    iter_source = iter(source_train_loader)
                source_data,source_label = next(iter_source)
            else:
                source_data,source_label = next(iter_source)
                if i % len_source_loader == 0:
                    iter_target = iter(target_train_loader)
                target_data,_ = next(iter_target)

            input_source_data,input_source_label = source_data.to(DEVICE),source_label.to(DEVICE).squeeze()
            input_target_data = target_data.to(DEVICE)

            optimizer.zero_grad()


            label_source_pred,source_output,target_output = cross_model(input_source_data, input_target_data)
            loss_adapt = adapt_loss_function(source_output,target_output)
            loss_cls = criterion(label_source_pred,input_source_label)
            lambda_1 = 2 / (1 + math.exp(-10 * (epoch) / epoches)) - 1
            loss = loss_cls + lambda_1 * loss_adapt
            
            
            if i%5 ==0:
                n_iter = (epoch-1)*len_target_loader+i
                writer.add_scalar('data/adapt loss',loss_adapt,n_iter)
                writer.add_scalar('data/cls loss',loss_cls,n_iter)
                writer.add_scalar('data/total loss',loss,n_iter)
                #print(optimizer.param_groups[0]['lr'])

            loss.backward()
            optimizer.step()

            #Print statistics
            running_loss += loss.item()
            if i%LOG_INTERVAL == 0: #Print every 30 mini-batches
                print('Epoch:[{}/{}],Batch:[{}/{}] loss: {}'.format(epoch,epoches,i,len_target_loader,running_loss/LOG_INTERVAL))
                running_loss = 0

        if epoch%TEST_INTERVAL ==0:   #Every 2 epoches
            
            acc_test,class_corr,class_total=cross_test(cross_model,target_test_loader,epoch)
            #log test acc
            writer.add_scalar('data/test accuracy',acc_test,epoch)
            #Store the best model
            if acc_test>best_result:
                model_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}.pth'.format(source_name,target_name,model_name,epoch,round(acc_test,3)))
                torch.save(cross_model,model_path)
                #log results for classes
                log_path = model_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}.csv'.format(source_name,target_name,model_name,epoch,round(acc_test,3)))
                log_to_csv(log_path,classes,class_corr,class_total)
                best_result = acc_test
            else:
                print('The results in this epoch cannot exceed the best results !')

    writer.close()

'''
===========================
Dann-onestep
===========================
'''
def Revgrad_train_onestep():

    #Basic parameters
    gpus = FLAG.gpus
    batch_size = FLAG.batch_size
    epoches = FLAG.epoch
    init_lr = FLAG.lr
    LOG_INTERVAL = 30
    TEST_INTERVAL = 2
    source_name = FLAG.source
    target_name = FLAG.target
    model_name = FLAG.arch
    adapt_mode = FLAG.adapt_mode
    momentum=0.9
    l2_decay= 5e-4

    #Loading dataset
    source_train,target_train,target_test,classes = my_cross_dataset(FLAG)
    source_train_loader = torch.utils.data.DataLoader(dataset=source_train,batch_size=batch_size,
                    shuffle=True,num_workers=8,drop_last=True)
    target_train_loader = torch.utils.data.DataLoader(dataset=target_train,batch_size=batch_size,
                    shuffle=True,num_workers=8,drop_last=True)
    target_test_loader = torch.utils.data.DataLoader(dataset=target_test,batch_size=batch_size,
                    shuffle=False,num_workers=8)
    #Define model
    cross_model = models.RevGrad_onestep(FLAG)
    
    if len(gpus)>1:
        gpus = gpus.split(',')
        gpus = [int(v) for v in gpus]
        cross_model = nn.DataParallel(cross_model,device_ids=gpus)
        cross_model.to(DEVICE)
        #Define Optimizer
        optimizer = optim.Adam(cross_model.module.parameters(),
                            lr=init_lr/10)

    else:
        cross_model.to(DEVICE)
        #Define Optimizer
        optimizer = optim.Adam(cross_model.parameters(),
                            lr=init_lr/10)

    #loss function
    criterion = torch.nn.CrossEntropyLoss()
    #Training
    
    best_result = 0.0
    #Model store
    model_dir = os.path.join('./cross_models/',adapt_mode+'-'+target_name+'-'+model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #Tensorboard configuration
    log_dir = os.path.join('./cross_logs/',adapt_mode+'-'+target_name+'-'+model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)

    for epoch in range(1,epoches+1):
        cross_model.train()
        len_source_loader= len(source_train_loader)
        len_target_loader = len(target_train_loader)
        iter_source = iter(source_train_loader)
        iter_target = iter(target_train_loader)
        dlabel_src = torch.zeros(batch_size).long()
        dlabel_tgt = torch.ones(batch_size).long()

        min_dataloader = min(len_source_loader,len_target_loader)

        for i in range(1,len_source_loader+1):
            
            optimizer.zero_grad()
            p = float(i+epoch*len_source_loader)/epoches/min_dataloader

            alpha  = 2./(1.+np.exp(-10*p)) -1
            #Domain label
            src_dlabel,tgt_dlabel = dlabel_src.to(DEVICE),dlabel_tgt.to(DEVICE) 
            #Source data
            source_data,source_label = next(iter_source)
            input_source_data,input_source_label = source_data.to(DEVICE),source_label.to(DEVICE).squeeze()
            class_output,domain_output = cross_model(input_data=input_source_data,alpha=alpha)
            err_s_label = criterion(class_output,input_source_label)
            err_s_domain = criterion(domain_output,src_dlabel)
            #Target data
            target_data,target_label = next(iter_target)
            if i % len_source_loader == 0:
                iter_target = iter(target_train_loader)
            input_target_data,input_target_label = target_data.to(DEVICE),target_label.to(DEVICE)
            _,domain_output = cross_model(input_data=input_target_data,alpha=alpha)
            err_t_domain = criterion(domain_output,tgt_dlabel)
            err = err_s_label+err_s_domain+err_t_domain

            err.backward()
            optimizer.step()
            
            if i%5 ==0:
                n_iter = (epoch-1)*len_source_loader+i
                writer.add_scalar('data/class_loss',err_s_label,n_iter)
                writer.add_scalar('data/s_domain_loss',err_s_domain,n_iter)
                writer.add_scalar('data/t_domain_loss',err_t_domain,n_iter)

            #Print statistics
            if i%LOG_INTERVAL == 0: #Print every 30 mini-batches
                print('Epoch:[{}/{}],Batch:[{}/{}] \tsource_label_loss: {:.4f} \tsource_domain_loss: {:.4f} \ttarget_domain_loss: {:.4f}'
                        .format(epoch,epoches,i,len_source_loader,err_s_label.item(),err_s_domain.item(),err_t_domain.item()))


        if epoch%TEST_INTERVAL ==0:   #Every 2 epoches
            
            acc_test,class_corr,class_total,_=rev_test(cross_model,target_test_loader,epoch)
            #log test acc
            writer.add_scalar('data/test accuracy',acc_test,epoch)
            #Store the best model
            if acc_test>best_result:
                model_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}.pth'.format(source_name,target_name,model_name,epoch,round(acc_test,3)))
                torch.save(cross_model,model_path)
                #log results for classes
                log_path = model_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}.csv'.format(source_name,target_name,model_name,epoch,round(acc_test,3)))
                log_to_csv(log_path,classes,class_corr,class_total)
                best_result = acc_test
            else:
                print('The results in this epoch cannot exceed the best results !')

    writer.close()

'''
===========================
MCD
===========================
'''

def Mcd_train():
    
    #Basic parameters
    gpus = FLAG.gpus
    batch_size = FLAG.batch_size
    epoches = FLAG.epoch
    init_lr = FLAG.lr
    LOG_INTERVAL = 10
    TEST_INTERVAL = 2
    source_name = FLAG.source
    target_name = FLAG.target
    model_name = FLAG.arch
    adapt_mode = FLAG.adapt_mode
    l2_decay = 5e-4
    num_k = FLAG.num_k

    #Loading dataset
    if FLAG.isLT:
        source_train,target_train,target_test,classes = cross_dataset_LT(FLAG)
    else:
        source_train,target_train,target_test,classes = my_cross_dataset(FLAG)
    source_train_loader = torch.utils.data.DataLoader(dataset=source_train,batch_size=batch_size,
                    shuffle=True,num_workers=12,drop_last=True)
    target_train_loader = torch.utils.data.DataLoader(dataset=target_train,batch_size=batch_size,
                    shuffle=True,num_workers=12,drop_last=True)
    target_test_loader = torch.utils.data.DataLoader(dataset=target_test,batch_size=batch_size,
                    shuffle=False,num_workers=4)

    #Define model
    G = models.MCD_G(FLAG)
    C1 = models.MCD_C(FLAG)
    C2 = models.MCD_C(FLAG)
    C1.apply(weights_init)
    C2.apply(weights_init)

    if len(gpus)>1:
        gpus = gpus.split(',')
        gpus = [int(v) for v in gpus]
        G = nn.DataParallel(G,device_ids=gpus)
        C1 = nn.DataParallel(C1,device_ids=gpus)
        C2 = nn.DataParallel(C2,device_ids=gpus)

    G.to(DEVICE)
    C1.to(DEVICE)
    C2.to(DEVICE)

    if len(gpus)>1:
        optimizer_g = optim.SGD(list(G.module.features.parameters()),
                            lr=init_lr/10,weight_decay=l2_decay)

        optimizer_f = optim.SGD(list(C1.module.parameters())+ list(C2.module.parameters()),
                            lr=init_lr/10,momentum=0.9,weight_decay=l2_decay)

    else:
        optimizer_g = optim.SGD(list(G.features.parameters()),
                            lr=init_lr/10,weight_decay=l2_decay)

        optimizer_f = optim.SGD(list(C1.parameters())+ list(C2.parameters()),
                            lr=init_lr/10,momentum=0.9,weight_decay=l2_decay)

    #loss function
    criterion = torch.nn.CrossEntropyLoss()
    #Training
    best_result = 0.0
    #Model store
    model_dir = os.path.join('./mcd_models/',adapt_mode+'-'+source_name+'2'+target_name+'-'+model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #Tensorboard configuration
    log_dir = os.path.join('./mcd_logs/',adapt_mode+'-'+source_name+'2'+target_name+'-'+model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)

    for epoch in range(1,epoches+1):
        G.train()
        C1.train()
        C2.train()

        len_source_loader= len(source_train_loader)
        len_target_loader = len(target_train_loader)
        iter_source = iter(source_train_loader)
        iter_target = iter(target_train_loader)


        if len_target_loader <= len_source_loader:
            iter_num = len_target_loader
            which_dataset = True
        else:
            iter_num = len_source_loader
            which_dataset = False
        #learning rate log

        writer.add_scalar('data/Generator lr',optimizer_g.param_groups[0]['lr'],epoch)
        writer.add_scalar('data/Classifier lr',optimizer_f.param_groups[0]['lr'],epoch)

        log_source_loss_1 = 0.0
        log_source_loss_2 = 0.0
        log_entropy_loss = 0.0
        log_dis_loss = 0.0
        for i in range(1,iter_num+1):

            if which_dataset:
                target_data,_ = next(iter_target)
                if i % len_target_loader == 0:
                    iter_source = iter(source_train_loader)
                source_data,source_label = next(iter_source)
            else:
                source_data,source_label = next(iter_source)
                if i % len_source_loader == 0:
                    iter_target = iter(target_train_loader)
                target_data,_ = next(iter_target)
            
            input_source_data,input_source_label = source_data.to(DEVICE),source_label.to(DEVICE).squeeze()
            input_target_data = target_data.to(DEVICE)

            #when pretraining network source only
            eta = 1.0
            input_all = torch.cat((input_source_data,input_target_data),0)

            #Step A - training on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output_g = G(input_all)
            output_c1 = C1(output_g)
            output_c2 = C2(output_g)

            output_s1 = output_c1[:batch_size,:]
            output_s2 = output_c2[:batch_size,:]
            output_t1 = output_c1[batch_size:,:]
            output_t2 = output_c2[batch_size:,:]
            output_t1 = F.softmax(output_t1,dim=1)
            output_t2 = F.softmax(output_t2,dim=1)

            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

            loss1 = criterion(output_s1, input_source_label)
            loss2 = criterion(output_s2, input_source_label)
            all_loss = loss1 + loss2 + 0.01 * entropy_loss
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            #Step B train classifier to maximize discrepancy
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            output = G(input_all)
            output1 = C1(output)
            output2 = C2(output)
            output_s1 = output1[:batch_size,:]
            output_s2 = output2[:batch_size,:]
            output_t1 = output1[batch_size:,:]
            output_t2 = output2[batch_size:,:]
            output_t1 = F.softmax(output_t1,dim=1)
            output_t2 = F.softmax(output_t2,dim=1)
            loss1 = criterion(output_s1, input_source_label)
            loss2 = criterion(output_s2, input_source_label)
            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))
            loss_dis = torch.mean(torch.abs(output_t1-output_t2))
            F_loss = loss1 + loss2 - eta*loss_dis  + 0.01 * entropy_loss
            F_loss.backward()
            optimizer_f.step()

            # Step C train genrator to minimize discrepancy
            for j in range(num_k):
                optimizer_g.zero_grad()
                output = G(input_all)
                output1 = C1(output)
                output2 = C2(output)

                output_s1 = output1[:batch_size,:]
                output_s2 = output2[:batch_size,:]
                output_t1 = output1[batch_size:,:]
                output_t2 = output2[batch_size:,:]

                loss1 = criterion(output_s1, input_source_label)
                loss2 = criterion(output_s2, input_source_label)
                output_t1 = F.softmax(output_t1,dim=1)
                output_t2 = F.softmax(output_t2,dim=1)
                loss_dis = torch.mean(torch.abs(output_t1-output_t2))
                entropy_loss = -torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

                loss_dis.backward()
                optimizer_g.step()
            
            if i%5 ==0:
                n_iter = (epoch-1)*len_target_loader+i
                writer.add_scalar('data/source loss1',loss1,n_iter)
                writer.add_scalar('data/source loss2',loss2,n_iter)
                writer.add_scalar('data/entropy loss',entropy_loss,n_iter)
                writer.add_scalar('data/discrepancy loss',loss_dis,n_iter)

            log_source_loss_1 += loss1.item()
            log_source_loss_2 += loss2.item()
            log_entropy_loss += entropy_loss.item()
            log_dis_loss += loss_dis.item()
            if i%LOG_INTERVAL == 0: #Print every 30 mini-batches
                print('Epoch:[{}/{}],Batch:[{}/{}] source_loss_1: {:.4f} source_loss_2: {:.4f} entropy loss: {:.4f} discrepancy loss {:.4f}'.format
                (epoch,epoches,i,len_target_loader,log_source_loss_1/LOG_INTERVAL,log_source_loss_2/LOG_INTERVAL,
                log_entropy_loss/LOG_INTERVAL,log_dis_loss/LOG_INTERVAL))
                
                log_source_loss_1 = 0.0
                log_source_loss_2 = 0.0
                log_entropy_loss = 0.0
                log_dis_loss = 0.0

        if epoch%TEST_INTERVAL ==0:   #Every 2 epoches
            
            acc_1,acc_2,corr1,corr2,class_total,testloss_1,testloss_2=mcd_test(G,C1,C2,target_test_loader,epoch)
            #log test acc
            writer.add_scalar('data/test accuracy 1',acc_1,epoch)
            writer.add_scalar('data/test accuracy 2',acc_2,epoch)
            writer.add_scalar('data/test loss 1',testloss_1,epoch)
            writer.add_scalar('data/test loss 2',testloss_2,epoch)
            acc_test = max(acc_1,acc_2)
            #Store the best model
            if acc_test>best_result:
                g_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}-g.pth'.format(source_name,target_name,
                            model_name,epoch,round(acc_test,3)))
                c1_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}-c1_acc_{}.pth'.format(source_name,target_name,
                            model_name,epoch,round(acc_test,4),round(acc_1,3)))
                c2_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}-c2_acc_{}.pth'.format(source_name,target_name,
                            model_name,epoch,round(acc_test,4),round(acc_2,3)))
                torch.save(G,g_path)
                torch.save(C1,c1_path)
                torch.save(C2,c2_path)
                #log results for classes
                log_path_1 = model_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}-c1.csv'.format(source_name,target_name,
                            model_name,epoch,round(acc_test,3)))
                log_path_2 = model_path = os.path.join(model_dir,
                            '{}-{}-{}-epoch_{}-accval_{}-c2.csv'.format(source_name,target_name,
                            model_name,epoch,round(acc_test,3)))
                log_to_csv(log_path_1,classes,corr1,class_total)
                log_to_csv(log_path_2,classes,corr2,class_total)
                best_result = acc_test
            else:
                print('The results in this epoch cannot exceed the best results !')

    writer.close()


 
if __name__ == "__main__":

    if FLAG.mode == 'single':
        baseline_train()

    elif FLAG.mode == 'cross':
        cross_train()

    elif FLAG.mode == 'rev':
        Revgrad_train()

    elif FLAG.mode == 'rev_2':
        Revgrad_train_onestep()

    elif FLAG.mode == 'mcd':
        Mcd_train()


    else:
        print('Training mode is wrong !')
