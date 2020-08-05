import os
import numpy as np 
import csv
import torch
from torch.autograd import Function
"""
========================
General functions
========================
"""


def log_to_csv(log_path,classes,class_corr,class_total):

    '''
    log test results for classes
    '''
    with open(log_path,'w') as f:

        writer = csv.writer(f,delimiter=',')
        writer.writerow(['Categories','num_corr','num_total','acc'])
        for i in range(len(classes)):
            acc = str(round(100*float(class_corr[i]/class_total[i]),3))+'%'
            writer.writerow([classes[i],class_corr[i],class_total[i],acc])

        f.close()

def adjust_lr(optimizer,epoch,FLAG):
    init_lr = FLAG.lr
    epoches = FLAG.epoch
    lr = init_lr/((1+10*(epoch-1)/epoches)**0.75)

    optimizer.param_groups[0]['lr'] = lr/10
    optimizer.param_groups[1]['lr'] = lr

    return optimizer   

def adjust_lr_v2(optimizer_fea,optimizer_critic,epoch,FLAG):
    init_lr = FLAG.lr
    epoches = FLAG.epoch
    lr = init_lr/((1+10*(epoch-1)/epoches)**0.75)

    optimizer_fea.param_groups[0]['lr'] = lr/10
    optimizer_fea.param_groups[1]['lr'] = lr

    optimizer_critic.param_groups[0]['lr']=lr

    return optimizer_fea,optimizer_critic   


'''
========================
DDC
========================
'''
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


'''
========================
CORAL
========================
'''
def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = torch.mm(xm.t(),xm)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = torch.mm(xmt.t(),xmt)

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss


'''
========================
MCD
========================
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)



'''
========================
DANN_2
========================
'''
class ReverseLayerF(Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None