import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from functions import ReverseLayerF
#from IPython import embed
import torch
import model.backbone as backbone
import torch.nn.functional as F

class DAANNet(nn.Module):

    def __init__(self, num_classes=65, base_net='ResNet50'):
        super(DAANNet, self).__init__()
        self.sharedNet = backbone.network_dict[base_net]()
        self.Inception = InceptionA(2048,64)
        #self.bottleneck = nn.Linear(288, 256)
        self.source_fc = nn.Linear(288, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.classes = num_classes
        self.domain_classifier1 = nn.Sequential()
        self.domain_classifier1.add_module('fc1', nn.Linear(64, 32))
        self.domain_classifier1.add_module('relu1', nn.ReLU(True))
        self.domain_classifier1.add_module('dpt1', nn.Dropout())
        self.domain_classifier1.add_module('fc2', nn.Linear(32, 2))
        self.domain_classifier2 = nn.Sequential()
        self.domain_classifier2.add_module('fc1', nn.Linear(64, 32))
        self.domain_classifier2.add_module('relu1', nn.ReLU(True))
        self.domain_classifier2.add_module('dpt1', nn.Dropout())
        self.domain_classifier2.add_module('fc2', nn.Linear(32, 2))
        self.domain_classifier3 = nn.Sequential()
        self.domain_classifier3.add_module('fc1', nn.Linear(96, 64))
        self.domain_classifier3.add_module('relu1', nn.ReLU(True))
        self.domain_classifier3.add_module('dpt1', nn.Dropout())
        self.domain_classifier3.add_module('fc2', nn.Linear(64, 2))
        self.domain_classifier4 = nn.Sequential()
        self.domain_classifier4.add_module('fc1', nn.Linear(64, 32))
        self.domain_classifier4.add_module('relu1', nn.ReLU(True))
        self.domain_classifier4.add_module('dpt1', nn.Dropout())
        self.domain_classifier4.add_module('fc2', nn.Linear(32, 2))
        # global domain discriminator
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(288, 1024))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(1024, 2))

        # local domain discriminator
        self.dcis = nn.Sequential()
        self.dci = {}
        for i in range(num_classes):
            self.dci[i] = nn.Sequential()
            self.dci[i].add_module('fc1', nn.Linear(288, 1024))
            self.dci[i].add_module('relu1', nn.ReLU(True))
            self.dci[i].add_module('dpt1', nn.Dropout())
            self.dci[i].add_module('fc2', nn.Linear(1024, 1024))
            self.dci[i].add_module('relu2', nn.ReLU(True))
            self.dci[i].add_module('dpt2', nn.Dropout())
            self.dci[i].add_module('fc3', nn.Linear(1024, 2)) 
            self.dcis.add_module('dci_'+str(i), self.dci[i])

    def forward(self, source, target, s_label, DEV, alpha=0.0):
        source_share = self.sharedNet(source)
        s1, s2, s3, s4 = self.Inception(source_share)
        #s1 =  torch.cat([s1, s2, s3, s4], 1)
        source_share = torch.cat([s1, s2, s3, s4], 1)#(64,352)
        #source_share = self.bottleneck(source_share)
        source = self.source_fc(source_share)
        p_source = self.softmax(source)

        target_share = self.sharedNet(target)
        t1, t2, t3, t4 = self.Inception(target_share)
        target = torch.cat([t1, t2, t3, t4], 1)#(64,352)
        #target = self.bottleneck(target)
        t_label = self.source_fc(target)
        p_target = self.softmax(t_label)
        t_label = t_label.data.max(1)[1]
        s_out = []
        t_out = []
        if self.training == True:
            # RevGrad
            s1_revese = ReverseLayerF.apply(s1, alpha)
            t1_revese = ReverseLayerF.apply(t1, alpha)
            s1_domain_output = self.domain_classifier1(s1_revese)
            t1_domain_output = self.domain_classifier1(t1_revese)

            s2_revese = ReverseLayerF.apply(s2, alpha)
            t2_revese = ReverseLayerF.apply(t2, alpha)
            s2_domain_output = self.domain_classifier2(s2_revese)
            t2_domain_output = self.domain_classifier2(t2_revese)

            s3_revese = ReverseLayerF.apply(s3, alpha)
            t3_revese = ReverseLayerF.apply(t3, alpha)
            s3_domain_output = self.domain_classifier3(s3_revese)
            t3_domain_output = self.domain_classifier3(t3_revese)

            s4_revese = ReverseLayerF.apply(s4, alpha)
            t4_revese = ReverseLayerF.apply(t4, alpha)
            s4_domain_output = self.domain_classifier4(s4_revese)
            t4_domain_output = self.domain_classifier4(t4_revese)
            
            s_reverse_feature = ReverseLayerF.apply(source_share, alpha)
            t_reverse_feature = ReverseLayerF.apply(target, alpha)
            s_domain_output = self.domain_classifier(s_reverse_feature)
            t_domain_output = self.domain_classifier(t_reverse_feature)

            # p*feature-> classifier_i ->loss_i
            for i in range(self.classes):
                ps = p_source[:, i].reshape((target.shape[0],1))
                fs = ps * s_reverse_feature
                pt = p_target[:, i].reshape((target.shape[0],1))
                ft = pt * t_reverse_feature
                outsi = self.dcis[i](fs)
                s_out.append(outsi)
                outti = self.dcis[i](ft)
                t_out.append(outti)
            return source,s_out,t_out, s1_domain_output,s2_domain_output,s3_domain_output,s4_domain_output,t1_domain_output,t2_domain_output,t3_domain_output,t4_domain_output,s_domain_output,t_domain_output
        else:
            #s_domain_output = 0
            #t_domain_output = 0
            s_out = [0]*self.classes
            t_out = [0]*self.classes
            return p_source


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

        self.avg_pool = nn.AvgPool2d(7, stride=1)

        #self.source_fc = nn.Linear(288, num_classes)

    def forward(self, source):
        s_branch1x1 = self.branch1x1(source)

        s_branch5x5 = self.branch5x5_1(source)
        s_branch5x5 = self.branch5x5_2(s_branch5x5)

        s_branch3x3dbl = self.branch3x3dbl_1(source)
        s_branch3x3dbl = self.branch3x3dbl_2(s_branch3x3dbl)
        s_branch3x3dbl = self.branch3x3dbl_3(s_branch3x3dbl)

        s_branch_pool = F.avg_pool2d(source, kernel_size=3, stride=1, padding=1)
        s_branch_pool = self.branch_pool(s_branch_pool)

        s_branch1x1 = self.avg_pool(s_branch1x1)
        s_branch5x5 = self.avg_pool(s_branch5x5)
        s_branch3x3dbl = self.avg_pool(s_branch3x3dbl)
        s_branch_pool = self.avg_pool(s_branch_pool)

        s_branch1x1 = s_branch1x1.view(s_branch1x1.size(0), -1)#(64,64)
        s_branch5x5 = s_branch5x5.view(s_branch5x5.size(0), -1)#(64,64)
        s_branch3x3dbl = s_branch3x3dbl.view(s_branch3x3dbl.size(0), -1)#(64,96)
        s_branch_pool = s_branch_pool.view(s_branch_pool.size(0), -1)#(64,64)


        return  s_branch1x1, s_branch5x5, s_branch3x3dbl, s_branch_pool

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)