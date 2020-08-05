'''
====================================
Model library for domain adaptation
Jiangang,Yang
May,31,2019
====================================
'''
import torch
import torchvision.models as models
from torchsummaryX import summary
import torch.nn as nn
from utils import mmd_linear,CORAL
from torch.autograd import Function
from utils import ReverseLayerF
'''
===================================
BaseNet
===================================
'''
class Models(object):

    def __init__(self,FLAG):
        self.flag = FLAG
        pass

    def model(self):
        
        is_train = self.flag.pretrain
        model_name = self.flag.arch
        num_class = self.flag.num_class
        try:
            model = getattr(models,model_name)(pretrained=is_train)
            #save_model = torch.load('/home/chenshuo/Damain-dehaze/haze_cs/eca_resnet18_k3577.pth')
           # model_dict = model.state_dict()
            #state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            #print(state_dict.keys()) # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
            #model_dict.update(state_dict)
            #model.load_state_dict(model_dict) """

        except NameError:
            print('{} doesn''t exist.'.format(model_name))

        print('Pretrained model {} has loaded.'.format(model_name))
        #Modify number of classes
        #print(model)
        if model_name == 'resnet18':
            model.fc = nn.Linear(512,num_class)

        elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'resnest50'or model_name== 'inception_v3':
            model.fc = nn.Linear(2048,num_class)  
         
        elif model_name == 'vgg19' or model_name == 'alexnet':
            #model.features[0] = nn.Conv2d(3,64,(7,7),(1,1),(3,3))
            model.classifier[6] = nn.Linear(4096,num_class)

        #print(model)
        #model = model.cuda()
        #Output model information
        #print('Done!')
        return model


'''
===================================
DDC
===================================
'''
class DDCNet(nn.Module):

    def __init__(self, FLAG):
        super(DDCNet, self).__init__()
        Nets = Models(FLAG)
        self.flag = FLAG
        model_name = self.flag.arch
        num_class = self.flag.num_class
        base_model = Nets.model()
        #print(base_model)
        if model_name == 'resnet18':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = nn.Linear(512,num_class)
        if model_name == 'resnet50' or model_name == 'resnest50':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = nn.Linear(2048,num_class)
        elif model_name == 'vgg19' or model_name =='alexnet':
            base_model.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
            #base_model.classifier = base_model.classifier[:-1]
            self.sharedNet = base_model
            self.cls_fc = nn.Linear(4096,num_class)
        #print(self.sharedNet)
            
        #print(self.sharedNet)

    def forward(self, source, target):
        source_output = self.sharedNet(source)
        source_output = source_output.view(source_output.size(0),-1)
        loss = 0

        if self.training == True:
            target_output = self.sharedNet(target)
            target_output = target_output.view(target_output.size(0),-1)
            final_source = self.cls_fc(source_output)
            return final_source,source_output,target_output

        else:
            final_source = self.cls_fc(source_output)
            return final_source,source_output,source_output




'''
===================================
Coral
===================================
'''
class DeepCoral(nn.Module):
    def __init__(self,FLAG):

        super(DeepCoral, self).__init__()
        Nets = Models(FLAG)
        self.flag = FLAG
        model_name = self.flag.arch
        num_class = self.flag.num_class
        base_model = Nets.model()
        #print(self.sharedNet)
        if model_name == 'resnet18':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = nn.Linear(512,num_class)
        if model_name == 'resnet50' or model_name == 'resnest50':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = nn.Linear(2048,num_class)
        elif model_name == 'vgg19' or model_name == 'alexnet':
            base_model.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
            self.sharedNet = base_model
            self.cls_fc = nn.Linear(4096,num_class)
        #print(self.sharedNet)

    def forward(self, source, target):
        source_output = self.sharedNet(source)
        source_output = source_output.view(source_output.size(0),-1)

        if self.training == True:
            target_output = self.sharedNet(target)
            target_output = target_output.view(target_output.size(0),-1)
            final_source = self.cls_fc(source_output)
            return final_source,source_output,target_output

        else:
            final_source = self.cls_fc(source_output)
            return final_source, source_output,source_output
'''
===================================
RevGrad_v2
===================================
'''
class RevGrad_v3(nn.Module):

    def __init__(self, FLAG):
        
        super(RevGrad_v3, self).__init__()
        Nets = Models(FLAG)
        self.flag = FLAG
        model_name = self.flag.arch
        num_class = self.flag.num_class
        base_model = Nets.model()
        #print(base_model)
        layers=[]
        if model_name == 'resnet18':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = nn.Linear(512,num_class)
            self.domain_fc = nn.Linear(512, 2)
        if model_name == 'resnet50':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = nn.Linear(2048,num_class)
            self.domain_fc = nn.Linear(2048, 2)
        elif model_name == 'vgg19':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = base_model.classifier

            num_unit = 25088
            self.domain_fc = nn.Linear(num_unit, 2)

        elif model_name == 'alexnet':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = base_model.classifier

            num_unit = 9216
            self.domain_fc = nn.Linear(num_unit, 2)


    def forward(self, data):
        data_output = self.sharedNet(data)
        data_output = data_output.view(data_output.size(0),-1)
        clabel_pred = self.cls_fc(data_output)
        dlabel_pred = self.domain_fc(data_output)

        return clabel_pred, dlabel_pred
'''
===================================
RevGrad_v1
===================================
'''
class RevGrad(nn.Module):

    def __init__(self, FLAG):
        
        super(RevGrad, self).__init__()
        Nets = Models(FLAG)
        self.flag = FLAG
        model_name = self.flag.model_name
        num_class = self.flag.num_class
        base_model = Nets.model()
   

        if model_name == 'resnet18':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = nn.Linear(512,num_class)
            self.domain_fc = nn.Linear(512, 2)
        elif model_name == 'resnest50' or model_name == 'resnet50':
            self.sharedNet = nn.Sequential(*list(base_model.children())[:-1])
            self.cls_fc = nn.Linear(2048,num_class)
            self.domain_fc = nn.Linear(2048, 2)
        elif model_name == 'vgg19' or model_name == 'alexnet':
            base_model.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
            self.sharedNet = base_model
            self.cls_fc = nn.Linear(4096,num_class)
            self.domain_fc = nn.Linear(4096, 2)
      
    def forward(self, data):
        data_output = self.sharedNet(data)
        data_output = data_output.view(data_output.size(0),-1)
        clabel_pred = self.cls_fc(data_output)
        dlabel_pred = self.domain_fc(data_output)

        return clabel_pred, dlabel_pred


'''
===================================
RevGrad_onestep
===================================
'''
class RevGrad_onestep(nn.Module):

    def __init__(self,FLAG):
        super(RevGrad_onestep, self).__init__()
        self.flag = FLAG 
        self.is_train = FLAG.pretrain
        self.model_name = FLAG.arch
        self.num_class = FLAG.num_class

        try:
            model = getattr(models,self.model_name)(pretrained=self.is_train)

        except NameError:
            print('{} doesn''t exist.'.format(self.model_name))

        print('Pretrained model {} has loaded.'.format(self.model_name))
        
        layer_c = []
        layer_d = []
        n_init = 0
        if self.model_name == 'resnet18':
            #print(model)
            self.feature = nn.Sequential(*list(model.children())[:-1])

            n_init = 512

        if self.model_name == 'resnet50':
            #print(model)
            self.feature = nn.Sequential(*list(model.children())[:-1])

            n_init = 2048   
        elif self.model_name == 'vgg19' or self.model_name == 'alexnet':
            #print(model)
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            self.feature = model
            n_init = 4096
        #print(self.features)

        
        layer_c.append(nn.Linear(n_init,100))
        layer_c.append(nn.BatchNorm1d(100))
        layer_c.append(nn.ReLU(True))
        layer_c.append(nn.Dropout2d())
        layer_c.append(nn.Linear(100,100))
        layer_c.append(nn.BatchNorm1d(100))
        layer_c.append(nn.ReLU(True))
        layer_c.append(nn.Linear(100,self.num_class))

        layer_d.append(nn.Linear(n_init,100))
        layer_d.append(nn.BatchNorm1d(100))
        layer_d.append(nn.ReLU(True))
        layer_d.append(nn.Linear(100,2))

        self.class_classifier = nn.Sequential(*layer_c)
        self.domain_classifier = nn.Sequential(*layer_d)

    def forward(self, input_data, alpha=0):

        feature = self.feature(input_data)
        feature = feature.view(feature.size(0),-1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

'''
===================================
Maximum Classifier discrepancy
===================================
'''
class MCD_G(nn.Module):

    def __init__(self,FLAG):
        super(MCD_G, self).__init__()
        self.flag = FLAG 
        self.is_train = FLAG.pretrain
        self.model_name = FLAG.arch
        self.num_class = FLAG.num_class

        try:
            model = getattr(models,self.model_name)(pretrained=self.is_train)

        except NameError:
            print('{} doesn''t exist.'.format(self.model_name))

        print('Pretrained model {} has loaded.'.format(self.model_name))
        #print(model)
        if self.model_name == 'resnet18':
            #print(model)
            self.features = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'resnest50' or self.model_name == 'resnet50':
            self.features = nn.Sequential(*list(model.children())[:-1])
        #elif self.model_name == 'vgg19' or 'alexnet':
            #print(model)
        #    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        #    self.features = model
        elif self.model_name == 'vgg19' or self.model_name == 'alexnet':
            self.features = nn.Sequential(*list(model.children())[:-1])

        #print(self.features)

    def forward(self,x):

        x = self.features(x)
        x = x.view(x.size(0),-1)

        return x

class MCD_C(nn.Module):

    def __init__(self,FLAG,prob=0.5,middle=1000,num_layer=2):
        super(MCD_C, self).__init__()
        self.model_name = FLAG.arch
        self.num_classes = FLAG.num_class

        #Three layers
        layers = []
        if self.model_name == 'resnet18':
            num_unit = 512
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))

            for i in range(num_layer-1):
                layers.append(nn.Dropout(p=prob))
                layers.append(nn.Linear(middle,middle))
                layers.append(nn.BatchNorm1d(middle,affine=True))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(middle,self.num_classes))
        if self.model_name == 'resnest50' or self.model_name == 'resnet50':
            num_unit = 2048
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))

            for i in range(num_layer-1):
                layers.append(nn.Dropout(p=prob))
                layers.append(nn.Linear(middle,middle))
                layers.append(nn.BatchNorm1d(middle,affine=True))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(middle,self.num_classes))
        elif self.model_name == 'vgg19' or self.model_name == 'alexnet':
            
            model = getattr(models,self.model_name)(pretrained=False)

            model.classifier[6] = nn.Linear(4096,self.num_classes)    
            layers = list(model.classifier.children())

        else:
            print('The model name is wrong!!')
    
        self.classifier = nn.Sequential(*layers)
        #print(self.classifier)
        #print(self.classifier)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x

class MCD_C_2(nn.Module):

    def __init__(self,FLAG,prob=0.5,middle=1000,num_layer=2):
        super(MCD_C_3, self).__init__()
        self.model_name = 'resnest50'
        self.num_classes = FLAG.num_class

        #Three layers
        layers = []
        if self.model_name == 'resnet18':
            num_unit = 512
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))

            for i in range(num_layer-1):
                layers.append(nn.Dropout(p=prob))
                layers.append(nn.Linear(middle,middle))
                layers.append(nn.BatchNorm1d(middle,affine=True))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(middle,self.num_classes))
        if self.model_name == 'resnest50'  or self.model_name == 'resnet50' :
            num_unit = 2048
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))

            for i in range(num_layer-1):
                layers.append(nn.Dropout(p=prob))
                layers.append(nn.Linear(middle,middle))
                layers.append(nn.BatchNorm1d(middle,affine=True))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(middle,self.num_classes))
        elif self.model_name == 'vgg19':

            num_unit = 25088
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit,middle))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle,middle))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(middle,self.num_classes))

        elif self.model_name == 'alexnet':

            num_unit = 9216
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit,middle))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle,middle))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(middle,self.num_classes))

        else:
            print('The model name is wrong!!')
        self.classifier = nn.Sequential(*layers)
        #print(self.classifier)
        #print(self.classifier)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x

class MCD_C_3(nn.Module):

    def __init__(self,FLAG,prob=0.5,middle=1000,num_layer=2):
        super(MCD_C_3, self).__init__()
        self.model_name = FLAG.model_name
        self.num_classes = FLAG.num_class

        #Three layers
        layers = []
        num_unit = 0
        if self.model_name == 'resnet18':
            num_unit = 512
        if self.model_name == 'resnest50'or self.model_name == 'resnet50':
            num_unit = 2048
        elif self.model_name == 'vgg19':
            num_unit = 25088

        elif self.model_name == 'alexnet':

            num_unit = 9216
        else:
            print('The model name is wrong!!')

        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit,middle))
        layers.append(nn.BatchNorm1d(middle,affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer-1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(middle,self.num_classes))

        self.classifier = nn.Sequential(*layers)
        #print(self.classifier)
        #print(self.classifier)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    def forward(self, x):
        return x.view_as(x)
    def backward(self, grad_output):
        return (grad_output*(-self.lambd))

def grad_reverse(x,lambd=1.0):
    return GradReverse(lambd)(x)


'''
===================================
Madda
===================================
'''
def get_model(name):
    if name == "lenet":
        model = EmbeddingNet(FLAG).cuda()
        return model

    if name == "disc":
        model = Discriminator(input_dims=512, hidden_dims=512, output_dims=2)
        return model

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, output_dims))

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
    
class EmbeddingNet(nn.Module):
    def __init__(self, FLAG):
        super(EmbeddingNet, self).__init__()
        Nets = Models(FLAG)
        self.flag = FLAG
        model_name = self.flag.arch
        num_class = self.flag.num_class
        base_model = Nets.model()
        #print(base_model)
        layers=[]
        if model_name == 'resnet18':
            self.convnet = nn.Sequential(*list(base_model.children())[:-1])
            self.fc = nn.Linear(512,256)
            self.domain_fc = nn.Linear(512, 2)

    def extract_features(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0),-1)
        return output

    def forward(self, x):
        output = self.convnet(x)     
        output = self.fc(output)
        output = output.view(output.size()[0], -1)
        return output

    def get_embedding(self, x):
        return self.forward(x)