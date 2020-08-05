import argparse

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='Parameters for training and testing')

#Dirs
parser.add_argument("--root_dir",type=str,default='/home/chenshuo/data',help='Data path.')
parser.add_argument("--source",type=str,default='clear1',help='source dataset name.')
parser.add_argument("--target",type=str,default='haze1',help='target dataset name.')
#Models
parser.add_argument('-a', '--arch', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument("--pretrain",type=bool,default=True,help='Pretrain or not.')
parser.add_argument("--num_class",type=int,default=12,help='Number of classes.')
parser.add_argument("--mode",type=str,default='rev_2',help='Different strategies: single or cross,rev,rev_2,mcd.')
parser.add_argument("--adapt_mode",type=str,default='dann-onestep',help='Different adaptation strategies: ddc,coral,dann,dann-onestep,mcd.')

#Training&Testing
parser.add_argument("--gpus",type=str,default='0,1,2,3',help='gpu group.')
parser.add_argument("--lr",type=float,default=1e-2,help='learning rate')
parser.add_argument("--epoch",type=int,default=50,help='Epoches.')
parser.add_argument("--batch_size",type=int,default=128,help='Batch size.')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update-only MCD')
parser.add_argument('--saliency_map_add', type=bool, default=False,
                    help='add saliency map channel is 3')
parser.add_argument('--saliency_map_cat', type=bool, default=False,
                    help='add saliency map channel is 4')      
parser.add_argument('--darkchannel_dehaze', type=bool, default=False,
                    help='add darkchannel_dehaze in target')             
parser.add_argument('--isLT', type=bool, default=False,
                    help='LabelShift')

#saliency_map_add and salinecy_map_cat can not both true.
