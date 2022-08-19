from __future__ import print_function, division
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from .model import ft_net, ft_net_dense, ft_net_swin, ft_net_NAS, PCB, PCB_test


try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/rzh/Desktop/re-id/Market-1501-v15.09.15/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='warm5_s1_b8_lr2_p0.5_circle_DG', type=str, help='save model path')
# parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()

###load config###
# load the training config
config_path = os.path.join('deep_sort_pytorch/deep_sort/model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.use_swin = config['use_swin']
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

if opt.use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
############### Ten Crop 

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])



######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('/home/rzh/Desktop/my_project4/deep_sort_pytorch/deep_sort/model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network



######################################################################
# Load Collected data Trained model
def get_model(opt):
    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses)
    elif opt.use_NAS:
        model_structure = ft_net_NAS(opt.nclasses)
    elif opt.use_swin:
        model_structure = ft_net_swin(opt.nclasses)
    else:
        model_structure = ft_net(opt.nclasses, stride = opt.stride)

    if opt.PCB:
        model_structure = PCB(opt.nclasses)

    model = load_network(model_structure)

    if opt.PCB:
        #if opt.fp16:
        #    model = PCB_test(model[1])
        #else:
            model = PCB_test(model)
    else:
        #if opt.fp16:
            #model[1].model.fc = nn.Sequential()
            #model[1].classifier = nn.Sequential()
        #else:
            model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    return model

def fliplr(img):
    '''flip horizontal'''
    
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_fea(model, img):
    features = torch.FloatTensor()
    ff = torch.FloatTensor(1, 512).zero_().cuda()
    if opt.PCB:
        ff = torch.FloatTensor(2048,6).zero_().cuda() # we have six parts
    img = Image.fromarray(img)
    img = data_transforms(img).unsqueeze(0)
    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        for scale in ms:
            if scale != 1:
                # bicubic is only  available in pytorch>= 1.1
                input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
            print("*************input_img**********************")
            print(input_img.type, input_img.shape)
            outputs = model(input_img) 
            ff += outputs
    # norm feature
    if opt.PCB:
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    features = ff.data.cpu()
    return features


def extract_features(img_input):
    model = get_model(opt)
    features = torch.FloatTensor()
    for img in img_input:
        with torch.no_grad():
            ff = extract_fea(model, img)
        features = torch.cat((features,ff.data.cpu()), 0)
    return features.cpu().numpy()  