import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import copy
import torch.nn.functional as F

from models.losses.triplet import online_triplet_loss
from models.losses.protonet import prototypical_loss
from models.losses.angproto import angular_proto_loss
from models.losses.amsoftmax import am_softmax

# comparison with peeler
from models.encoder.DSCNN import DSCNNS_PEELER, DSCNNL_PEELER
from models.losses.peeler import peeler_loss
from models.losses.dproto import dproto

class conv_cka(nn.Module):
    def __init__(self, orig_conv, opt):
        super(conv_cka, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        groups = self.conv.groups
        stride, _ = self.conv.stride
        # costum-keyword adapters
        if 'delta' not in opt['adapting.cka_opt']:
            self.ad_type = 'none'
        else:
            self.ad_type = opt['adapting.cka_ad_type']
            self.ad_form = opt['adapting.cka_ad_form']
        if self.ad_type == 'residual':
            if self.ad_form == 'matrix' or planes != in_planes:
                self.delta = nn.Parameter(torch.ones(planes, in_planes*groups, 1, 1))
            else:
                self.delta = nn.Parameter(torch.ones(1, planes, 1, 1))
        elif self.ad_type == 'serial':
            if self.ad_form == 'matrix':
                self.delta = nn.Parameter(torch.ones(planes, planes, 1, 1))
            else:
                self.delta = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias.requires_grad = True
        if self.ad_type != 'none':
            self.delta.requires_grad = True
        
        # if(opt['data.cuda']):
        #     self.delta = torch.nn.Parameter(self.delta).cuda()

    def forward(self, x):
        y = self.conv(x)
        if self.delta.device != x.device:
            self.delta = nn.Parameter(self.delta.to(x.device))
        if self.ad_type == 'residual':
            if self.delta.size(0) > 1:
                y = y + F.conv2d(x, self.delta, stride=self.conv.stride)
            else:
                # residual adaptation in channel-wise (vector)
                y = y + x * self.delta
        elif self.ad_type == 'serial':
            if self.delta.size(0) > 1:
                # serial adaptation in matrix form
                y = F.conv2d(y, self.delta) + self.alpha_bias
            else:
                # serial adaptation in channel-wise (vector)
                y = y * self.delta + self.alpha_bias
        return y

class Domain_transform(nn.Module):
    def __init__(self, planes):
        super(Domain_transform, self).__init__()
        self.planes = planes
        self.avg = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear=torch.nn.Linear(planes, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.detach().data 
        x = self.avg(x).view(-1, self.planes)
        x = self.linear(x)
        x = self.relu(x)
        domain_offset = x.mean()
        return domain_offset
    
class bn_cka(nn.Module):
    def __init__(self, orig_bn, opt):
        super(bn_cka, self).__init__()
        # the original bn layer
        self.BN = copy.deepcopy(orig_bn)
        if self.BN.affine:  
            self.BN.weight.requires_grad = False
            self.BN.bias.requires_grad = False
        
        planes = self.BN.num_features
        self.IN = nn.InstanceNorm2d(planes, affine=False)
        
        self.alpha = nn.Parameter(torch.FloatTensor([-5.0]), requires_grad=True)
        self.alpha_t = torch.Tensor([0.0])
        # if(opt['data.cuda']):
        #     self.alpha = torch.nn.Parameter(self.alpha).cuda()
        #     self.alpha_t = torch.nn.Parameter(self.alpha_t).cuda()
        self.domain_transform = Domain_transform(planes)
        if(opt['data.cuda']):
            self.domain_transform.cuda()
            # self.IN.cuda()
    def forward(self, x):
        if self.alpha.device != x.device:
            self.alpha = nn.Parameter(self.alpha.to(x.device))
        self.alpha_t = self.alpha + 0.01 * self.domain_transform(x) # default: self.alpha_t = self.alpha + 0.01 * self.domain_transform(x)
        # self.alpha_t = self.alpha

        if x.device == torch.device('cpu'):
            t = torch.sigmoid(self.alpha_t)
        else:
            t = torch.sigmoid(self.alpha_t).cuda()
        
        # t = 0
        # print('t:{}'.format(t))

        out_in = self.IN(x)
        out_bn = self.BN(x)
        out = t * out_in + (1 - t) * out_bn
        return out



class ReprModel_cka(nn.Module):
    def __init__(self, orig_model, opt, criterion, x_dim):
        super(ReprModel_cka, self).__init__()
        # freeze the pretrained encoder
        for k, v in orig_model.encoder.named_parameters():
                v.requires_grad=False

        #get cka settings
        self.cka_init = opt['adapting.cka_init']
        
        # attaching task-specific adapters (delta) to each convolutional layers
        # for block in orig_model.encoder:
        for name, m in orig_model.encoder.named_modules():
            # print(name,m)
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3: 
                # print("reset the conv2d")
                new_conv_cka = conv_cka(m, opt)
                
                parent_module = dict(orig_model.encoder.named_modules())[name.rsplit('.', 1)[0]]
                setattr(parent_module, name.rsplit('.', 1)[-1], new_conv_cka)
                
            if isinstance(m, nn.BatchNorm2d):
                new_bn_cka = bn_cka(m, opt)

                parent_module = dict(orig_model.encoder.named_modules())[name.rsplit('.', 1)[0]]
                setattr(parent_module, name.rsplit('.', 1)[-1], new_bn_cka)
            
        # print(orig_model.encoder)
        
        self.encoder = orig_model.encoder
        self.preprocessing = orig_model.preprocessing
        self.emb_norm = orig_model.emb_norm
        
        # get embedding size
        if(opt['data.cuda']):
            x_fake = torch.Tensor(1,x_dim[0],x_dim[1],x_dim[2] ).cuda()
        else:
            x_fake = torch.Tensor(1,x_dim[0],x_dim[1],x_dim[2] )
        z = self.encoder.forward(x_fake)
        z_dim = z.size(1)

        #setup loss
        if criterion['type'] == 'prototypical':
            self.criterion = prototypical_loss(criterion)
        elif criterion['type'] == 'triplet':
            self.criterion = online_triplet_loss(criterion)
        elif criterion['type'] == 'angproto':
            self.criterion = angular_proto_loss(criterion)
        elif criterion['type'] == 'normsoftmax':
            criterion['z_dim'] = z_dim
            criterion['margin'] = 0
            self.criterion = am_softmax(criterion, scale=1)
        elif criterion['type'] == 'amsoftmax':
            criterion['z_dim'] = z_dim
            self.criterion = am_softmax(criterion)
        elif criterion['type'] == 'peeler':
            criterion['z_dim'] = z_dim
            self.criterion = peeler_loss(criterion)
        elif criterion['type'] == 'dproto':
            criterion['z_dim'] = z_dim
            self.criterion = dproto(criterion)    

        self.feat_extractor = orig_model.feat_extractor
    
    def get_embeddings(self, x):
        # x is a batch of data
        if self.preprocessing:
            x = self.preprocessing.extract_features(x)
        if self.feat_extractor:
            zq = zq 
        zq = self.encoder.forward(x)
        if self.emb_norm:
            zq = F.normalize(zq, p=2.0, dim=-1)
        return zq

    def loss(self, x):
        # get information
        n_class = x.size(0)
        n_sample = x.size(1)

        #  inference
        x = x.view(n_class * n_sample, *x.size()[2:]).cuda()
        zq = self.get_embeddings(x)
        
        # loss
        loss_val = self.criterion.compute(zq, n_sample, n_class)

        return loss_val, {
            'loss': loss_val.item(),
        }

    def loss_class(self, x, labels):
        zq = self.get_embeddings(x)
        return self.criterion.compute(zq, labels)
    
    def reset(self, opt):

        # initialize custom-keyword adapters (delta)
        for k, v in self.encoder.named_parameters():
            if 'delta' in k:
                # initialize each adapter as an identity matrix
                if self.cka_init == 'eye':
                    if v.size(0) > 1:
                        v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
                    else:
                        v.data = torch.ones(v.size()).to(v.device)
                    # for residual adapter, each adapter is initialized as identity matrix scaled by 0.0001
                    if  opt['adapting.cka_ad_type'] == 'residual':
                        v.data = v.data * 0.0001
                    if 'bias' in k:
                        v.data = v.data * 0
                elif self.cka_init == 'random':
                    # randomly initialization
                    v.data = torch.rand(v.data.size()).data.normal_(0, 0.001).to(v.device)
        # initialize pre-classifier alignment mapping (beta)
        # v = self.beta.weight
        # self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
