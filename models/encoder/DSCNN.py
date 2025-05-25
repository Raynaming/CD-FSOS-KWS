import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder.baseUtil import Flatten, get_padding
from collections import OrderedDict

import math

class Domain_transform(nn.Module):
    def __init__(self, planes):
        super(Domain_transform, self).__init__()
        self.planes = planes
        self.avg = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear=torch.nn.Linear(planes, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.detach().data # detach()用于将一个张量从计算图中分离出来，使其不再参与梯度计算
        x = self.avg(x).view(-1, self.planes)
        x = self.linear(x)
        x = self.relu(x)
        domain_offset = x.mean()
        return domain_offset


class AN(nn.Module):

    def __init__(self, planes): # planes：num_features
        super(AN, self).__init__()
        self.IN = nn.InstanceNorm2d(planes, affine=False)
        self.BN = nn.BatchNorm2d(planes, affine=False)
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.alpha_t = torch.Tensor([0.0])
        self.domain_transform = Domain_transform(planes)

    def forward(self, x):
        # if gol.get_value('is_ft') and gol.get_value('use_transform'):
        # print(self.alpha)
        self.alpha_t = self.alpha + 0.02 * self.domain_transform(x) # default: self.alpha_t = self.alpha + 0.01 * self.domain_transform(x)
        # print(self.alpha_t)
        if x.device == torch.device('cpu'):
            t = torch.sigmoid(self.alpha_t)
        else:
            t = torch.sigmoid(self.alpha_t).cuda()
        # else:
        #     t = torch.sigmoid(self.alpha).cuda()
        out_in = self.IN(x)
        out_bn = self.BN(x)
        out = t * out_in + (1 - t) * out_bn
        return out
    
def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
  feature_augment = True
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    if self.feature_augment: # initialize {gamma, beta} with {0.3, 0.5}
      self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3, requires_grad=True)
      self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3, requires_grad=True)
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    # if self.weight.fast is not None and self.bias.fast is not None:
    #   weight = self.weight.fast
    #   bias = self.bias.fast
    # else:
    weight = self.weight
    bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    #   print(out.shape)
    else:
      out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)
    #   print(out.shape)

    # apply feature-wise transformation
    if self.feature_augment and self.training:
      gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
      beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      out = gamma*out + beta
    return out

class DSCNN(nn.Module):
    
    def __init__(self, t_dim, f_dim, model_size_info, padding_0, last_norm=True, return_feat_maps=False ):
        super(DSCNN, self).__init__()
        self.input_features = [t_dim,f_dim]
        self.return_feat_maps = return_feat_maps

        num_layers = model_size_info[0]
        conv_feat = [0 for x in range(num_layers)]
        conv_kt = [0 for x in range(num_layers)]
        conv_kf = [0 for x in range(num_layers)]
        conv_st = [0 for x in range(num_layers)]
        conv_sf = [0 for x in range(num_layers)]
        i=1
        for layer_no in range(0,num_layers):
            conv_feat[layer_no] = model_size_info[i]
            i += 1
            conv_kt[layer_no] = model_size_info[i]
            i += 1
            conv_kf[layer_no] = model_size_info[i]
            i += 1
            conv_st[layer_no] = model_size_info[i]
            i += 1
            conv_sf[layer_no] = model_size_info[i]
            i += 1
            
            
        ds_cnn_layers = []
        
        for layer_no in range(0,num_layers):
            num_filters = conv_feat[layer_no]
            kernel_size = (conv_kt[layer_no],conv_kf[layer_no])
            stride = (conv_st[layer_no],conv_sf[layer_no])

            if layer_no==0:
                # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
                padding = (   int( (conv_kt[layer_no]-1)  //2 ),  int( (conv_kf[layer_no]-1) //2) )
                ds_cnn_layers.append( nn.Conv2d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size, stride = stride, padding = padding_0, bias = True) )
                # ds_cnn_layers.append( AN(num_filters) )
                ds_cnn_layers.append( nn.BatchNorm2d(num_filters) )
                ds_cnn_layers.append( nn.ReLU() )
            else:
                ds_cnn_layers.append( nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = kernel_size, stride = stride, padding = (1,1), groups = num_filters, bias = True) )
                # ds_cnn_layers.append( AN(num_filters) )
                ds_cnn_layers.append( nn.BatchNorm2d(num_filters) )
                ds_cnn_layers.append( nn.ReLU() )
                ds_cnn_layers.append( nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = (1, 1), stride = (1, 1), bias = True) )
                if (last_norm== True) or (layer_no < num_layers-1):
                    # ds_cnn_layers.append( FeatureWiseTransformation2d_fw(num_filters) )
                    ds_cnn_layers.append( nn.BatchNorm2d(num_filters) )
                    # ds_cnn_layers.append( AN(num_filters) )
                    ds_cnn_layers.append( nn.ReLU() )
                elif (last_norm== 'Layer'):
                    ds_cnn_layers.append( nn.LayerNorm([num_filters, t_dim, f_dim], elementwise_affine=False) )
            
            t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
            f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

                
        self.dscnn = nn.Sequential(*ds_cnn_layers)
        self.embedding_features = num_filters

        self.avgpool = nn.AvgPool2d(kernel_size=(t_dim, f_dim), stride=1) 
        self.flatten = Flatten() 
        
        
    def forward(self, x):
        x = self.dscnn(x)
        if self.return_feat_maps:
            return x
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

        
            
# DSCNN_SMALL
model_size_info_DSCNNS = [5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]
padding_0_DSCNNS = (5,1)

def DSCNNS(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS )

def DSCNNS_NONORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS, last_norm=False  )

def DSCNNS_LAYERNORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS, last_norm='Layer'  )

# DSCNN_MEDIUM
model_size_info_DSCNNM = [5, 172, 10, 4, 2, 1, 172, 3, 3, 2, 2, 172, 3, 3, 1, 1, 172, 3, 3, 1, 1, 172, 3, 3, 1, 1]
padding_0_DSCNNM = (5,1)

def DSCNNM(x_dim):
    return DSCNN(x_dim[1], x_dim[2],model_size_info_DSCNNM, padding_0_DSCNNM  )

# DSCNN_LARGE
model_size_info_DSCNNL = [6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1]
padding_0_DSCNNL = (5,1)

def DSCNNL(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL  )

def DSCNNL_NONORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL , last_norm=False  )

def DSCNNL_LAYERNORM(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL  , last_norm='Layer'  )


# for peeler
def DSCNNS_PEELER(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNS, padding_0_DSCNNS, return_feat_maps=True )
def DSCNNL_PEELER(x_dim):
    return DSCNN(x_dim[1], x_dim[2], model_size_info_DSCNNL, padding_0_DSCNNL, return_feat_maps=True )
