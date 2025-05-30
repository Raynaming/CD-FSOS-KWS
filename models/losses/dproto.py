# adapted from https://github.com/Sha-Lab/FEAT/ to reproduce https://arxiv.org/abs/2206.13691


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils import euclidean_dist
from torch.autograd import Variable


class DeepSetsFunc_Unk(nn.Module):
    def __init__(self, z_dim):
        super(DeepSetsFunc_Unk, self).__init__()
        """
        DeepSets Function
        """
        self.L = 3
        self.z_dim = z_dim

        self.hidden = 32 #z_dim*4
        self.gen1 = nn.Linear(z_dim, self.hidden)
        self.gen2 = nn.Linear(self.hidden, z_dim * self.L, bias=False)


    def forward(self, set_input):
        """
        set_input, seq_length, set_size, dim
        """
        num_proto, dim = set_input.shape
        combined_mean = F.relu(self.gen1(set_input.view(-1, self.z_dim)))
#        print(combined_mean.size())
        combined_mean = combined_mean.max(0)[0]
#        print('after max: ', combined_mean.size())
        combined_mean = self.gen2(combined_mean)
        combined_mean = combined_mean.view(self.L, self.z_dim)
        return combined_mean

class dproto(nn.Module):
    def __init__(self, args):
        super(dproto, self).__init__()
        print(args)
        self.n_support = args['n_support']
        self.n_query = args['n_query']
        self.n_way_u = args['n_way_u']
        # if args['distance'] is not None:
        #     self.distance =  args['distance']
        # else:
        self.distance =  'euclidean'
        
        if self.distance ==  'cosine':
            init_w=10.0
            init_b=-5.0
            self.w = nn.Parameter(torch.tensor(init_w))
            self.b = nn.Parameter(torch.tensor(init_b))

        self.margin = args['margin']
        self.in_feats = args['z_dim']

        self.set_func = DeepSetsFunc_Unk(self.in_feats) 
        self.temp_knw = 1
        self.temp_unk = 3
        self.gamma = 0.1

        

    def compute(self, zq, n_sample, n_class):

        # split support and query samples
        assert n_class > self.n_way_u, "Amount of unknown classes is {} must be lower than classes per episode (currently: {}) per episode".format(self.n_way_u, n_class)
        assert n_sample == (self.n_support + self.n_query), "{} samples per batch expected, got {}".format(n_sample, self.n_support + self.n_query)
        zq = zq.view(n_class, n_sample, *zq.size()[1:])

        n_class_known = n_class - self.n_way_u
        support_samples = zq[:n_class_known,:self.n_support,:].contiguous()
        query_samples_knw = zq[:n_class_known,self.n_support:self.n_support+self.n_query,:].contiguous()
        query_samples_ukn = zq[n_class_known:,self.n_support:self.n_support+self.n_query,:].contiguous()

        #compute proto for known and unkwown
        proto_k = support_samples.mean(1)

        proto_u = self.set_func(proto_k) 
#        print(proto_k.size(), proto_u.size())

        # known class L^K
        query_samples_knw = query_samples_knw.view(n_class_known*self.n_query, *query_samples_knw.size()[2:])
        if self.distance ==  'euclidean':
            dists_k = euclidean_dist(query_samples_knw, proto_k) / self.temp_knw
    #        print('dist calc:', query_samples_knw.size() , proto_k.size(), dists_k.size())
            dist_u = euclidean_dist(query_samples_knw, proto_u)
            cd = torch.mm(F.gumbel_softmax(-dist_u, dim=0), proto_u)
    #        print('quaery and cd:', query_samples_knw.size() , cd.size())
            dist_u = torch.pow(cd - query_samples_knw, 2).sum(1).unsqueeze(1) / self.temp_unk
    #        print('dists: ', dist_u.size(), dists_k.size())

            score = - torch.cat([dists_k, dist_u], dim=1)
    #        	print('final: ', score.size())
        else:
#            print('going to compute the distance')
            input1 = query_samples_knw.unsqueeze(1).expand( n_class_known*self.n_query, n_class_known, query_samples_knw.size(-1))
            score_k = F.cosine_similarity(input1, proto_k, dim=2)
#            print('score_k:', input1.size(), proto_k.size(), score_k.size())

            input1 = query_samples_knw.unsqueeze(1).expand( n_class_known*self.n_query, self.set_func.L, query_samples_knw.size(-1))
            score_u = F.cosine_similarity(input1, proto_u, dim=2)
#            print('score_u:', query_samples_knw.size(), input1.size(), proto_u.size(), score_u.size())

            if self.margin > 0:
                for cl in range(n_class_known):
                    score_k[self.n_query*cl:self.n_query*cl+self.n_query,cl].add_(-self.margin)

            cd = torch.mm(F.gumbel_softmax(score_u, dim=0), proto_u)
            score_u = F.cosine_similarity(cd, query_samples_knw, dim=1).unsqueeze(1)
#            print('score_u', cd.size(), query_samples_knw.size(), score_u.size())

            score = torch.cat([score_k, score_u], dim=1)
#            print('final: ', score.size())
            # apply learned scaling
            torch.clamp(self.w, 1e-6)
            score = score * self.w + self.b



        log_p_y = F.log_softmax(score, dim=1)
        log_p_y = log_p_y.view(n_class_known, self.n_query, -1)

        # compute targets
        target_inds = torch.arange(0, n_class_known).view(n_class_known, 1, 1).expand(n_class_known, self.n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if zq.is_cuda:
            target_inds = target_inds.cuda()
#        print('target_inds: ', target_inds.size())

        # compute loss
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()


#        print('log_p_y: ', log_p_y.size())


        # known class L^UKN
        query_samples_ukn = query_samples_ukn.view(self.n_way_u*self.n_query, *query_samples_ukn.size()[2:])
        if self.distance ==  'euclidean':
            dists_k = euclidean_dist(query_samples_ukn, proto_k) / self.temp_knw
            dist_u = euclidean_dist(query_samples_ukn, proto_u)
            cd = torch.mm(F.gumbel_softmax(-dist_u, dim=0), proto_u)
    #        print(cd.size())
    #        print('quaery and cd:', query_samples_ukn.size() , cd.size())
            dist_u = torch.pow(cd - query_samples_ukn, 2).sum(1).unsqueeze(1) / self.temp_unk
            score = -torch.cat([dists_k, dist_u], dim=1)
        else:
            input1 = query_samples_ukn.unsqueeze(1).expand( self.n_way_u*self.n_query, n_class_known, query_samples_ukn.size(-1))
            score_k = F.cosine_similarity(input1, proto_k, dim=2)
            input1 = query_samples_ukn.unsqueeze(1).expand( self.n_way_u*self.n_query, self.set_func.L, query_samples_ukn.size(-1))            
            score_u = F.cosine_similarity(input1, proto_u, dim=2)
            if self.margin > 0:
                for cl in range(self.n_way_u):
                    score_k[self.n_query*cl:self.n_query*cl+self.n_query,cl].add_(-self.margin)

            cd = torch.mm(F.gumbel_softmax(score_u, dim=0), proto_u)
            score_u = F.cosine_similarity(cd, query_samples_ukn, dim=1).add_(-self.margin).unsqueeze(1)
#            print('score_u', cd.size(), query_samples_knw.size(), score_u.size())


            score = torch.cat([score_k, score_u], dim=1)
#            print('final: ', score.size())
            # apply learned scaling
            torch.clamp(self.w, 1e-6)
            score = score * self.w + self.b

        log_p_y = F.log_softmax(score, dim=1)
        log_p_y = log_p_y.view(self.n_way_u, self.n_query, -1)

        # compute targets of unknown : [n_class_known]
        target_inds = torch.zeros(self.n_way_u).add(n_class_known).view(self.n_way_u, 1, 1).expand(self.n_way_u, self.n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if zq.is_cuda:
            target_inds = target_inds.cuda()
#        print('target_inds: ', target_inds.size(), target_inds, log_p_y.size())
        
        # compute loss
        loss_val += -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() * self.gamma


#        print('final: ', loss_val, log_p_y.size())

        return loss_val




