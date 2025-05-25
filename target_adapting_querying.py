import sys
import os
import json
from functools import partial
from tqdm import tqdm
import time 
import numpy as np

# needed by the computing infrastructure, you can remove it!
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('_CONDOR_AssignedGPUs', 'CUDA1').replace('CUDA', '')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler 
import torchvision
import torchnet as tnt
from models.utils import get_model

from utils import filter_opt
import log as log_utils

# classification models
from classifiers.NCM import NearestClassMean
from classifiers.NCM_openmax import NCMOpenMax
from classifiers.peeler import PeelerClass
from classifiers.dproto import DProto
from metrics import compute_metrics 

# custom keyword adapters
from models.adapting_para_selection import get_cka_params
from models.CKAs_module import ReprModel_cka



def test_model(data_loader, classifier, unknow_id, force_unk_testdata=False):
    y_pred_tot = []
    y_true = []
    y_score = []
    y_pred_close_tot = []
    y_pred_ood_tot = []
    
    score_corr = 0
    score_wrong = 0
    
    # Initialize lists to accumulate confidences across batches
    all_correct_confidences = []
    all_incorrect_confidences = []

    for sample in tqdm(data_loader):

        x = sample['data']
        labels = sample['label'] # labela

        # replace labels with unknown
        if force_unk_testdata:
            labels = ['_unknown_' for item in labels]

        # perform classification
        p_y, target_ids = classifier.evaluate_batch(x, labels, return_probas=False)
        
        # compute the probabilities
        _, y_pred = p_y.max(1)
        conf_val = p_y.gather(1, y_pred.unsqueeze(1)).squeeze().view(-1)

        if '_unknown_' in classifier.word_to_index.keys():
            y_pred_ood = p_y[:,unknow_id]

            unknow_lab = torch.zeros(p_y.size(0)).add(unknow_id).long()
            y_pred_close = p_y[(1 - torch.nn.functional.one_hot(unknow_lab, p_y.shape[1])).bool()].reshape(p_y.shape[0], -1) 
        else:
            y_pred_close = p_y
            y_pred_ood = None
        y_pred_ood_tot += y_pred_ood.tolist()
        y_pred_close_tot += y_pred_close.tolist()

        y_pred_tot +=  y_pred.tolist()
        target_ids = target_ids.squeeze().tolist()
        y_true += [target_ids] if isinstance(target_ids, int) else target_ids

        conf_val = conf_val.tolist()
        y_score += [conf_val] if isinstance(conf_val, int) else conf_val
    
    return y_score, y_pred_tot, y_true, y_pred_close_tot, y_pred_ood_tot


#######################################################################
# Target Adapting & Querying: 
#   1. Target domain adapting by integrating the user-keyword adapters.
#   2. Target domain querying with Prototype Reprojection Module.
#######################################################################
if __name__ == '__main__':
    from parser_kws import *
    args = parser.parse_args()
    opt = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['choose_cuda']
    cuda = opt['data.cuda']
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = ['loss']
    opt['model.model_name'] = 'repr_conv'
    opt['model.encoding'] = 'DSCNNL_LAYERNORM'
    opt['train.loss'] = 'triplet'
    opt['train.margin'] = 0.5
    speech_args = filter_opt(opt, 'speech')
    model_opt = filter_opt(opt, 'model')
    model_type = model_opt['model_name']
    
    # prepare preprocessing
    if opt['model.preprocessing'] == 'mfcc':
        print('Setup Preprocessing configuration structure')
        model_opt['mfcc'] = { 
            'window_size_ms': speech_args['window_size'],
            'window_stride_ms': speech_args['window_stride'],
            'sample_rate': speech_args['sample_rate'],
            'n_mfcc': speech_args['n_mfcc'],
            'feature_bin_count': speech_args['num_features']
        }
        
    # Metric Learning Parameters
    n_support_s = opt['train.n_support']
    n_query = opt['train.n_query']

    # preparare loss
    print('Loss function: ', opt['train.loss'])
    model_opt['loss'] = {'type': opt['train.loss'], 'margin':  opt['train.margin']}
    if opt['train.loss'] == 'prototypical' or opt['train.loss'] == 'angproto':
        model_opt['loss']['n_support'] = n_support_s
        model_opt['loss']['n_query'] = n_query
    elif opt['train.loss'] == 'peeler' or opt['train.loss'] == 'dproto':
        model_opt['loss']['n_support'] = opt['train.n_support']
        model_opt['loss']['n_query'] = opt['train.n_query']
        model_opt['loss']['n_way_u'] = opt['train.n_way_u']
        
    # load the model
    # model = get_model(model_opt)
    # print(model)
    
    # import tasks: positive samples and optionative negative samples for open set
    # current limitations: tasks belongs to same dataset (separate eyword split)
    speech_args = filter_opt(opt, 'speech')
    dataset = opt['speech.dataset']
    data_dir = opt['speech.default_datadir'] 
    task = opt['speech.task'] 
    tasks = task.split(",")
    if len(tasks) == 2:
        pos_task, neg_task = tasks
    elif len(tasks) == 1:
        pos_task = tasks[0]
        neg_task = None

    if dataset == 'googlespeechcommand':
        from data.GSC import GSCSpeechDataset
        ds = GSCSpeechDataset(data_dir, pos_task, opt['data.cuda'], speech_args)
        num_classes = ds.num_classes()
        opt['model.num_classes'] = num_classes
        print("The task {} of the {} Dataset has {} classes".format(
                pos_task, dataset, num_classes))
        
        ds_neg = None
        if neg_task is not None:
            ds_neg = GSCSpeechDataset(data_dir, neg_task, 
                    opt['data.cuda'], speech_args)
            print("The task {} is used for negative samples".format(
                    neg_task))       
    elif dataset == 'MDSC':
        from data.MDSC import MDSCSpeechDataset
        ds = MDSCSpeechDataset(data_dir, pos_task, opt['data.cuda'], speech_args)
        num_classes = ds.num_classes()
        opt['model.num_classes'] = num_classes
        print("The task {} of the {} Dataset has {} classes".format(
                pos_task, dataset, num_classes))
        
        ds_neg = None
        if neg_task is not None:
            ds_neg = MDSCSpeechDataset(data_dir, neg_task, 
                    opt['data.cuda'], speech_args)
            print("The task {} is used for negative samples".format(
                    neg_task)) 
    elif dataset == 'UASpeech':
        from data.UASpeech import UASpeechDataset
        ds = UASpeechDataset(data_dir, pos_task, opt['data.cuda'], speech_args)
        num_classes = ds.num_classes()
        opt['model.num_classes'] = num_classes
        print("The task {} of the {} Dataset has {} classes".format(
                pos_task, dataset, num_classes))
        
        ds_neg = None
        if neg_task is not None:
            ds_neg = UASpeechDataset(data_dir, neg_task, 
                    opt['data.cuda'], speech_args)
            print("The task {} is used for negative samples".format(
                    neg_task))
    else:
        raise ValueError("Dataset not recognized")


    # Few-Shot Parameters to configure the classifier for testing
    # the test is done over n_episodes
    # n_support support samples of n_way classes are avaible at test time 
    n_way = opt['fsl.test.n_way']
    n_support = opt['fsl.test.n_support']
    n_episodes = opt['fsl.test.n_episodes']
    fixed_silence_unknown = opt['fsl.test.fixed_silence_unknown']
    
    # setup dataloader of support samples
    # support samples are retrived from the training split of the dataset
    # if include_unknown is True, the _unknown_ class is one of the num_classes
    sampler = ds.get_episodic_fixed_sampler(num_classes,  n_way, n_episodes, 
        fixed_silence_unknown = fixed_silence_unknown, include_unknown = speech_args['include_unknown'])
    train_episodic_loader = ds.get_episodic_dataloader('training', n_way, n_support, n_episodes, sampler=sampler)
    
    # Postprocess arguments
    #   list of log variables. may be turned into a configurable list usign opt['log.fields'] as 
    #   opt['log.fields'] = opt['log.fields'].split(',')
    opt['log.fields'] = ['aucROC','accuracy_pos', 'accuracy_neg', 'acc_prec95','frr_prec95']

    # import stats
    meters = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } 

    # evaluate the model on multiple episodes 
    print("Evaluating model in a few-shot setting ({}-way | {}-shots) for {} episodes on the task {} of the Dataset {}".format(
            n_way,n_support, n_episodes, task, dataset))
    output = {'test':{}}
    
    # 
    for ep, support_sample in enumerate(train_episodic_loader):
        '''
            1.load model & prepare data
        '''
        # load the model
        model = get_model(model_opt)
        # print(model)
        
        # initialize weights from a pretrained model store in model.model_path (not used currently)
        if os.path.isfile(opt['model.model_path']):   
            print('Load Pretrained Model from', opt['model.model_path'])
            enc_model = torch.load(opt['model.model_path'])
            model.encoder.load_state_dict(enc_model.encoder.state_dict(), strict=True) # strict很重要 strict=False 仅加载匹配的参数，而不会加载缺失的参数。
        else:
            raise ValueError("Model {} not valid".format(opt['model.model_path']))
                    
        # move to cuda
        if opt['data.cuda']:
            model.cuda()     
            if  'mfcc' in model_opt.keys():
                model.preprocessing.mfcc.cuda()
        
        support_samples = support_sample['data']
                
            
        '''
            2. target adapting 
        '''
        # freeze the pretrained backbone 
        model.eval()
        
        criterion = model_opt['loss'] if 'loss' in model_opt.keys() else False
        x_dim = model_opt['x_dim']
        
        # integrate the pretrained backbone and custom-keyword adapters (CKAs)
        model = ReprModel_cka(model, opt, criterion, x_dim)
        model.reset(opt)
        if opt['data.cuda']:
            model.cuda()
            if  'mfcc' in model_opt.keys():
                model.preprocessing.mfcc.cuda()
                
        # target adapted parameter settings
        cka_para = get_cka_params(model)
        optimizer_2nd = torch.optim.Adadelta(cka_para, lr=0.001) 
        
        # target adapting used target support set
        print("\n  ========The {}-th trial: Target domain adapting start ========  \n".format(ep))
        for i in tqdm(range(int(opt['adapting.episodes'])), desc="Target domain adapting:"):
            
            model.eval() 
            
            samples_t = support_samples 
            
            # delet the unknow data in samples_t
            class_list_t = support_sample['label'][0]
            word_to_index_t = {}
            for j,item in enumerate(class_list_t):
                word_to_index_t[item] = j
            unk_idx_t = word_to_index_t['_unknown_'] if '_unknown_' in word_to_index_t.keys() \
                                                    else None
            samples_t = torch.cat((samples_t[:unk_idx_t], samples_t[unk_idx_t+1:]))
            
            # shuffle samples_t's samples
            dim_size = samples_t.size(1)
            permuted_indices = torch.randperm(dim_size)
            samples_t = samples_t[:, permuted_indices, :, :]
            # shuffle samples_t's classes
            dim_size = samples_t.size(0)
            permuted_indices = torch.randperm(dim_size)
            samples_t = samples_t[permuted_indices, :, :, :]

            if cuda:
                samples_t = samples_t.cuda()
            
            optimizer_2nd.zero_grad()
            model.zero_grad()
            
            loss, _ = model.loss(samples_t)
            # print("loss: {}".format(loss))
            # print(loss.requires_grad) 
            loss.backward()

            optimizer_2nd.step()
          
            
        '''
            3. Open-set classifier setup 
        '''      
        # load the classifier
        opt['fsl.classifier'] = 'ncm'
        print('Using the classifier: ', opt['fsl.classifier'])
        print('Using prototype reprojection:{}'.format(opt['querying.prototype_reprojection']))
        if opt['fsl.classifier'] == 'ncm':
            classifier = NearestClassMean(backbone=model, cuda=opt['data.cuda'], reprojection=opt['querying.prototype_reprojection'])
        elif opt['fsl.classifier'] == 'ncm_openmax':
            classifier = NCMOpenMax(backbone=model, cuda=opt['data.cuda'])
        elif opt['fsl.classifier'] == 'peeler':
            classifier = PeelerClass(backbone=model, cuda=opt['data.cuda'])
        elif opt['fsl.classifier'] == 'dproto':
            classifier = DProto(backbone=model, cuda=opt['data.cuda'])
        else:
            raise ValueError("Classifier {} is not valid".format(opt['fsl.classifier']))

        print(classifier)
        
        # compute prototypes
        support_samples = support_sample['data']
        # extract label list          
        class_list = support_sample['label'][0]
        print(class_list)
        # fit the classifier on the support samples
        classifier.fit_batch_offline(support_samples, class_list)
        #get the index of the unknown class of the classifier
        unk_idx = classifier.word_to_index['_unknown_'] if '_unknown_' in classifier.word_to_index.keys() \
                                                        else None

        '''
            Few-shot test in open set
            NB: _unknown_ is the negative class as part of the class_list
        '''  
        # test on positive dataset     
        print('\n Test Episode {} with classes: {}'.format(ep, class_list))

        # load only samples from the target classes and not negative _unknown_
        query_loader = ds.get_iid_dataloader('testing', opt['fsl.test.batch_size'], 
            class_list = [x for x in class_list if 'unknown' not in x])
        y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos = test_model(query_loader, classifier, unk_idx)

        # test on the negative dataset (_unknown_) if present    
        if ds_neg is not None:
            neg_loader = ds_neg.get_iid_dataloader('testing', opt['fsl.test.batch_size'])
            y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg = test_model(neg_loader, classifier, unk_idx, force_unk_testdata=True)
        else:
            y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg = None, None, None, None, None
        
        # store and print metrics
        output_ep = compute_metrics(y_score_pos, y_pred_pos, y_true_pos, y_pred_close_pos, y_pred_ood_pos,
                        y_score_neg, y_pred_neg, y_true_neg, y_pred_close_neg, y_pred_ood_neg,
                        classifier.word_to_index, verbose=True)

        for field, meter in meters.items():
            meter.add(output_ep[field])
        output[str(ep)] = output_ep


    for field,meter in meters.items():
        mean, std = meter.value()
        output["test"][field] = {}
        output["test"][field]["mean"] = mean
        output["test"][field]["std"] = std
        print("Final Test: Avg {} is {} with std dev {}".format(field, mean, std))

    # write log
    if speech_args['include_unknown']:
        n_way = n_way - 1
    if classifier.backbone.emb_norm:
        fsl_z_norm = "NORM"
    else: 
        fsl_z_norm = "NOTN"
    
    
    output_log_file = 'Adapting_results_{}epi_PRM_{}_{}_{}_{}_{}_{}_{}'.format(opt['adapting.episodes'],opt['querying.prototype_reprojection'],opt['fsl.classifier'],fsl_z_norm,task,n_way,n_support,n_episodes)
    output_file = os.path.join(os.path.dirname(opt['model.model_path']), output_log_file)
    print('Writing log to:', output_file)

    with open(output_file, 'w') as fp:
        json.dump(output, fp)

