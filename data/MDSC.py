
import os
import random
import re
import hashlib
import math
import glob
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as F

from .data_utils import SetDataset
from functools import partial
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset

# Constants
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 1
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 0
RANDOM_SEED = 42

def prepare_words_list(wanted_words, silence, unknown):
    extra_words = []
    if silence:
        extra_words.append(SILENCE_LABEL)
    if unknown:
        extra_words.append(UNKNOWN_WORD_LABEL)
    return extra_words + wanted_words

class EpisodicFixedBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, fixed_silence_unknown = False, include_unknown=True):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        if fixed_silence_unknown:
            skip = 2
            fixed_class = torch.tensor([SILENCE_INDEX, UNKNOWN_WORD_INDEX])
            n_way = n_way-skip
            self.sampling = []
            for i in range(self.n_episodes): 
                selected = torch.randperm(self.n_classes - skip)[:n_way]
                selected = torch.cat((fixed_class, selected.add(skip)))
                self.sampling.append(selected)        
        else:
            self.sampling = [torch.randperm(self.n_classes)[:self.n_way] for i in range(self.n_episodes)]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield self.sampling[i]
            

class MDSCSpeechDataset:
    def __init__(self, data_dir, MDSCtype, cuda, args):
        self.sample_rate = args['sample_rate']
        self.clip_duration_ms = args['clip_duration'] 
        self.window_size_ms = args['window_size']
        self.window_stride_ms = args['window_stride']
        self.n_mfcc = args['n_mfcc']
        self.feature_bin_count = args['num_features']
        self.foreground_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']
        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)

        # by now silence and background are enabled by default        
        self.use_background = args['include_noise']
        self.background_volume = args['bg_volume']
        self.background_frequency= args['bg_frequency']
        
        self.silence = args['include_silence']
        self.silence_num_samples = args['num_silence']
        self.unknown = args['include_unknown']
        
        self.data_cache = {}
        self.data_dir = data_dir
        
        # this are properties of the dataset!
        MDSC_training_parameters = {
            'unknown_percentage':30.0,
            'testing_percentage':10.0,
        }
        unknown_words = ['打开空调','开灯','拉开窗帘','打开后一集','播放上一首']
        if MDSCtype == 'MDSC12':
            #self.silence = True
            target_words='小度小度,小爱同学,天猫精灵,你好小布,小艺小艺,小溪你好,Hey Siri,小德小德,灵犀灵犀,小冰小冰,'  # MDSC - 10 words
            print('10 word')
        elif MDSCtype == 'MDSC22':
            # self.silence = True
            target_words='关闭空调,关灯,关闭窗帘,全部关闭,打开蓝牙,帮忙暂停,降低音量,开启省电,音量调大,替我取消暂停,重新开始,播放下一首,全部打开,打开上一集,调到下一频道,'  
            unknown_words = []
            print('15 word')
            
        elif MDSCtype == 'MDSC10': #meta train task
            target_words='小度小度,小爱同学,天猫精灵,你好小布,小艺小艺,小溪你好,Hey Siri,小德小德,灵犀灵犀,小冰小冰,'
            unknown_words = []
            print('10 words for meta train taks')
        elif MDSCtype == 'MDSC5': #meta val task
            target_words='小度小度,小爱同学,天猫精灵,你好小布,小艺小艺,'
            unknown_words = []
            print('5 words for meta val taks')
        else:
            print("data type of illegality")
        wanted_words=(target_words).split(',')
        wanted_words.pop()
        MDSC_training_parameters['wanted_words'] = wanted_words
        MDSC_training_parameters['unknown_words'] = unknown_words


        self.generate_data_dictionary(MDSC_training_parameters)
        
        #try if can I include cuda here
        self.cuda = cuda
        self.max_class = len(wanted_words)
        
    def get_episodic_fixed_sampler(self, num_classes,  n_way, n_episodes, fixed_silence_unknown = False, include_unknown = True):
        return EpisodicFixedBatchSampler(num_classes, n_way, n_episodes, fixed_silence_unknown = fixed_silence_unknown, include_unknown=include_unknown)
                    
    def get_episodic_dataloader(self, set_index, n_way, n_samples, n_episodes, sampler='episodic', 
            include_silence=True, include_unknown=True, unique_speaker=False):

        # exclude silence and unknown from the list
        class_list = []
        for item in self.words_list:
            if not include_silence and item == SILENCE_LABEL:
                continue
            if not include_unknown and item == UNKNOWN_WORD_LABEL:
                continue
            class_list.append(item)
        
        if sampler == 'episodic':
            sampler = self.get_episodic_fixed_sampler(len(class_list),  
                            n_way, n_episodes)

        dl_list=[]        
        if set_index in ['training', 'testing']:
            for keyword in class_list:
                ts_ds = self.get_transform_dataset(self.data_set[set_index], [keyword])
                
                if n_samples <= 0:
                    n_support = len(ts_ds)
                
                dl = torch.utils.data.DataLoader(ts_ds, batch_size=n_samples, 
                        shuffle=True, num_workers=0)
                dl_list.append(dl)

            ds = SetDataset(dl_list)

            data_loader_params = dict(batch_sampler = sampler,  num_workers =8, 
                    pin_memory=not self.cuda)   
            dl = torch.utils.data.DataLoader(ds, **data_loader_params)
        else:
            raise ValueError("Set index = {} in episodic dataset is not correct.".format(set_index))

        return dl
    
    def get_iid_dataloader(self, set_index, batch_size, class_list = False, include_silence=True, include_unknown=True, unique_speaker=False):
        
        # exclude silence and unknown from the list
        if not class_list:
            class_list = []
            for item in self.words_list:
                if not include_silence and item == SILENCE_LABEL:
                    continue
                if not include_unknown and item == UNKNOWN_WORD_LABEL:
                    continue
                class_list.append(item)
            
        ts_ds = self.get_transform_dataset(self.data_set[set_index], class_list)
        dl = torch.utils.data.DataLoader(ts_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        return dl
    
    def dataset_filter_class(self, dslist, classes):
    # FIXME: by now unique_speaker are not handled
        filtered_ds = []
        for item in dslist:
            label = item['label']
            if label in classes:
                filtered_ds.append(item)
            
        return filtered_ds
    
    def get_transform_dataset(self, file_dict, classes, filters=None):
        # file dict include is [{ 'label': LABEL_str, 'file': file_path, 'speaker': spkr_id}, .. ]
        # classes is a list of classes
        transforms = compose([
                partial(self.load_audio, 'file', 'label', 'data'),
                partial(self.adjust_volume, 'data'),
                partial(self.shift_and_pad, 'data'),
                partial(self.label_to_idx, 'label', 'label_idx')

        ])
        file_dict = self.dataset_filter_class(file_dict, classes)
        ls_ds = ListDataset(file_dict)
        ts_ds = TransformDataset(ls_ds, transforms)
        
        return ts_ds
    
    def num_classes(self):
        return len(self.words_list)
    
    def label_to_idx(self, k, key_out, d):
        label_index = self.word_to_index[d[k]]
        d[key_out] = torch.LongTensor([label_index]).squeeze()
        return d
    
    def shift_and_pad(self, key, d):
        audio = d[key]
        time_shift = int((self.time_shift_ms * self.sample_rate) / 1000)
        if time_shift > 0:
            time_shift_amount = np.random.randint(-time_shift, time_shift)
        else:
            time_shift_amount = 0
        
        if time_shift_amount > 0:
            time_shift_padding = (time_shift_amount, 0)
            time_shift_offset = 0
        else:
            time_shift_padding = (0, -time_shift_amount)
            time_shift_offset = -time_shift_amount
        
        
        # Ensure data length is equal to the number of desired samples
        audio_len = audio.size(1)
        if audio_len < self.desired_samples:
            pad = (0,self.desired_samples-audio_len)
            audio=F.pad(audio, pad, 'constant', 0) 
        elif audio_len > self.desired_samples:
            print('shift & pad: audio_len > self.desired_samples')
            start = np.random.randint(0, audio_len - self.desired_samples + 1)
            audio = audio[:, start:start + self.desired_samples]
            
        padded_foreground = F.pad(audio, time_shift_padding, 'constant', 0)
        sliced_foreground = torch.narrow(padded_foreground, 1, time_shift_offset, self.desired_samples)

        d[key] = sliced_foreground
        return d
    
    def adjust_volume(self, key, d):
        d[key] =  d[key] * self.foreground_volume
        return d
    
    def load_audio(self, key_path, key_label, out_field, d):
            #{'label': '小度小度', 'file': '../../data/MDSC/wav/CF0005/CF0005_0001.wav', 'speaker': 'CF0005'}
            sound, _ = torchaudio.load(filepath=d[key_path], normalize=True,
                                            num_frames=self.desired_samples)
            
            channels = sound.shape[0]
            if channels == 2:
                # transform 2 channels to 1 channel
                sound = sound.mean(dim=0, keepdim=True)
            # For silence samples, remove any sound
            if d[key_label] == SILENCE_LABEL:
                sound.zero_()
            d[out_field] = sound
            return d
    
    def generate_data_dictionary(self, training_parameters):
        # For each data set, generate a dictionary containing the path to each file, its label, and its speaker.
        # Make sure the shuffling and picking of unknowns is deterministic.
        wanted_words_index = {}
        unknown_words = training_parameters['unknown_words']
        
        global SILENCE_INDEX
        skip = 0
        # if self.silence:
        #     skip +=1
        if self.unknown:
            skip +=1
        else:
            SILENCE_INDEX = SILENCE_INDEX -1
        
        for index, wanted_word in enumerate(training_parameters['wanted_words']):
            wanted_words_index[wanted_word] = index + skip

        # Prepare data sets
        self.data_set = {'testing': [], 'training': []}
        unknown_set = {'testing': [], 'training': []}
        all_words = {}
        
        # Read data transcript
        data_df = pd.read_csv(os.path.join(self.data_dir, 'transcript', 'label.txt'), sep='\s+', header=None, names=['file', 'label'])
        data_df['label'] = data_df['label'].replace('Hey', 'Hey Siri')
        # print(data_df['label'].unique())
        
        # If it's a known class (wanted_words), store its detail
        for label in training_parameters['wanted_words']:
            # print(label)
            label_data = data_df[data_df['label'] == label]
            num_samples = len(label_data)
            num_train = num_samples // 2 # 1:1 division of training and test sets
            num_test = num_samples - num_train
            train_data = label_data.sample(num_train, random_state=RANDOM_SEED)
            test_data = label_data.drop(train_data.index)
            
            # Save training set
            for _, row in train_data.iterrows():
                file_path = os.path.join(self.data_dir, 'wav', row['file'][:6], row['file'] + '.wav')
                speaker = row['file'][:6]
                self.data_set['training'].append({'label': row['label'], 'file': file_path, 'speaker': speaker})
            
            # Save testing set
            for _, row in test_data.iterrows():
                file_path = os.path.join(self.data_dir, 'wav', row['file'][:6], row['file'] + '.wav')
                speaker = row['file'][:6]
                self.data_set['testing'].append({'label': row['label'], 'file': file_path, 'speaker': speaker})

            all_words[row['label']] = True

        #for unknown_words, add it to the list.
        if training_parameters['unknown_words']:
            unknown_data = data_df[data_df['label'].isin(training_parameters['unknown_words'])]
            unknown_sampled_data = unknown_data.sample(frac=training_parameters['unknown_percentage'] / 100, random_state=RANDOM_SEED)
            unknown_sampled_data = unknown_sampled_data.sample(frac=1, random_state=RANDOM_SEED)
            half_size = len(unknown_sampled_data) // 2
            unknown_train_data = unknown_sampled_data.iloc[:half_size]
            unknown_test_data = unknown_sampled_data.iloc[half_size:]
            # if self.unknown, we'll use to train the unknown label.
            if self.unknown:
                for _, row in unknown_train_data.iterrows():
                    file_path = os.path.join(self.data_dir, 'wav', row['file'][:6], row['file'] + '.wav')
                    speaker = row['file'][:6]
                    self.data_set['training'].append({'label': UNKNOWN_WORD_LABEL, 'file': file_path, 'speaker': speaker})
                
                for _, row in unknown_test_data.iterrows():
                    file_path = os.path.join(self.data_dir, 'wav', row['file'][:6], row['file'] + '.wav')
                    speaker = row['file'][:6]
                    self.data_set['testing'].append({'label': UNKNOWN_WORD_LABEL, 'file': file_path, 'speaker': speaker})
        
        # Make sure the ordering is random.
        random.shuffle(self.data_set['training'])
        random.shuffle(self.data_set['testing'])

        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(training_parameters['wanted_words'], self.silence, self.unknown)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
#            elif word in unknown_words:
#                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        # if self.silence:
        #     self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
        if self.unknown:
            self.word_to_index[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX
    
    

