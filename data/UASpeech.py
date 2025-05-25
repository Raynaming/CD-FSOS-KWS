import os
import random
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
# MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
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
            
class UASpeechDataset:
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
            # 'unknown_percentage':30.0,
            # 'testing_percentage':10.0,
        }
        unknown_words = ['use','an','each','which','she'] # UASPEECH - 5 words - CW41-CW45

        if MDSCtype == 'UASpeech12':
            #self.silence = True
            target_words='it,he,was,for,on,are,as,with,his,they,'  # UASPEECH - 10 words - CW10-CW19: it,he,was,for,on,are,as,with,his,they

            print('10 word')
        elif MDSCtype == 'UASpeech22':
            # self.silence = True
            target_words='I,at,be,this,have,from,or,had,by,word,but,not,what,all,were,we,when,your,can,said,' # UASPEECH - 20 words - CW20-CW39
            unknown_words = []
            print('15 word')
            
        elif MDSCtype == 'UASpeech10': #meta train task
            target_words='it,he,was,for,on,are,as,with,his,they,'
            unknown_words = []
            print('10 words for meta train taks')
        elif MDSCtype == 'UASpeech5': #meta val task
            target_words='it,he,was,for,on,'
            unknown_words = []
            print('5 words for meta val taks')
        else:
            print("data type of illegality")
        wanted_words=(target_words).split(',')
        wanted_words.pop()
        MDSC_training_parameters['wanted_words'] = wanted_words
        MDSC_training_parameters['unknown_words'] = unknown_words


        self.generate_data_dictionary(MDSC_training_parameters)
        
        # self.background_data = self.load_background_data()
        
        # try if can I include cuda here
        self.cuda = cuda
        self.max_class = len(wanted_words)
    
    def get_episodic_fixed_sampler(self, num_classes,  n_way, n_episodes, fixed_silence_unknown = False, include_unknown = True):
        return EpisodicFixedBatchSampler(num_classes, n_way, n_episodes, fixed_silence_unknown = fixed_silence_unknown, include_unknown=include_unknown)
                    
    def get_episodic_dataloader(self, set_index, n_way, n_samples, n_episodes, sampler='episodic', 
            include_silence=True, include_unknown=True, unique_speaker=False):

        # if cuda:
        #   self.transforms.append(CudaTransform())
        # transforms = compose(self.transforms)

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
                # partial(self.mix_background, self.use_background,'data', 'label'),
#                partial(self.extract_features, 'data', 'feat'),
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
            
        padded_foreground = F.pad(audio, time_shift_padding, 'constant', 0)
        sliced_foreground = torch.narrow(padded_foreground, 1, time_shift_offset, self.desired_samples)
        d[key] = sliced_foreground
        return d

    
    def adjust_volume(self, key, d):
        d[key] =  d[key] * self.foreground_volume
        return d
    
    def load_audio(self, key_path, key_label, out_field, d):
        #{'label': 'stop', 'file': '../../data/speech_commands/GSC/stop/879a2b38_nohash_3.wav', 'speaker': '879a2b38'}
        sound, _ = torchaudio.load(filepath=d[key_path], normalize=True,
                                         num_frames=self.desired_samples)
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
            
        # 1. Load speaker wordlist
        wordlist_path = os.path.join(self.data_dir, 'doc', 'speaker_wordlist.xls')
        word_df = pd.read_excel(wordlist_path, sheet_name="Word_filename", header=None, names=['word', 'id'])
        # word_df = pd.read_excel(wordlist_path, header=None, names=['word', 'id'])

        # Create a dictionary mapping words to their IDs
        word_to_id = dict(zip(word_df['word'], word_df['id']))

        # Get the word IDs for wanted words and unknown words
        wanted_word_ids = [word_to_id[word] for word in training_parameters['wanted_words']]
        unknown_word_ids = [word_to_id[word] for word in training_parameters['unknown_words']] if training_parameters['unknown_words'] else []

        # 2. Initialize data sets
        self.data_set = {'testing': [], 'training': []}
        unknown_set = {'testing': [], 'training': []}
        all_words = {}

        # 3. Iterate over each speaker directory in self.data_dir/audio/noisereduce/
        audio_dir = os.path.join(self.data_dir, 'audio', 'noisereduce')
        speakers = os.listdir(audio_dir)
        
        # Patients with severe dysarthria were excluded
        df = pd.read_excel(wordlist_path, sheet_name='Speaker', header=3)
        df['Intelligibility (%)'] = df['Intelligibility (%)'].fillna('').astype(str)
        exclude_intelligibility = ['Very low', 'Low', 'not obtained yet']
        filtered_speakers = df[~df['Intelligibility (%)'].str.contains('|'.join(exclude_intelligibility), case=False)]['Speaker']
        filtered_speakers = filtered_speakers.tolist()
        speakers = [speaker for speaker in speakers if speaker not in df['Speaker'].tolist() or speaker in filtered_speakers]
        # print(speakers)
        
        # distinguish Control speakers & Dysarthria speakers
        c_speakers = []  # Control speakers (no dysarthria)
        d_speakers = []  # Dysarthria speakers

        for speaker in speakers:
            if speaker.startswith('C'):
                c_speakers.append(speaker)
            elif speaker.startswith('M') or speaker.startswith('F'):
                d_speakers.append(speaker)
        

        for speaker in speakers:
            speaker_dir = os.path.join(audio_dir, speaker)
            audio_files = os.listdir(speaker_dir)

            # 4. Iterate over each audio file
            for audio_file in audio_files:
                parts = audio_file.split('_')
                if len(parts) != 4:
                    continue

                # Extract relevant parts from the filename
                _, block, word_id, _ = parts

                # Check if this word_id is in the wanted_word_ids or unknown_word_ids
                if word_id in wanted_word_ids:
                    label = list(word_to_id.keys())[list(word_to_id.values()).index(word_id)]  # Convert word_id to the actual word
                    file_path = os.path.join(speaker_dir, audio_file)
                    if block in ['B1', 'B3']:
                        self.data_set['training'].append({'label': label, 'file': file_path, 'speaker': speaker})
                    elif block == 'B2':
                        self.data_set['testing'].append({'label': label, 'file': file_path, 'speaker': speaker})
                    
                    all_words[label] = True

                elif word_id in unknown_word_ids:
                    label = list(word_to_id.keys())[list(word_to_id.values()).index(word_id)]  # Convert word_id to the actual word
                    file_path = os.path.join(speaker_dir, audio_file)
                    if block in ['B1', 'B3']:
                        unknown_set['training'].append({'label': UNKNOWN_WORD_LABEL, 'file': file_path, 'speaker': speaker})
                    elif block == 'B2':
                        unknown_set['testing'].append({'label': UNKNOWN_WORD_LABEL, 'file': file_path, 'speaker': speaker})

        # 5. If unknown words are to be included, add them to the respective data set
        if self.unknown:
            self.data_set['training'].extend(unknown_set['training'])
            self.data_set['testing'].extend(unknown_set['testing'])

        # Shuffle the data sets to ensure random ordering
        random.shuffle(self.data_set['training'])
        random.shuffle(self.data_set['testing'])

        # Prepare the words list and word-to-index mapping
        self.words_list = prepare_words_list(training_parameters['wanted_words'], self.silence, self.unknown)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
        
        # self.word_to_index = {word: index for index, word in enumerate(training_parameters['wanted_words'])}
        if self.unknown:
            self.word_to_index[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX
    