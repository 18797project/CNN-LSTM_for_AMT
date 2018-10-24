
# coding: utf-8

# In[3]:


#preprocessing.py

import re,os
import os.path as osp
import glob
import numpy as np
import random
import librosa as lb
import pretty_midi
import zipfile
import sys
from multiprocessing import Pool
from functools import partial

#import h5py
eps=sys.float_info.epsilon
pretty_midi.pretty_midi.MAX_TICK = 1e10
## Paramater setting ##
RangeMIDInotes=[21,108]
sr=44100.
bins_per_octave=36
n_octave=7
data_path='C:\proj18797\data'
#test_list=['ENSTDkAm','ENSTDkCl']   #real piano
val_rate=1./7
n_workers=1

def preprocessing(data_path, sr=sr, bins_per_octave=bins_per_octave, n_octave=n_octave,#win_width=3,
                  RangeMIDInotes=RangeMIDInotes, save_path=None,n_worker=n_workers,delete=True):
    # Convert the raw data(wav/mid) into input/output data from the train/test directories

    # data_path = None or any other with train/test dirs inside

    # output_path: Path to save the processed data with format hdf5
    # None=only preprocessed data,no output file;
    # '' generate an output directory in current directory(without preprocessed data)

    # output_name: name the hf file, data.h5 by default
    # sr:Raw audio sampling rate
    # RangeMIDInotes: by default for the 88 key piano

    # Default data path
    if save_path == None:
        save_path = osp.join(osp.dirname(osp.realpath(data_path)), 'preprocessed_data')
        if not osp.exists(save_path):
            os.makedirs(save_path)
    output_train = osp.join(save_path, 'train')
    output_val = osp.join(save_path, 'val')
    output_test = osp.join(save_path, 'test')

    if not osp.exists(output_train):
        os.makedirs(output_train)

    if not osp.exists(output_val):
        os.makedirs(output_val)

    if not osp.exists(output_test):
        os.makedirs(output_test)

    # train/test inside
    train_list = glob.glob(osp.join(data_path, 'train') + '/*')
    val_list = glob.glob(osp.join(data_path, 'val') + '/*')
    test_list = glob.glob(osp.join(data_path, 'test') + '/*')

    train_name = []
    val_name= []
    test_name = []
    for i in train_list:
        train_name.append(i[:-3])
    for i in val_list:
        val_name.append(i[:-3])
    for i in test_list:
        test_name.append(i[:-3])
    train_name = list(set(train_name)) #remove repeated name
    val_name = list(set(val_name))
    test_name = list(set(test_name))

    n_bins=n_octave*bins_per_octave


    # training set processing
    for i in train_name:
        processing(i, n_bins, output_train, sr=sr, bins_per_octave=bins_per_octave,
                RangeMIDInotes=RangeMIDInotes)
    for i in val_name:
        processing(i, n_bins, output_val, sr=sr, bins_per_octave=bins_per_octave,
                RangeMIDInotes=RangeMIDInotes)
    # testing set processing# testing set processing
    for i in test_name:
        processing(i, n_bins, output_test, sr=sr, bins_per_octave=bins_per_octave,
                RangeMIDInotes=RangeMIDInotes)

    print('Data preprocessing completed')
    if delete:
        os.system("rm -r %s" % (data_path))

def processing(data_path,n_bins,output,sr=sr, bins_per_octave=bins_per_octave,
                  RangeMIDInotes=RangeMIDInotes):
    save_path=osp.join(output,data_path.split('\\')[-1][:-1])
    # input:  CQT spectrum form raw audio
    audio_path_train = data_path + 'wav'
    print(audio_path_train)
    x, sr = lb.load(audio_path_train, sr=sr)
    CQT_spectrum = lb.cqt(x, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins,
                                fmin=lb.note_to_hz('A0'))
    CQT = np.transpose(np.abs(CQT_spectrum))

    # Ground-truth: convert midi to pianoroll
    midi_path_train = data_path + 'mid'
    Ground_truth_mat=midi2mat(midi_path_train, len(x), CQT.shape[0], sr, RangeMIDInotes=RangeMIDInotes)
    midi_train = np.transpose(Ground_truth_mat)

    if midi_train.shape[0]<CQT.shape[0]:
    #midi length<CQT length, cut CQT 
        CQT=CQT[:midi_train.shape[0],:]
    np.save(save_path + '_CQT.npy', CQT)
    np.save(save_path + '_label.npy', midi_train)
    print("Preprocessing of file %s completed..." % (data_path[:-1]))

def midi2mat(midi_path_train, length, CQT_len, sr, RangeMIDInotes=RangeMIDInotes):
    midi_data = pretty_midi.PrettyMIDI(midi_path_train)
    pianoRoll = midi_data.instruments[0].get_piano_roll(fs=CQT_len * sr/length)
    Ground_truth_mat = (pianoRoll[RangeMIDInotes[0]:RangeMIDInotes[1] + 1, :CQT_len] > 0) #bool mat
    return Ground_truth_mat

        
        
        


# In[10]:



save_path = osp.join(osp.dirname(osp.realpath(data_path)), 'preprocessed_data')
if not osp.exists(save_path):
    os.makedirs(save_path)
output_train = osp.join(save_path, 'train')
output_val = osp.join(save_path, 'val')
output_test = osp.join(save_path, 'test')


# In[11]:


output_train 


# In[12]:


if not osp.exists(output_train):
    os.makedirs(output_train)

if not osp.exists(output_val):
    os.makedirs(output_val)

if not osp.exists(output_test):
    os.makedirs(output_test)


# In[13]:


train_list = glob.glob(osp.join(data_path, 'train') + '/*')
val_list = glob.glob(osp.join(data_path, 'val') + '/*')
test_list = glob.glob(osp.join(data_path, 'test') + '/*')

train_name = []
val_name= []
test_name = []


# In[14]:


train_list


# In[15]:


for i in train_list:
    train_name.append(i[:-3])
for i in val_list:
    val_name.append(i[:-3])
for i in test_list:
    test_name.append(i[:-3])


# In[16]:


train_name


# In[17]:


train_name = list(set(train_name))
val_name = list(set(val_name))
test_name = list(set(test_name))


# In[18]:


train_name


# In[19]:


n_bins=n_octave*bins_per_octave


# In[20]:


n_bins


# In[22]:


data_path=train_name[1]


# In[23]:


data_path


# In[25]:


output=output_train


# In[26]:


output


# In[34]:


save_path=osp.join(output,data_path.split('\\')[-1][:-1])


# In[35]:


save_path


# In[37]:


data_path.split('\\')[-1][:-1]


# In[32]:


data_path.split('\\')


# In[38]:


audio_path_train = data_path + 'wav'


# In[39]:


audio_path_train


# In[40]:


x, sr = lb.load(audio_path_train, sr=sr)


# In[41]:


x


# In[42]:


sr


# In[44]:


CQT_spectrum = lb.cqt(x, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins,
                                fmin=lb.note_to_hz('A0'))


# In[45]:


CQT_spectrum


# In[47]:


CQT_spectrum.shape


# In[48]:


x.shape


# In[49]:


CQT = np.transpose(np.abs(CQT_spectrum))


# In[54]:


CQT.shape


# In[56]:


midi_path_train = data_path + 'mid'
midi_path_train


# In[61]:


midi_data = pretty_midi.PrettyMIDI(midi_path_train)


# In[62]:


midi_data


# In[65]:


CQT_len=CQT.shape[0]
CQT_len


# In[67]:


length=len(x)
length


# In[76]:


pianoRoll = midi_data.instruments[0].get_piano_roll(fs=100)
pianoRoll.shape


# In[69]:


midi_data.instruments[0]


# In[71]:


fs=CQT_len * sr/length
fs


# In[77]:


Ground_truth_mat = (pianoRoll[RangeMIDInotes[0]:RangeMIDInotes[1] + 1, :CQT_len] > 0)


# In[78]:


Ground_truth_mat


# In[80]:


Ground_truth_mat=midi2mat(midi_path_train, len(x), CQT.shape[0], sr, RangeMIDInotes=RangeMIDInotes)
midi_train = np.transpose(Ground_truth_mat)

if midi_train.shape[0]<CQT.shape[0]:
#midi length<CQT length, cut CQT 
    CQT=CQT[:midi_train.shape[0],:]


# In[81]:


CQT.shape


# In[82]:


np.save(save_path + '_CQT.npy', CQT)
np.save(save_path + '_label.npy', midi_train)


# In[90]:


preprocessing(data_path)


# In[91]:


train_name


# In[4]:


preprocessing(data_path)


# In[99]:


val_list = glob.glob(osp.join(data_path, 'val') + '/*')


# In[100]:


val_list


# In[101]:


for i in val_list:
    val_name.append(i[:-3])


# In[102]:


val_name


# In[103]:


val_name = list(set(val_name))


# In[104]:


val_name

