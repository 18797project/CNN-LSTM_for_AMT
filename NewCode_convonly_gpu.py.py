#!/usr/bin/env python
# coding: utf-8

# https://musicinformationretrieval.com/stft.html
# https://music.stackexchange.com/questions/34402/understanding-midi-files
# 

# In[65]:


#%matplotlib inline
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import re

from multiprocessing import Pool
from functools import partial

import IPython.display as ipd
RangeMIDInotes=[21,108]
MIDInotes=[21,108]
sr=44100
bins_per_octave=36
n_octave=7
n_bins=n_octave*bins_per_octave
import torch
from torch.utils import data
import os,glob
import os.path as osp
from torch import nn
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from importlib import import_module
import time

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
print(sr)

#import midi_to_mat as mm
import pretty_midi
import torch.cuda

if torch.cuda.is_available():
    print('cuda available')
else:
    print('cuda not available')

# In[54]:


n_workers=10
start_lr=0.01
weight_decay=1e-4
nb_epochs=50
save_freq=30
win_width=32  
batch_size=32
kernel_size=7


# In[44]:



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
        save_path = osp.join(osp.dirname(osp.realpath(data_path)), 'Newpreprocessed')
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
    print('filename list obtained')
    n_bins=n_octave*bins_per_octave


    # training set processing
    print('n_workers:',n_worker)
    if n_workers==0 or 1:
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
    else:
        pool = Pool(processes=n_worker)
        partial_processing = partial(processing, n_bins=n_bins, output=output_train, sr=sr, bins_per_octave=bins_per_octave,
                    RangeMIDInotes=RangeMIDInotes)
        _ = pool.map(partial_processing, train_name)
        pool.close()
        pool.join()

        pool = Pool(processes=n_worker)
        partial_processing = partial(processing, n_bins=n_bins,output=output_val, sr=sr,
                                     bins_per_octave=bins_per_octave,
                                     RangeMIDInotes=RangeMIDInotes)
        _ = pool.map(partial_processing, val_name)
        pool.close()
        pool.join()

        pool = Pool(processes=n_worker)
        partial_processing = partial(processing, n_bins=n_bins, output=output_test, sr=sr,
                                     bins_per_octave=bins_per_octave,
                                     RangeMIDInotes=RangeMIDInotes)
        _ = pool.map(partial_processing, test_name)
        pool.close()
        pool.join()

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


# In[17]:


def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


# In[18]:


def analyse_audio(audio_file, midi_file):
    x, _ = librosa.load(audio_file, sr=sr)
    print("Music file length=%s, sampling_rate=%s" % (x.shape[0],sr))
    plt.figure(figsize=(14, 5))
    plt.title('Music Sample Waveplot')
    librosa.display.waveplot(x, sr=sr)
    x_stft_spectrum = lb.stft(x, n_fft=1024,hop_length=512,center=True, dtype=np.complex64)
    x_stft = librosa.amplitude_to_db(abs(x_stft_spectrum))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(lb.amplitude_to_db(x_stft, ref=np.max), sr=sr, fmin=lb.note_to_hz('A0'), x_axis='time', y_axis='linear',cmap='coolwarm')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.figure(figsize=(14, 5))
    x_cqt = np.abs(librosa.cqt(x, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins, fmin=lb.note_to_hz('A0')))
    librosa.display.specshow(librosa.amplitude_to_db(x_cqt, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note',cmap='coolwarm')
    print("CQT Matrix shape", x_cqt.shape)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    n_frames=x_cqt.shape[1]
    
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    plt.figure(figsize=(12, 4))
    plot_piano_roll(midi_data, 24, 84)
    print('There are {} time signature changes'.format(len(midi_data.time_signature_changes)))
    print('There are {} instruments'.format(len(midi_data.instruments)))
    print('Instrument 1 has {} notes'.format(len(midi_data.instruments[0].notes)))
    pianoRoll = midi_data.instruments[0].get_piano_roll(fs=n_frames * 44100. / len(x))
    midi_mat = (pianoRoll[MIDInotes[0]:MIDInotes[1] + 1, :n_frames] > 0)
    print("MIDI Matrix shape", midi_mat.shape)
    plt.figure()
    
    librosa.display.specshow(midi_mat, sr=sr, bins_per_octave=12, fmin=lb.note_to_hz('A0'), x_axis='time', y_axis='cqt_note')
    n_pitch_frame=np.sum(midi_mat, axis=1)
    print(n_pitch_frame)
    plt.bar(range(MIDInotes[0],MIDInotes[1]+1),n_pitch_frame/np.sum(n_pitch_frame).astype(np.float))
    plt.xticks(range(MIDInotes[0],MIDInotes[1]+1,12), lb.midi_to_note(range(MIDInotes[0], MIDInotes[1]+1,12)))
    plt.xlabel('Midi note')
    plt.ylabel('Note probability')
    
    
    
    


# In[55]:


#from here
class data_loader(Dataset):
    def __init__(self, data_dir, win_width, kernel_size, overlap=True, phase='train'):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        filelist= glob.glob(os.path.join(data_dir,phase)+'/*') #namelist of CQT files and label files
        print(phase+'filelist:')
        print(filelist)
        CQT_name=[f for f in filelist if (f[-7:-4]=='CQT')]
        self.input=[]
        self.nb_sample=[]

        for i in CQT_name:
            self.input.append(np.expand_dims(cut(np.load(i),win_width,kernel_size,overlap=overlap),axis=1))   # 64s,no need to paralellize, I/O is limited ,246s with 5 workers
            self.nb_sample.append(self.input[-1].shape[0])
        print(phase+'data loaded')
        label_name=[f[:-7]+'label.npy' for f in CQT_name]
        self.label=[]
        for i in label_name:
            self.label.append(np.expand_dims(cut(np.load(i),win_width,kernel_size,overlap=False).transpose(0,2,1),axis=3))
        print(phase+'label loaded')

    def __getitem__(self,idx):
        #if self.phase!='test':  no real test in our senerios
        nb_list, sub_nb = index(idx,self.nb_sample)
        return torch.from_numpy(self.input[nb_list][sub_nb].astype(np.float32)),torch.from_numpy(self.label[nb_list][sub_nb].astype(np.float32))   #(1,106,252)/(1,100,88)

    def __len__(self):
        return sum(self.nb_sample)

def cut(matrix,win_width,kernel_size,overlap=True,axis=0):  #window cut module
# cut the tensor along the first axis by the win_width with a single frame hop
    #matrix=np.load(matrix)
    l=matrix.shape[0]
    cut_matrix=[]
    nb_win=math.floor(l/win_width)  #integer division=floor
    if not overlap:
        for i in range(nb_win):
            cut_matrix.append(matrix[i*win_width:(i+1)*win_width,:])
    else:
        w=matrix.shape[1]
        matrix_1=np.concatenate([np.zeros([math.floor(kernel_size/2),w]),matrix,np.zeros([math.floor(kernel_size/2),w])],axis=0)  #padding
        cut_matrix = []
        for i in range(nb_win):
            cut_matrix.append(matrix_1[i * win_width:(i + 1) * win_width+kernel_size-1,:])    #0-104,100-204,...
    cut_matrix = np.asarray(cut_matrix)
    return cut_matrix

def index(idx,nb_sample):
    l=len(nb_sample)
    accum_nb =0
    nb_list=0
    sub_nb=0
    for i in range(l):
        accum_nb+=nb_sample[i]
        if idx < accum_nb:
            nb_list, sub_nb= i, idx+nb_sample[i]-accum_nb
            break
    return nb_list,sub_nb


def LoadData_main(data_dir, win_width, kernel_size,overlap=True):
    trainset=data_loader(data_dir,win_width,kernel_size,overlap=True,phase='train');
    valset=data_loader(data_dir,win_width,kernel_size,overlap=True,phase='val');
    testset=data_loader(data_dir,win_width,kernel_size,overlap=True,phase='test');
    print('all data and label loaded!')
    return trainset,valset,testset


# In[56]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), return_indices=False)

        self.conv1= nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(5,25), stride=1),
            nn.BatchNorm2d(50),
            nn.ELU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(3,5), stride=1),
            nn.BatchNorm2d(50),
            nn.ELU(inplace=True))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=38400,  out_features=9600),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=9600,  out_features=4800),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.fc3 = nn.Linear(in_features=4800,  out_features=32*88)
        self.conv3=nn.Sequential(
            nn.Conv2d(50, 1000, kernel_size=(1,24)),   #kernel_size !!!!! conv_mode,padding=0 by default
            #nn.ReLU(inplace=True),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.conv4 = nn.Sequential(
            nn.Conv2d(1000, 500, kernel_size=1),  # kernel_size !!!!! conv_mode,padding=0 by default
            nn.ELU(inplace=True),
        #nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.conv5=nn.Sequential(
            nn.Conv2d(500, 88, kernel_size=1))#,
            #nn.Sigmoid()


    def forward(self, x):
#         print("1.",x.shape)
        x = self.conv1(x)
#         print("2.",x.shape)
        x = self.maxpool(x)
#         print("3.",x.shape)
        x = self.conv2(x)  
#         print("4.",x.shape)
        x = self.maxpool(x)
#         print("5.",x.shape)
#         x = x.view(-1, 38400)
# #         print("6.",x.shape)
#         x = self.fc1(x)
# #         print("7.",x.shape)
#         x = self.fc2(x)
# #         print("8.",x.shape)
#         x = self.fc3(x)
# #         print("9.",x.shape)
#         x = x.view(-1, 88, 32, 1)
#         print("10.",x.shape)

        x = self.conv3(x)# (8L, 1000L, 32L, 1L)
        x = self.conv4(x)#(8L, 500L, 32L, 1L)
        x = self.conv5(x)#(8L, 88L, 32L, 1L)
        return x


class Loss(nn.Module):
    def __init__(self, num_hard=0):
        super(Loss, self).__init__()

        self.classify_loss = nn.BCEWithLogitsLoss() 
    def forward(self, output, labels, train=True):
        loss = self.classify_loss(
           output,labels)
        pos = (torch.sigmoid(output) >= 0.5).type(torch.FloatTensor).cuda()
        pos_recall=labels.sum()
        pos_precision=pos.sum()
        TP=(pos*labels).sum()
        return [loss, TP.item(), pos_precision.item(), pos_recall.item()] #F-score must be computed by whole epoch

def get_model():
#     print("Fetching Model")
    net = Net()
    loss = Loss()
    return net, loss


# In[57]:


#paths and dir
data_path='C:\\Users\\Li_Sh\\18797proj\\trymycode\\NewStart'
parent_dir=osp.dirname(osp.realpath(data_path))
print('parent',parent_dir)
save_path0=osp.join(parent_dir, 'Newpreprocessed')
if not osp.exists(save_path0): 
    preprocessing(data_path)
data_dir=save_path0 #cqt label
save_dir=osp.join(osp.dirname(osp.realpath(data_path)), 'Newsavemodel') #trained ckpt
if not osp.exists(save_dir): 
    os.makedirs(save_dir) 


# In[58]:


net, loss= get_model()
net = DataParallel(net) 
net = net.cuda()
loss = loss.cuda()


# In[59]:


dataset=data_loader(data_dir,win_width, kernel_size,overlap=True,phase='train')
train_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = n_workers,
        pin_memory=True)   #train/val pin_memory=True, test pin_memory=False

dataset=data_loader(data_dir, win_width, kernel_size,overlap=True,phase='val')
val_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_workers,
        pin_memory=True)   #train/val pin_memory=True, test pin_memory=False


# In[60]:


optimizer = optim.SGD(
        net.parameters(),
        start_lr,
        momentum=0.9,
        weight_decay = weight_decay)


# In[61]:


def get_lr(epoch,nb_epochs,start_lr):
    if epoch <= nb_epochs * 0.5:
        lr = start_lr
    elif epoch <= nb_epochs * 0.8:
        lr = 0.1 * start_lr
    else:
        lr = 0.01 * start_lr
    return lr


# In[70]:


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,nb_epochs,start_lr):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch,nb_epochs,start_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target) in enumerate(data_loader):
        data = Variable(data).cuda()
        target = Variable(target).cuda()

        output = net(data)
        #print("Output",output.shape,"Target",target.shape)
        loss_output = loss(output,target)#(8L, 88L, 32L, 1L)/(8L, 1L, 32L, 88L)
        
#         print (loss_output[0])
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict
            },
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)

    print('Epoch %03d (lr %.5f),time %3.2f' % (epoch, lr,end_time - start_time))  # Framewise and notewise Accuracy precision,recall,F-score
    TP=np.sum(metrics[:, 1])
    Precision=TP/np.sum(metrics[:, 2])
    Recall=TP/np.sum(metrics[:, 3])
    Fscore=2*Precision*Recall/(Precision+Recall)
    print('Train:　loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics[:,0]),Precision,Recall,Fscore))
    print
    return (np.mean(metrics[:,0]),Precision,Recall,Fscore)


# In[71]:


def validate(data_loader, net, loss):
    start_time = time.time()

    net.eval()

    metrics = []
    
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data).cuda()
        target = Variable(target).cuda()

        output = net(data)
        loss_output = loss(output, target, train=False)

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    TP=np.sum(metrics[:, 1])
    Precision=TP/np.sum(metrics[:, 2])
    Recall=TP/np.sum(metrics[:, 3])
    Fscore=2*Precision*Recall/(Precision+Recall)
    print('Validation: Loss %2.4f,Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics[:,0]),Precision,Recall,Fscore))
    print
    print
    return (np.mean(metrics[:,0]),Precision,Recall,Fscore)


# In[64]:


start_epoch=0
showmetric_train=[]
showmetric_val=[]

for epoch in range(start_epoch,nb_epochs):
    temp_train=train(train_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,nb_epochs,start_lr)
    temp_val=validate(val_loader, net, loss)
    showmetric_train.append(temp_train)
    showmetric_val.append(temp_val)
    
show_train=np.array(showmetric_train)
show_val=np.array(showmetric_val)

#save the metrics 
np.save(osp.join(save_dir, 'metric_train.npy') , showmetric_train)
np.save(osp.join(save_dir, 'metric_val.npy') , showmetric_val)

if (epoch+1) % save_freq == 0:
    state_dict = net.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'save_dir': save_dir,
        'state_dict': state_dict
        },
        os.path.join(save_dir, '%03d.ckpt' % epoch))

#plot metric
plt.plot(range(1,nb_epochs+1),show_train[:,0])
plt.xlabel('epoch number')
plt.ylabel('loss - train')
plt.show()

plt.plot(range(1,nb_epochs+1),show_train[:,1])
plt.xlabel('epoch number')
plt.ylabel('precision - train')
plt.show()
plt.savefig('precision - train.jpg')

plt.plot(range(1,nb_epochs+1),show_train[:,2])
plt.xlabel('epoch number')
plt.ylabel('recall - train')
plt.show()

plt.plot(range(1,nb_epochs+1),show_train[:,3])
plt.xlabel('epoch number')
plt.ylabel('F-score - train')
plt.show()

plt.plot(range(1,nb_epochs+1),show_val[:,0])
plt.xlabel('epoch number')
plt.ylabel('loss - valid')
plt.show()
        
plt.plot(range(1,nb_epochs+1),show_val[:,1])
plt.xlabel('epoch number')
plt.ylabel('precision - valid')
plt.show()

plt.plot(range(1,nb_epochs+1),show_val[:,2])
plt.xlabel('epoch number')
plt.ylabel('recall - valid')
plt.show()

plt.plot(range(1,nb_epochs+1),show_val[:,3])
plt.xlabel('epoch number')
plt.ylabel('F-score - valid')
plt.show()

    


# In[66]:


# #demo: load test data
# dataset=data_loader(data_dir, win_width, kernel_size,overlap=True,phase='test')
# test_loader = DataLoader(
#         dataset,
#         batch_size = batch_size,
#         shuffle = False,
#         num_workers = n_workers,
#         pin_memory=True)   #train/val pin_memory=True, test pin_memory=False


# In[67]:


# #demo: 1. use trained network to transcribe the testset music
# correct = 0
# total = 0
# outputlist=torch.tensor([])
# for i, (data, target) in enumerate(test_loader):
#         data = Variable(data, volatile=True)
#         target = Variable(target, volatile=True)
#         output = net(data)
#         output = (output > 0.5).type(torch.FloatTensor)

#         outputlist=torch.cat((outputlist,output),0)
        
        
#         target1 = torch.flatten(target)
#         output1 = torch.flatten(output)
#         total += target1.size(0)                    # Increment the total count
#         correct += (output1 == target1).sum()     # Increment the correct count

# print("Correct Percentage:",correct.item()/total)


# In[68]:


# # for demo: convert roll matrix to midi music
# from midiutil.MidiFile import MIDIFile #need to install python lib nidiutil

# def write_midi_roll_to_midi(x, out_path):
#     """Write out midi_roll to midi file. 
    
#     Args: 
#       x: (n_time, n_pitch), midi roll. 
#       out_path: string, path to write out the midi. 
#     """
#     step_sec = float(512)/44100
    
#     def _get_bgn_fin_pairs(ary):
#         pairs = []
#         bgn_fr, fin_fr = -1, -1
#         for i2 in range(1, len(ary)):
#             if ary[i2-1] == 0 and ary[i2] == 0:
#                 pass
#             elif ary[i2-1] == 0 and ary[i2] == 1:
#                 bgn_fr = i2
#             elif ary[i2-1] == 1 and ary[i2] == 0:
#                 fin_fr = i2
#                 if fin_fr > bgn_fr:
#                     pairs.append((bgn_fr, fin_fr))
#             elif ary[i2-1] == 1 and ary[i2] == 1:
#                 pass
#             else:
#                 raise Exception("Input must be binary matrix!")
            
#         return pairs
    
#     # Get (pitch, bgn_frame, fin_frame) triple. 
#     triples = []
#     (n_time, n_pitch) = x.shape
#     for i1 in range(n_pitch):
#         ary = x[:, i1]
#         pairs_per_pitch = _get_bgn_fin_pairs(ary)
#         if pairs_per_pitch:
#             triples_per_pitch = [(i1,) + pair for pair in pairs_per_pitch]
#             triples += triples_per_pitch
    
#     # Sort by begin frame. 
#     triples = sorted(triples, key=lambda x: x[1])
    
#     # Write out midi. 
#     MyMIDI = MIDIFile(1)    # Create the MIDIFile Object with 1 track
#     track = 0   
#     time = 0
#     tempo = 120
#     beat_per_sec = 60. / float(tempo)
#     MyMIDI.addTrackName(track, time, "Sample Track")  # Add track name 
#     MyMIDI.addTempo(track, time, tempo)   # Add track tempo
    
#     for triple in triples:
#         (midi_pitch, bgn_fr, fin_fr) = triple
#         print("pitch, bgn, end for each note",triple)
#         bgn_beat = bgn_fr * step_sec / float(beat_per_sec)
#         fin_beat = fin_fr * step_sec / float(beat_per_sec)
#         dur_beat = fin_beat - bgn_beat
#         #print("note begin:",bgn_beat)
#         #print("duration (beat) of this note:",dur_beat)
#         MyMIDI.addNote(track=0,     # The track to which the note is added.
#                     channel=0,   # the MIDI channel to assign to the note. [Integer, 0-15]
#                     pitch=midi_pitch,    # the MIDI pitch number [Integer, 0-127].
#                     time=bgn_beat,      # the time (in beats) at which the note sounds [Float].
#                     duration=dur_beat,  # the duration of the note (in beats) [Float].
#                     volume=100)  # the volume (velocity) of the note. [Integer, 0-127].
#     out_file = open(out_path, 'wb')
#     MyMIDI.writeFile(out_file)
#     out_file.close()
#     print("successfully wrote the midi file!")




# In[69]:


# #for demo: convert nn output to roll matrix and wirte midi accordingly, show the resulting roll mat
# import os.path as osp
# import os
# import matplotlib.pyplot as plt


# print(outputlist.shape)
# Alloutput=outputlist.permute(1,2,3,0)
# print(Alloutput.shape)

# Alloutput=Alloutput.contiguous()
# Alloutput=Alloutput.view(88,-1)#get the midiroll matrix
# Alloutput=Alloutput.permute(1,0) #transpose the midiroll mat to fit in the write midi function
# Alloutput=Alloutput.numpy()
# print(Alloutput.shape) 

# pitch_bgn=21
# roll_mat = np.zeros((Alloutput.shape[0]+1, 128))
# roll_mat[1:, pitch_bgn : pitch_bgn + 88] = Alloutput
# print("roll_mat shape:",roll_mat.shape)

# roll_mat_b=(roll_mat==1)
# print(roll_mat_b.shape)

# if not osp.exists(testoutput_dir): 
#     os.makedirs(testoutput_dir) 
# outaudio_path=osp.join(testoutput_dir, 'outaudio.mid') 
# write_midi_roll_to_midi(roll_mat_b, outaudio_path) #write roll mat to midi

# #save roll testoutput_dir='C:\\Users\\Li_Sh\\18797proj\\testoutput'
# np.save(osp.join(testoutput_dir, 'rollmat.npy') , roll_mat_b)

# #plot the roll matrix
# fig = plt.figure()
# plt.matshow(roll_mat.T,origin='lower', aspect='auto')
# plt.xlabel('time frame')
# plt.ylabel('note frame')
# plt.title('roll matrix after transcription', fontsize=10)
# plt.show()

# #reread the midi file and plot its roll 
# x_midi = pretty_midi.PrettyMIDI(outaudio_path)
# plt.figure(figsize=[12,4])
# start_pitch=0
# end_pitch=127
# librosa.display.specshow(x_midi.get_piano_roll(100)[start_pitch:end_pitch],
#                              hop_length=1, sr=100, x_axis='time', y_axis='cqt_note',
#                              fmin=pretty_midi.note_number_to_hz(start_pitch))
# plt.title('roll matrix after transcription', fontsize=10)
# plt.show()


# In[20]:


# import numpy as np
# patht='C:\\Users\\Li_Sh\\Downloads\\metric_train.npy'
# path='C:\\Users\\Li_Sh\\Downloads\\metric_val.npy'
# ht=np.load(patht)
# h=np.load(path)
# # print(h)


# In[24]:


# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

# n=h.shape[0]
# ax=plt.figure().gca()
# plt.plot(range(1,n+1),ht[:,0])
# plt.plot(range(1,n+1),h[:,0])
# plt.legend(['train loss','valid loss'], loc='upper right')
# plt.xlabel('epoch number')
# plt.ylabel('loss')
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.show()
# #plt.savefig(osp.join(save_dir,'loss - train.jpg'))


# In[26]:


# n=h.shape[0]
# ax=plt.figure().gca()
# plt.plot(range(1,n+1),ht[:,1])
# plt.plot(range(1,n+1),h[:,1])
# plt.legend(['train precision','valid precision'], loc='upper left')
# plt.xlabel('epoch number')
# plt.ylabel('precision')
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.show()

