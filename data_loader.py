
# coding: utf-8

# In[1]:


import torch


# In[96]:


# data_loader.py
import math
import numpy as np
import os,glob
from torch.utils.data import Dataset
import torch

# win_width=100
# kernel_size=7  #7*252=42**2=1764
# data_dir= 'C:\proj18797\preprocessed_data'

class data_loader(Dataset):
    def __init__(self, data_dir, win_width, kernel_size,overlap=True,phase='train'):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        filelist= glob.glob(os.path.join(data_dir,phase)+'/*') #namelist of CQT files and label files
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


# In[6]:


phase = 'train'


# In[7]:


data_dir = 'C:\proj18797\preprocessed_data'


# In[8]:


filelist= glob.glob(os.path.join(data_dir,phase)+'/*')


# In[9]:


filelist


# In[11]:


CQT_name=[f for f in filelist if (f[-7:-4]=='CQT')]
input=[]
nb_sample=[]


# In[12]:


CQT_name


# In[13]:


label_name=[f[:-7]+'label.npy' for f in CQT_name]


# In[27]:


label=[]   
win_width=10
kernel_size=3
for i in label_name:
    label.append(np.expand_dims(cut(np.load(i),win_width,kernel_size,overlap=False).transpose(0,2,1),axis=3))


# In[29]:


label[][][]


# In[33]:


haha=cut(np.load(i),win_width,kernel_size,overlap=False)


# In[34]:


haha.shape


# In[37]:


haha=haha.transpose(0,2,1)


# In[38]:


haha.shape


# In[45]:


trainset=data_loader(data_dir,10,4,phase='train')


# In[47]:


len(trainset)


# In[51]:


trainset[1]


# In[54]:


trainset.nb_sample


# In[69]:


dete=trainset.input[0]
dete=np.array(dete)
dete.shape


# In[71]:


trainset.__getitem__(1)


# In[79]:


N=trainset.__len__()


# In[80]:


N


# In[78]:


sum(trainset.nb_sample)


# In[92]:


trainset,valset,testset = LoadData_main(data_dir, win_width, kernel_size,overlap=True)


# In[95]:


valset.nb_sample

