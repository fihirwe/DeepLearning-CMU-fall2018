#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
from matplotlib import pyplot as plt
import time
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from phoneme_list import PHONEME_MAP as phonemes
#import torchvision.transforms as transforms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE




# In[2]:


phonemes


# In[3]:



#getting data
trainx=np.array(np.load('wsj0_dev.npy', encoding='bytes'))
trainy=np.array(np.load('wsj0_dev_merged_labels.npy', encoding='bytes'))
valx=np.array(np.load('wsj0_dev.npy', encoding='bytes'))
valy=np.array(np.load('wsj0_dev_merged_labels.npy', encoding='bytes'))
TEST=np.array(np.load('wsj0_test.npy', encoding='bytes'))
#phonemes=np.array(np.load('phoneme_list.py', encoding='bytes'))
get_ipython().system(u'chmod 777 *')
print("done")


# In[ ]:





# In[4]:






# In[ ]:





# In[63]:


class dataAPI(Dataset):
    def __init__(self,x_data,y_data, transform=None):
        #self.transform=transforms.ToTensor()
        self.x=x_data
        self.y=y_data
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        ut=self.x[idx]
        lab=self.y[idx]
        return ut,lab
    
    
    
class datatestAPI(Dataset):
    def __init__(self,x_data, transform=None):
        #self.transform=transforms.ToTensor()
        self.x=x_data
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        ut=self.x[idx]
        return ut,lab


# In[106]:


class LanguageModel(nn.Module):
    """
        TODO: Define your model here
    """
    def __init__(self, ccount):
        super(LanguageModel, self).__init__()
        self.linear1 = nn.Linear(40,256)
        self.LSTM1=nn.LSTM(256, 256,bidirectional=True)
        self.LSTM2=nn.LSTM(512, 256,bidirectional=True)
        self.LSTM3=nn.LSTM(512, 256,bidirectional=True)
        self.projection = nn.Linear(512, ccount)

    def forward(self,features):
        input1 = features
        # Embedding layer
        input1 = self.linear1(input1)
        # lstm models
        states = []
        input1, state = self.LSTM1(input1)
        states.append(state)
        input1, state = self.LSTM2(input1)
        states.append(state)
        input1, state = self.LSTM3(input1)
        states.append(state)
        # Projection layer
        logits_out2 = self.projection(input1)
        return logits_out2


# In[ ]:





# In[107]:


def processBatch(ins):
    ins=np.array(ins)
    data1 = ins[:,0]
    labels1= ins[:,1]
    #print(data1)
    X_lengths = [len(utterance) for utterance in data1]
    X_lengths=np.array(pd.DataFrame(X_lengths).sort_values(0,ascending=False).reset_index(level=0))
    
    # create an empty matrix with paddings
    longest_Utter =X_lengths[0][1]
    size = data1.shape[0]
    inputs = np.zeros((size, longest_Utter,40))
     # copy over the actual sequences
    for i, x_len in enumerate(X_lengths[:,1]):
        sequence = data1[X_lengths[:,0][i]]
        inputs[i, 0:x_len] = sequence
    #Retreive their labels based on sorting indices
    labels=[labels1[index] for index in X_lengths[:,0]]
    labels=np.concatenate(labels, axis=0)
    inputs = torch.from_numpy(inputs).float()
    labels = torch.from_numpy(labels).float()
    return inputs,labels


def processBatchTest(ins):
    ins=np.array(ins)
    data1 = ins[:,0]
    #print(data1)
    X_lengths = [len(utterance) for utterance in data1]
    X_lengths=np.array(pd.DataFrame(X_lengths).sort_values(0,ascending=False).reset_index(level=0))
    
    # create an empty matrix with paddings
    longest_Utter =X_lengths[0][1]
    size = data1.shape[0]
    inputs = np.zeros((size, longest_Utter,40))
     # copy over the actual sequences
    for i, x_len in enumerate(X_lengths[:,1]):
        sequence = data1[X_lengths[:,0][i]]
        inputs[i, 0:x_len] = sequence
    inputs = torch.from_numpy(inputs).float()
    return inputs,labels


# In[108]:


# model trainer

class SpeechModelTrainer:
    def __init__(self, model, loader, max_epochs=3):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.epochs = 0
        self.max_epochs = max_epochs

        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)

    def train(self):
        
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0
        k=0;
        for batch_num, (inputs, targets) in enumerate(self.loader):
            #epoch_loss +=
            self.train_batch(inputs, targets)
        #epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs, self.max_epochs, epoch_loss))
        #self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """ 
            TODO: Define code for training a single batch of inputs
        
        """
        inputs = Variable(inputs).to(DEVICE)
        targets = Variable(targets).to(DEVICE)
        outputs = self.model(inputs) # 3D
        print(outputs.mean())



# In[109]:


BATCH_SIZE=20
NUM_EPOCHS=2


# In[112]:


model = LanguageModel(len(phonemes)+1)
trainn=dataAPI(trainx,trainy)
trainloader = torch.utils.data.DataLoader(trainn, batch_size=BATCH_SIZE,shuffle=True,collate_fn = processBatch)
trainer = SpeechModelTrainer(model=model, loader=trainloader, max_epochs=NUM_EPOCHS)


# In[ ]:


best_dist = 1e30  # set to super large value at first
for epoch in range(NUM_EPOCHS):
    trainer.train()


# In[ ]:


class ER_pred:

    def __init__(self):
        self.label_map = [' '] + phonemes
        self.decoder = CTCBeamDecoder(
            labels=self.label_map,
            blank_id=0
        )

    def __call__(self, prediction):
        return self.forward(prediction, target)

    def forward(self, prediction):
        logits = prediction[0]
        feature_lengths = prediction[1].int()
        logits = torch.transpose(logits, 0, 1)
        logits = logits.cpu()
        probs = F.softmax(logits, dim=2)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=feature_lengths)

        pos = 0
        ls = 0.
        for i in range(output.size(0)):
            pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])

def run_test(model, test_dataset):
    error_rate_op = ER_pred()
    loader = DataLoader(test_dataset, shuffle=False, batch_size=100,collate_fn = processBatchtest)
    predictions = []
    feature_lengths = []
    labels = []
    for data_batch, labels_batch in loader:
        if torch.cuda.is_available():
            data_batch = data_batch.cuda()
        predictions_batch, feature_lengths_batch = model(data_batch.unsqueeze(1))
        predictions.append(predictions_batch.to("cpu"))
    predictions = torch.cat(predictions, dim=1)
    feature_lengths = torch.cat(feature_lengths, dim=0)
    to_ret = error_rate_op((predictions, feature_lengths), labels.view(-1))
    return to_ret


# In[ ]:


test_set=dataAPI(TEST)
prediction=run_test(model, test_set)
np.savetxt('test.csv', prediction, delimiter=',')

