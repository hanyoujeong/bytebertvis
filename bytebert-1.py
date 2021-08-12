#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax


# In[2]:


import torch
import torch.onnx as onnx
import torchvision.models as models


# In[8]:


import os
os.getcwd()


# In[7]:


os.chdir('C:\\Users\\MSI10\\PycharmProjects\\')


# In[9]:


from byteBERT.src.model import BERTLM, BERT
from byteBERT.src.trainer.optim_schedule import ScheduledOptim


# In[18]:


model = torch.load("C:\\Users\\MSI10\\PycharmProjects\\byteBERT\\bert.model.ep26")


# In[19]:


model


# In[1]:


from transformers import AutoTokenizer, AutoModel
from bertviz import model_view

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
inputs = tokenizer.encode("The cat sat on the mat", return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 
model_view(attention, tokens)


# In[2]:


from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show

model_type = 'bert'
model_version = 'bert-base-uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version)
# tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
show(model, model_type, tokenizer, sentence_a, sentence_b, display_mode='dark', layer=2, head=0)


# In[ ]:




