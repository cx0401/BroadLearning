#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import jieba
import os
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # data processing

# In[3]:


# 分词
def tokenizer(text): 
    return [word for word in jieba.lcut(text) if word not in stop_words]


# In[4]:


# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt',encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

stop_words = get_stop_words()  # 加载停用词表


# In[7]:


text = data.Field(sequential=True,
                  lower=True,
                  tokenize=tokenizer,
                  stop_words=stop_words)
label = data.Field(sequential=False)


# In[5]:


train, val = data.TabularDataset.splits(
    path='data/',
    skip_header=True,
    train='train.tsv',
    validation='validation.tsv',
    format='tsv',
    fields=[('index', None), ('label', label), ('text', text)],
)


# In[6]:


print(train[2].text)
print(train[5].__dict__.keys())


# In[7]:


#加载Google训练的词向量
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('data/myvector.vector', binary=False)


# In[8]:


cache = 'data/.vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)
vectors = Vectors(name='data/myvector.vector', cache=cache)
# 指定Vector缺失值的初始化方式，没有命中的token的初始化方式
#vectors.unk_init = nn.init.xavier_uniform_

text.build_vocab(train, val, vectors=vectors)#加入测试集的vertor


# In[9]:


#text.build_vocab(train, val, vectors=Vectors(name='data/myvector.vector'))#加入测试集的vertor
label.build_vocab(train, val)

embedding_dim = text.vocab.vectors.size()[-1]
vectors = text.vocab.vectors


# In[10]:


text.vocab.freqs.most_common(10)
print(text.vocab.vectors.shape)


# In[11]:


batch_size=128
train_iter, val_iter = data.Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.text),
            batch_sizes=(batch_size, len(val)), # 训练集设置batch_size,验证集整个集合用于测试
    )

vocab_size = len(text.vocab)
label_num = len(label.vocab)


# In[73]:


batch = next(iter(train_iter))
data = batch.text
print(batch.text.shape)
print(batch.text)


# # model

# In[84]:


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, num_layers):
        super(BiLSTM_Attention, self).__init__()
        # embedding之后的shape: torch.Size([200, 8, 300])
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = self.word_embeddings.from_pretrained(
            vectors, freeze=False)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=False,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            num_hiddens * 2, num_hiddens * 2))
        self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        self.decoder = nn.Linear(2*num_hiddens, 2)

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        # inputs的形状是(seq_len,batch_size)
        embeddings = self.word_embeddings(inputs)
        # 提取词特征，输出形状为(seq_len,batch_size,embedding_dim)
        # rnn.LSTM只返回最后一层的隐藏层在各时间步的隐藏状态。
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # outputs形状是(seq_len,batch_size, 2 * num_hiddens)
        x = outputs.permute(1, 0, 2)
        # x形状是(batch_size, seq_len, 2 * num_hiddens)
        
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
       # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
       # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
       # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
       # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束
        
        feat = torch.sum(scored_x, dim=1)
       # feat形状是(batch_size, 2 * num_hiddens)
        outs = self.decoder(feat)
       # out形状是(batch_size, 2)
        return outs


# In[85]:


embedding_dim, num_hiddens, num_layers = 100, 64, 1
net = BiLSTM_Attention(vocab_size, embedding_dim, num_hiddens, num_layers)
print(net)


# # train

# In[86]:


def evaluate_accuracy(data_iter,net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            X, y = batch.text, batch.label
         #   X = X.permute(1, 0)
            y.data.sub_(1)  #X转置 y为啥要减1
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


# In[87]:


def train(train_iter, test_iter, net, loss, optimizer, num_epochs):
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch_idx, batch in enumerate(train_iter):
            X, y = batch.text, batch.label
           # X = X.permute(1, 0)
            y.data.sub_(1)  #X转置 y为啥要减1
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               test_acc, time.time() - start))


# In[88]:


lr, num_epochs = 0.01, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, val_iter, net, loss, optimizer, num_epochs)


# In[ ]:




