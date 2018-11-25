# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
import random
import numpy as np
import pickle
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.utils.data as Data
class Transform():
    def pad_sequences(self,data_num,padding_token=0,padding_sentence_length=None):
        max_sentence_length=padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in data_num])
        for sentence in data_num:
            while len(sentence) > max_sentence_length:
                sentence.pop()
            else:
                sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
        return np.array(data_num)

    def rows0to0(self,embed_matrix):
        embeddings = np.delete(embed_matrix, 0, axis=0)
        zero = np.zeros(len(embed_matrix[0]), dtype=np.int32)
        embeddings = np.row_stack((zero, embeddings))
        return embeddings
def shuffle(data_num):
    for line in data_num:
        random.shuffle(line)
def load_data(args,sf=True):
    T=Transform()
    with open(args.embedding_path, 'rb') as f:
        embed_matrix = pickle.load(f)
    embed_matrix=T.rows0to0(embed_matrix)
    # print(embed_matrix[0])
    with open(args.train_path, 'rb') as f:
        data_num = pickle.load(f)
    if sf:
        shuffle(data_num)
    df=pd.read_csv(r'D:\localE\code\DaGuang\train_set_filter.csv')
    label=df['class']-1
    label=np.array(label)
    padding_data_num=T.pad_sequences(data_num,padding_sentence_length=args.max_text_len)
    print('切分训练数据、label和验证集数据、label......')
    train_data,val_data,train_label,val_label=train_test_split(padding_data_num,label,test_size=args.split_rate,
                                                                    random_state=1)
    val_data=torch.from_numpy(val_data).long()
    val_label=torch.from_numpy(val_label)
    val_torch_data=Data.TensorDataset(val_data,val_label)
    val_loader=Data.DataLoader(dataset=val_torch_data,batch_size=args.batch_size,shuffle=False)

    train_data=torch.from_numpy(train_data).long()
    train_label=torch.from_numpy(train_label)
    train_torch_data=Data.TensorDataset(train_data,train_label)
    train_loader=Data.DataLoader(dataset=train_torch_data,batch_size=args.batch_size
                                 ,shuffle=True)
    with open(args.test_path, 'rb') as f:
        test_num = pickle.load(f)
    test_num=T.pad_sequences(data_num=test_num,padding_sentence_length=args.max_text_len)
    test_num=torch.from_numpy(test_num).long()
    torch_test=Data.TensorDataset(test_num)
    test_loader=Data.DataLoader(dataset=torch_test,batch_size=args.batch_size,shuffle=False)
    return train_loader,val_loader,test_loader,embed_matrix,len(embed_matrix),len(embed_matrix[0])