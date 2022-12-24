import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import nltk
import jieba

# 初始化类Lang
class Lang:
    def __init__(self):
        # 初始化容器，保存单词和对应的索引
        self.word2index = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}  # 单词 ——> 索引
        self.word2count = {}  # 单词 ——> 频率
        self.index2word = ["<PAD>", "<BOS>", "<EOS>"]  # 索引 ——> 单词
        self.n_words = 3  # 累计
        self.words = []

    def process_data(self, en_data, language="en"):
        # 对语句分词，将每一个单词加入到容器中
        for sentence in en_data:
            if language == "en":
                sentence_list = nltk.word_tokenize(sentence.lower())
            elif language == "ch":
                sentence_list = list(jieba.cut(sentence, cut_all=False))
            else:
                sentence_list = list(sentence)
            #sentence_list = sentence.split(' ') if language == " " else sentence
            self.words.append(sentence_list)
            for word in sentence_list:
                # 判断单词是否已经在容器中，如果不存在，则添加。同时，统计字符出现的频率
                if word not in self.word2index:
                    self.word2index[word] = self.n_words  # 单词对应的索引
                    self.word2count[word] = 1  # 单词频率
                    self.index2word.append(word)  # 索引对应的单词
                    self.n_words += 1  # 索引加1
                else:
                    self.word2count[word] += 1  # 如果单词已经存在，则频率加1
        print(language, " 个数 : ", self.n_words,"\n")

def batch_data_process(batch_datas):
    PAD_i,BOS_i,EOS_i = 0,1,2
    en_index, ch_index = [], []
    en_len, ch_len,ch_lennew = [], [],[]

    for en,_, ch,_ in batch_datas:
        en_index.append(en)
        ch_index.append(ch)
        en_len.append(len(en))
        ch_len.append(len(ch))
        ch_lennew.append(len(ch)+2)

    max_en_len = max(en_len)
    max_ch_len = max(ch_len)

    en_index = [i + [PAD_i] * (max_en_len - len(i)) for i in en_index]
    ch_index = [[BOS_i] + i + [EOS_i] + [PAD_i] * (max_ch_len - len(i)) for i in ch_index]

    return en_index, en_len, ch_index, ch_lennew

class MyDataset(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file, delimiter='\t', header=None)
        self.en_data = list(df[0])
        self.en_laug = Lang()
        self.en_laug.process_data(self.en_data, "en")

        self.ch_data = list(df[1])
        self.ch_laug = Lang()
        self.ch_laug.process_data(self.ch_data, "ch")

    def __getitem__(self, index):
        en = self.en_laug.words[index]
        ch = self.ch_laug.words[index]

        en_index = [self.en_laug.word2index[i] for i in en]
        ch_index = [self.ch_laug.word2index[i] for i in ch]

        return en_index, len(en_index), ch_index, len(ch_index)

    def __len__(self):
        assert len(self.en_data) == len(self.ch_data)
        return len(self.ch_data)



if __name__ == '__main__':
    filePath = "D:/02dataset/cmn_zhsim.txt"
    mydataset = MyDataset(filePath)



    print("")
