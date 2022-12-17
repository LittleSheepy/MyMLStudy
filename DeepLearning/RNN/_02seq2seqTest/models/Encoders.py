import torch
from torch import nn

# LSTM batch_first
class Encoder_LSTM_Batch0(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len):   # 100 150 78
        super(Encoder_LSTM_Batch0, self).__init__()
        self.embedding = nn.Embedding(en_corpus_len, encoder_embedding_num)     # 78 100
        self.lstm = nn.LSTM(encoder_embedding_num, encoder_hidden_num, batch_first=True)
    def forward(self, en_index):    # en_index:torch.Size([10, 7]) [[3,4,5,0,0,0,0],[]]
        en_embedding = self.embedding(en_index)     # torch.Size([10, 7, 50])
        _, encoder_hidden = self.lstm(en_embedding)
        return encoder_hidden   # encoder_hidden (torch.Size([1, 10, 100]),torch.Size([1, 10, 100]))

# GRU
class Encoder_GRU(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len, num_layers=1):
        super(Encoder_GRU, self).__init__()
        self.embedding = nn.Embedding(en_corpus_len, encoder_embedding_num)     # 78 100
        self.gru = nn.GRU(encoder_embedding_num, encoder_hidden_num, num_layers=num_layers)
    # tensor([ 3,  3,  6,  9,  3, 16, 16, 21, 23,  3], device='cuda:0')  size=10
    def forward(self, src):
        embedded = self.embedding(src)  # torch.Size([10, 50])
        embedded = embedded.unsqueeze(0)  # torch.Size([1, 10, 50])
        outputs, hidden = self.gru(embedded)
        return outputs, hidden  # outputs: torch.Size([1, 10, 100]) ***** hidden: torch.Size([1, 10, 100])

# seq2seq的编码器 test4
class Encoder_GRU_Batch0(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len, num_layers=1, dropout=0.2):
        super(Encoder_GRU_Batch0, self).__init__()
        self.embedding = nn.Embedding(en_corpus_len, encoder_embedding_num)     # 78 100
        self.gru = nn.GRU(encoder_embedding_num, encoder_hidden_num, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    """
    batch_x
   [[ 3,  4,  5,  0,  0,  0,  0],
    [ 3,  4,  5,  0,  0,  0,  0],
    [ 6,  7,  8,  5,  0,  0,  0],
    [ 9, 10,  4, 11, 12,  0,  0],
    [ 3, 13, 14, 14, 15, 12,  0],
    [16, 17, 11, 18, 19,  5,  0],
    [16, 17, 20, 15,  8, 12,  0],
    [21, 22, 17,  8, 15, 12,  0],
    [23, 22, 13, 13, 18, 24, 12],
    [ 3, 13, 17, 18, 10,  8,  5]]
    batch_x_sorted
   [[23, 22, 13, 13, 18, 24, 12],
    [ 3, 13, 17, 18, 10,  8,  5],
    [21, 22, 17,  8, 15, 12,  0],
    [16, 17, 20, 15,  8, 12,  0],
    [ 3, 13, 14, 14, 15, 12,  0],
    [16, 17, 11, 18, 19,  5,  0],
    [ 9, 10,  4, 11, 12,  0,  0],
    [ 6,  7,  8,  5,  0,  0,  0],
    [ 3,  4,  5,  0,  0,  0,  0],
    [ 3,  4,  5,  0,  0,  0,  0]]
    """
    # batch_x:[batch_size, setence_len] torch.Size([10, 7])  lengths : [batch_size] tensor([3, 3, 4, 5, 6, 6, 6, 6, 7, 7], device='cuda:0')
    def forward(self, batch_x, lengths):
        # 基于每个batch中句子的实际长度倒序（后续使用pad_packed_sequence要求句子长度需要倒排序）
        sorted_lengths, sorted_index = lengths.sort(0, descending=True) # sorted_index : tensor([8, 9, 7, 6, 4, 5, 3, 2, 1, 0], device='cuda:0')
        batch_x_sorted = batch_x[sorted_index.long()]   # torch.Size([10, 7])

        embed = self.embedding(batch_x_sorted)  # [batch_size, sentence_len, hidden_size] torch.Size([10, 7, 50])
        embed = self.dropout(embed)

        """ 采用 PackedSequence 方式  packed_embed packed_out 都是PackedSequence类型"""
        # 将句子末尾添加的padding去掉，使得GRU只对实际有效语句进行编码
        sorted_lengths_np = sorted_lengths.long().cpu().data.numpy()
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths_np, batch_first=True)
        packed_out, hidden = self.gru(packed_embed)  # packed_out为PackedSequence类型数据，hidden[1, batch_size, hidden_size]

        # unpacked，恢复数据为tensor tensor([7, 7, 6, 6, 6, 6, 5, 4, 3, 3])
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [batch_size, sentence_len, hidden_size] torch.Size([10, 7, 100])

        # 恢复batch中sentence原始的顺序
        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()
        hidden = hidden[:, original_index.long()].contiguous()

        return out, hidden

# seq2seq的编码器(双向gru对源语言进行编码)
class Encoder_GRU_Batch0_bid_att(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len, decoder_hidden_num, dropout=0.2,directions=1):
        super(Encoder_GRU_Batch0_bid_att, self).__init__()
        self.embedding = nn.Embedding(en_corpus_len, encoder_embedding_num)     # 6224 50
        self.gru = nn.GRU(encoder_embedding_num, encoder_hidden_num, batch_first=True, bidirectional=(directions == 2))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(encoder_hidden_num * 2, decoder_hidden_num)   # in_features out_features

    # torch.Size([10, 3]) tensor([2, 2, 2, 2, 2, 3, 3, 3, 2, 3], device='cuda:0')
    def forward(self, batch_x, lengths):
        # batch_x: [batch_size, max_x_setence_len]
        # lengths: [batch_size]

        # 基于每个batch中句子的实际长度倒序（后续使用pad_packed_sequence要求句子长度需要倒排序）
        sorted_lengths, sorted_index = lengths.sort(0, descending=True)  # sorted_index:tensor([7, 6, 5, 9, 1, 0, 8, 4, 2, 3], device='cuda:0')
        batch_x_sorted = batch_x[sorted_index.long()]

        embed = self.embedding(batch_x_sorted)  # [batch_size, max_x_sentence_len, embed_size]  torch.Size([10, 3, 50])
        embed = self.dropout(embed) # torch.Size([10, 3, 50])

        # 将句子末尾添加的padding去掉，使得GRU只对实际有效语句进行编码
        sorted_lengths_np = sorted_lengths.long().cpu().data.numpy()
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths_np, batch_first=True)
        packed_out, hidden = self.gru(packed_embed)  # packed_out为PackedSequence类型数据，hidden为tensor类型:[2, batch_size, enc_hidden_size]

        # unpacked，恢复数据为tensor
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out,
                                                  batch_first=True)  # [batch_size, max_x_sentence_len, enc_hidden_size * 2]

        # 恢复batch中sentence原始的顺序
        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()   # torch.Size([10, 3, 200])
        hidden = hidden[:, original_index.long()].contiguous()  # torch.Size([2, 10, 100])
        # torch.Size([1, 10, 100])
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)  # [batch_size, enc_hidden_size*2] torch.Size([10, 200])

        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)  # [1, batch_size, dec_hidden_size] torch.Size([1, 10, 100])

        return out, hidden  # [batch_size, max_x_sentence_len, enc_hidden_size*2], [1, batch_size, dec_hidden_size]









