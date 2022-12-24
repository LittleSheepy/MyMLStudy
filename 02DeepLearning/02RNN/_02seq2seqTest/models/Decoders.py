import torch
from torch import nn
import torch.nn.functional as F

# LSTM batch_first
class Decoder_LSTM_Batch0(nn.Module):
    def __init__(self, decoder_embedding_num, decoder_hidden_num, ch_corpus_len):   # 150 100 2698
        super().__init__()
        self.embedding = nn.Embedding(ch_corpus_len, decoder_embedding_num)     # 2698 150
        self.lstm = nn.LSTM(decoder_embedding_num, decoder_hidden_num, batch_first=True)
    # decoder_input     : torch.Size([10, 6])    [[1,3,4,2,0,0],[1,5,6,4,2,0]]
    # hidden            : <tuple> (torch.Size([1, 10, 100]),torch.Size([1, 10, 100]))
    def forward(self, decoder_input, hidden):
        embedding = self.embedding(decoder_input)   # torch.Size([10, 6, 150])
        decoder_output, decoder_hidden = self.lstm(embedding, hidden)
        # decoder_output    :torch.Size([2, 6, 100])
        # decoder_hidden    :(torch.Size([1, 10, 100]),torch.Size([1, 10, 100]))
        return decoder_output, decoder_hidden

# GRU
class Decoder_GRU(nn.Module):
    def __init__(self, decoder_embedding_num, decoder_hidden_num, ch_corpus_len):   # 150 100 2698
        super(Decoder_GRU, self).__init__()
        self.embedding = nn.Embedding(ch_corpus_len, decoder_embedding_num)     # 2698 150
        self.gru = nn.GRU(decoder_embedding_num, decoder_hidden_num)
        self.out = nn.Linear(decoder_hidden_num, ch_corpus_len)  # 线性层
        self.softmax = nn.LogSoftmax(dim=1)  # LogSoftmax函数
        self.output_dim = ch_corpus_len

    def forward(self, input, hidden):
        # reshape the input to (1, batch_size)
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden

# seq2seq的解码器 test4
class Decoder_GRU_Batch0(nn.Module):
    def __init__(self, decoder_embedding_num, decoder_hidden_num, ch_corpus_len, dropout=0.2):   # 150 100 2698
        super(Decoder_GRU_Batch0, self).__init__()
        self.embedding = nn.Embedding(ch_corpus_len, decoder_embedding_num)     # 2698 150
        self.gru = nn.GRU(decoder_embedding_num, decoder_hidden_num, batch_first=True)
        self.liner = nn.Linear(decoder_hidden_num, ch_corpus_len)
        self.dropout = nn.Dropout(dropout)
    """
    batch_y
       [[ 1,  3,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  5,  6,  4,  2,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  5,  7,  8,  9, 10, 10, 10, 10, 10, 10,  4],
        [ 1, 11, 11, 12,  2,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  5,  6,  4,  2,  0,  0,  0,  0,  0,  0,  0],
        [ 1, 13, 14, 15,  4,  2,  0,  0,  0,  0,  0,  0],
        [ 1, 14, 16, 17,  4,  2,  0,  0,  0,  0,  0,  0],
        [ 1, 18, 19, 20,  4,  2,  0,  0,  0,  0,  0,  0],
        [ 1, 21, 22, 23,  2,  0,  0,  0,  0,  0,  0,  0],
        [ 1, 24,  8, 17,  4,  2,  0,  0,  0,  0,  0,  0]]
    """
    # batch_y: [batch_size, setence_len] torch.Size([10, 12])
    # lengths: [batch_size] tensor([ 3,  4, 12,  4,  4,  5,  5,  5,  4,  5], device='cuda:0') size=10
    # encoder_hidden: [1, batch_size, hidden_size] torch.Size([1, 10, 100])
    def forward(self, batch_y, lengths, encoder_hidden):
        # 基于每个batch中句子的实际长度倒序
        sorted_lengths, sorted_index = lengths.sort(0, descending=True)
        batch_y_sorted = batch_y[sorted_index.long()]
        hidden = encoder_hidden[:, sorted_index.long()]

        embed = self.embedding(batch_y_sorted)  # [batch_size, setence_len, hidden_size]
        embed = self.dropout(embed)
        sorted_lengths_np = sorted_lengths.long().cpu().data.numpy()
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths_np, batch_first=True)
        # 解码器的输入为编码器的输出，上一个词，然后预测下一个词
        packed_out, hidden = self.gru(packed_embed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()
        hidden = hidden[:, original_index.long()].contiguous()
        out = self.liner(out)  # [batch_size, sentence_len, vocab_size]

        # log_softmax求出每个输出的概率分布，最大概率出现的位置就是预测的词在词表中的位置
        out = F.log_softmax(out, dim=-1)  # [batch_size, sentence_len, vocab_size]
        return out, hidden


# attention(计算源语言编码与目标语言编码中每个词间的相似度)
class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.enc_hidden_size = encoder_hidden_size
        self.dec_hidden_size = decoder_hidden_size

        self.linear_in = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size, bias=False)
        self.linear_out = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size, decoder_hidden_size)

    def forward(self, output, context, masks):
        # output [batch_size, max_y_sentence_len, dec_hidden_size]
        # context [batch_size, max_x_sentence_len, enc_hidden_size*2] torch.Size([10, 3, 200])
        # masks [batch_size, max_y_sentence_len, max_x_sentence_len]

        batch_size = output.size(0) # 10
        y_len = output.size(1)  # 8
        x_len = context.size(1) # 3

        x = context.view(batch_size * x_len, -1)  # [batch_size * max_x_sentence_len, enc_hidden_size*2] torch.Size([30, 200])
        x = self.linear_in(x)  # [batch_size * max_x_len, dec_hidden_size]  torch.Size([30, 100])

        context_in = x.view(batch_size, x_len, -1)  # [batch_size, max_x_sentence_len, dec_hidden_size] torch.Size([10, 3, 100])
        context_in_trans = context_in.transpose(1, 2) # torch.Size([10, 3, 100]) -> torch.Size([10, 100, 3])
        atten = torch.bmm(output, context_in_trans)  # [batch_size, max_y_sentence_len, max_x_sentence_len] torch.Size([10, 8, 3])

        atten.data.masked_fill_(masks.bool(), -1e-6)

        atten = F.softmax(atten, dim=2)  # [batch_size, max_y_sentence_len, max_x_sentence_len] torch.Size([10, 8, 3])

        # 目标语言上一个词(因为目标语言做了shift处理，所以是上一个词)的编码与源语言所有词经attention的加权，concat上目标语言上一个词的编码，目标语言当前词的预测编码
        context = torch.bmm(atten, context)  # [batch_size, max_y_sentence_len, enc_hidden_size*2] torch.Size([10, 8, 200])
        output = torch.cat((context, output),   # torch.Size([10, 8, 200])  torch.Size([10, 8, 100])
                           dim=2)  # [batch_size, max_y_sentence_len, enc_hidden_size*2+dec_hidden_size] torch.Size([10, 8, 300])

        output = output.view(batch_size * y_len,
                             -1)  # [batch_size * max_y_sentence_len, enc_hidden_size*2+dec_hidden_size] torch.Size([80, 300])
        output = torch.tanh(self.linear_out(output)) # torch.Size([80, 100])

        output = output.view(batch_size, y_len, -1)  # [batch_size, max_y_sentence_len, dec_hidden_size] torch.Size([10, 8, 100])

        return output, atten


# seq2seq的解码器
class Decoder_GRU_Batch0_bid_att(nn.Module):
    def __init__(self, decoder_embedding_num, decoder_hidden_num, ch_corpus_len, encoder_hidden_num, dropout=0.2):   # 150 100 2698
        super(Decoder_GRU_Batch0_bid_att, self).__init__()
        self.embedding = nn.Embedding(ch_corpus_len, decoder_embedding_num)     # 2698 150
        self.attention = Attention(encoder_hidden_num, decoder_hidden_num)    # 100 200
        self.gru = nn.GRU(decoder_embedding_num, decoder_hidden_num, batch_first=True)
        self.liner = nn.Linear(decoder_hidden_num, ch_corpus_len)
        self.dropout = nn.Dropout(dropout)

    def create_atten_masks(self, x_len, y_len,device):
        # 创建attention的masks
        # 超出句子有效长度部分的attention用一个很小的数填充，使其在softmax后的权重很小
        max_x_len = x_len.max() # 3
        max_y_len = y_len.max() # 8
        """
        [[0, 1, 2]]
        x_len : [2, 2, 2, 2, 2, 3, 3, 3, 2, 3]
        x_len[:, None] : 
        [[2],
        [2],
        [2],
        [2],
        [2],
        [3],
        [3],
        [3],
        [2],
        [3]]
        
        [[ True,  True, False],
        [ True,  True, False],
        [ True,  True, False],
        [ True,  True, False],
        [ True,  True, False],
        [ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True],
        [ True,  True, False],
        [ True,  True,  True]]
        """
        x_masks = torch.arange(max_x_len, device=device)[None, :] < x_len[:, None]  # [batch_size, max_x_sentence_len] torch.Size([10, 3])
        """
        y_len: tensor([3, 3, 8, 3, 3, 5, 5, 4, 3, 5], device='cuda:0')
        
        [[ True,  True,  True, False, False, False, False, False],
        [ True,  True,  True, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True],
        [ True,  True,  True, False, False, False, False, False],
        [ True,  True,  True, False, False, False, False, False],
        [ True,  True,  True,  True,  True, False, False, False],
        [ True,  True,  True,  True,  True, False, False, False],
        [ True,  True,  True,  True, False, False, False, False],
        [ True,  True,  True, False, False, False, False, False],
        [ True,  True,  True,  True,  True, False, False, False]]
        """
        y_masks = torch.arange(max_y_len, device=device)[None, :] < y_len[:, None]  # [batch_size, max_y_sentence_len] torch.Size([10, 8])

        # x_masks[:, :, None] [batch_size, max_x_sentence_len, 1]
        # y_masks[:, None, :][batch_size, 1, max_y_sentence_len]
        # masked_fill_填充的是True所在的维度，所以取反(~)
        """
        torch.Size([10, 8, 1]) * torch.Size([10, 1, 3])
        8*1
        [[ True],
        [ True],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [False]]
        
        x_masks[:, None, :] : torch.Size([10, 1, 3])
        1*3
        [[ True,  True, False]]
        
        mask[0]
        [[0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]
        """
        masks = (
            ~(y_masks[:, :, None] * x_masks[:, None, :])).byte()  # [batch_size, max_y_sentence_len, max_x_sentence_len]

        return masks  # [batch_size, max_y_sentence_len, max_x_sentence_len]
    # torch.Size([10, 3, 200])
    def forward(self, encoder_out, x_lengths, batch_y, y_lengths, encoder_hidden):
        # batch_y: [batch_size, max_x_setence_len]
        # lengths: [batch_size]
        # encoder_hidden: [1, batch_size, dec_hidden_size*2]

        # 基于每个batch中句子的实际长度倒序
        sorted_lengths, sorted_index = y_lengths.sort(0, descending=True)
        batch_y_sorted = batch_y[sorted_index.long()]
        hidden = encoder_hidden[:, sorted_index.long()]

        embed = self.embedding(batch_y_sorted)  # [batch_size, max_x_setence_len, embed_size]
        embed = self.dropout(embed)

        #packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(),
        #if sorted_lengths
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(), batch_first=True)
        # 目标语言编码，h0为编码器中最后一个unit输出的hidden
        packed_out, hidden = self.gru(packed_embed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()  # [batch_size, max_y_sentence_len, dec_hidden_size]
        hidden = hidden[:, original_index.long()].contiguous()  # [1, batch_size, dec_hidden_size] torch.Size([1, 10, 100])
        """ Attention """
        """
        [[[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 1]],

        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]]
        """
        atten_masks = self.create_atten_masks(x_lengths,        # tensor([2, 2, 2, 2, 2, 3, 3, 3, 2, 3], device='cuda:0')
                                              y_lengths,        # tensor([3, 3, 8, 3, 3, 5, 5, 4, 3, 5], device='cuda:0')
                                              encoder_out.get_device())  # [batch_size, max_y_sentcnec_len, max_x_sentcnec_len] torch.Size([10, 8, 3])

        out, atten = self.attention(out, encoder_out,   # torch.Size([10, 8, 100])  torch.Size([10, 3, 200])
                                    atten_masks         # torch.Size([10, 8, 3])
                                    )  # out [batch_size, max_y_sentence_len, dec_hidden_size] torch.Size([10, 8, 100])
        """ Attention """
        out = self.liner(out)  # [batch_size, cn_sentence_len, vocab_size]

        # log_softmax求出每个输出的概率分布，最大概率出现的位置就是预测的词在词表中的位置
        out = F.log_softmax(out, dim=-1)  # [batch_size, cn_sentence_len, vocab_size]
        return out, hidden
