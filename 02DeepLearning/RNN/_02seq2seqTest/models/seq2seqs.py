import torch
from torch import nn
from models.Encoders import Encoder_LSTM_Batch0, Encoder_GRU, Encoder_GRU_Batch0, Encoder_GRU_Batch0_bid_att
from models.Decoders import Decoder_LSTM_Batch0, Decoder_GRU, Decoder_GRU_Batch0, Decoder_GRU_Batch0_bid_att

# test5
class Seq2SeqSimple(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len,    # 50  100 78
                 decoder_embedding_num, decoder_hidden_num, ch_corpus_len):         # 150 100 2698
        super().__init__()
        self.encoder = Encoder_LSTM_Batch0(encoder_embedding_num, encoder_hidden_num, en_corpus_len)    # 50  100 78
        self.decoder = Decoder_LSTM_Batch0(decoder_embedding_num, decoder_hidden_num, ch_corpus_len)    # 150 100 2698
        self.classifier = nn.Linear(decoder_hidden_num, ch_corpus_len)  # 250 2698
        self.cross_loss = nn.CrossEntropyLoss()
    """
        [
        [ 1,  3,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  5,  6,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  5,  7,  8,  9, 10, 10, 10, 10, 10, 10,  4,  2],
        [ 1, 11, 11, 12,  2,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  5,  6,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 1, 13, 14, 15,  4,  2,  0,  0,  0,  0,  0,  0,  0],
        [ 1, 14, 16, 17,  4,  2,  0,  0,  0,  0,  0,  0,  0],
        [ 1, 18, 19, 20,  4,  2,  0,  0,  0,  0,  0,  0,  0],
        [ 1, 21, 22, 23,  2,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 1, 24,  8, 17,  4,  2,  0,  0,  0,  0,  0,  0,  0]]
    """
    # en_index:torch.Size([10, 7]) [[1,3,4,2,0,0,0],[1,5,6,4,2,0,0],[1,5,7,8,9,4,2]]hi.   # ch_index:torch.Size([10, 7])
    def forward(self, en_index, ch_index):
        decoder_input = ch_index[:, :-1]    # torch.Size([2, 4])    [[3590, 2085,    0, 3591]]
        label = ch_index[:, 1:]             # torch.Size([2, 4])    [[2085,    0, 3591, 3589]]
        encoder_hidden = self.encoder(en_index) # encoder_hidden (torch.Size([1, 2, 100]),torch.Size([1, 2, 100]))
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden) # decoder_output:torch.Size([10, 6, 100])
        pre = self.classifier(decoder_output)   # torch.Size([10, 6, 2698])
        loss = self.cross_loss(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))   # tensor(8.2197)
        return loss

# test2
class Seq2SeqTest4(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len,    # 50  100 78
                 decoder_embedding_num, decoder_hidden_num, ch_corpus_len,          # 150 100 2698
                 num_layers=1
                 ):
        super().__init__()
        self.encoder = Encoder_GRU(encoder_embedding_num, encoder_hidden_num, en_corpus_len)    # 50  100 78
        self.decoder = Decoder_GRU(decoder_embedding_num, decoder_hidden_num, ch_corpus_len)    # 150 100 2698
        self.device = "cuda"
    """
    source
   [[ 3,  3,  6,  9,  3, 16, 16, 21, 23,  3],
    [ 4,  4,  7, 10, 13, 17, 17, 22, 22, 13],
    [ 5,  5,  8,  4, 14, 11, 20, 17, 13, 17],
    [ 0,  0,  5, 11, 14, 18, 15,  8, 13, 18],
    [ 0,  0,  0, 12, 15, 19,  8, 15, 18, 10],
    [ 0,  0,  0,  0, 12,  5, 12, 12, 24,  8],
    [ 0,  0,  0,  0,  0,  0,  0,  0, 12,  5]]
    target
   [[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 3,  5,  5, 11,  5, 13, 14, 18, 21, 24],
    [ 4,  6,  7, 11,  6, 14, 16, 19, 22,  8],
    [ 2,  4,  8, 12,  4, 15, 17, 20, 23, 17],
    [ 0,  2,  9,  2,  2,  4,  4,  4,  2,  4],
    [ 0,  0, 10,  0,  0,  2,  2,  2,  0,  2],
    [ 0,  0, 10,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0, 10,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0, 10,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0, 10,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0, 10,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  4,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  2,  0,  0,  0,  0,  0,  0,  0]]
    """
    # source:torch.Size([7, 10])
    # target:torch.Size([13, 10])
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        SOS_token = 0  # 开始标志
        input_length = source.size(0)  # 获取输入的长度
        batch_size = target.shape[1]
        target_length = target.shape[0]         # 13
        vocab_size = self.decoder.output_dim    # 2698
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)    # torch.Size([13, 10, 2698])
        # 为语句中的每一个word编码
        for i in range(input_length):   # source[0] tensor([ 3,  3,  6,  9,  3, 16, 16, 21, 23,  3], device='cuda:0')
            encoder_output, encoder_hidden = self.encoder(source[i])    # encoder_output: torch.Size([1, 10, 100]) ***** encoder_hidden: torch.Size([1, 10, 100])
        # 使用encoder的hidden层作为decoder的hidden层
        decoder_hidden = encoder_hidden.to(self.device)     # torch.Size([1, 10, 100])
        # 在预测前，添加一个token
        decoder_input = torch.tensor([SOS_token], device=device)
        # 获取list中的top_k
        # 根据当前的target，预测output word
        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if (teacher_force == False and input.item() == EOS_token):
                break
        return outputs

class Seq2Seq_GRU_Batch0(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len,    # 50  100 78
                 decoder_embedding_num, decoder_hidden_num, ch_corpus_len,          # 150 100 2698
                 num_layers=1
                 ):
        super().__init__()
        self.encoder = Encoder_GRU_Batch0(encoder_embedding_num, encoder_hidden_num, en_corpus_len)    # 50  100 78
        self.decoder = Decoder_GRU_Batch0(decoder_embedding_num, decoder_hidden_num, ch_corpus_len)    # 150 100 2698

    def forward(self, x, x_lengths, y, y_lengths):
        decoder_input = y[:, :-1]    # torch.Size([2, 4])    [[3590, 2085,    0, 3591]]
        label = y[:, 1:]             # torch.Size([2, 4])    [[2085,    0, 3591, 3589]]
        y_lengths = y_lengths - 1
        y_lengths[y_lengths <= 0] = 1
        encoder_out, encoder_hid = self.encoder(x, x_lengths)  # 英文编码
        output, hidden = self.decoder(decoder_input, y_lengths, encoder_hid)  # 解码出中文   torch.Size([10, 10, 2698])
        # 计算loss
        # mask
        y_lengths = y_lengths.unsqueeze(1)  # [batch_size, 1]
        batch_target_masks = torch.arange(y_lengths.max().item(), device=y.get_device()) < y_lengths
        batch_target_masks = batch_target_masks.float()

        loss = self.loss(output, label, batch_target_masks)

        return output, loss

    def translate(self, x, x_lengths, y, max_length=50):
        # 翻译en2cn
        # max_length表示翻译的目标句子可能的最大长度
        encoder_out, encoder_hidden = self.encoder(x, x_lengths)  # 将输入的英文进行编码
        predicts = []
        batch_size = x.size(0)
        # 目标语言（中文）的输入只有”BOS“表示句子开始，因此y的长度为1
        # 每次都用上一个词(y)与编码器的输出预测下一个词，因此y的长度一直为1
        y_length = torch.ones(batch_size).long().to(y.device)
        for i in range(max_length):
            # 每次用上一次的输出y和编码器的输出encoder_hidden预测下一个词
            output, hidden = self.decoder(y, y_length, encoder_hidden)
            # output: [batch_size, 1, vocab_size]

            # output.max(2)[1]表示找出output第二个维度的最大值所在的位置（即预测词在词典中的index）
            y = output.max(2)[1].view(batch_size, 1)  # [batch_size, 1]
            predicts.append(y)
        predicts = torch.cat(predicts, 1)  # [batch_size, max_length]
        return predicts

    def loss(self, predicts, targets, masks):
        # predicts [batch_size, sentence_len, vocab_size]
        # target [batch_size, sentence_len]
        # masks [batch_size, sentence_len]
        predicts = predicts.contiguous().view(-1, predicts.size(2))  # [batch_size * sentence_len, vocab_size] torch.Size([100, 2698])
        targets = targets.contiguous().view(-1, 1)  # [batch_size*sentence_len, 1]  torch.Size([120, 1])
        masks = masks.contiguous().view(-1, 1)  # [batch_size*sentence_len, 1]  # torch.Size([100, 1])

        # predicts.gather(1, targets)为predicts[i][targets[i]]
        # 乘上masks，即只需要计算句子有效长度的预测
        # 负号：因为采用梯度下降法，所以要最大化目标词语的概率，即最小化其相反数
        output = -predicts.gather(1, targets) * masks
        output = torch.sum(output) / torch.sum(masks)  # 平均

        return output


# seq2seq模型
# 训练时，decoder中输入了整个目标语言，decoder会先采用一个双向的gru对目标语言进行编码(其输入为shift处理后的目标语言和编码器最后一个unit输出的源语言编码)，
# 因此，训练时，decoder的某一个unit的输出是具有目标语言上下文信息的
# 实际应用时，目标语言是未知的，decoder中双向gru的输入只有上一个unit的输出和编码器最后一个unit输出的源语言编码
# 所以，训练与实际使用的decoder输入是不一致的，为什么输入不一样还能有效？？？？
class Seq2Seq(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len,    # 50  100 6224
                 decoder_embedding_num, decoder_hidden_num, ch_corpus_len           # 150 100 10476
                 ):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder_GRU_Batch0_bid_att(encoder_embedding_num, encoder_hidden_num, en_corpus_len, decoder_hidden_num, directions=2)    # 50  100 6224
        self.decoder = Decoder_GRU_Batch0_bid_att(decoder_embedding_num, decoder_hidden_num, ch_corpus_len, encoder_hidden_num)    # 150 100 2698
    # torch.Size([10, 3]) tensor([2, 2, 2, 2, 2, 3, 3, 3, 2, 3], device='cuda:0')  torch.Size([10, 9]) tensor([4, 4, 9, 4, 4, 6, 6, 5, 4, 6], device='cuda:0')
    def forward(self, x, x_lengths, y, y_lengths):
        decoder_input = y[:, :-1]    # torch.Size([10, 8])    [[3590, 2085,    0, 3591]]
        label = y[:, 1:]             # torch.Size([10, 8])    [[2085,    0, 3591, 3589]]
        y_lengths = y_lengths - 1
        y_lengths[y_lengths <= 0] = 1
        encoder_out, encoder_hid = self.encoder(x, x_lengths)  # 源语言编码
        output, hidden = self.decoder(encoder_out, x_lengths, decoder_input, y_lengths, encoder_hid)  # 解码出目标

        # 计算loss
        # mask
        y_lengths = y_lengths.unsqueeze(1)  # [batch_size, 1]
        batch_target_masks = torch.arange(y_lengths.max().item(), device=y.get_device()) < y_lengths
        batch_target_masks = batch_target_masks.float()

        loss = self.loss(output, label, batch_target_masks)
        return output, loss

    def translate(self, x, x_lengths, y, max_length=50):
        # 翻译en2cn
        # max_length表示翻译的目标句子可能的最大长度
        encoder_out, encoder_hidden = self.encoder(x, x_lengths)  # 将输入的英文进行编码
        predicts = []
        batch_size = x.size(0)
        # 目标语言（中文）的输入只有”BOS“表示句子开始，因此y的长度为1
        # 每次都用上一个词(y)与编码器的输出预测下一个词，因此y的长度一直为1
        y_length = torch.ones(batch_size).long().to(y.device)
        for i in range(max_length):
            # 每次用上一次的输出y和编码器的输出encoder_hidden预测下一个词
            output, hidden = self.decoder(encoder_out, x_lengths, y, y_length, encoder_hidden)
            # output: [batch_size, 1, vocab_size]

            # output.max(2)[1]表示找出output第二个维度的最大值所在的位置（即预测词在词典中的index）
            y = output.max(2)[1].view(batch_size, 1)  # [batch_size, 1]
            predicts.append(y)

        predicts = torch.cat(predicts, 1)  # [batch_size, max_length]

        return predicts

    def loss(self, predicts, targets, masks):
        # predicts [batch_size, sentence_len, vocab_size]
        # target [batch_size, sentence_len]
        # masks [batch_size, sentence_len]
        predicts = predicts.contiguous().view(-1, predicts.size(2))  # [batch_size * sentence_len, vocab_size] torch.Size([100, 2698])
        targets = targets.contiguous().view(-1, 1)  # [batch_size*sentence_len, 1]  torch.Size([120, 1])
        masks = masks.contiguous().view(-1, 1)  # [batch_size*sentence_len, 1]  # torch.Size([100, 1])

        # predicts.gather(1, targets)为predicts[i][targets[i]]
        # 乘上masks，即只需要计算句子有效长度的预测
        # 负号：因为采用梯度下降法，所以要最大化目标词语的概率，即最小化其相反数
        output = -predicts.gather(1, targets) * masks
        output = torch.sum(output) / torch.sum(masks)  # 平均

        return output
