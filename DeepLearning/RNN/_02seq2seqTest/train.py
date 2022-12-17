import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from datasets.data import MyDataset,batch_data_process
from models.seq2seqs import Seq2SeqSimple,Seq2Seq
# 自定义损失函数（mask的交叉熵损失）
# 目的：使句子中添加的padding部分不参与损失计算
class MaskCriterion(nn.Module):
    def __init__(self):
        super(MaskCriterion, self).__init__()

    def forward(self, predicts, targets, masks):
        # predicts [batch_size, sentence_len, vocab_size]
        # target [batch_size, sentence_len]
        # masks [batch_size, sentence_len]
        predicts = predicts.contiguous().view(-1, predicts.size(2))  # [batch_size * sentence_len, vocab_size]
        targets = targets.contiguous().view(-1, 1)  # [batch_size*sentence_len, 1]
        masks = masks.contiguous().view(-1, 1)  # [batch_size*sentence_len, 1]

        # predicts.gather(1, targets)为predicts[i][targets[i]]
        # 乘上masks，即只需要计算句子有效长度的预测
        # 负号：因为采用梯度下降法，所以要最大化目标词语的概率，即最小化其相反数
        output = -predicts.gather(1, targets) * masks
        output = torch.sum(output) / torch.sum(masks)  # 平均

        return output
if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder_embedding_num = 50
    encoder_hidden_num = 100
    decoder_embedding_num = 150
    decoder_hidden_num = 100

    batch_size = 10
    epoch = 40
    lr = 0.001
    filePath = "D:/02dataset/cmn_zhsim.txt"
    mydataset = MyDataset(filePath)

    ch_corpus_len = mydataset.ch_laug.n_words
    en_corpus_len = mydataset.en_laug.n_words
    dataloader = DataLoader(mydataset, batch_size, shuffle=False, collate_fn=batch_data_process)






    ###### 有长度  target去掉最后
    model = Seq2Seq(encoder_embedding_num, encoder_hidden_num, en_corpus_len,
                    decoder_embedding_num, decoder_hidden_num, ch_corpus_len)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for en_index,en_index_len, ch_index, ch_index_len in dataloader:
            en_index = torch.tensor(en_index, device=device)
            en_index_len = torch.tensor(en_index_len, device=device)

            ch_index = torch.from_numpy(np.array(ch_index)).to(device).long()
            ch_index_len = torch.tensor(np.array(ch_index_len)).to(device).long()
            # en_index = en_index.transpose(0, 1)
            # ch_index = ch_index.transpose(0, 1)
            _,loss = model(en_index,en_index_len, ch_index, ch_index_len)  # torch.Size([2, 3]) torch.Size([2, 5])

            # onnx_name = "D:/000/Seq2Seq/test5/Seq2Seq.onnx"
            # onnx_model = torch.onnx.export(model, (en_index, ch_index),
            #                                onnx_name,
            #                                opset_version=12)

            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")
    ###########seq2seq_LSTM_Batch0
    # model = Seq2SeqSimple(encoder_embedding_num, encoder_hidden_num, en_corpus_len, decoder_embedding_num, decoder_hidden_num,
    #                 ch_corpus_len)
    # model = model.to(device)
    #
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    #
    # for e in range(epoch):
    #     for en_index, ch_index in dataloader:
    #         en_index = torch.tensor(en_index, device=device)
    #         ch_index = torch.tensor(ch_index, device=device)
    #         #input = data.transpose(0, 1)
    #         loss = model(en_index, ch_index)  # torch.Size([2, 3]) torch.Size([2, 5])
    #
    #         # onnx_name = "D:/000/Seq2Seq/test5/Seq2Seq.onnx"
    #         # onnx_model = torch.onnx.export(model, (en_index, ch_index),
    #         #                                onnx_name,
    #         #                                opset_version=12)
    #
    #         loss.backward()
    #         opt.step()
    #         opt.zero_grad()
    #
    #     print(f"loss:{loss:.3f}")