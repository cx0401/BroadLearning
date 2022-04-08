import torch
import torchtext


class imdbdata():
    def __init__(self, batchsize=0, seq_len=200, out_dim=3):
        self.TEXT = torchtext.legacy.data.Field(lower=True, fix_length=seq_len, batch_first=True)  # 每篇提取200个单词
        self.LABEL = torchtext.legacy.data.Field(sequential=False)
        self.embeding_dim = 10
        self.out_dim = out_dim
        self.hidden_size = 300
        self.seq_len = seq_len
        self.train, self.test = torchtext.legacy.datasets.IMDB.splits(self.TEXT, self.LABEL)
        # train, test[25000]
        self.TEXT.build_vocab(self.train, max_size=10000, min_freq=10, vectors=None)
        self.LABEL.build_vocab(self.train)
        self.train_iter, self.test_iter = torchtext.legacy.data.BucketIterator.splits((self.train, self.test),
                                                                                      batch_size=batchsize)
        self.vocab_size = len(self.TEXT.vocab.stoi)


    def data_convert(self):
        self.train_iter, self.test_iter = torchtext.legacy.data.BucketIterator.splits((self.train, self.test),
                                                                                      batch_size=len(self.train.examples))
        for i in self.train_iter:
            x, y = i.text, i.label
        y = y - 1
        y = torch.unsqueeze(y, 1)
        y = torch.zeros(y.shape[0], self.out_dim).scatter_(1, y, 1)
        self.train_x = x.detach().numpy()
        self.train_y = y.detach().numpy()
        for i in self.test_iter:
            x, y = i.text, i.label
        y = y - 1
        y = torch.unsqueeze(y, 1)
        y = torch.zeros(y.shape[0], self.out_dim).scatter_(1, y, 1)
        self.test_x = x.detach().numpy()
        self.test_y = y.detach().numpy()

