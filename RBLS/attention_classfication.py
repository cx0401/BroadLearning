import torch
import numpy as np
from net.RNNet import CorpusNet
from torch.autograd import Variable
from utils.utils import parse_args
import time
from tqdm import tqdm
from predata.datapre import CorpusData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader(object):
    def __init__(self, src_sents, label, max_len, cuda=True,
                 batch_size=64, shuffle=True, evaluation=False):
        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation

        self._batch_size = batch_size
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts, max_len):
            inst_data = np.array([inst + [0] * (max_len - len(inst)) for inst in insts])

            inst_data_tensor = Variable(torch.from_numpy(inst_data), volatile=self.evaluation)
            inst_data_tensor = inst_data_tensor.to(device)
            return inst_data_tensor

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step * self._batch_size
        _bsz = self._batch_size
        self._step += 1
        data = pad_to_longest(self._src_sents[_start:_start + _bsz], self._max_len)
        label = Variable(torch.from_numpy(self._label[_start:_start + _bsz]),
                         volatile=self.evaluation)
        label = label.to(device)

        return data, label


class CorpusTrain():
    def __init__(self):
        args = parse_args()
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        seed = 1111
        cuda_able = True
        dropout = 0.5
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        bidirectional = args.bidirectional
        attention_size = args.attention_size
        seq_len = args.seq_len
        torch.manual_seed(seed)
        if args.use_attention:
            PATH = "model/corpus_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
                args.hidden_dim) + "seq_len" + str(args.seq_len) + "attention_size" + str(args.attention_size)
        else:
            PATH = "model/corpus_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
                args.hidden_dim) + "seq_len" + str(args.seq_len)
        if bidirectional:
            PATH += "bi"
        PATH += ".pth"
        self.PATH = PATH
        datafile = './data/corpus.pt'
        self.data = torch.load(datafile)
        self.lstm_attn = CorpusNet(out_dim=self.data['dict']['label_size'], hidden_dim=hidden_dim,
                                   vocab_size=self.data['dict']['vocab_size'], embedding_dim=embedding_dim,
                                   bidirectional=bidirectional, dropout=dropout, attention_size=attention_size,
                                   seq_len=seq_len,
                                   use_attention=args.use_attention).to(device)

    def train(self):
        start = time.time()
        optimizer = torch.optim.Adam(self.lstm_attn.parameters(), lr=0.001, weight_decay=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        training_data = DataLoader(self.data['train']['src'], self.data['train']['label'],
                                   self.data["max_len"], batch_size=self.batch_size)
        self.lstm_attn.train()

        for epoch in range(self.epochs):
            total_loss = 0

            for data, label in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
                optimizer.zero_grad()

                target = self.lstm_attn(data)
                loss = criterion(target, label)

                loss.backward()
                optimizer.step()

                total_loss += loss.data
            loss = total_loss / training_data.sents_size
            print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - start, loss))

    def test(self):
        start = time.time()
        self.lstm_attn.eval()
        criterion = torch.nn.CrossEntropyLoss()
        corrects = eval_loss = 0
        validation_data = DataLoader(self.data['valid']['src'], self.data['valid']['label'],
                                     self.data['max_len'], batch_size=self.batch_size, shuffle=False, )
        _size = validation_data.sents_size

        for data, label in tqdm(validation_data, mininterval=0.2,
                                desc='Evaluate Processing', leave=False):
            pred = self.lstm_attn(data)
            loss = criterion(pred, label)

            eval_loss += loss.data
            corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
        loss, corrects, acc, size = eval_loss / _size, corrects, corrects * 100.0 / _size, _size
        print('| time: {:2.2f}s | loss {:.4f} | accuracy {}%({}/{})'.format(time.time() - start, loss, acc, corrects,
                                                                            size))
        torch.save(self.lstm_attn.state_dict(), self.PATH)


if __name__ == '__main__':
    x = CorpusTrain()
    x.train()
    x.test()
