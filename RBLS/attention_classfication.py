import torch
import numpy as np
from net.RNNet import corpusNet
from torch.autograd import Variable
from utils.utils import parse_args

args = parse_args()
lr = 0.001
epochs = args.epochs
batch_size = args.batch_size
seed = 1111
cuda_able = True
dropout = 0.5
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
bidirectional = args.bidirectional
weight_decay = 0.001
attention_size = args.attention_size
seq_len = args.seq_len
torch.manual_seed(seed)
use_cuda = torch.cuda.is_available() and cuda_able
if args.use_attention:
    PATH = "model/corpus_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
        args.hidden_dim) + "seq_len" + str(args.seq_len) + "attention_size" + str(args.attention_size)
else:
    PATH = "model/corpus_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
        args.hidden_dim) + "seq_len" + str(args.seq_len)
if bidirectional:
    PATH += "bi"
PATH += ".pth"
data = './data/corpus.pt'


###################################################
# load data

class DataLoader(object):
    def __init__(self, src_sents, label, max_len, cuda=True,
                 batch_size=64, shuffle=True, evaluation=False):
        self.cuda = cuda
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
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
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
        if self.cuda:
            label = label.cuda()

        return data, label


data = torch.load(data)
max_len = data["max_len"]
vocab_size = data['dict']['vocab_size']
output_size = data['dict']['label_size']

training_data = DataLoader(data['train']['src'],
                           data['train']['label'],
                           max_len,
                           batch_size=batch_size,
                           cuda=use_cuda)
validation_data = DataLoader(data['valid']['src'],
                             data['valid']['label'],
                             max_len,
                             batch_size=batch_size,
                             shuffle=False,
                             cuda=use_cuda)

###############################################
# build model

lstm_attn = corpusNet(output_size=output_size,
                      hidden_dim=hidden_dim,
                      vocab_size=vocab_size,
                      embedding_dim=embedding_dim,
                      bidirectional=bidirectional,
                      dropout=dropout,
                      use_cuda=use_cuda,
                      attention_size=attention_size,
                      seq_len=seq_len,
                      use_attention=args.use_attention)
if use_cuda:
    lstm_attn = lstm_attn.cuda()

optimizer = torch.optim.Adam(lstm_attn.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()

###################################################
# training
import time
from tqdm import tqdm

train_loss = []
valid_loss = []
accuracy = []


def evaluate():
    lstm_attn.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size

    for data, label in tqdm(validation_data, mininterval=0.2,
                            desc='Evaluate Processing', leave=False):
        pred = lstm_attn(data)
        loss = criterion(pred, label)

        eval_loss += loss.data
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
    return eval_loss / _size, corrects, corrects * 100.0 / _size, _size


def train():
    lstm_attn.train()
    total_loss = 0
    for data, label in tqdm(training_data, mininterval=1,
                            desc='Train Processing', leave=False):
        optimizer.zero_grad()

        target = lstm_attn(data)
        loss = criterion(target, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.data
    return total_loss / training_data.sents_size


#################################################
# saving
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    epoch_start_time = time.time()
    for epoch in range(1, epochs + 1):
        loss = train()
        train_loss.append(loss * 1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                              time.time() - epoch_start_time,
                                                                              loss))

        loss, corrects, acc, size = evaluate()
        valid_loss.append(loss * 1000.)
        accuracy.append(acc)

        print('-' * 10)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {}%({}/{})'.format(epoch,
                                                                                                 time.time() - epoch_start_time,
                                                                                                 loss,
                                                                                                 acc,
                                                                                                 corrects,
                                                                                                 size))
        print('-' * 10)
except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time) / 60.0))

model_state_dict = lstm_attn.state_dict()
torch.save(lstm_attn.state_dict(), PATH)