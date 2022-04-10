import time
import torch
import numpy as np
from predata.datapre import TwitterData, ImdbData, ReutersData, CorpusData
import utils.utils as utils
from net.RNNet import TwitterNet, ImdbNet, ReutersNet, CorpusNet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
from torch.autograd import Variable
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwitterTrain():
    def __init__(self):
        args = utils.TwitterUtils()
        self.batch_size = args.batch_size
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        n_layers = args.n_layers
        self.seq_length = args.seq_len
        self.epochs = args.epochs
        self.lstm_data = TwitterData()
        vocab_size = len(self.lstm_data.vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
        self.net = TwitterNet(vocab_size, self.lstm_data.out_dim, embedding_dim, hidden_dim, n_layers)
        self.model_name = "model/Twitter/Train" + str(args.epochs) + "_Embed" + str(
            args.embedding_dim) + "_Hidden" + str(
            args.hidden_dim) + "_Seq" + str(args.seq_len) + ".pth"
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        clip = 5  # gradient clipping
        print_every = 100
        lr = 0.001
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        train_data = TensorDataset(torch.from_numpy(self.lstm_data.train_x), torch.from_numpy(self.lstm_data.train_y))
        valid_data = TensorDataset(torch.from_numpy(self.lstm_data.val_x), torch.from_numpy(self.lstm_data.val_y))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size)
        self.net.to(device)
        time_start = time.time()
        self.net.train()
        # train for some number of epochs
        counter = 0
        for e in range(self.epochs):
            # initialize hidden state
            h = self.net.init_hidden(self.batch_size)
            # batch loop
            for inputs, labels in train_loader:
                counter += 1
                inputs, labels = inputs.to(device), labels.to(device)
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])
                # zero accumulated gradients
                self.net.zero_grad()
                # get the output from the model
                output, h = self.net(inputs, h)
                # calculate the loss and perform backprop
                loss = self.criterion(output, labels)
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                optimizer.step()
                # loss stats
                num_correct = 0
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = self.net.init_hidden(self.batch_size)
                    val_losses = []
                    self.net.eval()
                    for inputs, labels in valid_loader:
                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        inputs, labels = inputs.to(device), labels.to(device)

                        output, val_h = self.net(inputs, val_h)
                        val_loss = self.criterion(output, labels)

                        val_losses.append(val_loss.item())
                        pred = torch.argmax(output, axis=1)
                        num_correct += (pred == labels).sum().item()

                    self.net.train()
                    print("Epoch: {}/{}...".format(e + 1, self.epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)),
                          "ACC: {:.6f}".format(num_correct / len(valid_loader.dataset))
                          )
        print("train over,time:", time.time() - time_start)
        torch.save(obj=self.net.state_dict(), f=self.model_name)

    def test(self):
        time_start = time.time()
        test_losses = []  # track loss
        num_correct = 0
        test_data = TensorDataset(torch.from_numpy(self.lstm_data.test_x), torch.from_numpy(self.lstm_data.test_y))
        test_loader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size)
        # init hidden state
        h = self.net.init_hidden(self.batch_size)

        self.net.eval()
        # iterate over test data
        for inputs, labels in test_loader:
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            inputs, labels = inputs.to(device), labels.to(device)

            # get predicted outputs
            output, h = self.net(inputs, h)

            # calculate loss
            test_loss = self.criterion(output, labels)
            test_losses.append(test_loss.item())

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.argmax(output, axis=1)
            num_correct += (pred == labels).sum().item()

        # -- stats! -- ##
        # avg test loss
        print("Test over, time:", time.time() - time_start)
        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        # accuracy over all test data
        test_acc = num_correct / len(test_loader.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))

    def word_test(self):
        test_review = "@AmericanAir you have my money, you change my flight, and don't answer your phones! Any other suggestions so I can make my commitment??"
        punctuation = self.lstm_data.punctuation

        def tokenize_review(test_review):
            test_review = test_review.lower()  # lowercase
            # get rid of punctuation
            test_text = ''.join([c for c in test_review if c not in punctuation])
            # splitting by spaces
            test_words = test_text.split()
            # get rid of web address, twitter id, and digit
            new_text = []
            for word in test_words:
                if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
                    new_text.append(word)
            # tokens
            test_ints = []
            test_ints.append([self.lstm_data.vocab_to_int[word] for word in new_text])
            return test_ints

        # test code and generate tokenized review
        test_ints = tokenize_review(test_review)
        print(test_ints)
        # test sequence padding
        features = self.lstm_data.pad_features(test_ints, self.seq_length)
        print(features)
        # test conversion to tensor and pass into your model
        feature_tensor = torch.from_numpy(features)
        print(feature_tensor.size())

        def predict(net, test_review, sequence_length=30):
            net.eval()
            # tokenize review
            test_ints = tokenize_review(test_review)
            # pad tokenized sequence
            seq_length = sequence_length
            features = self.lstm_data.pad_features(test_ints, seq_length)
            # convert to tensor to pass into your model
            feature_tensor = torch.from_numpy(features)
            batch_size = feature_tensor.size(0)
            # initialize hidden state
            h = net.init_hidden(batch_size)
            feature_tensor = feature_tensor.to(device)
            # get the output from the model
            output, h = net(feature_tensor, h)
            # convert output probabilities to predicted class (0 or 1)
            pred = torch.argmax(output, axis=1)
            # print custom response
            if (pred == 1):
                print("Non-negative review detected.")
            else:
                print("Negative review detected.")

        seq_length = 30  # good to use the length that was trained on
        # call function on negative review
        test_review_neg = "@AmericanAir you have my money, you change my flight, and don't answer your phones! Any other suggestions so I can make my commitment??"
        predict(self.net, test_review_neg, seq_length)
        # call function on positive review
        test_review_pos = "@AmericanAir thank you we got on a different flight to Chicago."
        predict(self.net, test_review_pos, seq_length)
        # call function on neutral review
        test_review_neu = "@AmericanAir i need someone to help me out"
        predict(self.net, test_review_neu, seq_length)


class ImdbTrain():
    def __init__(self):
        args = utils.ImdbUtils()
        vocab_size = args.vocab_size  # imdb’s vocab_size 即词汇表大小
        seq_len = args.seq_len  # max length
        self.batch_size = args.batch_size
        embedding_dim = args.embedding_dim  # embedding size
        hidden_dim = args.hidden_dim  # lstm hidden size
        DROPOUT = 0.2
        self.epochs = args.epochs
        self.data = ImdbData(batchsize=self.batch_size, seq_len=args.seq_len)
        self.model_name = "model/Imdb/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
            args.hidden_dim) + "_Seq" + str(args.seq_len) + ".pth"  # 定义模型保存路径
        self.model = ImdbNet(vocab_size, embedding_dim, hidden_dim, DROPOUT).to(device)

    def train(self):  # 训练模型
        start = time.time()
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        train_data = TensorDataset(torch.LongTensor(self.data.train_x), torch.LongTensor(self.data.train_y))
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        for epoch in range(self.epochs):  # 10个epoch
            acc = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_ = self.model(x)
                loss = criterion(y_, y)  # 得到loss
                pred = y_.max(-1, keepdim=True)[1]  # .max() 2输出，分别为最大值和最大值的index
                acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
                loss.backward()
                optimizer.step()
            print('Epoch:{} Train Accuracy: ({:.0f}%) Time:{}'.format(epoch, 100. * acc / len(train_loader.dataset),
                                                                      time.time() - start))
        torch.save(self.model.state_dict(), self.model_name)

    def test(self):  # 测试模型
        start = time.time()
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加loss
        test_loss = 0.0
        acc = 0
        test_data = TensorDataset(torch.LongTensor(self.data.test_x), torch.LongTensor(self.data.test_y))
        test_sampler = SequentialSampler(test_data)
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_ = self.model(x)
            test_loss += criterion(y_, y)
            pred = y_.max(-1, keepdim=True)[1]  # .max() 2输出，分别为最大值和最大值的index
            acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, acc, len(test_loader.dataset),
            100. * acc / len(test_loader.dataset)))
        print("test time:", time.time() - start)


class ReutersTrain():
    def __init__(self):
        args = utils.ReutersUtils()
        vocab_size = args.vocab_size  # imdb’s vocab_size 即词汇表大小
        seq_len = args.seq_len  # max length
        self.batch_size = args.batch_size
        embedding_dim = args.embedding_dim  # embedding size
        hidden_dim = args.hidden_dim  # lstm hidden size
        DROPOUT = 0.2
        self.epochs = args.epochs
        self.data = ReutersData(batchsize=self.batch_size, seq_len=args.seq_len, out_dim=args.out_dim)
        self.model_name = "model/Reuters/Train" + str(args.epochs) + "_Embed" + str(
            args.embedding_dim) + "_Hidden" + str(
            args.hidden_dim) + "_Seq" + str(args.seq_len) + ".pth"  # 定义模型保存路径
        self.model = ReutersNet(vocab_size, embedding_dim, hidden_dim, DROPOUT).to(device)

    def train(self):  # 训练模型
        start = time.time()
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        train_data = TensorDataset(torch.LongTensor(self.data.train_x), torch.LongTensor(self.data.train_y))
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        for epoch in range(self.epochs):  # 10个epoch
            acc = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_ = self.model(x)
                loss = criterion(y_, y)  # 得到loss
                pred = y_.max(-1, keepdim=True)[1]  # .max() 2输出，分别为最大值和最大值的index
                acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
                loss.backward()
                optimizer.step()
            print('Epoch:{} Train Accuracy: ({:.0f}%) Time:{}'.format(epoch, 100. * acc / len(train_loader.dataset),
                                                                      time.time() - start))
        torch.save(self.model.state_dict(), self.model_name)

    def test(self):  # 测试模型
        start = time.time()
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加loss
        test_loss = 0.0
        acc = 0
        test_data = TensorDataset(torch.LongTensor(self.data.test_x), torch.LongTensor(self.data.test_y))
        test_sampler = SequentialSampler(test_data)
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_ = self.model(x)
            test_loss += criterion(y_, y)
            pred = y_.max(-1, keepdim=True)[1]  # .max() 2输出，分别为最大值和最大值的index
            acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, acc, len(test_loader.dataset),
            100. * acc / len(test_loader.dataset)))
        print("test time:", time.time() - start)


class CorpusDataLoader(object):
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
        args = utils.CorpusUtils()
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
            PATH = "model/Corpus/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
                args.hidden_dim) + "_Seq" + str(args.seq_len) + "_Attention" + str(args.attention_size)
        else:
            PATH = "model/Corpus/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
                args.hidden_dim) + "_Seq" + str(args.seq_len)
        if bidirectional:
            PATH += "_bi"
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
        training_data = CorpusDataLoader(self.data['train']['src'], self.data['train']['label'],
                                   self.data["max_len"], batch_size=self.batch_size)
        self.lstm_attn.train()

        for epoch in range(self.epochs):
            total_loss = 0
            corrects = 0
            for data, label in tqdm(training_data, mininterval=1, desc='Train Processing', leave=False):
                optimizer.zero_grad()

                target = self.lstm_attn(data)
                loss = criterion(target, label)

                loss.backward()
                optimizer.step()

                total_loss += loss.data
                corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()
            acc = corrects * 100.0 / training_data.sents_size
            loss = total_loss / training_data.sents_size
            print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f} | acc {}'.format(epoch, time.time() - start,
                                                                                           loss, acc))

    def test(self):
        start = time.time()
        self.lstm_attn.eval()
        criterion = torch.nn.CrossEntropyLoss()
        corrects = eval_loss = 0
        validation_data = CorpusDataLoader(self.data['valid']['src'], self.data['valid']['label'],
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


def Twitter():
    x = TwitterTrain()
    x.train()
    x.test()
    x.word_test()


def Imdb():
    x = ImdbTrain()
    x.train()
    x.test()


def Reuters():
    x = ReutersTrain()
    x.train()
    x.test()


def Corpus():
    x = CorpusTrain()
    x.train()
    x.test()


if __name__ == '__main__':
    # Twitter()
    Imdb()
    Reuters()
    Corpus()
