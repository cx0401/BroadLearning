import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import *
import utils.utils as utils
from predata.datapre import ImdbData
import time
from net.RNNet import ImdbNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class imdb_train():
    def __init__(self):
        args = utils.parse_args()
        vocab_size = args.vocab_size  # imdb’s vocab_size 即词汇表大小
        seq_len = args.seq_len      # max length
        self.batch_size = args.batch_size
        embedding_dim = args.embedding_dim   # embedding size
        hidden_dim = args.hidden_dim   # lstm hidden size
        DROPOUT = 0.2
        self.epochs = args.epochs
        self.data = ImdbData(batchsize=self.batch_size,seq_len=args.seq_len,out_dim=args.out_dim)
        self.model_name = "model/Imdb/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
            args.hidden_dim) + "_Seq" + str(args.seq_len) + ".pth" # 定义模型保存路径
        self.model = ImdbNet(vocab_size, embedding_dim, hidden_dim, DROPOUT).to(device)

    def train(self):   # 训练模型
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
            print('Epoch{}Train Accuracy: ({:.0f}%)'.format(epoch, 100. * acc / len(train_loader.dataset)))
        torch.save(self.model.state_dict(), self.model_name)
        print("train time:", time.time()-start)

    def test(self):    # 测试模型
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
            pred = y_.max(-1, keepdim=True)[1]   # .max() 2输出，分别为最大值和最大值的index
            acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, acc, len(test_loader.dataset),
            100. * acc / len(test_loader.dataset)))
        print("test time:", time.time()-start)

if __name__=='__main__':
    x = imdb_train()
    x.train()
    x.test()

