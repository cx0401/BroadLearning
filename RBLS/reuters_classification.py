import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import *
import utils.utils as utils
from predata.datapre import ReutersData
from net.RNNet import ReutersNet
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReutersTrain():
    def __init__(self):
        args = utils.parse_args()
        vocab_size = args.vocab_size  # imdb’s vocab_size 即词汇表大小
        seq_len = args.seq_len      # max length
        BATCH_SIZE = args.batch_size
        EMB_SIZE = args.embedding_dim   # embedding size
        HID_SIZE = args.hidden_dim   # lstm hidden size
        DROPOUT = 0.2
        
        epochs = args.epochs
        
        self.data = ReutersData(batchsize=BATCH_SIZE,seq_len=args.seq_len,out_dim=args.out_dim)
        self.model = ReutersNet(vocab_size, EMB_SIZE, HID_SIZE, DROPOUT).to(device)
        
        x_train, y_train, x_test, y_test = data.train_x, data.train_y, data.test_x, data.test_y
        
        train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
        test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
        
        
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
        
        test_sampler = SequentialSampler(test_data)
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    def train(self, model, device, train_loader, optimizer, epoch):   # 训练模型
        model.train()
        criterion = nn.CrossEntropyLoss()
        acc = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_ = model(x)
            loss = criterion(y_, y)  # 得到loss
            pred = y_.max(-1, keepdim=True)[1]  # .max() 2输出，分别为最大值和最大值的index
            acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
            loss.backward()
            optimizer.step()
        print('\nTrain Accuracy: ({:.0f}%)'.format(100. * acc / len(train_loader.dataset)))

    def test(self, model, device, test_loader):    # 测试模型
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加loss
        test_loss = 0.0 
        acc = 0 
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_ = model(x)
            test_loss += criterion(y_, y)
            pred = y_.max(-1, keepdim=True)[1]   # .max() 2输出，分别为最大值和最大值的index
            acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, acc, len(test_loader.dataset),
            100. * acc / len(test_loader.dataset)))
        return acc / len(test_loader.dataset)


print(model)
optimizer = optim.Adam(model.parameters())

PATH = "model/reuterskeras_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
               args.hidden_dim) + "seq_len" + str(args.seq_len) + "out_dim" + str(args.out_dim) + ".pth"  # 定义模型保存路径

train_start = time.time()
for epoch in range(epochs):  # 10个epoch
    train(model, device, train_loader, optimizer, epoch)
    print("epoch:", epoch)
print("train time:", time.time()-train_start)

# 检验保存的模型
torch.save(model.state_dict(), PATH)
test_start = time.time()
test(model, device, test_loader)
print("test time:", time.time()-test_start)
